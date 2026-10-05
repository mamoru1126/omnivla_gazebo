#!/usr/bin/env python3
"""OmniVLA を Gazebo で集めたデータで LoRA ファインチューニングする.

公式 vla-scripts/train_omnivla.py との対応:
  * 同じ: 全 Linear 層への LoRA (r=32, alpha=min(r,16), gaussian init), pose_projector / action_head は全パラメータ学習,
          損失 = MSE(行動) + 0.1 * 平滑化項, AdamW lr=1e-4, bf16 autocast, ラベル/プロンプト形式
  * 違い: - TRAIN_MODE フラグ不要 (公式は既定 False で勾配が流れない)
          - MBRA (外部リポジトリ + モデル) 不要: Gazebo の真値オドメトリから行動ラベルを作るため
          - 1 GPU では torchrun 不要 (torchrun でマルチ GPU DDP も可)
          - チェックポイントは LoRA アダプタ + ヘッドのみ (数百 MB)。15GB のマージ済みモデルは保存しない
            (必要なら training/merge_lora.py で後から作る)
          - gradient checkpointing (既定 ON) で RTX 3090/4090 (24GB) でも batch 2 程度で学習可能

例:
  python3 training/finetune_omnivla.py --config training/configs/finetune_gazebo.yaml
  python3 training/finetune_omnivla.py --data_dirs /data/raw/office_0 /data/raw/park_0 --max_steps 3000
  torchrun --standalone --nproc-per-node 2 training/finetune_omnivla.py --config ...
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from omnivla_nav.data_utils import AugmentConfig, GoalSamplingConfig  # noqa: E402
from omnivla_nav.omnivla_model import (FINETUNE_META, build_heads, collate, find_checkpoint_step,  # noqa: E402
                                       forward_actions, load_base_vla, num_vision_patches, read_finetune_meta)
from omnivla_nav.trajectory_io import find_trajectories  # noqa: E402
from prismatic.vla.action_tokenizer import ActionTokenizer  # noqa: E402

from common import action_loss, evaluate  # noqa: E402
from gazebo_dataset import (GazeboDatasetConfig, GazeboNavDataset, WeightedEpochSampler,  # noqa: E402
                            parse_modality_weights, split_trajectories)


@dataclass
class FinetuneConfig:
    # --- model ---
    vla_path: str = "/checkpoints/omnivla-original"   # 学習の起点 (公式チェックポイント)
    base_step: int = -1                               # -1: 自動検出 (omnivla-original は 120000)
    resume_from: str = ""                             # 本スクリプトのチェックポイントから再開 (…/step_XXXXXX)
    # --- data ---
    data_dirs: List[str] = field(default_factory=lambda: ["/data/raw"])
    val_ratio: float = 0.05
    metric_waypoint_spacing: float = 0.1
    waypoint_spacing: int = 1
    max_goal_dist: float = 30.0
    image_goal_offset: List[int] = field(default_factory=lambda: [2, 30])
    pose_goal_offset: List[int] = field(default_factory=lambda: [2, 300])
    modality_weights: Dict[str, float] = field(default_factory=lambda: {"image": 0.5, "image_pose": 0.25,
                                                                        "pose": 0.25})
    turn_sample_ratio: float = 0.5     # 学習で「この先曲がる」サンプルを引く割合 (0 で一様 = 以前と同じ)
    turn_threshold_deg: float = 45.0   # 曲がるサンプルの判定: turn_horizon フレーム以内に何度以上曲がるか
    turn_horizon: int = 10
    val_turn_ratio: float = 0.5        # 検証サンプルに含める曲がるサンプルの割合 (turn/ade で別集計)
    augment: bool = True
    crop_v: float = 0.1
    crop_h: float = 0.05
    flip_prob: float = 0.5
    color_jitter: float = 0.2
    num_workers: int = 6
    # --- optimization ---
    batch_size: int = 2
    grad_accumulation_steps: int = 4
    max_steps: int = 5000              # optimizer step 数
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    lr_warmup_steps: int = 100
    lr_decay_step: int = 0             # >0 ならこの step で lr を lr_decay_gamma 倍 (公式は 100k step で 0.1 倍)
    lr_decay_gamma: float = 0.1
    max_grad_norm: float = 1.0
    smooth_loss_weight: float = 0.1
    gradient_checkpointing: bool = True
    # --- LoRA ---
    lora_rank: int = 32
    lora_dropout: float = 0.0
    lora_target: str = "all"           # all: 公式と同じ全 Linear / llm: 言語モデルのみ (省メモリ)
    train_heads: bool = True
    # --- logging / checkpoints ---
    run_root: str = "/runs"
    run_name: str = ""
    log_freq: int = 10
    val_freq: int = 500
    val_batches: int = 50
    val_at_start: bool = True
    num_viz: int = 8
    save_freq: int = 1000
    keep_last: int = 3
    save_optimizer: bool = False
    wandb_project: str = ""
    wandb_entity: str = ""
    seed: int = 42
    dry_run: bool = False              # 数 step だけ回して配管と VRAM を確認


# ---------------------------------------------------------------------------
# config handling
# ---------------------------------------------------------------------------
def _str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("1", "true", "yes", "y", "on"):
        return True
    if v.lower() in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"boolean expected, got {v}")


def parse_config(argv=None) -> FinetuneConfig:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="", help="YAML file (CLI options override it)")
    for f in dataclasses.fields(FinetuneConfig):
        name = f"--{f.name}"
        default = argparse.SUPPRESS
        tstr = str(f.type)  # "from __future__ import annotations" なので型は文字列
        if "List[int]" in tstr:
            ap.add_argument(name, nargs="+", type=int, default=default)
        elif "List[str]" in tstr:
            ap.add_argument(name, nargs="+", type=str, default=default)
        elif "Dict" in tstr:
            ap.add_argument(name, type=json.loads, default=default, help="JSON, e.g. '{\"image\": 1.0}'")
        elif tstr in ("bool", "<class 'bool'>"):
            ap.add_argument(name, type=_str2bool, default=default)
        elif tstr in ("int", "<class 'int'>"):
            ap.add_argument(name, type=int, default=default)
        elif tstr in ("float", "<class 'float'>"):
            ap.add_argument(name, type=float, default=default)
        else:
            ap.add_argument(name, type=str, default=default)
    args = vars(ap.parse_args(argv))
    values = {}
    cfg_file = args.pop("config", "")
    if cfg_file:
        with open(cfg_file) as fh:
            values.update(yaml.safe_load(fh) or {})
    values.update(args)
    known = {f.name for f in dataclasses.fields(FinetuneConfig)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    return FinetuneConfig(**values)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
class HeadWrapper(nn.Module):
    """action_head.predict_action を forward として呼ぶ (DDP で勾配同期させるため)."""

    def __init__(self, head: nn.Module):
        super().__init__()
        self.head = head

    def forward(self, hidden, modality):
        return self.head.predict_action(hidden, modality)


def unwrap(m):
    return m.module if isinstance(m, DDP) else m


def lora_targets(vla, mode: str) -> List[str]:
    names = []
    for name, module in vla.named_modules():
        if isinstance(module, nn.Linear):
            if mode == "llm" and not name.startswith("language_model."):
                continue
            names.append(name)
    if not names:
        raise ValueError(f"no LoRA target modules for mode={mode}")
    return names


class Logger:
    def __init__(self, run_dir: Path, cfg: FinetuneConfig, enabled: bool):
        self.enabled = enabled
        self.path = run_dir / "metrics.csv"
        self.rows = []
        self.wandb = None
        if enabled and cfg.wandb_project:
            import wandb

            self.wandb = wandb
            wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity or None, name=run_dir.name,
                       config=dataclasses.asdict(cfg))

    def log(self, step: int, data: Dict[str, float]):
        if not self.enabled:
            return
        row = {"step": step, **{k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                               for k, v in data.items()}}
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")
        if self.wandb is not None:
            self.wandb.log(data, step=step)


def save_checkpoint(run_dir: Path, step: int, cfg: FinetuneConfig, vla, action_head, pose_projector, optimizer,
                    base_vla_path: str, base_step: int, extra: Dict):
    ckpt_root = run_dir / "checkpoints"
    d = ckpt_root / f"step_{step:06d}"
    d.mkdir(parents=True, exist_ok=True)
    unwrap(vla).save_pretrained(str(d / "lora_adapter"))
    torch.save(unwrap(action_head).head.state_dict(), d / f"action_head--{step}_checkpoint.pt")
    torch.save(unwrap(pose_projector).state_dict(), d / f"pose_projector--{step}_checkpoint.pt")
    if cfg.save_optimizer:
        torch.save(optimizer.state_dict(), d / "optimizer.pt")
    meta = {
        "base_vla_path": base_vla_path,
        "base_step": base_step,
        "step": step,
        "metric_waypoint_spacing": cfg.metric_waypoint_spacing,
        "waypoint_spacing": cfg.waypoint_spacing,
        "lora_rank": cfg.lora_rank,
        "lora_target": cfg.lora_target,
        "data_dirs": cfg.data_dirs,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        **extra,
    }
    with open(d / FINETUNE_META, "w") as f:
        json.dump(meta, f, indent=2)
    with open(ckpt_root / "latest.txt", "w") as f:
        f.write(str(d) + "\n")
    # 古いチェックポイントを削除
    if cfg.keep_last > 0:
        olds = sorted(p for p in ckpt_root.glob("step_*") if p.is_dir())
        for p in olds[:-cfg.keep_last]:
            shutil.rmtree(p, ignore_errors=True)
    return d


def make_loader(ds, cfg: FinetuneConfig, shuffle: bool, distributed: bool, rank: int, world: int, collate_fn):
    if shuffle and cfg.turn_sample_ratio > 0:
        # 曲がるサンプルを turn_sample_ratio の割合で引く (元データはほぼ直進なので、そのままだと直進ばかり学習する)
        sampler = WeightedEpochSampler(ds.sample_weights(cfg.turn_sample_ratio), len(ds) // world, cfg.seed, rank)
    elif distributed:
        sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=shuffle, seed=cfg.seed)
    else:
        sampler = None
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=(shuffle and sampler is None), sampler=sampler,
                      num_workers=cfg.num_workers, collate_fn=collate_fn, drop_last=shuffle,
                      pin_memory=True, persistent_workers=False), sampler


# ---------------------------------------------------------------------------
def main(argv=None):
    cfg = parse_config(argv)
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    else:
        rank, world, local_rank = 0, 1, 0
    is_main = rank == 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for fine-tuning OmniVLA (7B)")
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    random.seed(cfg.seed + rank)
    np.random.seed(cfg.seed + rank)
    torch.manual_seed(cfg.seed + rank)

    run_name = cfg.run_name or f"omnivla_gazebo_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(cfg.run_root) / run_name
    if is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "config.yaml", "w") as f:
            yaml.safe_dump(dataclasses.asdict(cfg), f, sort_keys=False)
    log = Logger(run_dir, cfg, is_main)

    def say(*a):
        if is_main:
            print("[finetune]", *a, flush=True)

    # ---------------- data split ----------------
    trajs = find_trajectories(cfg.data_dirs)
    if len(trajs) < 2:
        raise RuntimeError(f"need >= 2 trajectories, found {len(trajs)} in {cfg.data_dirs}")
    train_dirs, val_dirs = split_trajectories(trajs, cfg.val_ratio, cfg.seed)
    if is_main:
        with open(run_dir / "split.json", "w") as f:
            json.dump({"train": train_dirs, "val": val_dirs}, f, indent=1)
    say(f"trajectories: train={len(train_dirs)} val={len(val_dirs)}")

    # ---------------- model ----------------
    from peft import LoraConfig, PeftModel, get_peft_model

    start_step = 0
    if cfg.resume_from:
        meta = read_finetune_meta(cfg.resume_from)
        base_vla_path = meta.get("base_vla_path", cfg.vla_path)
        base_step = int(meta.get("base_step", -1))
        heads_dir, heads_step = cfg.resume_from, find_checkpoint_step(cfg.resume_from)
        start_step = int(meta.get("step", heads_step))
        if abs(float(meta.get("metric_waypoint_spacing", cfg.metric_waypoint_spacing))
               - cfg.metric_waypoint_spacing) > 1e-9:
            raise ValueError("metric_waypoint_spacing differs from the resumed checkpoint")
    else:
        base_vla_path = cfg.vla_path
        base_step = cfg.base_step if cfg.base_step >= 0 else find_checkpoint_step(cfg.vla_path)
        heads_dir, heads_step = cfg.vla_path, base_step
    say(f"loading base model {base_vla_path} (heads from {heads_dir} @ {heads_step})")
    vla, processor = load_base_vla(base_vla_path, device)
    num_patches = num_vision_patches(vla)
    llm_dim = int(vla.llm_dim)
    if cfg.gradient_checkpointing:
        vla.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        vla.language_model.config.use_cache = False
    if cfg.resume_from:
        vla = PeftModel.from_pretrained(vla, os.path.join(cfg.resume_from, "lora_adapter"), is_trainable=True)
    else:
        lora_cfg = LoraConfig(r=cfg.lora_rank, lora_alpha=min(cfg.lora_rank, 16), lora_dropout=cfg.lora_dropout,
                              target_modules=lora_targets(vla, cfg.lora_target), init_lora_weights="gaussian")
        vla = get_peft_model(vla, lora_cfg)
    if is_main:
        vla.print_trainable_parameters()
    action_head, pose_projector = build_heads(llm_dim, heads_dir, heads_step, device)
    for prm in list(action_head.parameters()) + list(pose_projector.parameters()):
        prm.requires_grad = bool(cfg.train_heads)
    head = HeadWrapper(action_head)
    if distributed:
        vla = DDP(vla, device_ids=[local_rank], find_unused_parameters=True, gradient_as_bucket_view=True)
        if cfg.train_heads:
            head = DDP(head, device_ids=[local_rank])
            pose_projector = DDP(pose_projector, device_ids=[local_rank])

    params = [p for p in vla.parameters() if p.requires_grad]
    if cfg.train_heads:
        params += list(unwrap(head).parameters()) + list(unwrap(pose_projector).parameters())
    say(f"trainable params: {sum(p.numel() for p in params) / 1e6:.1f}M")
    optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    if cfg.resume_from and os.path.exists(os.path.join(cfg.resume_from, "optimizer.pt")):
        optimizer.load_state_dict(torch.load(os.path.join(cfg.resume_from, "optimizer.pt"), map_location="cpu"))
        say("optimizer state restored")

    def lr_lambda(step: int) -> float:
        s = step + start_step
        f = 1.0
        if cfg.lr_warmup_steps > 0 and s < cfg.lr_warmup_steps:
            f = 0.1 + 0.9 * s / cfg.lr_warmup_steps
        if cfg.lr_decay_step > 0 and s >= cfg.lr_decay_step:
            f *= cfg.lr_decay_gamma
        return f

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---------------- datasets ----------------
    ds_cfg = GazeboDatasetConfig(
        metric_waypoint_spacing=cfg.metric_waypoint_spacing,
        waypoint_spacing=cfg.waypoint_spacing,
        max_goal_dist=cfg.max_goal_dist,
        goal=GoalSamplingConfig(image_goal_offset=tuple(cfg.image_goal_offset),
                                pose_goal_offset=tuple(cfg.pose_goal_offset),
                                modality_weights=parse_modality_weights(cfg.modality_weights)),
        aug=AugmentConfig(enabled=cfg.augment, crop_v=cfg.crop_v, crop_h=cfg.crop_h, flip_prob=cfg.flip_prob,
                          color_jitter=cfg.color_jitter),
        turn_horizon=cfg.turn_horizon,
        turn_threshold_deg=cfg.turn_threshold_deg,
    )
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    train_ds = GazeboNavDataset(train_dirs, processor, action_tokenizer, ds_cfg, train=True, seed=cfg.seed)
    val_ds = GazeboNavDataset(val_dirs, processor, action_tokenizer, ds_cfg, train=False, seed=cfg.seed,
                              max_samples=cfg.val_batches * cfg.batch_size,
                              turn_ratio=cfg.val_turn_ratio) if val_dirs else None
    say(f"samples: train={len(train_ds)} ({train_ds.num_frames()} frames)"
        + (f", val={len(val_ds)} (turn {val_ds.turn_fraction() * 100:.0f}%)" if val_ds else ""))
    say(f"turning samples in train data: {train_ds.turn_fraction() * 100:.1f}% "
        f"(> {cfg.turn_threshold_deg:.0f}deg within {cfg.turn_horizon} frames) -> sampled at "
        + (f"{cfg.turn_sample_ratio * 100:.0f}%" if cfg.turn_sample_ratio > 0 else "natural rate"))

    def collate_fn(items):
        return collate(items, processor.tokenizer.pad_token_id, processor.tokenizer.model_max_length)

    train_loader, train_sampler = make_loader(train_ds, cfg, True, distributed, rank, world, collate_fn)
    val_loader = None
    if val_ds is not None and is_main:
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                                collate_fn=collate_fn)

    def run_validation(step: int):
        if val_loader is None:
            return {}
        vla.eval()
        head.eval()
        pose_projector.eval()
        m = evaluate(unwrap(vla), unwrap(head), unwrap(pose_projector), val_loader, num_patches, device,
                     cfg.metric_waypoint_spacing, max_batches=cfg.val_batches,
                     viz_dir=str(run_dir / "viz" / f"step_{step:06d}"), num_viz=cfg.num_viz)
        vla.train()
        head.train()
        pose_projector.train()
        say(f"[val step {step}] ADE={m.get('ade', float('nan')):.3f}m FDE={m.get('fde', float('nan')):.3f}m "
            f"turn: ADE={m.get('turn/ade', float('nan')):.3f}m FDE={m.get('turn/fde', float('nan')):.3f}m "
            f"heading_err={m.get('turn/yaw_err_deg', float('nan')):.1f}deg | "
            + " ".join(f"{k}={v:.3f}" for k, v in m.items() if k.endswith("/ade")))
        log.log(step, {f"val/{k}": v for k, v in m.items()})
        return m

    # ---------------- training loop ----------------
    vla.train()
    head.train()
    pose_projector.train()
    best = {}
    if cfg.val_at_start and is_main and not cfg.dry_run:
        best = {"val_at_start": run_validation(start_step)}
    if distributed:
        dist.barrier()
    max_steps = 3 if cfg.dry_run else cfg.max_steps
    epoch = 0
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)
    train_ds.set_epoch(epoch)
    it = iter(train_loader)
    step = 0
    t_last, last_log_step = time.time(), 0
    hist: Dict[str, List[float]] = {}
    torch.cuda.reset_peak_memory_stats(device)
    while step < max_steps:
        optimizer.zero_grad(set_to_none=True)
        for micro in range(cfg.grad_accumulation_steps):
            try:
                batch = next(it)
            except StopIteration:
                epoch += 1
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)
                train_ds.set_epoch(epoch)
                it = iter(train_loader)
                batch = next(it)
            pred = forward_actions(vla, head, pose_projector, batch, num_patches, device)
            loss, parts = action_loss(pred, batch["actions"].to(device), cfg.smooth_loss_weight)
            (loss / cfg.grad_accumulation_steps).backward()
            for k, v in {"loss": loss.detach(), **parts}.items():
                hist.setdefault(k, []).append(float(v))
        grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm) if cfg.max_grad_norm > 0 else None
        optimizer.step()
        scheduler.step()
        step += 1
        gstep = start_step + step
        if is_main and (step % cfg.log_freq == 0 or step == 1 or cfg.dry_run):
            dt = (time.time() - t_last) / max(1, step - last_log_step)
            t_last, last_log_step = time.time(), step
            data = {f"train/{k}": float(np.mean(v)) for k, v in hist.items()}
            data.update({"train/lr": scheduler.get_last_lr()[0], "train/sec_per_step": dt,
                         "train/max_mem_gb": torch.cuda.max_memory_allocated(device) / 1e9, "epoch": epoch})
            if grad_norm is not None:
                data["train/grad_norm"] = float(grad_norm)
            hist = {}
            say(f"step {gstep} loss={data['train/loss']:.4f} lr={data['train/lr']:.2e} "
                f"{dt:.2f}s/step mem={data['train/max_mem_gb']:.1f}GB")
            log.log(gstep, data)
        if not cfg.dry_run and step % cfg.val_freq == 0 and is_main:
            best[f"val_step_{gstep}"] = run_validation(gstep)
        if is_main and (step % cfg.save_freq == 0 or step == max_steps):
            m = best.get(f"val_step_{gstep}", {})
            d = save_checkpoint(run_dir, gstep, cfg, vla, head, pose_projector, optimizer, base_vla_path, base_step,
                                {"val": m})
            say(f"saved {d}")
        if distributed:
            dist.barrier()
    if is_main and not cfg.dry_run and step % cfg.val_freq != 0:
        run_validation(start_step + step)
    say(f"done. run dir: {run_dir}")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
