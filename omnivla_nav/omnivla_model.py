"""OmniVLA (7B, OpenVLA-OFT ベース) のロード・入力作成・順伝播.

公式 inference/run_omnivla.py と vla-scripts/train_omnivla.py の処理を、
  * グローバル変数に依存しない関数群
  * 推論 (policy.py) と学習 (training/finetune_omnivla.py) で共通の forward
に整理したもの。前処理・プロンプト・隠れ状態の取り出し方は公式と同一にしている。

チェックポイントの形式
  (a) 公式: <dir>/model-0000x-of-00004.safetensors, config.json, ...,
            action_head--<step>_checkpoint.pt, proprio_projector--<step>_checkpoint.pt
  (b) 本リポジトリのファインチューニング出力 (training/finetune_omnivla.py):
            <ckpt>/lora_adapter/, action_head--<step>_checkpoint.pt,
            pose_projector--<step>_checkpoint.pt, finetune_meta.json
"""
from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn.utils.rnn import pad_sequence

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction_MMNv1
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import L1RegressionActionHead_idcat
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.projectors import ProprioProjector
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, POSE_DIM
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

NO_LANGUAGE_PROMPT = "No language instruction"
FINETUNE_META = "finetune_meta.json"
NUM_IMAGES_IN_INPUT = 2  # current + goal (公式と同じ。変更不可: 注意マスクが 256patch x 2 を前提)


# ---------------------------------------------------------------------------
# checkpoint helpers
# ---------------------------------------------------------------------------
def register_auto_classes() -> None:
    """公式と同じくローカルの OmniVLA クラス (MMNv1) を HF Auto クラスに登録 (多重登録は無視)."""
    for fn in (
        lambda: AutoConfig.register("openvla", OpenVLAConfig),
        lambda: AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor),
        lambda: AutoProcessor.register(OpenVLAConfig, PrismaticProcessor),
        lambda: AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction_MMNv1),
    ):
        try:
            fn()
        except ValueError:
            pass


def find_checkpoint_step(directory: str, module: str = "action_head") -> int:
    files = glob.glob(os.path.join(directory, f"{module}--*_checkpoint.pt"))
    steps = []
    for f in files:
        m = re.search(rf"{module}--(\d+)_checkpoint\.pt$", os.path.basename(f))
        if m:
            steps.append(int(m.group(1)))
    if not steps:
        raise FileNotFoundError(f"no '{module}--<step>_checkpoint.pt' found in {directory}")
    return max(steps)


def strip_ddp_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}


def load_module_state(directory: str, module: str, step: int) -> Dict[str, torch.Tensor]:
    path = os.path.join(directory, f"{module}--{step}_checkpoint.pt")
    if not os.path.exists(path) and module == "pose_projector":
        path = os.path.join(directory, f"proprio_projector--{step}_checkpoint.pt")  # 公式チェックポイントの名前
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return strip_ddp_prefix(torch.load(path, map_location="cpu"))


def read_finetune_meta(directory: str) -> dict:
    path = os.path.join(directory, FINETUNE_META)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def is_finetuned_dir(directory: Optional[str]) -> bool:
    return bool(directory) and os.path.isdir(os.path.join(directory, "lora_adapter"))


@dataclass
class OmniVLAComponents:
    vla: nn.Module
    processor: object
    action_head: nn.Module
    pose_projector: nn.Module
    action_tokenizer: ActionTokenizer
    num_patches: int
    device: torch.device
    meta: dict


def load_base_vla(vla_path: str, device: torch.device, dtype=torch.bfloat16):
    register_auto_classes()
    # processor は公式と同じく trust_remote_code=True、モデルはローカル登録した MMNv1 クラスを使う
    processor = AutoProcessor.from_pretrained(vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(vla_path, torch_dtype=dtype, low_cpu_mem_usage=True)
    if not isinstance(vla, OpenVLAForActionPrediction_MMNv1):
        raise TypeError(f"unexpected model class {type(vla)}; is {vla_path} an OmniVLA checkpoint?")
    vla.vision_backbone.set_num_images_in_input(NUM_IMAGES_IN_INPUT)
    vla.to(dtype=dtype, device=device)
    return vla, processor


def build_heads(llm_dim: int, state_dir: Optional[str], step: Optional[int], device: torch.device):
    """公式と同じ構成: pose_projector は fp32, action_head は bf16."""
    pose_projector = ProprioProjector(llm_dim=llm_dim, proprio_dim=POSE_DIM)
    action_head = L1RegressionActionHead_idcat(input_dim=llm_dim, hidden_dim=llm_dim, action_dim=ACTION_DIM)
    if state_dir is not None:
        pose_projector.load_state_dict(load_module_state(state_dir, "pose_projector", step))
        action_head.load_state_dict(load_module_state(state_dir, "action_head", step))
    pose_projector = pose_projector.to(device)
    action_head = action_head.to(torch.bfloat16).to(device)
    return action_head, pose_projector


def num_vision_patches(vla) -> int:
    base = vla.get_base_model() if hasattr(vla, "get_base_model") else vla
    n = base.vision_backbone.get_num_patches() * base.vision_backbone.get_num_images_in_input()
    return n + 1  # + goal pose token


def resolve_checkpoint(vla_path: Optional[str], step: Optional[int], finetuned_dir: Optional[str]):
    """(base_vla_path, base_step, heads_dir, heads_step, adapter_dir, meta) を決める."""
    meta = {}
    if finetuned_dir:
        finetuned_dir = os.path.abspath(finetuned_dir)
        if not is_finetuned_dir(finetuned_dir):
            raise FileNotFoundError(f"{finetuned_dir} has no lora_adapter/ (expected output of finetune_omnivla.py)")
        meta = read_finetune_meta(finetuned_dir)
        base = vla_path or meta.get("base_vla_path")
        if not base:
            raise ValueError("base vla_path is unknown (pass vla_path)")
        return base, None, finetuned_dir, find_checkpoint_step(finetuned_dir), \
            os.path.join(finetuned_dir, "lora_adapter"), meta
    if not vla_path:
        raise ValueError("vla_path is required")
    meta = read_finetune_meta(vla_path)  # merge_lora.py の出力なら存在する
    base_step = step if step is not None and step >= 0 else find_checkpoint_step(vla_path)
    return vla_path, base_step, vla_path, base_step, None, meta


def load_omnivla(vla_path: Optional[str], step: Optional[int] = None, finetuned_dir: Optional[str] = None,
                 device: str = "cuda:0", merge_lora: bool = True) -> OmniVLAComponents:
    """推論用にモデル一式をロードする (eval モード)."""
    dev = torch.device(device if torch.cuda.is_available() or not device.startswith("cuda") else "cpu")
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    base, _, heads_dir, heads_step, adapter_dir, meta = resolve_checkpoint(vla_path, step, finetuned_dir)
    print(f"[omnivla] loading base model from {base}")
    vla, processor = load_base_vla(base, dev)
    if adapter_dir is not None:
        from peft import PeftModel

        print(f"[omnivla] applying LoRA adapter {adapter_dir} (merge={merge_lora})")
        vla = PeftModel.from_pretrained(vla, adapter_dir)
        if merge_lora:
            vla = vla.merge_and_unload()
        vla.to(device=dev)
    print(f"[omnivla] loading heads from {heads_dir} (step {heads_step})")
    action_head, pose_projector = build_heads(_llm_dim(vla), heads_dir, heads_step, dev)
    vla.eval()
    action_head.eval()
    pose_projector.eval()
    return OmniVLAComponents(vla, processor, action_head, pose_projector, ActionTokenizer(processor.tokenizer),
                             num_vision_patches(vla), dev, meta)


def _llm_dim(vla) -> int:
    base = vla.get_base_model() if hasattr(vla, "get_base_model") else vla
    return int(base.llm_dim)


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def build_prompt(tokenizer, action_tokenizer: ActionTokenizer, instruction: Optional[str],
                 actions: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """公式 dataset / inference と同じプロンプトとラベルを作る.

    actions (8,4) はトークン化されて「行動トークンの位置」を示すだけ (埋め込みはモデル内で 0 にされる)。
    """
    current_action_string = action_tokenizer(np.asarray(actions[0]))
    future_actions_string = "".join(action_tokenizer(np.asarray(actions[1:])))
    action_chunk_string = current_action_string + future_actions_string
    action_chunk_len = len(action_chunk_string)
    human = NO_LANGUAGE_PROMPT if not instruction else f"What action should the robot take to {instruction}?"
    builder = PurePromptBuilder("openvla")
    builder.add_turn("human", human)
    builder.add_turn("gpt", action_chunk_string)
    input_ids = torch.tensor(tokenizer(builder.get_prompt(), add_special_tokens=True).input_ids)
    labels = input_ids.clone()
    labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
    return input_ids, labels


def count_action_tokens(labels: torch.Tensor) -> int:
    return int(((labels != IGNORE_INDEX) & (labels > ACTION_TOKEN_BEGIN_IDX)).sum())


def make_sample(image_transform: Callable, current: Image.Image, goal: Image.Image, input_ids: torch.Tensor,
                labels: torch.Tensor, goal_pose: np.ndarray, modality: int,
                actions: Optional[np.ndarray] = None) -> dict:
    return dict(
        pixel_values=image_transform(current.convert("RGB")),
        pixel_values_goal=image_transform(goal.convert("RGB")),
        input_ids=input_ids,
        labels=labels,
        actions=torch.as_tensor(np.zeros((NUM_ACTIONS_CHUNK, ACTION_DIM)) if actions is None else actions,
                                dtype=torch.float32),
        goal_pose=torch.as_tensor(goal_pose, dtype=torch.float32),
        modality_id=int(modality),
    )


def collate(instances: Sequence[dict], pad_token_id: int, model_max_length: int) -> dict:
    input_ids = pad_sequence([x["input_ids"] for x in instances], batch_first=True, padding_value=pad_token_id)
    labels = pad_sequence([x["labels"] for x in instances], batch_first=True, padding_value=IGNORE_INDEX)
    input_ids, labels = input_ids[:, :model_max_length], labels[:, :model_max_length]
    out = dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=input_ids.ne(pad_token_id),
        # (B, 6+6, 224, 224): [current(dino, siglip), goal(dino, siglip)]
        pixel_values=torch.cat([torch.stack([x["pixel_values"] for x in instances]),
                                torch.stack([x["pixel_values_goal"] for x in instances])], dim=1),
        actions=torch.stack([torch.as_tensor(x["actions"], dtype=torch.float32) for x in instances]),
        goal_pose=torch.stack([torch.as_tensor(x["goal_pose"], dtype=torch.float32) for x in instances]),
        modality_id=torch.tensor([float(x["modality_id"]) for x in instances], dtype=torch.float32),
    )
    for key in instances[0]:
        if key not in out and key not in ("pixel_values_goal",):
            out[key] = [x[key] for x in instances]  # メタ情報 (可視化用) はリストのまま
    return out


# ---------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------
def forward_actions(vla, action_head_fn: Callable, pose_projector, batch: dict, num_patches: int,
                    device: torch.device) -> torch.Tensor:
    """OmniVLA の順伝播. 戻り値: 正規化 waypoint (B, 8, 4) [bf16].

    action_head_fn(hidden_states, modality_id) は `action_head.predict_action` か、DDP 用ラッパ。
    勾配の有無は呼び出し側 (torch.no_grad など) で制御する。
    """
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    modality = batch["modality_id"].to(device=device, dtype=torch.bfloat16)
    use_amp = device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
        output = vla(
            input_ids=input_ids,
            attention_mask=batch["attention_mask"].to(device),
            pixel_values=batch["pixel_values"].to(device=device, dtype=torch.bfloat16),
            modality_id=modality,
            labels=labels,
            output_hidden_states=True,
            proprio=batch["goal_pose"].to(device=device, dtype=torch.bfloat16),
            proprio_projector=pose_projector,
            use_film=False,
            use_cache=False,
        )
    gt_token_ids = labels[:, 1:]
    action_mask = get_current_action_mask(gt_token_ids) | get_next_actions_mask(gt_token_ids)
    last_hidden = output.hidden_states[-1]                # (B, seq, D)
    text_hidden = last_hidden[:, num_patches:-1]          # 画像パッチ+pose トークンの後ろ
    bsz = input_ids.shape[0]
    actions_hidden = text_hidden[action_mask].reshape(bsz, NUM_ACTIONS_CHUNK * ACTION_DIM, -1).to(torch.bfloat16)
    return action_head_fn(actions_hidden, modality)


@torch.no_grad()
def image_embeddings(vla, image_transform: Callable, images: Sequence[Image.Image],
                     device: torch.device) -> np.ndarray:
    """DINOv2 (OmniVLA の視覚エンコーダの片方) のパッチ特徴を平均した L2 正規化ベクトル (N, D)."""
    base = vla.get_base_model() if hasattr(vla, "get_base_model") else vla
    px = torch.stack([image_transform(im.convert("RGB")) for im in images]).to(device=device, dtype=torch.bfloat16)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        feats = base.vision_backbone.featurizer(px[:, :3])  # (N, 256, 1024)
    emb = F.normalize(feats.float().mean(dim=1), dim=-1)
    return emb.cpu().numpy()
