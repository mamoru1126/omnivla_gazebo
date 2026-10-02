"""OmniVLA 推論の高水準ラッパ (ROS 非依存).

公式 inference/run_omnivla.py との違い:
  * 現在画像・ゴール画像・ゴール姿勢・言語指示を「引数」で受け取る (公式はファイルパスと GPS が固定)
  * グローバル変数を使わないので、ROS ノードや評価スクリプトから import して使える
  * ファインチューニング済み LoRA アダプタ + ヘッドを読める
  * 出力はメートル単位の waypoint (ロボット座標: x前, y左) と yaw の cos/sin

使い方:
    policy = OmniVLAPolicy(PolicyConfig(vla_path="/checkpoints/omnivla-original"))
    out = policy.predict(current_pil, goal_image=goal_pil, modality="image")
    out.waypoints  # (8, 4) [x[m], y[m], cos, sin]
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image

from .data_utils import (IMAGE_MODALITIES, LANGUAGE_MODALITIES, MODALITY_NAMES, POSE_MODALITIES,
                         SUPPORTED_MODALITIES, denormalize_actions, modality_id, normalize_goal_pose)
from .omnivla_model import (OmniVLAComponents, build_prompt, collate, count_action_tokens, forward_actions,
                            image_embeddings, load_omnivla, make_sample)
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK

ImageLike = Union[Image.Image, np.ndarray]


@dataclass
class PolicyConfig:
    vla_path: Optional[str] = "/checkpoints/omnivla-original"
    step: Optional[int] = None                  # None/-1: ディレクトリ内の最新 step を自動検出
    finetuned_dir: Optional[str] = None         # finetune_omnivla.py の出力 (lora_adapter/ を含む)
    device: str = "cuda:0"
    metric_waypoint_spacing: Optional[float] = None  # None: finetune_meta.json の値 or 0.1 (公式推論と同じ)
    max_goal_dist: float = 30.0                 # 公式 thres_dist
    merge_lora: bool = True
    seed: int = 0


@dataclass
class PolicyOutput:
    waypoints: np.ndarray       # (8, 4) [x[m], y[m], cos, sin]
    normalized: np.ndarray      # (8, 4) モデル生出力
    modality: int
    latency: float              # [s]
    goal_pose_input: np.ndarray  # (4,) モデルに入れた正規化 goal pose


def _to_pil(img: ImageLike) -> Image.Image:
    if isinstance(img, np.ndarray):
        return Image.fromarray(img.astype(np.uint8)).convert("RGB")
    return img.convert("RGB")


class OmniVLAPolicy:
    def __init__(self, cfg: PolicyConfig, components: Optional[OmniVLAComponents] = None):
        self.cfg = cfg
        self.c = components or load_omnivla(cfg.vla_path, cfg.step, cfg.finetuned_dir, cfg.device, cfg.merge_lora)
        meta_spacing = self.c.meta.get("metric_waypoint_spacing")
        if cfg.metric_waypoint_spacing is None:
            self.metric_spacing = float(meta_spacing) if meta_spacing else 0.1
        else:
            self.metric_spacing = float(cfg.metric_waypoint_spacing)
            if meta_spacing and abs(float(meta_spacing) - self.metric_spacing) > 1e-6:
                print(f"[policy] WARNING: metric_waypoint_spacing={self.metric_spacing} differs from "
                      f"the value used for fine-tuning ({meta_spacing})")
        self.tokenizer = self.c.processor.tokenizer
        self.image_transform = self.c.processor.image_processor.apply_transform
        self._rng = np.random.default_rng(cfg.seed)
        self._prompt_cache: Dict[Optional[str], Tuple[torch.Tensor, torch.Tensor]] = {}
        self._emb_cache: Dict[str, np.ndarray] = {}
        print(f"[policy] ready (device={self.c.device}, metric_waypoint_spacing={self.metric_spacing})")

    # ------------------------------------------------------------------
    def _prompt(self, instruction: Optional[str]):
        if instruction not in self._prompt_cache:
            for _ in range(10):
                # 公式 inference と同じくダミー行動 (乱数) で行動トークンの位置だけを作る
                dummy = self._rng.random((NUM_ACTIONS_CHUNK, ACTION_DIM))
                ids, labels = build_prompt(self.tokenizer, self.c.action_tokenizer, instruction, dummy)
                if count_action_tokens(labels) == NUM_ACTIONS_CHUNK * ACTION_DIM:
                    break
            else:
                raise RuntimeError("failed to build a prompt with the expected number of action tokens")
            self._prompt_cache[instruction] = (ids, labels)
        return self._prompt_cache[instruction]

    @torch.no_grad()
    def predict(self, current: ImageLike, goal_image: Optional[ImageLike] = None,
                goal_pose: Optional[Sequence[float]] = None, instruction: Optional[str] = None,
                modality: Union[str, int] = "image") -> PolicyOutput:
        """goal_pose: ロボット座標での相対ゴール (x[m], y[m], dyaw[rad])."""
        mid = modality_id(modality)
        if mid not in SUPPORTED_MODALITIES:
            raise ValueError(f"modality {MODALITY_NAMES[mid]} (satellite) is not supported in this wrapper")
        if mid in IMAGE_MODALITIES and goal_image is None:
            raise ValueError(f"modality {MODALITY_NAMES[mid]} requires goal_image")
        if mid in POSE_MODALITIES and goal_pose is None:
            raise ValueError(f"modality {MODALITY_NAMES[mid]} requires goal_pose")
        if mid in LANGUAGE_MODALITIES and not instruction:
            raise ValueError(f"modality {MODALITY_NAMES[mid]} requires instruction")
        t0 = time.time()
        cur = _to_pil(current)
        # 使わないモダリティは注意マスクで遮断されるので中身は何でもよい (現在画像/ゼロで埋める)
        goal = _to_pil(goal_image) if goal_image is not None else cur
        if goal_pose is not None and mid in POSE_MODALITIES:
            gp = normalize_goal_pose(float(goal_pose[0]), float(goal_pose[1]), float(goal_pose[2]),
                                     self.metric_spacing, self.cfg.max_goal_dist)
        else:
            gp = np.zeros(4, dtype=np.float32)
        ids, labels = self._prompt(instruction if mid in LANGUAGE_MODALITIES else None)
        sample = make_sample(self.image_transform, cur, goal, ids, labels, gp, mid)
        batch = collate([sample], self.tokenizer.pad_token_id, self.tokenizer.model_max_length)
        pred = forward_actions(self.c.vla, self.c.action_head.predict_action, self.c.pose_projector, batch,
                               self.c.num_patches, self.c.device)
        norm = pred[0].float().cpu().numpy()
        if self.c.device.type == "cuda":
            torch.cuda.synchronize(self.c.device)
        return PolicyOutput(denormalize_actions(norm, self.metric_spacing), norm, mid, time.time() - t0, gp)

    # ------------------------------------------------------------------
    def embed(self, image: ImageLike, cache_key: Optional[str] = None) -> np.ndarray:
        if cache_key is not None and cache_key in self._emb_cache:
            return self._emb_cache[cache_key]
        emb = image_embeddings(self.c.vla, self.image_transform, [_to_pil(image)], self.c.device)[0]
        if cache_key is not None:
            self._emb_cache[cache_key] = emb
        return emb

    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
