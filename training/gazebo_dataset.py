"""Gazebo で集めた GNM 形式の軌跡から OmniVLA の学習サンプルを作る Dataset.

公式の GNM_Dataset (prismatic/vla/datasets/gnm_dataset.py) と同じ出力形式・正規化にしているが、
  * 外部リポジトリ (visualnav-transformer / MBRA) や LMDB キャッシュに依存しない
  * 入力画像サイズを決め打ちしない (クロップは画像サイズに比例, 最後に 224x224 へ resize-naive)
  * 軌跡末尾は最終姿勢でパディング (= ゴール付近では「止まる」ことを学習)
  * ゴール画像は同じ軌跡の未来フレーム (hindsight relabeling), ゴール姿勢は真値オドメトリから計算
"""
from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from omnivla_nav.data_utils import (AugmentConfig, GoalSamplingConfig, augment_pair, build_sample_index,  # noqa: E402
                                    make_targets, modality_id)
from omnivla_nav.omnivla_model import build_prompt, make_sample  # noqa: E402
from omnivla_nav.trajectory_io import image_path, load_trajectory  # noqa: E402


@dataclass
class GazeboDatasetConfig:
    metric_waypoint_spacing: float = 0.1   # 正規化に使う 1 単位 [m] (公式推論は 0.1)
    waypoint_spacing: int = 1              # 何フレームおきに waypoint を取るか
    max_goal_dist: float = 30.0
    goal: GoalSamplingConfig = field(default_factory=GoalSamplingConfig)
    aug: AugmentConfig = field(default_factory=AugmentConfig)


def parse_modality_weights(weights: Dict) -> Dict[int, float]:
    return {modality_id(k): float(v) for k, v in weights.items()}


class GazeboNavDataset(Dataset):
    def __init__(self, traj_dirs: Sequence[str], processor, action_tokenizer, cfg: GazeboDatasetConfig,
                 train: bool = True, force_modality: Optional[int] = None, max_samples: Optional[int] = None,
                 seed: int = 0):
        self.cfg = cfg
        self.train = train
        self.force_modality = force_modality
        self.seed = seed
        self.tokenizer = processor.tokenizer
        self.image_transform = processor.image_processor.apply_transform
        self.action_tokenizer = action_tokenizer
        self.trajs: List[dict] = []
        for d in traj_dirs:
            data = load_trajectory(d)
            n = len(data["position"])
            if n < 2 or not os.path.exists(image_path(d, n - 1)):
                print(f"[dataset] skip {d} (frames={n})")
                continue
            self.trajs.append({"dir": d, "name": os.path.basename(d), "position": data["position"],
                               "yaw": data["yaw"], "n": n})
        self.index = build_sample_index([t["n"] for t in self.trajs])
        if max_samples is not None and len(self.index) > max_samples:
            rng = np.random.default_rng(seed)
            keep = np.sort(rng.choice(len(self.index), size=max_samples, replace=False))
            self.index = [self.index[i] for i in keep]
        if not self.index:
            raise ValueError("dataset is empty")

    def __len__(self) -> int:
        return len(self.index)

    def num_frames(self) -> int:
        return int(sum(t["n"] for t in self.trajs))

    epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rng(self, i: int) -> np.random.Generator:
        if self.train:
            # worker ごとに異なる torch の seed + epoch + index から作る (ゴール/modality/augment が毎回変わる)
            return np.random.default_rng([torch.initial_seed() % (2 ** 32), self.epoch, i])
        return np.random.default_rng([self.seed, i])  # 検証は毎回同じサンプル

    def __getitem__(self, i: int) -> dict:
        ti, t = self.index[i]
        tr = self.trajs[ti]
        rng = self._rng(i)
        mod, goal_t, actions, goal_pose = make_targets(
            rng, tr["position"], tr["yaw"], t, self.cfg.goal, self.cfg.metric_waypoint_spacing,
            self.cfg.waypoint_spacing, self.cfg.max_goal_dist, self.force_modality)
        cur = Image.open(image_path(tr["dir"], t)).convert("RGB")
        goal = Image.open(image_path(tr["dir"], goal_t)).convert("RGB")
        cur, goal, actions, goal_pose = augment_pair(rng, cur, goal, actions, goal_pose, self.cfg.aug, self.train)
        input_ids, labels = build_prompt(self.tokenizer, self.action_tokenizer, None, actions)
        sample = make_sample(self.image_transform, cur, goal, input_ids, labels, goal_pose, mod, actions)
        sample["meta"] = {"traj_dir": tr["dir"], "t": int(t), "goal_t": int(goal_t), "modality": mod}
        return sample


def split_trajectories(traj_dirs: Sequence[str], val_ratio: float, seed: int):
    """軌跡単位で train/val に分ける (同じ軌跡のフレームが両方に入らないように)."""
    dirs = sorted(traj_dirs)
    rng = random.Random(seed)
    rng.shuffle(dirs)
    n_val = int(round(len(dirs) * val_ratio))
    if val_ratio > 0 and len(dirs) >= 2:
        n_val = max(1, n_val)
    n_val = min(n_val, len(dirs) - 1)
    return sorted(dirs[n_val:]), sorted(dirs[:n_val])
