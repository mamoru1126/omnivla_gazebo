"""データ収集用のエキスパート経路追従 (pure pursuit, ROS 非依存)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .geometry import to_local


@dataclass
class FollowerConfig:
    speed: float = 0.3            # [m/s]
    lookahead: float = 0.6        # [m]
    max_angular: float = 0.8      # [rad/s]
    goal_tolerance: float = 0.25  # [m]
    rotate_threshold: float = 1.0  # [rad] これ以上ずれていたらその場旋回
    slow_radius: float = 0.6      # ゴール手前で減速を始める距離 [m]


class PathFollower:
    def __init__(self, path: np.ndarray, cfg: FollowerConfig):
        self.path = np.asarray(path, dtype=np.float64)
        self.cfg = cfg
        seg = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
        self.remaining_from = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
        self.idx = 0

    def remaining(self, xy) -> float:
        p = np.asarray(xy, dtype=np.float64)
        return float(self.remaining_from[self.idx] + np.linalg.norm(self.path[self.idx] - p))

    def step(self, pose: Tuple[float, float, float]) -> Tuple[float, float, bool]:
        """(v, w, reached) を返す."""
        p = np.asarray(pose[:2], dtype=np.float64)
        win = self.path[self.idx:self.idx + 60]
        self.idx += int(np.argmin(np.linalg.norm(win - p, axis=1)))
        dist_goal = float(np.linalg.norm(self.path[-1] - p))
        if dist_goal < self.cfg.goal_tolerance:
            return 0.0, 0.0, True
        d_all = np.linalg.norm(self.path[self.idx:] - p, axis=1)
        ahead = np.nonzero(d_all >= self.cfg.lookahead)[0]
        target = self.path[self.idx + int(ahead[0])] if len(ahead) else self.path[-1]
        local = to_local(target, p, pose[2])
        alpha = math.atan2(local[1], local[0])
        wmax = self.cfg.max_angular
        if abs(alpha) > self.cfg.rotate_threshold:
            return 0.0, math.copysign(wmax, alpha), False
        lt = max(float(np.hypot(local[0], local[1])), 1e-3)
        v = self.cfg.speed * max(0.3, 1.0 - abs(alpha) / 1.2)
        v *= float(np.clip(dist_goal / self.cfg.slow_radius, 0.35, 1.0))
        w = float(np.clip(2.0 * v * math.sin(alpha) / lt, -wmax, wmax))
        return float(v), w, False


def unicycle_step(pose, v: float, w: float, dt: float):
    x, y, yaw = pose
    return x + v * math.cos(yaw) * dt, y + v * math.sin(yaw) * dt, yaw + w * dt
