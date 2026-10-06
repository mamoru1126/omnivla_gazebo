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


@dataclass
class PerturbConfig:
    """データ収集中にお手本の走行をわざと乱す (DART: Disturbances for Augmenting Robot Trajectories).

    外乱でロボットを経路から外し、その後お手本が経路へ戻る様子を記録する。
    外乱中のフレームは正解ラベルに使わず (training 側で除外)、外乱直後の「立て直し」だけを学習させる。
    これが無いと、モデルは経路の上を走る状態しか見たことがなく、少しずれると予測が崩れる (曲がり角で衝突した原因)。
    """
    enabled: bool = True
    interval: Tuple[float, float] = (3.0, 8.0)     # 外乱と外乱の間隔 [s]
    duration: Tuple[float, float] = (0.8, 2.5)     # 1 回の外乱の長さ [s] (経路から最大 0.7m 程度外れる)
    angular: Tuple[float, float] = (0.4, 0.9)      # 外乱中の角速度の大きさ [rad/s] (向きはランダム)
    linear: Tuple[float, float] = (0.1, 0.3)       # 外乱中の速度 [m/s]
    min_clearance: float = 0.85                    # この空きが無い場所では外乱を始めない [m]
    abort_clearance: float = 0.55                  # 外乱中にこれより障害物に近づいたら打ち切る [m]
    min_goal_dist: float = 1.2                     # ゴール手前では外乱しない [m]
    tail: float = 0.35                             # 外乱終了後もこの秒数はラベル除外フレーム扱い [s]


class Perturber:
    def __init__(self, cfg: PerturbConfig, rng):
        self.cfg = cfg
        self.rng = rng
        self.active = False
        self.end = 0.0
        self.flag_until = -1e9
        self.cmd = (0.0, 0.0)
        self.next_time = None
        self.count = 0

    def reset(self, t: float) -> None:
        self.active = False
        self.flag_until = -1e9
        self.next_time = t + float(self.rng.uniform(*self.cfg.interval))

    def flagged(self, t: float) -> bool:
        """この時刻に記録したフレームを「外乱中」として扱うか."""
        return self.active or t <= self.flag_until

    def step(self, t: float, clearance: float, dist_goal: float, expert_cmd: Tuple[float, float]):
        """(v, w) を返す. 外乱中でなければお手本の指令をそのまま返す."""
        c = self.cfg
        if not c.enabled:
            return expert_cmd
        if self.next_time is None:
            self.reset(t)
        if self.active:
            if t >= self.end or clearance < c.abort_clearance:
                self.active = False
                self.flag_until = t + c.tail
                self.next_time = t + float(self.rng.uniform(*c.interval))
            else:
                return self.cmd
        if t >= self.next_time and clearance >= c.min_clearance and dist_goal >= c.min_goal_dist:
            sign = 1.0 if self.rng.random() < 0.5 else -1.0
            self.cmd = (float(self.rng.uniform(*c.linear)), sign * float(self.rng.uniform(*c.angular)))
            self.end = t + float(self.rng.uniform(*c.duration))
            self.active = True
            self.count += 1
            return self.cmd
        return expert_cmd
