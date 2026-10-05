"""OmniVLA が出力した waypoint 列を (v, w) 速度指令に変換するコントローラ (ROS 非依存).

mode="upstream":
    OmniVLA 公式 inference/run_omnivla.py の制御則を忠実に再現したもの。
    (waypoint[4] を DT=1/3 で追従 -> 0.5 / 1.0 でクリップ -> v<=0.3, w<=0.3 に曲率を保って制限)
    公式コードで未定義だった clip_angle もここで実装している。
    ただし目標点が後方 (dx<0) のとき公式の atan(dy/dx) は旋回方向が逆になるため atan2 に直している。
mode="pure_pursuit":
    予測軌跡上の前方注視点に向かう pure pursuit。上流より滑らかだが、学習時の想定とは異なる。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


def clip_angle(theta: float) -> float:
    """[-pi, pi] に正規化 (visualnav-transformer の clip_angle と同じ挙動)."""
    theta = math.fmod(theta, 2.0 * math.pi)
    if theta < 0.0:
        theta += 2.0 * math.pi
    if theta > math.pi:
        theta -= 2.0 * math.pi
    return theta


@dataclass
class ControllerConfig:
    mode: str = "upstream"          # "upstream" | "pure_pursuit"
    # --- upstream ---
    waypoint_index: int = 4         # 公式は waypoints[0][4] を使う
    dt: float = 1.0 / 3.0           # 公式の DT (tick_rate=3Hz)
    max_linear_raw: float = 0.5     # 公式: np.clip(linear, 0, 0.5)
    max_angular_raw: float = 1.0    # 公式: np.clip(angular, -1, 1)
    max_v: float = 0.3              # 公式: maxv
    max_w: float = 0.3              # 公式: maxw
    # --- pure pursuit ---
    lookahead: float = 0.5          # [m]
    pp_speed: float = 0.3           # [m/s]
    pp_max_w: float = 0.8           # [rad/s]
    rotate_in_place_angle: float = 1.2  # この角度以上ずれていたらその場旋回 [rad]


def limit_velocity(v: float, w: float, maxv: float, maxw: float) -> Tuple[float, float]:
    """公式 run_omnivla.py の "Velocity limitation" ブロックそのまま (曲率半径を保ったまま制限)."""
    if abs(v) <= maxv:
        if abs(w) <= maxw:
            return v, w
        rd = v / w
        return maxw * np.sign(v) * abs(rd), maxw * np.sign(w)
    if abs(w) <= 0.001:
        return maxv * np.sign(v), 0.0
    rd = v / w
    if abs(rd) >= maxv / maxw:
        return maxv * np.sign(v), maxv * np.sign(w) / abs(rd)
    return maxw * np.sign(v) * abs(rd), maxw * np.sign(w)


def upstream_command(waypoint: np.ndarray, cfg: ControllerConfig) -> Tuple[float, float]:
    """waypoint = [dx, dy, cos(yaw), sin(yaw)] (dx, dy はメートル, ロボット座標)."""
    dx, dy, hx, hy = [float(x) for x in waypoint[:4]]
    eps = 1e-8
    dt = cfg.dt
    if abs(dx) < eps and abs(dy) < eps:
        v = 0.0
        w = clip_angle(math.atan2(hy, hx)) / dt
    elif abs(dx) < eps:
        v = 0.0
        w = float(np.sign(dy)) * math.pi / (2.0 * dt)
    else:
        v = dx / dt
        # 公式は atan(dy/dx) だが、目標点が後方 (dx<0) だと左右が逆になる
        # (例: 左後ろの点で右旋回)。dx>0 では atan2 と同じ値なので、後方のときだけ挙動が変わる。
        w = math.atan2(dy, dx) / dt
    v = float(np.clip(v, 0.0, cfg.max_linear_raw))
    w = float(np.clip(w, -cfg.max_angular_raw, cfg.max_angular_raw))
    v, w = limit_velocity(v, w, cfg.max_v, cfg.max_w)
    return float(v), float(w)


def pure_pursuit_command(waypoints_xy: np.ndarray, cfg: ControllerConfig) -> Tuple[float, float]:
    pts = np.asarray(waypoints_xy, dtype=np.float64)[:, :2]
    dists = np.linalg.norm(pts, axis=1)
    if dists.max() < 0.05:  # ほぼ停止軌跡
        return 0.0, 0.0
    idx = int(np.argmax(dists >= cfg.lookahead)) if np.any(dists >= cfg.lookahead) else len(pts) - 1
    tx, ty = pts[idx]
    ld = max(float(np.hypot(tx, ty)), 1e-3)
    alpha = math.atan2(ty, tx)
    if abs(alpha) > cfg.rotate_in_place_angle:
        return 0.0, float(np.sign(alpha) * cfg.pp_max_w)
    v = cfg.pp_speed * min(1.0, ld / max(cfg.lookahead, 1e-3))
    curvature = 2.0 * math.sin(alpha) / ld
    w = curvature * v
    if abs(w) > cfg.pp_max_w:
        v = v * cfg.pp_max_w / abs(w)
        w = float(np.sign(w) * cfg.pp_max_w)
    return float(v), float(w)


def compute_command(waypoints: np.ndarray, cfg: ControllerConfig) -> Tuple[float, float]:
    """waypoints: (8, 4) = [x[m], y[m], cos, sin] (ロボット座標)."""
    waypoints = np.asarray(waypoints, dtype=np.float64)
    if cfg.mode == "upstream":
        idx = int(np.clip(cfg.waypoint_index, 0, len(waypoints) - 1))
        return upstream_command(waypoints[idx], cfg)
    if cfg.mode == "pure_pursuit":
        return pure_pursuit_command(waypoints, cfg)
    raise ValueError(f"unknown controller mode: {cfg.mode}")
