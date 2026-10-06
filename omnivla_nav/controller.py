"""OmniVLA が出力した waypoint 列を (v, w) 速度指令に変換するコントローラ (ROS 非依存).

mode="upstream" (公式):
    OmniVLA 公式 inference/run_omnivla.py の制御則を忠実に再現したもの。
    (waypoint[4] を DT=1/3 で追従 -> 0.5 / 1.0 でクリップ -> v<=0.3, w<=0.3 に曲率を保って制限)
    公式コードで未定義だった clip_angle もここで実装している。
    ただし目標点が後方 (dx<0) のとき公式の atan(dy/dx) は旋回方向が逆になるため atan2 に直している。
    注意: 予測した動きをそのまま実行する式ではない。
      * 角速度が maxw=0.3rad/s で頭打ち (学習データのお手本は最大 0.8rad/s で曲がる)
      * 0.5/1.0 のクリップと曲率保存の制限により、旋回半径 0.5m 未満の予測は 0.5m 以上に引き伸ばされる
        (クリップが掛からない範囲でも曲率 atan(dy/dx)/dx は予測点を通る円弧の曲率 2dy/(dx²+dy²) の約半分)
    そのため鋭く曲がる予測ほどロボットは外側に膨らみ、経路から外れる (office_0 の机の角で 0.6m 外れて衝突した)。
mode="trajectory" (既定):
    予測軌跡そのものを再現する。waypoint k は学習データの記録周期 (3Hz) で (k+1)*dt 秒後の姿勢なので、
    先頭 track_horizon+1 点の「向きの変化」と「進んだ道のり」を時間で最小二乗フィットして (v, w) を決める。
    モデルが予測した旋回の速さ・曲率・減速をそのまま実行する (ゴール方向へ向ける等の規則は入れない)。
mode="pure_pursuit":
    予測軌跡上の前方注視点に向かう pure pursuit。
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
    mode: str = "trajectory"        # "trajectory" | "upstream" | "pure_pursuit"
    # --- trajectory ---
    track_horizon: int = 4          # wp0..wp{track_horizon} (≒1.7 秒分) に合わせる
    track_max_v: float = 0.4        # 学習データ (お手本) の速度 0.3m/s に余裕を持たせた上限
    track_max_w: float = 1.0        # お手本の最大角速度 0.8rad/s に余裕を持たせた上限
    track_phi_scale: float = 0.15   # 予測点までの距離がこれより近いときは、位置より予測した向きで旋回量を決める [m]
    # --- upstream ---
    waypoint_index: int = 4         # 公式は waypoints[0][4] を使う
    dt: float = 1.0 / 3.0           # 公式の DT (tick_rate=3Hz)
    max_linear_raw: float = 0.5     # 公式: np.clip(linear, 0, 0.5)
    max_angular_raw: float = 1.0    # 公式: np.clip(angular, -1, 1)
    max_v: float = 0.3              # 公式: maxv
    max_w: float = 0.3              # 公式: maxw
    # モデルの速度を尊重する (公式には無い): 予測 waypoint k は (k+1)*dt 秒後の位置なので、
    # その到達に必要な速度 |wp_k| / ((k+1)*dt) を上限にする。予測軌跡が短い (= 減速/停止の予測) と遅くなる。
    # 公式の v = dx/dt は 0.1m 先でも上限速度に張り付き、壁の手前でも減速しない。曲率 (v/w) は保つ。
    respect_predicted_speed: bool = True
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


def predicted_speed(waypoints: np.ndarray, index: int, dt: float) -> float:
    """予測 waypoint index (= (index+1)*dt 秒後) まで行くのに必要な速度 [m/s]."""
    wp = np.asarray(waypoints, dtype=np.float64)[int(index)]
    return float(math.hypot(wp[0], wp[1])) / ((int(index) + 1) * dt)


def cap_to_predicted_speed(v: float, w: float, v_pred: float) -> Tuple[float, float]:
    """v を v_pred 以下にする. 曲率半径 v/w を保つため w も同じ割合で小さくする (その場旋回 v=0 はそのまま)."""
    if v <= 1e-6 or v <= v_pred:
        return v, w
    s = max(v_pred, 0.0) / v
    return v * s, w * s


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


def trajectory_command(waypoints: np.ndarray, cfg: ControllerConfig) -> Tuple[float, float]:
    """予測軌跡 (各点は (k+1)*dt 秒後) を最もよく再現する一定の (v, w)."""
    wps = np.asarray(waypoints, dtype=np.float64)
    K = int(np.clip(cfg.track_horizon, 0, len(wps) - 1))
    t = (np.arange(K + 1) + 1.0) * cfg.dt
    xy = wps[:K + 1, :2]
    yaw = np.unwrap(np.arctan2(wps[:K + 1, 3], wps[:K + 1, 2]))       # 予測された向き
    chord = np.linalg.norm(xy, axis=1)
    phi = 2.0 * np.arctan2(xy[:, 1], xy[:, 0])                        # その点を通る円弧での向きの変化
    # 道のり: 円弧長 = 弦 * (phi/2) / sin(phi/2)
    half = np.abs(phi) / 2.0
    arc = np.where(half > 1e-6, chord * half / np.maximum(np.sin(half), 1e-6), chord)
    # 向きの変化: モデルが予測した向きと、位置から決まる円弧の向きの重み付き平均
    # (点が近い = ゆっくり/その場旋回のときは位置の誤差で phi が暴れるので、予測した向きを重視する)
    a = 0.5 * chord ** 2 / (chord ** 2 + cfg.track_phi_scale ** 2)
    turn = a * phi + (1.0 - a) * yaw
    v = float(np.dot(t, arc) / np.dot(t, t))
    w = float(np.dot(t, turn) / np.dot(t, t))
    v = max(v, 0.0)
    # 速度上限: 曲率 (w/v) と比率を保ったまま全体を遅くする
    s = 1.0
    if v > cfg.track_max_v:
        s = min(s, cfg.track_max_v / v)
    if abs(w) > cfg.track_max_w:
        s = min(s, cfg.track_max_w / abs(w))
    return v * s, w * s


def compute_command(waypoints: np.ndarray, cfg: ControllerConfig) -> Tuple[float, float]:
    """waypoints: (8, 4) = [x[m], y[m], cos, sin] (ロボット座標)."""
    waypoints = np.asarray(waypoints, dtype=np.float64)
    if cfg.mode == "upstream":
        idx = int(np.clip(cfg.waypoint_index, 0, len(waypoints) - 1))
        v, w = upstream_command(waypoints[idx], cfg)
        if cfg.respect_predicted_speed:
            v, w = cap_to_predicted_speed(v, w, predicted_speed(waypoints, idx, cfg.dt))
        return v, w
    if cfg.mode == "trajectory":
        return trajectory_command(waypoints, cfg)
    if cfg.mode == "pure_pursuit":
        return pure_pursuit_command(waypoints, cfg)
    raise ValueError(f"unknown controller mode: {cfg.mode}")


class StuckDetector:
    """前進指令を出しているのに timeout 秒間ほとんど動いていなければ「動けない」(壁に押し付けている) と判定."""

    def __init__(self, timeout: float = 4.0, min_move: float = 0.05, min_turn: float = 0.1, min_cmd_v: float = 0.05):
        self.timeout, self.min_move, self.min_turn, self.min_cmd_v = timeout, min_move, min_turn, min_cmd_v
        self.ref = None   # (t, x, y, yaw) 動き出し/最後に動いた時点

    def reset(self) -> None:
        self.ref = None

    def update(self, t: float, pose, cmd_v: float) -> bool:
        if pose is None or self.timeout <= 0:
            return False
        if cmd_v < self.min_cmd_v:          # 止まる/その場旋回の指令中は判定しない
            self.ref = None
            return False
        if self.ref is None:
            self.ref = (t, pose[0], pose[1], pose[2])
            return False
        t0, x0, y0, yaw0 = self.ref
        moved = math.hypot(pose[0] - x0, pose[1] - y0)
        turned = abs((pose[2] - yaw0 + math.pi) % (2 * math.pi) - math.pi)
        if moved > self.min_move or turned > self.min_turn:
            self.ref = (t, pose[0], pose[1], pose[2])
            return False
        return t - t0 > self.timeout
