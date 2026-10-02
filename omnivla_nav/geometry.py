"""2D 幾何ユーティリティ (ROS 非依存).

座標系の約束:
  - ワールド/odom 座標: ROS (ENU) と同じ。yaw は x 軸から反時計回り [rad]
  - ロボット座標: x = 前方, y = 左  (OmniVLA / GNM の waypoint と同じ)
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple

import numpy as np

Pose2D = Tuple[float, float, float]  # (x, y, yaw)


def wrap_angle(angle):
    """角度を [-pi, pi) に正規化 (スカラー / ndarray 両対応)."""
    if isinstance(angle, np.ndarray):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quaternion_from_yaw(yaw: float) -> Tuple[float, float, float, float]:
    """(x, y, z, w)"""
    return (0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5))


def rotation_matrix(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def to_local(points_xy, origin_xy: Sequence[float], origin_yaw: float) -> np.ndarray:
    """ワールド座標の点列 (N,2) or (2,) を origin のロボット座標 (x前, y左) に変換."""
    pts = np.asarray(points_xy, dtype=np.float64)
    delta = pts - np.asarray(origin_xy, dtype=np.float64)
    # R(-yaw) を掛ける。行ベクトルなので delta @ R(-yaw)^T = delta @ R(yaw)
    return delta @ rotation_matrix(origin_yaw)


def to_world(local_xy, origin_xy: Sequence[float], origin_yaw: float) -> np.ndarray:
    """ロボット座標の点列 (N,2) or (2,) をワールド座標に変換 (to_local の逆)."""
    pts = np.asarray(local_xy, dtype=np.float64)
    return pts @ rotation_matrix(origin_yaw).T + np.asarray(origin_xy, dtype=np.float64)


def relative_pose(current: Pose2D, target: Pose2D) -> Pose2D:
    """current から見た target の相対姿勢 (x前, y左, dyaw)."""
    local = to_local(np.array(target[:2]), current[:2], current[2])
    return float(local[0]), float(local[1]), float(wrap_angle(target[2] - current[2]))


def path_length(points_xy: Iterable[Sequence[float]]) -> float:
    pts = np.asarray(list(points_xy), dtype=np.float64)
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
