"""デバッグ用の可視化 (PIL のみ, ROS 非依存).

- 現在画像に予測軌跡を地面投影して重畳
- ゴール画像
- 俯瞰の軌跡プロットとテキスト情報
を 1 枚 (640x480) にまとめる。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

PANEL = (320, 240)


@dataclass
class CameraModel:
    """前向き水平カメラ (pitch=0) の簡易モデル. robot 座標 (x前, y左, z上)."""
    hfov: float = 1.75          # [rad] Gazebo のカメラ設定と合わせる
    height: float = 0.35        # 地面からの高さ [m]
    x_offset: float = 0.17      # ロボット原点からの前方オフセット [m]
    y_offset: float = 0.0

    def project(self, pts_xy: np.ndarray, width: int, height: int):
        pts = np.asarray(pts_xy, dtype=np.float64).reshape(-1, 2)
        fx = (width / 2.0) / math.tan(self.hfov / 2.0)
        fy = fx
        zc = pts[:, 0] - self.x_offset
        xc = -(pts[:, 1] - self.y_offset)
        yc = np.full_like(zc, self.height)
        valid = zc > 0.15
        zc_safe = np.where(valid, zc, 1.0)
        u = width / 2.0 + fx * xc / zc_safe
        v = height / 2.0 + fy * yc / zc_safe
        return u, v, valid


def _to_pil(img) -> Optional[Image.Image]:
    if img is None:
        return None
    if isinstance(img, np.ndarray):
        return Image.fromarray(img.astype(np.uint8))
    return img.convert("RGB")


def overlay_trajectory(img: Image.Image, waypoints_xy: np.ndarray, cam: CameraModel,
                       color=(255, 60, 0)) -> Image.Image:
    img = img.copy()
    w, h = img.size
    pts = np.concatenate([[[cam.x_offset + 0.16, 0.0]], np.asarray(waypoints_xy)[:, :2]], axis=0)
    u, v, ok = cam.project(pts, w, h)
    d = ImageDraw.Draw(img)
    coords = [(float(a), float(b)) for a, b, k in zip(u, v, ok) if k]
    if len(coords) >= 2:
        d.line(coords, fill=color, width=max(2, w // 160))
    for (a, b) in coords[1:]:
        r = max(2, w // 120)
        d.ellipse([a - r, b - r, a + r, b + r], outline=(255, 255, 255), fill=color)
    return img


def topdown_plot(waypoints_xy: Optional[np.ndarray], goal_local: Optional[Tuple[float, float]] = None,
                 extra_paths: Sequence[Tuple[np.ndarray, Tuple[int, int, int]]] = (),
                 size=PANEL, x_range=(-0.5, 2.5)) -> Image.Image:
    """ロボット座標の俯瞰図. 上が前方 (x), 左が +y."""
    w, h = size
    img = Image.new("RGB", size, (250, 250, 250))
    d = ImageDraw.Draw(img)
    scale = h / (x_range[1] - x_range[0])  # px / m

    def to_px(x, y):
        return w / 2.0 - y * scale, h - (x - x_range[0]) * scale

    # grid (0.5m)
    for gx in np.arange(math.ceil(x_range[0] * 2) / 2, x_range[1] + 1e-6, 0.5):
        _, py = to_px(gx, 0)
        d.line([(0, py), (w, py)], fill=(225, 225, 225))
    for gy in np.arange(-3.0, 3.01, 0.5):
        px, _ = to_px(0, gy)
        d.line([(px, 0), (px, h)], fill=(225, 225, 225))
    # robot
    rx, ry = to_px(0, 0)
    d.polygon([(rx, ry - 10), (rx - 7, ry + 6), (rx + 7, ry + 6)], fill=(60, 60, 60))
    for path, color in extra_paths:
        pts = [to_px(p[0], p[1]) for p in np.asarray(path)[:, :2]]
        if len(pts) >= 2:
            d.line([(rx, ry)] + pts, fill=color, width=2)
    if waypoints_xy is not None:
        pts = [to_px(p[0], p[1]) for p in np.asarray(waypoints_xy)[:, :2]]
        d.line([(rx, ry)] + pts, fill=(0, 90, 255), width=3)
        for (px, py) in pts:
            d.ellipse([px - 3, py - 3, px + 3, py + 3], fill=(0, 90, 255))
    if goal_local is not None:
        gx, gy = goal_local
        px, py = to_px(gx, gy)
        clipped = not (0 <= px < w and 0 <= py < h)
        px, py = min(max(px, 6), w - 6), min(max(py, 6), h - 6)
        color = (230, 30, 30) if not clipped else (230, 140, 30)
        d.regular_polygon((px, py, 7), 5, fill=color)
    return img


def render_debug(current, goal=None, waypoints: Optional[np.ndarray] = None,
                 cam: Optional[CameraModel] = None, goal_local: Optional[Tuple[float, float]] = None,
                 lines: Sequence[str] = (), gt_waypoints: Optional[np.ndarray] = None) -> Image.Image:
    """waypoints (予測, 青/橙) と gt_waypoints (正解, 緑) は [x[m], y[m], ...] (ロボット座標)."""
    cam = cam or CameraModel()
    canvas = Image.new("RGB", (PANEL[0] * 2, PANEL[1] * 2), (30, 30, 30))
    cur = _to_pil(current)
    if cur is not None:
        if gt_waypoints is not None:
            cur = overlay_trajectory(cur, gt_waypoints, cam, color=(40, 200, 40))
        if waypoints is not None:
            cur = overlay_trajectory(cur, waypoints, cam)
        canvas.paste(cur.resize(PANEL), (0, 0))
    g = _to_pil(goal)
    if g is not None:
        canvas.paste(g.resize(PANEL), (PANEL[0], 0))
    extra = [(gt_waypoints, (40, 180, 40))] if gt_waypoints is not None else []
    canvas.paste(topdown_plot(waypoints, goal_local, extra_paths=extra), (0, PANEL[1]))
    d = ImageDraw.Draw(canvas)
    d.text((6, 4), "current", fill=(255, 255, 0))
    d.text((PANEL[0] + 6, 4), "goal", fill=(255, 255, 0))
    y = PANEL[1] + 8
    for line in lines:
        d.text((PANEL[0] + 8, y), str(line), fill=(235, 235, 235))
        y += 14
    return canvas
