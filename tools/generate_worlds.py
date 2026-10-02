#!/usr/bin/env python3
"""Gazebo (Harmonic) 用のワールドとテクスチャを手続き的に生成する.

実写で学習された OmniVLA が Gazebo でもそれなりに見えるように、
単色ボックスではなくテクスチャ付き (木目床・レンガ・ポスター・草地など) の環境を作る。
seed を変えるとレイアウトが変わるので、データ収集時のドメインランダム化にも使える。

生成物:
  ros2_ws/src/omnivla_gazebo/models/omnivla_textures/materials/textures/*.png
  ros2_ws/src/omnivla_gazebo/worlds/<name>.sdf
  ros2_ws/src/omnivla_gazebo/maps/<name>.{pgm,yaml,png}   (2D 占有格子, 確認用)

例:
  python3 tools/generate_worlds.py                       # 既定 (office_0, park_0) + テクスチャ
  python3 tools/generate_worlds.py --worlds office --seeds 1 2 3
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
PKG = os.path.join(REPO, "ros2_ws", "src", "omnivla_gazebo")
TEX_MODEL = "omnivla_textures"
TEX_DIR = os.path.join(PKG, "models", TEX_MODEL, "materials", "textures")
TEX_URI = f"model://{TEX_MODEL}/materials/textures"
ROBOT_NAME = "omnivla_robot"
TEX_SIZE = 256


# =============================================================================
# Textures
# =============================================================================
def _noise(rng: np.random.Generator, size: int, base: int, octaves: int = 4, persistence: float = 0.5,
           stretch: Tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
    """値ノイズ (0..1). stretch で縦横に引き伸ばす (木目など)."""
    total = np.zeros((size, size), np.float64)
    amp, norm = 1.0, 0.0
    for o in range(octaves):
        cells = max(2, int(base * (2 ** o)))
        gw = max(2, int(cells / stretch[0]))
        gh = max(2, int(cells / stretch[1]))
        grid = rng.random((gh, gw))
        img = Image.fromarray((grid * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC)
        total += amp * (np.asarray(img, np.float64) / 255.0)
        norm += amp
        amp *= persistence
    total /= norm
    return (total - total.min()) / (total.max() - total.min() + 1e-9)


def _tint(noise: np.ndarray, c0, c1) -> np.ndarray:
    c0, c1 = np.asarray(c0, np.float64), np.asarray(c1, np.float64)
    return (c0[None, None] + noise[..., None] * (c1 - c0)[None, None]).clip(0, 255)


def _img(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.asarray(arr).clip(0, 255).astype(np.uint8), "RGB")


def tex_wood_floor(rng):
    s = TEX_SIZE
    arr = np.zeros((s, s, 3))
    plank_h = 32
    for row in range(s // plank_h):
        offset = int(rng.integers(0, s))
        for seg in range(3):
            base = np.array([150, 100, 60]) + rng.integers(-25, 25, 3)
            grain = _noise(rng, s, 2, 3, stretch=(8.0, 1.0))
            col = _tint(grain, base * 0.75, base * 1.1)
            x0 = (offset + seg * s // 2) % s
            sl = slice(row * plank_h, (row + 1) * plank_h)
            for x in range(s // 2 + 1):
                arr[sl, (x0 + x) % s] = col[sl, (x0 + x) % s]
        arr[row * plank_h:row * plank_h + 2] *= 0.55
    return _img(arr)


def tex_tile_floor(rng, color=(205, 200, 190)):
    s, n = TEX_SIZE, 4
    arr = _tint(_noise(rng, s, 6, 3), np.array(color) * 0.9, np.array(color) * 1.05)
    t = s // n
    for i in range(n):
        for j in range(n):
            arr[i * t:(i + 1) * t, j * t:(j + 1) * t] *= 0.93 + 0.12 * rng.random()
    for k in range(n + 1):
        arr[max(0, k * t - 2):k * t + 2] = 120
        arr[:, max(0, k * t - 2):k * t + 2] = 120
    return _img(arr)


def tex_carpet(rng, color):
    s = TEX_SIZE
    arr = _tint(_noise(rng, s, 32, 2), np.array(color) * 0.75, np.array(color) * 1.1)
    yy, xx = np.mgrid[0:s, 0:s]
    arr *= (0.92 + 0.08 * (((xx + yy) // 8) % 2))[..., None]
    return _img(arr)


def tex_concrete(rng):
    s = TEX_SIZE
    arr = _tint(_noise(rng, s, 4, 5), (120, 120, 118), (185, 185, 180))
    sp = rng.random((s, s)) < 0.01
    arr[sp] *= 0.6
    return _img(arr)


def tex_grass(rng):
    s = TEX_SIZE
    arr = _tint(_noise(rng, s, 3, 5), (45, 95, 30), (110, 165, 60))
    img = _img(arr)
    d = ImageDraw.Draw(img)
    for _ in range(1400):
        x, y = rng.integers(0, s, 2)
        l = int(rng.integers(3, 8))
        a = rng.uniform(-0.6, 0.6)
        c = tuple(int(v) for v in (rng.integers(30, 90), rng.integers(110, 190), rng.integers(20, 60)))
        d.line([(x, y), (x + l * math.sin(a), y - l * math.cos(a))], fill=c, width=1)
    return img


def tex_asphalt(rng):
    s = TEX_SIZE
    arr = _tint(_noise(rng, s, 16, 3), (45, 45, 48), (85, 85, 88))
    sp = rng.random((s, s)) < 0.03
    arr[sp] = 150
    return _img(arr)


def tex_paving(rng):
    s, n = TEX_SIZE, 8
    arr = _tint(_noise(rng, s, 8, 3), (150, 140, 130), (200, 190, 175))
    t = s // n
    for i in range(n):
        off = (t // 2) * (i % 2)
        for j in range(n + 1):
            x0 = j * t + off
            arr[i * t:(i + 1) * t, max(0, x0 - 1):min(s, x0 + 1)] = 95
        arr[max(0, i * t - 1):i * t + 1] = 95
    return _img(arr)


def tex_brick(rng, base=(160, 70, 50), mortar=(190, 185, 175)):
    s = TEX_SIZE
    arr = np.zeros((s, s, 3)) + np.array(mortar)
    rows, cols = 10, 4
    bh, bw = s / rows, s / cols
    nz = _noise(rng, s, 16, 2)
    for r in range(rows):
        off = 0 if r % 2 == 0 else bw / 2
        for c in range(-1, cols + 1):
            x0, x1 = int(c * bw + off + 2), int((c + 1) * bw + off - 2)
            y0, y1 = int(r * bh + 2), int((r + 1) * bh - 2)
            x0c, x1c = max(0, x0), min(s, x1)
            if x1c <= x0c:
                continue
            tint = np.array(base) * (0.8 + 0.35 * rng.random())
            arr[y0:y1, x0c:x1c] = tint[None, None] * (0.85 + 0.25 * nz[y0:y1, x0c:x1c, None])
    return _img(arr)


def tex_paint(rng, color):
    s = TEX_SIZE
    arr = _tint(_noise(rng, s, 3, 4), np.array(color) * 0.92, np.array(color) * 1.03)
    img = _img(arr)
    d = ImageDraw.Draw(img)
    # 小物 (スイッチ・コンセント・掲示) で特徴点を作る
    for _ in range(int(rng.integers(1, 4))):
        x, y = rng.integers(20, s - 40, 2)
        w, h = rng.integers(8, 30, 2)
        c = tuple(int(v) for v in rng.integers(40, 250, 3))
        d.rectangle([x, y, x + w, y + h], fill=c, outline=(60, 60, 60))
    return img


def tex_poster(rng):
    s = TEX_SIZE
    bg = tuple(int(v) for v in rng.integers(150, 255, 3))
    img = Image.new("RGB", (s, s), bg)
    d = ImageDraw.Draw(img)
    for _ in range(int(rng.integers(4, 9))):
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        x0, y0 = rng.integers(0, s - 30, 2)
        x1, y1 = x0 + rng.integers(30, 140), y0 + rng.integers(30, 140)
        kind = rng.integers(0, 3)
        if kind == 0:
            d.rectangle([x0, y0, x1, y1], fill=c)
        elif kind == 1:
            d.ellipse([x0, y0, x1, y1], fill=c)
        else:
            d.polygon([(x0, y1), ((x0 + x1) // 2, y0), (x1, y1)], fill=c)
    for k in range(int(rng.integers(2, 5))):  # 文字列風のバー
        y = int(rng.integers(10, s - 20))
        d.rectangle([20, y, 20 + int(rng.integers(60, 200)), y + 6], fill=(30, 30, 30))
    d.rectangle([0, 0, s - 1, s - 1], outline=(20, 20, 20), width=6)
    return img


def tex_bookshelf(rng):
    s = TEX_SIZE
    img = Image.new("RGB", (s, s), (110, 75, 45))
    d = ImageDraw.Draw(img)
    shelves = 5
    sh = s // shelves
    for r in range(shelves):
        y0 = r * sh + 6
        x = 6
        while x < s - 12:
            w = int(rng.integers(6, 16))
            h = int(rng.integers(sh - 22, sh - 8))
            c = tuple(int(v) for v in rng.integers(20, 230, 3))
            d.rectangle([x, y0 + (sh - 8 - h), x + w, y0 + sh - 8], fill=c, outline=(20, 20, 20))
            x += w + 1
        d.rectangle([0, r * sh + sh - 6, s, r * sh + sh], fill=(80, 55, 30))
    return img


def tex_facade(rng, wall):
    s = TEX_SIZE
    arr = _tint(_noise(rng, s, 6, 3), np.array(wall) * 0.88, np.array(wall) * 1.05)
    img = _img(arr)
    d = ImageDraw.Draw(img)
    nx, ny = 3, 3
    for i in range(nx):
        for j in range(ny):
            x0 = int(s * (i + 0.2) / nx)
            y0 = int(s * (j + 0.2) / ny)
            x1 = int(s * (i + 0.8) / nx)
            y1 = int(s * (j + 0.75) / ny)
            glass = tuple(int(v) for v in (60 + rng.integers(0, 40), 90 + rng.integers(0, 50), 120 + rng.integers(0, 60)))
            d.rectangle([x0, y0, x1, y1], fill=glass, outline=(235, 235, 235), width=4)
            d.line([((x0 + x1) // 2, y0), ((x0 + x1) // 2, y1)], fill=(235, 235, 235), width=3)
    return img


def tex_bark(rng):
    s = TEX_SIZE
    return _img(_tint(_noise(rng, s, 4, 4, stretch=(1.0, 10.0)), (60, 40, 25), (125, 90, 60)))


def tex_leaves(rng):
    s = TEX_SIZE
    img = _img(_tint(_noise(rng, s, 8, 3), (30, 70, 25), (70, 120, 40)))
    d = ImageDraw.Draw(img)
    for _ in range(500):
        x, y = rng.integers(0, s, 2)
        r = int(rng.integers(3, 9))
        c = tuple(int(v) for v in (rng.integers(20, 100), rng.integers(80, 170), rng.integers(15, 60)))
        d.ellipse([x - r, y - r, x + r, y + r], fill=c)
    return img


def tex_metal(rng, color=(150, 155, 160)):
    s = TEX_SIZE
    arr = _tint(_noise(rng, s, 2, 3, stretch=(12.0, 1.0)), np.array(color) * 0.8, np.array(color) * 1.1)
    img = _img(arr)
    d = ImageDraw.Draw(img)
    for x in (12, s - 12):
        for y in (12, s - 12):
            d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(90, 90, 95))
    d.rectangle([0, 0, s - 1, s - 1], outline=(70, 70, 75), width=4)
    return img


def tex_crate(rng):
    s = TEX_SIZE
    arr = _tint(_noise(rng, s, 2, 3, stretch=(8.0, 1.0)), (140, 100, 55), (200, 155, 95))
    img = _img(arr)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, s - 1, s - 1], outline=(95, 65, 35), width=18)
    d.line([(10, 10), (s - 10, s - 10)], fill=(105, 72, 40), width=16)
    return img


def tex_cardboard(rng):
    s = TEX_SIZE
    img = _img(_tint(_noise(rng, s, 16, 2), (175, 135, 85), (210, 170, 115)))
    d = ImageDraw.Draw(img)
    d.rectangle([0, s // 2 - 14, s, s // 2 + 14], fill=(215, 200, 160))
    d.polygon([(s // 2 - 20, 40), (s // 2, 15), (s // 2 + 20, 40)], fill=(40, 40, 40))
    return img


def tex_fabric(rng, color):
    s = TEX_SIZE
    arr = _tint(_noise(rng, s, 24, 2), np.array(color) * 0.8, np.array(color) * 1.1)
    yy, xx = np.mgrid[0:s, 0:s]
    arr *= (0.9 + 0.1 * ((xx % 4 < 2) ^ (yy % 4 < 2)))[..., None]
    return _img(arr)


def tex_rock(rng):
    s = TEX_SIZE
    return _img(_tint(_noise(rng, s, 6, 5), (85, 85, 80), (160, 155, 145)))


def tex_stripes(rng, c0=(240, 110, 20), c1=(245, 245, 245)):
    s = TEX_SIZE
    yy = np.mgrid[0:s, 0:s][0]
    band = ((yy // 32) % 2 == 0)[..., None]
    return _img(np.where(band, np.array(c0), np.array(c1)))


PAINTS = {
    "paint_cream": (235, 225, 200), "paint_blue": (150, 180, 215), "paint_green": (170, 205, 165),
    "paint_salmon": (230, 170, 150), "paint_gray": (190, 190, 195), "paint_yellow": (235, 215, 140),
}
FABRICS = {"fabric_red": (170, 50, 50), "fabric_blue": (50, 80, 160), "fabric_green": (60, 120, 70),
           "fabric_gray": (110, 110, 115)}
CARPETS = {"carpet_blue": (70, 90, 140), "carpet_gray": (120, 120, 125), "carpet_red": (140, 70, 70)}
CARS = {"car_red": (180, 30, 30), "car_blue": (30, 60, 160), "car_white": (225, 225, 225),
        "car_black": (40, 40, 45), "car_yellow": (220, 190, 40)}


def generate_textures(seed: int = 0) -> List[str]:
    rng = np.random.default_rng(seed)
    os.makedirs(TEX_DIR, exist_ok=True)
    tex: Dict[str, Image.Image] = {
        "wood_floor": tex_wood_floor(rng),
        "tile_floor": tex_tile_floor(rng),
        "tile_floor_dark": tex_tile_floor(rng, (150, 150, 155)),
        "concrete": tex_concrete(rng),
        "grass": tex_grass(rng),
        "asphalt": tex_asphalt(rng),
        "paving": tex_paving(rng),
        "brick_red": tex_brick(rng),
        "brick_white": tex_brick(rng, base=(215, 210, 200), mortar=(150, 150, 150)),
        "bookshelf": tex_bookshelf(rng),
        "bark": tex_bark(rng),
        "leaves": tex_leaves(rng),
        "metal": tex_metal(rng),
        "crate": tex_crate(rng),
        "cardboard": tex_cardboard(rng),
        "rock": tex_rock(rng),
        "stripes_orange": tex_stripes(rng),
        "facade_beige": tex_facade(rng, (210, 195, 165)),
        "facade_gray": tex_facade(rng, (165, 165, 170)),
        "facade_red": tex_facade(rng, (170, 95, 80)),
    }
    for i in range(8):
        tex[f"poster_{i}"] = tex_poster(rng)
    for name, c in PAINTS.items():
        tex[name] = tex_paint(rng, c)
    for name, c in FABRICS.items():
        tex[name] = tex_fabric(rng, c)
    for name, c in CARPETS.items():
        tex[name] = tex_carpet(rng, c)
    for name, c in CARS.items():
        tex[name] = tex_metal(rng, c)
    for name, img in tex.items():
        img = img.filter(ImageFilter.SMOOTH)
        img.save(os.path.join(TEX_DIR, f"{name}.png"), optimize=True)
    _write_texture_model()
    return sorted(tex)


def _write_texture_model() -> None:
    mdir = os.path.join(PKG, "models", TEX_MODEL)
    with open(os.path.join(mdir, "model.config"), "w") as f:
        f.write(f"""<?xml version="1.0"?>
<model>
  <name>{TEX_MODEL}</name>
  <version>1.0</version>
  <sdf version="1.9">model.sdf</sdf>
  <description>Procedurally generated textures for omnivla_gazebo worlds (tools/generate_worlds.py).</description>
</model>
""")
    with open(os.path.join(mdir, "model.sdf"), "w") as f:
        f.write(f"""<?xml version="1.0"?>
<sdf version="1.9">
  <model name="{TEX_MODEL}_sample">
    <static>true</static>
    <link name="link">
      <visual name="visual">
        <geometry><box><size>1 1 1</size></box></geometry>
        {material('brick_red')}
      </visual>
    </link>
  </model>
</sdf>
""")


# =============================================================================
# SDF helpers
# =============================================================================
def material(tex: Optional[str] = None, color=(0.8, 0.8, 0.8), roughness: float = 0.85,
             metalness: float = 0.0) -> str:
    if tex:
        color = (1.0, 1.0, 1.0)
    c = " ".join(f"{v:.3f}" for v in color) + " 1"
    s = f"<material><ambient>{c}</ambient><diffuse>{c}</diffuse><specular>0.08 0.08 0.08 1</specular>"
    if tex:
        s += (f"<pbr><metal><albedo_map>{TEX_URI}/{tex}.png</albedo_map>"
              f"<roughness>{roughness}</roughness><metalness>{metalness}</metalness></metal></pbr>")
    return s + "</material>"


def geom_box(sx, sy, sz):
    return f"<box><size>{sx:.3f} {sy:.3f} {sz:.3f}</size></box>"


def geom_cyl(r, h):
    return f"<cylinder><radius>{r:.3f}</radius><length>{h:.3f}</length></cylinder>"


def geom_sphere(r):
    return f"<sphere><radius>{r:.3f}</radius></sphere>"


def geom_plane(sx, sy):
    return f"<plane><normal>0 0 1</normal><size>{sx:.3f} {sy:.3f}</size></plane>"


def pose_str(x, y, z, yaw=0.0, roll=0.0, pitch=0.0):
    return f"<pose>{x:.3f} {y:.3f} {z:.3f} {roll:.4f} {pitch:.4f} {yaw:.4f}</pose>"


@dataclass
class Part:
    geom: str
    mat: str
    pose: Tuple[float, ...]   # (x, y, z, yaw[, roll, pitch]) リンク座標
    collide: bool = True


@dataclass
class StaticModel:
    name: str
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    parts: List[Part] = field(default_factory=list)

    def to_sdf(self) -> str:
        items = []
        for i, p in enumerate(self.parts):
            ps = pose_str(*p.pose[:3], *(p.pose[3:4] or (0.0,)), *p.pose[4:6])
            if p.collide:
                items.append(f'<collision name="c{i}">{ps}<geometry>{p.geom}</geometry></collision>')
            items.append(f'<visual name="v{i}">{ps}<geometry>{p.geom}</geometry>{p.mat}</visual>')
        body = "\n        ".join(items)
        return f"""    <model name="{self.name}">
      <static>true</static>
      {pose_str(self.x, self.y, 0.0, self.yaw)}
      <link name="link">
        {body}
      </link>
    </model>"""


# footprints for placement checks: ("rect", cx, cy, yaw, sx, sy) or ("circle", cx, cy, r)
Footprint = Tuple


def _fp_points(fp: Footprint, n: int = 16) -> np.ndarray:
    if fp[0] == "circle":
        _, cx, cy, r = fp
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1)
    _, cx, cy, yaw, sx, sy = fp
    c, s = math.cos(yaw), math.sin(yaw)
    loc = np.array([[sx / 2, sy / 2], [-sx / 2, sy / 2], [-sx / 2, -sy / 2], [sx / 2, -sy / 2]])
    # 辺上の点も含める
    pts = []
    for a, b in zip(loc, np.roll(loc, -1, axis=0)):
        for t in np.linspace(0, 1, 5, endpoint=False):
            pts.append(a + t * (b - a))
    pts = np.asarray(pts)
    return pts @ np.array([[c, s], [-s, c]]) + np.array([cx, cy])


def _fp_dist(a: Footprint, b: Footprint) -> float:
    """足跡同士の (近似) 最短距離. 負なら重なり."""
    pa, pb = _fp_points(a), _fp_points(b)
    d = np.linalg.norm(pa[:, None] - pb[None], axis=2).min()
    # 片方がもう片方の内側にある場合の判定 (中心で近似)
    ca = np.array(a[1:3])
    cb = np.array(b[1:3])
    if _inside(ca, b) or _inside(cb, a):
        return -1.0
    return float(d)


def _inside(p, fp: Footprint) -> bool:
    if fp[0] == "circle":
        return np.linalg.norm(p - np.array(fp[1:3])) <= fp[3]
    _, cx, cy, yaw, sx, sy = fp
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = p[0] - cx, p[1] - cy
    lx, ly = c * dx + s * dy, -s * dx + c * dy
    return abs(lx) <= sx / 2 and abs(ly) <= sy / 2


class World:
    def __init__(self, name: str, seed: int):
        self.name = name
        self.rng = np.random.default_rng(seed)
        self.models: List[StaticModel] = []
        self.footprints: List[Footprint] = []
        self.keepouts: List[Footprint] = []  # 通路・ドアなど物を置かない領域
        self._count = 0

    def uid(self, prefix: str) -> str:
        self._count += 1
        return f"{prefix}_{self._count}"

    def add(self, model: StaticModel, footprint: Optional[Footprint] = None):
        self.models.append(model)
        if footprint is not None:
            self.footprints.append(footprint)

    def free_for(self, fp: Footprint, clearance: float) -> bool:
        for other in self.footprints:
            if _fp_dist(fp, other) < clearance:
                return False
        for k in self.keepouts:
            if _fp_dist(fp, k) < 0.05:
                return False
        return True

    def to_sdf(self, spawn: Tuple[float, float, float], scene_sky: bool = True, ambient: float = 0.55,
               sun_dir=(-0.4, 0.25, -0.88)) -> str:
        models = "\n".join(m.to_sdf() for m in self.models)
        sky = "<sky></sky>" if scene_sky else ""
        return f"""<?xml version="1.0" ?>
<!-- Generated by tools/generate_worlds.py (seeded procedural world). Do not edit by hand; re-generate instead. -->
<sdf version="1.9">
  <world name="{self.name}">
    <physics name="2ms" type="ignored">
      <max_step_size>0.002</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>

    <scene>
      <ambient>{ambient} {ambient} {ambient} 1</ambient>
      <background>0.70 0.80 0.95 1</background>
      <shadows>true</shadows>
      <grid>false</grid>
      {sky}
    </scene>

    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 20 0 0 0</pose>
      <diffuse>0.95 0.93 0.88 1</diffuse>
      <specular>0.3 0.3 0.3 1</specular>
      <direction>{sun_dir[0]} {sun_dir[1]} {sun_dir[2]}</direction>
    </light>

    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>{geom_plane(300, 300)}</geometry>
          <surface><friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction></surface>
        </collision>
        <visual name="visual">
          <geometry>{geom_plane(300, 300)}</geometry>
          {material(color=(0.35, 0.37, 0.33))}
        </visual>
      </link>
    </model>

{models}

    <include>
      <uri>model://{ROBOT_NAME}</uri>
      <name>{ROBOT_NAME}</name>
      {pose_str(spawn[0], spawn[1], 0.02, spawn[2])}
    </include>
  </world>
</sdf>
"""


# =============================================================================
# Builders shared by worlds
# =============================================================================
def floor_tiles(world: World, x0, y0, x1, y1, tile: float, tex_fn, z: float = 0.005, name="floor"):
    parts = []
    nx = int(math.ceil((x1 - x0) / tile))
    ny = int(math.ceil((y1 - y0) / tile))
    for i in range(nx):
        for j in range(ny):
            cx0, cy0 = x0 + i * tile, y0 + j * tile
            sx, sy = min(tile, x1 - cx0), min(tile, y1 - cy0)
            cx, cy = cx0 + sx / 2, cy0 + sy / 2
            tex = tex_fn(cx, cy)
            if tex is None:
                continue
            parts.append(Part(geom_plane(sx, sy), material(tex, roughness=0.95), (cx, cy, z), collide=False))
    world.add(StaticModel(world.uid(name), parts=parts))


def wall_line(world: World, p0, p1, height: float, thickness: float, tex_choices: Sequence[str],
              doors: Sequence[Tuple[float, float]] = (), seg_len: float = 2.0, landmark_p: float = 0.15,
              landmark_tex: Sequence[str] = (), name="wall"):
    """p0->p1 の直線壁. doors = [(中心の距離 s, 幅)] は開口部."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    length = float(np.linalg.norm(p1 - p0))
    yaw = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    u = (p1 - p0) / length
    # 開口部を除いた区間
    intervals = [(0.0, length)]
    for (sc, w) in doors:
        new = []
        for a, b in intervals:
            lo, hi = sc - w / 2, sc + w / 2
            if hi <= a or lo >= b:
                new.append((a, b))
                continue
            if lo > a:
                new.append((a, lo))
            if hi < b:
                new.append((hi, b))
        intervals = new
    base_tex = tex_choices[int(world.rng.integers(len(tex_choices)))]
    parts = []
    for a, b in intervals:
        n = max(1, int(math.ceil((b - a) / seg_len)))
        for k in range(n):
            s0 = a + (b - a) * k / n
            s1 = a + (b - a) * (k + 1) / n
            c = p0 + u * (s0 + s1) / 2
            tex = base_tex
            if landmark_tex and world.rng.random() < landmark_p:
                tex = landmark_tex[int(world.rng.integers(len(landmark_tex)))]
            parts.append(Part(geom_box(s1 - s0 + 0.002, thickness, height), material(tex),
                              (float(c[0]), float(c[1]), height / 2, yaw)))
            world.footprints.append(("rect", float(c[0]), float(c[1]), yaw, s1 - s0, thickness))
    world.models.append(StaticModel(world.uid(name), parts=parts))


def place_random(world: World, region: Tuple[float, float, float, float], make_fn, n: int,
                 clearance: float = 0.9, tries: int = 200, wall_snap: bool = False):
    """region 内に make_fn(x, y, yaw) -> (StaticModel, footprint) を n 個置く."""
    x0, y0, x1, y1 = region
    placed = 0
    for _ in range(tries):
        if placed >= n:
            break
        x = float(world.rng.uniform(x0, x1))
        y = float(world.rng.uniform(y0, y1))
        yaw = float(world.rng.choice([0.0, math.pi / 2, math.pi, -math.pi / 2])) if wall_snap \
            else float(world.rng.uniform(-math.pi, math.pi))
        model, fp = make_fn(x, y, yaw)
        if world.free_for(fp, clearance):
            world.add(model, fp)
            placed += 1
    return placed


# ---- furniture / props ----------------------------------------------------
def make_box_obj(world, prefix, x, y, yaw, size, tex):
    sx, sy, sz = size
    m = StaticModel(world.uid(prefix), x, y, yaw, [Part(geom_box(sx, sy, sz), material(tex), (0, 0, sz / 2))])
    return m, ("rect", x, y, yaw, sx, sy)


def make_desk(world, x, y, yaw):
    return make_box_obj(world, "desk", x, y, yaw, (1.4, 0.7, 0.75), "wood_floor")


def make_cabinet(world, x, y, yaw):
    return make_box_obj(world, "cabinet", x, y, yaw, (0.5, 0.5, 1.2), "metal")


def make_shelf(world, x, y, yaw):
    return make_box_obj(world, "shelf", x, y, yaw, (1.0, 0.35, 1.8), "bookshelf")


def make_sofa(world, x, y, yaw):
    tex = world.rng.choice(list(FABRICS))
    sx, sy = 1.8, 0.8
    m = StaticModel(world.uid("sofa"), x, y, yaw, [
        Part(geom_box(sx, sy, 0.45), material(tex), (0, 0, 0.225)),
        Part(geom_box(sx, 0.2, 0.45), material(tex), (0, -sy / 2 + 0.1, 0.675)),
    ])
    return m, ("rect", x, y, yaw, sx, sy)


def make_crate(world, x, y, yaw):
    s = float(world.rng.uniform(0.45, 0.8))
    tex = "crate" if world.rng.random() < 0.5 else "cardboard"
    parts = [Part(geom_box(s, s, s), material(tex), (0, 0, s / 2))]
    if world.rng.random() < 0.4:
        s2 = s * 0.7
        parts.append(Part(geom_box(s2, s2, s2), material("cardboard"), (0.05, 0.03, s + s2 / 2, 0.3)))
    return StaticModel(world.uid("crate"), x, y, yaw, parts), ("rect", x, y, yaw, s, s)


def make_plant(world, x, y, yaw):
    r = 0.22
    m = StaticModel(world.uid("plant"), x, y, yaw, [
        Part(geom_cyl(r, 0.4), material("brick_red"), (0, 0, 0.2)),
        Part(geom_sphere(0.42), material("leaves"), (0, 0, 0.75)),
    ])
    return m, ("circle", x, y, 0.42)


def make_bin(world, x, y, yaw):
    colors = ["metal", "car_blue", "car_yellow", "fabric_green"]
    tex = colors[int(world.rng.integers(len(colors)))]
    m = StaticModel(world.uid("bin"), x, y, yaw, [Part(geom_cyl(0.2, 0.6), material(tex), (0, 0, 0.3))])
    return m, ("circle", x, y, 0.2)


def make_pillar(world, x, y, yaw, height=2.4):
    m = StaticModel(world.uid("pillar"), x, y, yaw, [Part(geom_cyl(0.22, height), material("concrete"),
                                                          (0, 0, height / 2))])
    return m, ("circle", x, y, 0.22)


def make_tree(world, x, y, yaw):
    tr = float(world.rng.uniform(0.14, 0.24))
    th = float(world.rng.uniform(2.2, 3.2))
    fr = float(world.rng.uniform(1.0, 1.7))
    m = StaticModel(world.uid("tree"), x, y, yaw, [
        Part(geom_cyl(tr, th), material("bark"), (0, 0, th / 2)),
        Part(geom_sphere(fr), material("leaves"), (0, 0, th + fr * 0.6), collide=False),
    ])
    return m, ("circle", x, y, tr)


def make_bench(world, x, y, yaw):
    m = StaticModel(world.uid("bench"), x, y, yaw, [
        Part(geom_box(1.6, 0.45, 0.08), material("wood_floor"), (0, 0, 0.45)),
        Part(geom_box(1.6, 0.06, 0.4), material("wood_floor"), (0, -0.2, 0.75)),
        Part(geom_box(0.08, 0.45, 0.45), material("metal"), (0.7, 0, 0.225)),
        Part(geom_box(0.08, 0.45, 0.45), material("metal"), (-0.7, 0, 0.225)),
    ])
    return m, ("rect", x, y, yaw, 1.6, 0.45)


def make_lamp(world, x, y, yaw):
    m = StaticModel(world.uid("lamp"), x, y, yaw, [
        Part(geom_cyl(0.07, 3.2), material("metal"), (0, 0, 1.6)),
        Part(geom_sphere(0.18), material(color=(1.0, 0.95, 0.8)), (0, 0, 3.3), collide=False),
    ])
    return m, ("circle", x, y, 0.07)


def make_cone(world, x, y, yaw):
    m = StaticModel(world.uid("cone"), x, y, yaw, [
        Part(geom_box(0.4, 0.4, 0.04), material(color=(0.1, 0.1, 0.1)), (0, 0, 0.02)),
        Part(geom_cyl(0.13, 0.6), material("stripes_orange"), (0, 0, 0.32)),
    ])
    return m, ("rect", x, y, yaw, 0.4, 0.4)


def make_rock(world, x, y, yaw):
    r = float(world.rng.uniform(0.3, 0.6))
    m = StaticModel(world.uid("rock"), x, y, yaw, [Part(geom_sphere(r), material("rock"), (0, 0, r * 0.4))])
    return m, ("circle", x, y, r)


def make_car(world, x, y, yaw):
    tex = world.rng.choice(list(CARS))
    m = StaticModel(world.uid("car"), x, y, yaw, [
        Part(geom_box(4.2, 1.8, 0.8), material(tex, roughness=0.4, metalness=0.3), (0, 0, 0.6)),
        Part(geom_box(2.2, 1.6, 0.6), material(tex, roughness=0.4, metalness=0.3), (-0.3, 0, 1.3)),
        Part(geom_cyl(0.33, 0.25), material(color=(0.05, 0.05, 0.05)), (1.3, 0.85, 0.33, 0, 1.5708, 0)),
        Part(geom_cyl(0.33, 0.25), material(color=(0.05, 0.05, 0.05)), (1.3, -0.85, 0.33, 0, 1.5708, 0)),
        Part(geom_cyl(0.33, 0.25), material(color=(0.05, 0.05, 0.05)), (-1.3, 0.85, 0.33, 0, 1.5708, 0)),
        Part(geom_cyl(0.33, 0.25), material(color=(0.05, 0.05, 0.05)), (-1.3, -0.85, 0.33, 0, 1.5708, 0)),
    ])
    return m, ("rect", x, y, yaw, 4.2, 1.8)


def make_building(world, x, y, yaw, size, tex):
    sx, sy, sz = size
    m = StaticModel(world.uid("building"), x, y, yaw, [Part(geom_box(sx, sy, sz), material(tex), (0, 0, sz / 2))])
    return m, ("rect", x, y, yaw, sx, sy)


# =============================================================================
# Office (indoor)
# =============================================================================
WALL_TEX = list(PAINTS) + ["brick_white", "brick_red"]
LANDMARK_TEX = [f"poster_{i}" for i in range(8)] + ["bookshelf"]


def build_office(seed: int) -> Tuple[World, Tuple[float, float, float]]:
    w = World(f"office_{seed}", seed)
    rng = w.rng
    W, H = 22.0, 14.0
    xmin, xmax, ymin, ymax = -W / 2, W / 2, -H / 2, H / 2
    cw = 1.3  # corridor half width
    wall_h, wall_t = 2.2, 0.15
    door_w = 1.4

    # rooms: split x positions for top and bottom rows
    def splits():
        while True:
            a = float(rng.uniform(-6.0, -2.0))
            b = float(rng.uniform(2.0, 6.0))
            if b - a >= 5.0:
                return [xmin, a, b, xmax]

    top, bottom = splits(), splits()
    room_floor = {}
    floor_choices = ["wood_floor", "carpet_blue", "carpet_gray", "carpet_red", "tile_floor", "paving"]
    rooms = []
    for row, xs, (ry0, ry1) in (("top", top, (cw, ymax)), ("bottom", bottom, (ymin, -cw))):
        for i in range(3):
            rooms.append((row, xs[i], xs[i + 1], ry0, ry1))
            room_floor[(row, i)] = floor_choices[int(rng.integers(len(floor_choices)))]

    def floor_tex(cx, cy):
        if abs(cy) <= cw:
            return "tile_floor_dark"
        row = "top" if cy > 0 else "bottom"
        xs = top if row == "top" else bottom
        for i in range(3):
            if xs[i] <= cx <= xs[i + 1]:
                return room_floor[(row, i)]
        return "concrete"

    floor_tiles(w, xmin, ymin, xmax, ymax, 2.0, floor_tex)

    # outer walls
    wall_line(w, (xmin, ymin), (xmax, ymin), wall_h, wall_t, WALL_TEX, landmark_tex=LANDMARK_TEX)
    wall_line(w, (xmin, ymax), (xmax, ymax), wall_h, wall_t, WALL_TEX, landmark_tex=LANDMARK_TEX)
    wall_line(w, (xmin, ymin), (xmin, ymax), wall_h, wall_t, WALL_TEX, landmark_tex=LANDMARK_TEX)
    wall_line(w, (xmax, ymin), (xmax, ymax), wall_h, wall_t, WALL_TEX, landmark_tex=LANDMARK_TEX)

    # corridor walls with one door per room
    for row, xs, yline in (("top", top, cw), ("bottom", bottom, -cw)):
        doors = []
        for i in range(3):
            a, b = xs[i], xs[i + 1]
            c = float(rng.uniform(a + 1.2, b - 1.2))
            doors.append((c - xmin, door_w))
            # ドアの前後 (部屋側・廊下側とも 1.8m) には物を置かない
            w.keepouts.append(("rect", c, yline, 0.0, door_w + 0.6, 3.6))
        wall_line(w, (xmin, yline), (xmax, yline), wall_h, wall_t, WALL_TEX, doors=doors,
                  landmark_tex=LANDMARK_TEX, landmark_p=0.25)
        # partitions between rooms (optionally with a connecting door)
        y0, y1 = (cw, ymax) if row == "top" else (ymin, -cw)
        for xi in xs[1:3]:
            pdoors = []
            if rng.random() < 0.6:
                c = float(rng.uniform(1.3, (y1 - y0) - 1.3))
                pdoors.append((c, door_w))
                w.keepouts.append(("rect", xi, y0 + c, 0.0, 3.6, door_w + 0.6))
            wall_line(w, (xi, y0), (xi, y1), wall_h, wall_t, WALL_TEX, doors=pdoors, landmark_tex=LANDMARK_TEX)
    # keep the corridor center free
    w.keepouts.append(("rect", 0.0, 0.0, 0.0, W, 0.9))

    # furniture
    for (row, x0, x1, y0, y1) in rooms:
        inner = (x0 + 0.6, y0 + 0.6, x1 - 0.6, y1 - 0.6)
        place_random(w, inner, lambda x, y, yaw: make_desk(w, x, y, yaw), int(rng.integers(1, 3)),
                     clearance=0.9, wall_snap=True)
        place_random(w, inner, lambda x, y, yaw: make_shelf(w, x, y, yaw), int(rng.integers(0, 2)),
                     clearance=0.9, wall_snap=True)
        place_random(w, inner, lambda x, y, yaw: make_cabinet(w, x, y, yaw), int(rng.integers(0, 2)),
                     clearance=0.9, wall_snap=True)
        if rng.random() < 0.5:
            place_random(w, inner, lambda x, y, yaw: make_sofa(w, x, y, yaw), 1, clearance=0.9, wall_snap=True)
        place_random(w, inner, lambda x, y, yaw: make_plant(w, x, y, yaw), int(rng.integers(0, 3)), clearance=0.9)
        place_random(w, inner, lambda x, y, yaw: make_crate(w, x, y, yaw), int(rng.integers(0, 3)), clearance=0.9)
        place_random(w, inner, lambda x, y, yaw: make_bin(w, x, y, yaw), int(rng.integers(0, 2)), clearance=0.9)
    # corridor props near walls (center stays free via keepout)
    corridor = (xmin + 0.5, -cw + 0.3, xmax - 0.5, cw - 0.3)
    place_random(w, corridor, lambda x, y, yaw: make_plant(w, x, y, yaw), 3, clearance=1.2)
    place_random(w, corridor, lambda x, y, yaw: make_bin(w, x, y, yaw), 2, clearance=1.2)
    spawn = (float(rng.uniform(-3.0, 3.0)), 0.0, 0.0 if rng.random() < 0.5 else math.pi)
    return w, spawn


# =============================================================================
# Park (outdoor)
# =============================================================================
def build_park(seed: int) -> Tuple[World, Tuple[float, float, float]]:
    w = World(f"park_{seed}", seed)
    rng = w.rng
    L = 17.0
    path_w = 2.6
    diag = rng.random() < 0.5

    def on_path(x, y, margin=0.0):
        if abs(x) <= path_w / 2 + margin or abs(y) <= path_w / 2 + margin:
            return True
        if diag and abs(x - y) / math.sqrt(2) <= path_w / 2 + margin and abs(x) < L:
            return True
        return False

    parking = (6.0, -15.5, 15.5, -9.0)  # x0, y0, x1, y1

    def ground_tex(cx, cy):
        if parking[0] <= cx <= parking[2] and parking[1] <= cy <= parking[3]:
            return "asphalt"
        return "grass"

    floor_tiles(w, -L, -L, L, L, 4.0, ground_tex, z=0.004, name="ground")
    # paths (slightly above grass)
    path_tex = "paving" if rng.random() < 0.5 else "concrete"
    parts = []
    for k in range(int(2 * L / 2.0)):
        c = -L + 1.0 + 2.0 * k
        parts.append(Part(geom_plane(2.0, path_w), material(path_tex), (c, 0.0, 0.008), collide=False))
        parts.append(Part(geom_plane(path_w, 2.0), material(path_tex), (0.0, c, 0.009), collide=False))
    if diag:
        n = int(2 * L / 2.0)
        for k in range(n):
            c = -L + 1.0 + 2.0 * k
            parts.append(Part(geom_plane(2.0 * math.sqrt(2), path_w), material(path_tex),
                              (c, c, 0.010, math.pi / 4), collide=False))
    w.add(StaticModel(w.uid("paths"), parts=parts))
    w.keepouts.append(("rect", 0.0, 0.0, 0.0, 2 * L, path_w + 0.4))
    w.keepouts.append(("rect", 0.0, 0.0, 0.0, path_w + 0.4, 2 * L))
    if diag:
        w.keepouts.append(("rect", 0.0, 0.0, math.pi / 4, 2 * L * math.sqrt(2), path_w + 0.4))

    # fence boundary
    fence_tex = ["metal", "crate", "brick_white"]
    for p0, p1 in (((-L, -L), (L, -L)), ((L, -L), (L, L)), ((L, L), (-L, L)), ((-L, L), (-L, -L))):
        wall_line(w, p0, p1, 1.1, 0.12, [fence_tex[int(rng.integers(len(fence_tex)))]], seg_len=2.5,
                  name="fence")
    # buildings outside the fence (landmarks)
    facades = ["facade_beige", "facade_gray", "facade_red", "brick_red"]
    for side in range(4):
        n = int(rng.integers(2, 4))
        for k in range(n):
            t = -L + (k + 0.5) * (2 * L / n) + float(rng.uniform(-2, 2))
            depth = float(rng.uniform(5, 9))
            width = float(rng.uniform(6, 10))
            height = float(rng.uniform(5, 14))
            off = L + 1.5 + depth / 2
            x, y, yaw = [(t, -off, 0.0), (off, t, math.pi / 2), (t, off, 0.0), (-off, t, math.pi / 2)][side]
            m, fp = make_building(w, x, y, yaw, (width, depth, height), facades[int(rng.integers(len(facades)))])
            w.add(m, None)
    # inner kiosks
    inner = (-L + 2.0, -L + 2.0, L - 2.0, L - 2.0)
    place_random(w, inner, lambda x, y, yaw: make_building(w, x, y, yaw, (3.0, 3.0, 2.6),
                                                           facades[int(rng.integers(len(facades)))]),
                 2, clearance=2.5, wall_snap=True)
    # parked cars
    for k in range(int(rng.integers(3, 6))):
        x = parking[0] + 1.5 + k * 2.4
        if x > parking[2] - 1.0:
            break
        if rng.random() < 0.8:
            m, fp = make_car(w, x, (parking[1] + parking[3]) / 2, math.pi / 2 + float(rng.uniform(-0.05, 0.05)))
            if w.free_for(fp, 0.3):
                w.add(m, fp)
    # vegetation and props
    place_random(w, inner, lambda x, y, yaw: make_tree(w, x, y, yaw), int(rng.integers(18, 28)), clearance=2.0)
    near_path = (-L + 1.0, -L + 1.0, L - 1.0, L - 1.0)

    def along_path(fn):
        def make(x, y, yaw):
            # 道の脇 (道の縁から 0.5-1.5m) に寄せる
            if rng.random() < 0.5:
                y = float(np.sign(y) or 1.0) * (path_w / 2 + rng.uniform(0.6, 1.4))
                yaw = 0.0
            else:
                x = float(np.sign(x) or 1.0) * (path_w / 2 + rng.uniform(0.6, 1.4))
                yaw = math.pi / 2
            return fn(x, y, yaw)
        return make

    place_random(w, near_path, along_path(lambda x, y, yaw: make_bench(w, x, y, yaw)), 6, clearance=1.5)
    place_random(w, near_path, along_path(lambda x, y, yaw: make_lamp(w, x, y, yaw)), 8, clearance=1.5)
    place_random(w, near_path, along_path(lambda x, y, yaw: make_bin(w, x, y, yaw)), 5, clearance=1.5)
    place_random(w, inner, lambda x, y, yaw: make_rock(w, x, y, yaw), int(rng.integers(5, 10)), clearance=1.5)
    place_random(w, inner, lambda x, y, yaw: make_cone(w, x, y, yaw), int(rng.integers(3, 7)), clearance=1.2)
    place_random(w, inner, lambda x, y, yaw: make_crate(w, x, y, yaw), int(rng.integers(2, 5)), clearance=1.2)
    spawn = (0.0, float(rng.uniform(-8.0, -3.0)), math.pi / 2)
    return w, spawn


BUILDERS = {"office": build_office, "park": build_park}


# =============================================================================
def check_spawn(sdf_text: str, spawn, min_clear: float = 0.6) -> float:
    from omnivla_nav.sim_map import build_occupancy_from_sdf

    grid = build_occupancy_from_sdf(sdf_text, resolution=0.05)
    return grid.clearance(spawn[0], spawn[1])


def write_world(kind: str, seed: int, write_map: bool = True) -> str:
    world, spawn = BUILDERS[kind](seed)
    sdf = world.to_sdf(spawn, scene_sky=True, ambient=0.6 if kind == "office" else 0.45)
    clear = check_spawn(sdf, spawn)
    if clear < 0.6:
        # スポーンが障害物に近すぎる場合は空いている場所に動かす
        from omnivla_nav.sim_map import PathPlanner, build_occupancy_from_sdf

        grid = build_occupancy_from_sdf(sdf, resolution=0.05)
        planner = PathPlanner(grid, inflate=0.6)
        x, y = planner.sample_free(np.random.default_rng(seed), min_clearance=0.8)
        spawn = (x, y, spawn[2])
        sdf = world.to_sdf(spawn, scene_sky=True, ambient=0.6 if kind == "office" else 0.45)
    out = os.path.join(PKG, "worlds", f"{world.name}.sdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(sdf)
    if write_map:
        from omnivla_nav.sim_map import build_occupancy_from_sdf

        grid = build_occupancy_from_sdf(out, resolution=0.05)
        prefix = os.path.join(PKG, "maps", world.name)
        grid.save(prefix)
        grid.preview(inflate=0.4, points=[spawn[:2]]).save(prefix + ".png")
    n_models = len(world.models)
    print(f"[world] {out}  models={n_models} spawn=({spawn[0]:.2f}, {spawn[1]:.2f}, {spawn[2]:.2f})")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worlds", nargs="+", default=["office", "park"], choices=sorted(BUILDERS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--texture-seed", type=int, default=0)
    ap.add_argument("--skip-textures", action="store_true")
    ap.add_argument("--no-map", action="store_true")
    args = ap.parse_args(argv)
    if not args.skip_textures:
        names = generate_textures(args.texture_seed)
        print(f"[textures] {len(names)} textures -> {TEX_DIR}")
    for kind in args.worlds:
        for seed in args.seeds:
            write_world(kind, seed, write_map=not args.no_map)


if __name__ == "__main__":
    main()
