"""Gazebo の SDF ワールドから 2D 占有格子地図を作り、経路計画を行う (ROS 非依存).

用途:
  - 自動データ収集 (auto_explorer) で、障害物を避ける「エキスパート経路」を A*/Dijkstra で生成
  - 評価 (eval_runner) で、スタート/ゴールのサンプリングと衝突判定・最短経路長 (SPL) の計算
  - ルート上のサブゴール画像 (topomap) を自動生成

対応ジオメトリ: box / cylinder / sphere / capsule / ellipsoid (mesh, plane, heightmap は無視)。
ロボットの高さ帯 (z_band) と交差する collision のみを障害物として扱う。
"""
from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    from scipy import ndimage
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
except ImportError as e:  # pragma: no cover
    raise ImportError("sim_map requires scipy (pip install scipy)") from e


# ---------------------------------------------------------------------------
# SDF parsing
# ---------------------------------------------------------------------------
def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ ry @ rx  # SDF: extrinsic X-Y-Z (= Rz*Ry*Rx)


def pose_to_matrix(text: Optional[str], degrees: bool = False) -> np.ndarray:
    t = np.eye(4)
    if text is None or not text.strip():
        return t
    vals = [float(v) for v in text.split()]
    if len(vals) == 3:
        vals = vals + [0.0, 0.0, 0.0]
    if len(vals) != 6:
        raise ValueError(f"unsupported pose '{text}' (quaternion poses are not supported)")
    x, y, z, r, p, yw = vals
    if degrees:
        r, p, yw = math.radians(r), math.radians(p), math.radians(yw)
    t[:3, :3] = _rpy_to_matrix(r, p, yw)
    t[:3, 3] = (x, y, z)
    return t


def _elem_pose(elem: ET.Element) -> np.ndarray:
    pose = elem.find("pose")
    if pose is None:
        return np.eye(4)
    deg = str(pose.attrib.get("degrees", "false")).lower() in ("true", "1")
    return pose_to_matrix(pose.text, degrees=deg)


def _floats(elem: Optional[ET.Element], default=None) -> Optional[List[float]]:
    if elem is None or elem.text is None:
        return default
    return [float(v) for v in elem.text.split()]


@dataclass
class Obstacle:
    name: str
    z_min: float
    z_max: float
    polygon: Optional[np.ndarray] = None            # (K,2) 凸多角形 (反時計回り)
    circle: Optional[Tuple[float, float, float]] = None  # (cx, cy, r)

    def bounds(self) -> Tuple[float, float, float, float]:
        if self.polygon is not None:
            mn, mx = self.polygon.min(axis=0), self.polygon.max(axis=0)
            return float(mn[0]), float(mn[1]), float(mx[0]), float(mx[1])
        cx, cy, r = self.circle
        return cx - r, cy - r, cx + r, cy + r


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Andrew の monotone chain. 反時計回りの頂点列を返す."""
    pts = np.unique(np.round(np.asarray(points, dtype=np.float64), 9), axis=0)
    if len(pts) <= 2:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))
    return np.array(lower[:-1] + upper[:-1], dtype=np.float64)


def _geometry_to_obstacle(name: str, geom: ET.Element, tf: np.ndarray) -> Optional[Obstacle]:
    rot, trans = tf[:3, :3], tf[:3, 3]
    vertical = abs(rot[2, 2]) > 0.995  # ローカル z 軸がほぼ鉛直か

    box = geom.find("box")
    if box is not None:
        sx, sy, sz = _floats(box.find("size"), [1, 1, 1])
        corners = np.array([[ix * sx / 2, iy * sy / 2, iz * sz / 2]
                            for ix in (-1, 1) for iy in (-1, 1) for iz in (-1, 1)])
        world = corners @ rot.T + trans
        return Obstacle(name, float(world[:, 2].min()), float(world[:, 2].max()),
                        polygon=convex_hull(world[:, :2]))

    def _round_shape(radius: float, half_len: float, endcap: float = 0.0) -> Obstacle:
        if vertical:
            return Obstacle(name, float(trans[2] - half_len - endcap), float(trans[2] + half_len + endcap),
                            circle=(float(trans[0]), float(trans[1]), float(radius)))
        ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        pts = []
        for zc in (-half_len - endcap, half_len + endcap):
            ring = np.stack([radius * np.cos(ang), radius * np.sin(ang), np.full_like(ang, zc)], axis=1)
            pts.append(ring)
        world = np.concatenate(pts) @ rot.T + trans
        return Obstacle(name, float(world[:, 2].min()), float(world[:, 2].max()),
                        polygon=convex_hull(world[:, :2]))

    cyl = geom.find("cylinder")
    if cyl is not None:
        r = _floats(cyl.find("radius"), [0.5])[0]
        length = _floats(cyl.find("length"), [1.0])[0]
        return _round_shape(r, length / 2)
    cap = geom.find("capsule")
    if cap is not None:
        r = _floats(cap.find("radius"), [0.5])[0]
        length = _floats(cap.find("length"), [1.0])[0]
        return _round_shape(r, length / 2, endcap=r)
    sph = geom.find("sphere")
    if sph is not None:
        r = _floats(sph.find("radius"), [0.5])[0]
        return Obstacle(name, float(trans[2] - r), float(trans[2] + r),
                        circle=(float(trans[0]), float(trans[1]), float(r)))
    ell = geom.find("ellipsoid")
    if ell is not None:
        rx, ry, rz = _floats(ell.find("radii"), [0.5, 0.5, 0.5])
        rr = max(rx, ry)
        return Obstacle(name, float(trans[2] - rz), float(trans[2] + rz),
                        circle=(float(trans[0]), float(trans[1]), float(rr)))
    return None  # plane / mesh / heightmap / polyline などは無視


def _walk_model(model: ET.Element, parent_tf: np.ndarray, prefix: str, out: List[Obstacle],
                exclude: Sequence[str]) -> None:
    name = model.attrib.get("name", "model")
    if name in exclude:
        return
    tf_model = parent_tf @ _elem_pose(model)
    for link in model.findall("link"):
        tf_link = tf_model @ _elem_pose(link)
        for col in link.findall("collision"):
            geom = col.find("geometry")
            if geom is None:
                continue
            obs = _geometry_to_obstacle(f"{prefix}{name}/{link.attrib.get('name')}/{col.attrib.get('name')}",
                                        geom, tf_link @ _elem_pose(col))
            if obs is not None:
                out.append(obs)
    for sub in model.findall("model"):
        _walk_model(sub, tf_model, f"{prefix}{name}::", out, exclude)


def load_world_root(sdf: str) -> ET.Element:
    """sdf: ファイルパス or XML 文字列."""
    if os.path.exists(sdf):
        root = ET.parse(sdf).getroot()
    else:
        root = ET.fromstring(sdf)
    world = root.find("world") if root.tag != "world" else root
    if world is None:
        raise ValueError("no <world> element found in SDF")
    return world


def extract_obstacles(sdf: str, z_band: Tuple[float, float] = (0.03, 0.6),
                      exclude_models: Sequence[str] = ("omnivla_robot",)) -> List[Obstacle]:
    world = load_world_root(sdf)
    obstacles: List[Obstacle] = []
    for model in world.findall("model"):
        _walk_model(model, np.eye(4), "", obstacles, exclude_models)
    lo, hi = z_band
    return [o for o in obstacles if o.z_max > lo and o.z_min < hi]


def world_name(sdf: str) -> str:
    return load_world_root(sdf).attrib.get("name", "default")


def robot_spawn_pose(sdf: str, robot_name: str = "omnivla_robot") -> Optional[Tuple[float, float, float]]:
    """<include> されたロボットの初期姿勢 (x, y, yaw)."""
    world = load_world_root(sdf)
    for inc in world.findall("include"):
        name = inc.findtext("name") or ""
        uri = inc.findtext("uri") or ""
        if name == robot_name or uri.rstrip("/").endswith(robot_name):
            tf = _elem_pose(inc)
            yaw = math.atan2(tf[1, 0], tf[0, 0])
            return float(tf[0, 3]), float(tf[1, 3]), yaw
    return None


# ---------------------------------------------------------------------------
# Occupancy grid
# ---------------------------------------------------------------------------
@dataclass
class OccupancyGrid:
    occupied: np.ndarray            # (H, W) bool, row = y, col = x
    resolution: float
    origin: Tuple[float, float]     # セル (0,0) の左下角のワールド座標
    _dist: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.occupied.shape

    def world_to_cell(self, x, y):
        ix = np.floor((np.asarray(x) - self.origin[0]) / self.resolution).astype(np.int64)
        iy = np.floor((np.asarray(y) - self.origin[1]) / self.resolution).astype(np.int64)
        return ix, iy

    def cell_to_world(self, ix, iy):
        x = self.origin[0] + (np.asarray(ix) + 0.5) * self.resolution
        y = self.origin[1] + (np.asarray(iy) + 0.5) * self.resolution
        return x, y

    def in_bounds(self, ix, iy):
        h, w = self.shape
        ix, iy = np.asarray(ix), np.asarray(iy)
        return (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)

    def distance_map(self) -> np.ndarray:
        """各セルから最も近い障害物までの距離 [m] (地図外は障害物扱いしない)."""
        if self._dist is None:
            self._dist = ndimage.distance_transform_edt(~self.occupied) * self.resolution
        return self._dist

    def clearance(self, x: float, y: float) -> float:
        ix, iy = self.world_to_cell(x, y)
        if not self.in_bounds(ix, iy):
            return 0.0
        return float(self.distance_map()[iy, ix])

    def free_mask(self, inflate: float) -> np.ndarray:
        return self.distance_map() > inflate

    # --- I/O ------------------------------------------------------------
    def save(self, path_prefix: str) -> Tuple[str, str]:
        """ROS map_server 形式 (.pgm + .yaml) で保存."""
        os.makedirs(os.path.dirname(os.path.abspath(path_prefix)), exist_ok=True)
        img = np.where(self.occupied, 0, 254).astype(np.uint8)[::-1]  # 画像は上が +y
        pgm = path_prefix + ".pgm"
        Image.fromarray(img, mode="L").save(pgm)
        yaml_path = path_prefix + ".yaml"
        with open(yaml_path, "w") as f:
            f.write(f"image: {os.path.basename(pgm)}\n")
            f.write(f"resolution: {self.resolution}\n")
            f.write(f"origin: [{self.origin[0]}, {self.origin[1]}, 0.0]\n")
            f.write("negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n")
        return pgm, yaml_path

    def preview(self, inflate: float = 0.0, path: Optional[Sequence[Tuple[float, float]]] = None,
                points: Iterable[Tuple[float, float]] = (), scale: int = 2) -> Image.Image:
        rgb = np.full(self.shape + (3,), 255, np.uint8)
        if inflate > 0:
            rgb[~self.free_mask(inflate)] = (255, 210, 210)
        rgb[self.occupied] = (40, 40, 40)
        if path is not None and len(path):
            p = np.asarray(path)
            ix, iy = self.world_to_cell(p[:, 0], p[:, 1])
            ok = self.in_bounds(ix, iy)
            rgb[iy[ok], ix[ok]] = (0, 90, 255)
        for (x, y) in points:
            ix, iy = self.world_to_cell(x, y)
            if self.in_bounds(ix, iy):
                rgb[max(0, iy - 2):iy + 3, max(0, ix - 2):ix + 3] = (230, 30, 30)
        img = Image.fromarray(rgb[::-1])
        if scale != 1:
            img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
        return img


def rasterize(obstacles: Sequence[Obstacle], resolution: float = 0.05,
              bounds: Optional[Tuple[float, float, float, float]] = None, margin: float = 1.0) -> OccupancyGrid:
    if bounds is None:
        if not obstacles:
            raise ValueError("no obstacles and no bounds given")
        b = np.array([o.bounds() for o in obstacles])
        bounds = (b[:, 0].min() - margin, b[:, 1].min() - margin, b[:, 2].max() + margin, b[:, 3].max() + margin)
    x0, y0, x1, y1 = bounds
    w = int(math.ceil((x1 - x0) / resolution))
    h = int(math.ceil((y1 - y0) / resolution))
    grid = np.zeros((h, w), dtype=bool)
    og = OccupancyGrid(grid, resolution, (x0, y0))
    for o in obstacles:
        bx0, by0, bx1, by1 = o.bounds()
        ix0, iy0 = og.world_to_cell(bx0, by0)
        ix1, iy1 = og.world_to_cell(bx1, by1)
        ix0, iy0 = max(int(ix0), 0), max(int(iy0), 0)
        ix1, iy1 = min(int(ix1), w - 1), min(int(iy1), h - 1)
        if ix1 < ix0 or iy1 < iy0:
            continue
        cx, cy = og.cell_to_world(np.arange(ix0, ix1 + 1), np.arange(iy0, iy1 + 1))
        gx, gy = np.meshgrid(cx, cy)
        if o.circle is not None:
            ccx, ccy, r = o.circle
            # セル中心判定 + 半セル分の余裕 (細い柱を取りこぼさない)
            inside = (gx - ccx) ** 2 + (gy - ccy) ** 2 <= (r + 0.5 * resolution) ** 2
        else:
            poly = o.polygon
            if len(poly) < 3:
                continue
            inside = np.ones_like(gx, dtype=bool)
            tol = 0.5 * resolution
            for i in range(len(poly)):
                a, b2 = poly[i], poly[(i + 1) % len(poly)]
                edge = b2 - a
                n = math.hypot(edge[0], edge[1])
                if n < 1e-12:
                    continue
                # 反時計回り凸多角形: 内側は cross >= 0. tol だけ外側に広げる
                cross = (edge[0] * (gy - a[1]) - edge[1] * (gx - a[0])) / n
                inside &= cross >= -tol
        grid[iy0:iy1 + 1, ix0:ix1 + 1] |= inside
    return og


def build_occupancy_from_sdf(sdf: str, resolution: float = 0.05, z_band=(0.03, 0.6),
                             exclude_models: Sequence[str] = ("omnivla_robot",),
                             margin: float = 1.0) -> OccupancyGrid:
    return rasterize(extract_obstacles(sdf, z_band, exclude_models), resolution, margin=margin)


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------
_NEIGHBORS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]


class PathPlanner:
    """膨張した占有格子上の Dijkstra (scipy.sparse.csgraph) による経路計画."""

    def __init__(self, grid: OccupancyGrid, inflate: float, clearance_weight: float = 3.0,
                 clearance_scale: float = 0.4):
        self.grid = grid
        self.inflate = float(inflate)
        self.free = grid.free_mask(self.inflate)
        dist = grid.distance_map()
        # 障害物に近いほどコストが高い (壁際を避ける)
        self.base_cost = 1.0 + clearance_weight * np.exp(-np.maximum(dist - self.inflate, 0.0) / clearance_scale)
        self.labels, _ = ndimage.label(self.free, structure=np.ones((3, 3)))
        self.component_slices = ndimage.find_objects(self.labels)  # label-1 -> (slice_y, slice_x)

    # -- helpers --------------------------------------------------------
    def is_free(self, x: float, y: float) -> bool:
        ix, iy = self.grid.world_to_cell(x, y)
        return bool(self.grid.in_bounds(ix, iy) and self.free[iy, ix])

    def component(self, x: float, y: float) -> int:
        ix, iy = self.grid.world_to_cell(x, y)
        if not self.grid.in_bounds(ix, iy):
            return 0
        return int(self.labels[iy, ix])

    def nearest_free(self, x: float, y: float, max_radius: float = 1.0) -> Optional[Tuple[float, float]]:
        if self.is_free(x, y):
            return x, y
        ix, iy = self.grid.world_to_cell(x, y)
        r = int(math.ceil(max_radius / self.grid.resolution))
        h, w = self.free.shape
        ys, xs = np.nonzero(self.free[max(0, iy - r):min(h, iy + r + 1), max(0, ix - r):min(w, ix + r + 1)])
        if len(xs) == 0:
            return None
        xs = xs + max(0, ix - r)
        ys = ys + max(0, iy - r)
        k = int(np.argmin((xs - ix) ** 2 + (ys - iy) ** 2))
        wx, wy = self.grid.cell_to_world(xs[k], ys[k])
        return float(wx), float(wy)

    def sample_free(self, rng: np.random.Generator, component: Optional[int] = None,
                    min_clearance: Optional[float] = None) -> Tuple[float, float]:
        mask = self.free.copy()
        if min_clearance is not None:
            mask &= self.grid.distance_map() > min_clearance
        if component is not None and component > 0:
            mask &= self.labels == component
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            raise RuntimeError("no free cell available for sampling")
        k = int(rng.integers(len(xs)))
        x, y = self.grid.cell_to_world(xs[k], ys[k])
        return float(x), float(y)

    def segment_free(self, p0, p1, extra_clearance: float = 0.0) -> bool:
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        n = max(2, int(np.linalg.norm(p1 - p0) / (0.5 * self.grid.resolution)) + 1)
        pts = p0[None] + np.linspace(0, 1, n)[:, None] * (p1 - p0)[None]
        ix, iy = self.grid.world_to_cell(pts[:, 0], pts[:, 1])
        if not np.all(self.grid.in_bounds(ix, iy)):
            return False
        return bool(np.all(self.grid.distance_map()[iy, ix] > self.inflate + extra_clearance))

    # -- planning -------------------------------------------------------
    def plan(self, start: Sequence[float], goal: Sequence[float], rng: Optional[np.random.Generator] = None,
             noise: float = 0.0, smooth: bool = True, spacing: float = 0.05) -> Optional[np.ndarray]:
        """start/goal (x, y) の経路を (N,2) で返す. 到達不能なら None.

        noise > 0 のとき、セルコストにランダムな滑らかな場を掛けて毎回異なる経路にする (データ多様化)。
        """
        s = self.nearest_free(start[0], start[1])
        g = self.nearest_free(goal[0], goal[1])
        if s is None or g is None:
            return None
        comp = self.component(*s)
        if comp == 0 or self.component(*g) != comp:
            return None
        # スタートの連結成分を囲む矩形だけでグラフを作る (広い地図でも速い)
        sl_y, sl_x = self.component_slices[comp - 1]
        oy, ox = sl_y.start, sl_x.start
        free = self.free[sl_y, sl_x] & (self.labels[sl_y, sl_x] == comp)
        cost = self.base_cost[sl_y, sl_x]
        if noise > 0 and rng is not None:
            field_ = ndimage.gaussian_filter(rng.standard_normal(free.shape), sigma=8)
            field_ /= (np.abs(field_).max() + 1e-9)
            cost = cost * (1.0 + noise * (field_ + 1.0))
        h, w = free.shape
        idx = np.arange(h * w).reshape(h, w)
        rows, cols, vals = [], [], []
        for dx, dy in _NEIGHBORS:
            ys0, ys1 = max(0, -dy), h - max(0, dy)
            xs0, xs1 = max(0, -dx), w - max(0, dx)
            a = (slice(ys0, ys1), slice(xs0, xs1))
            b = (slice(ys0 + dy, ys1 + dy), slice(xs0 + dx, xs1 + dx))
            ok = free[a] & free[b]
            if dx != 0 and dy != 0:  # 斜め移動は両隣も空いている時のみ (角の擦り抜け防止)
                ok &= free[ys0:ys1, xs0 + dx:xs1 + dx] & free[ys0 + dy:ys1 + dy, xs0:xs1]
            step = math.hypot(dx, dy) * self.grid.resolution
            rows.append(idx[a][ok])
            cols.append(idx[b][ok])
            vals.append(step * 0.5 * (cost[a][ok] + cost[b][ok]))
        graph = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                           shape=(h * w, h * w)).tocsr()
        six, siy = self.grid.world_to_cell(*s)
        gix, giy = self.grid.world_to_cell(*g)
        src, dst = int(idx[siy - oy, six - ox]), int(idx[giy - oy, gix - ox])
        _, pred = dijkstra(graph, directed=True, indices=src, return_predecessors=True)
        if src != dst and pred[dst] < 0:
            return None
        cells = [dst]
        while cells[-1] != src:
            cells.append(int(pred[cells[-1]]))
        cells = cells[::-1]
        cy, cx = np.divmod(np.asarray(cells), w)
        wx, wy = self.grid.cell_to_world(cx + ox, cy + oy)
        path = np.stack([wx, wy], axis=1)
        path[0] = s
        path[-1] = g
        if smooth:
            path = self.smooth(path)
        return resample_path(path, spacing)

    def smooth(self, path: np.ndarray, iterations: int = 3) -> np.ndarray:
        # 1) 見通しが取れる点同士をショートカット (クリアランスに少し余裕を持たせる)
        pts = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1 and not self.segment_free(path[i], path[j], extra_clearance=0.1):
                j -= 1
            pts.append(path[j])
            i = j
        out = np.asarray(pts)
        # 2) Chaikin で角を丸める (衝突するなら丸めない)
        for _ in range(iterations):
            if len(out) < 3:
                break
            new = [out[0]]
            for a, b in zip(out[:-1], out[1:]):
                new.append(0.75 * a + 0.25 * b)
                new.append(0.25 * a + 0.75 * b)
            new.append(out[-1])
            cand = np.asarray(new)
            if all(self.segment_free(p, q) for p, q in zip(cand[:-1], cand[1:])):
                out = cand
            else:
                break
        return out


def resample_path(path: np.ndarray, spacing: float) -> np.ndarray:
    path = np.asarray(path, dtype=np.float64)
    if len(path) < 2:
        return path
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-9:
        return path[:1]
    n = max(2, int(math.ceil(s[-1] / spacing)) + 1)
    t = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(t, s, path[:, 0]), np.interp(t, s, path[:, 1])], axis=1)


def path_headings(path: np.ndarray) -> np.ndarray:
    d = np.diff(path, axis=0)
    yaw = np.arctan2(d[:, 1], d[:, 0])
    return np.concatenate([yaw, yaw[-1:]]) if len(yaw) else np.zeros(len(path))
