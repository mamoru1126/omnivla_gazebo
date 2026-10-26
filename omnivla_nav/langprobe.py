"""言葉の指示がどれくらい効くかを測る (ROS 非依存).

eval_runner の mode:=language で使う。ランダムな場所から 1 つの指示だけを一定時間与え、
動いた結果が指示どおりかを判定する:
    left     : 左に 45 度以上曲がった
    right    : 右に 45 度以上曲がった
    straight : 向きの変化が 20 度以内で、前に 0.6m 以上進んだ
    object   : 指示した物 (例 "move toward red sofa") の 0.8m 以内まで近づいた
物の名前と色は Gazebo のワールド (SDF) のモデル名とテクスチャから作る。
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .geometry import to_local, wrap_angle
from .sim_map import Obstacle, extract_obstacles, load_world_root

Pose = Tuple[float, float, float]

# モデル名の先頭 -> 指示に使う名前 (壁・床・地面・道などは対象にしない)
NOUNS: Dict[str, str] = {
    "sofa": "sofa", "desk": "desk", "cabinet": "cabinet", "crate": "box", "shelf": "bookshelf", "plant": "plant",
    "tree": "tree", "building": "building", "rock": "rock", "lamp": "street lamp", "bench": "bench",
    "bin": "trash bin", "cone": "traffic cone", "car": "car",
}
# テクスチャ -> 色の言葉 (無いものは色を付けない)
COLORS: Dict[str, str] = {
    "fabric_red": "red", "fabric_blue": "blue", "fabric_green": "green", "fabric_gray": "gray",
    "car_black": "black", "car_yellow": "yellow", "car_red": "red", "car_white": "white", "car_blue": "blue",
    "stripes_orange": "orange", "metal": "gray", "cardboard": "brown", "brick_red": "red",
    "paint_blue": "blue", "paint_green": "green", "paint_yellow": "yellow", "paint_salmon": "pink",
    "paint_cream": "white", "paint_gray": "gray", "facade_red": "red", "facade_gray": "gray", "facade_beige": "beige",
}

DEFAULT_INSTRUCTIONS = ("left:turn left;left:go left;right:turn right;right:go right;"
                        "straight:go straight;straight:move forward")
KINDS = ("left", "right", "straight", "object")


@dataclass
class SceneObject:
    name: str                    # モデル名 (sofa_591)
    noun: str                    # sofa
    color: str                   # red / "" (無し)
    obstacles: List[Obstacle] = field(default_factory=list, repr=False)

    @property
    def phrase(self) -> str:
        return f"{self.color} {self.noun}".strip()

    @property
    def center(self) -> np.ndarray:
        pts = []
        for o in self.obstacles:
            if o.circle is not None:
                pts.append(o.circle[:2])
            else:
                pts.append(o.polygon.mean(axis=0))
        return np.mean(np.asarray(pts, dtype=np.float64), axis=0)

    def distance(self, x: float, y: float) -> float:
        """点から物の足元 (衝突形状を上から見た形) までの距離 [m]. 中なら 0."""
        return min(_obstacle_distance(o, x, y) for o in self.obstacles)


def _obstacle_distance(o: Obstacle, x: float, y: float) -> float:
    if o.circle is not None:
        cx, cy, r = o.circle
        return max(0.0, math.hypot(x - cx, y - cy) - r)
    poly = np.asarray(o.polygon, dtype=np.float64)
    p = np.array([x, y], dtype=np.float64)
    inside = True
    best = float("inf")
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        e = b - a
        n2 = float(e @ e)
        t = 0.0 if n2 < 1e-12 else float(np.clip((p - a) @ e / n2, 0.0, 1.0))
        best = min(best, float(np.linalg.norm(a + t * e - p)))
        if e[0] * (p[1] - a[1]) - e[1] * (p[0] - a[0]) < 0:  # 反時計回りの凸多角形: 内側は左
            inside = False
    return 0.0 if inside and len(poly) >= 3 else best


def _model_textures(model: ET.Element) -> List[str]:
    out = []
    for e in model.iter():
        if e.tag == "albedo_map" and e.text:
            out.append(e.text.strip().split("/")[-1].rsplit(".", 1)[0])
    return out


def scene_objects(sdf: str) -> List[SceneObject]:
    """ワールドの中の「指示で名前を言える物」と、その足元の形."""
    world = load_world_root(sdf)
    obstacles = extract_obstacles(sdf)
    by_model: Dict[str, List[Obstacle]] = {}
    for o in obstacles:
        by_model.setdefault(o.name.split("/")[0].split("::")[0], []).append(o)
    out = []
    for model in world.findall("model"):
        name = model.attrib.get("name", "")
        noun = NOUNS.get(name.split("_")[0])
        if noun is None or name not in by_model:
            continue
        tex = [t for t in _model_textures(model) if t in COLORS]
        color = COLORS[Counter(tex).most_common(1)[0][0]] if tex else ""
        out.append(SceneObject(name, noun, color, by_model[name]))
    return out


def unique_phrase_objects(objects: Sequence[SceneObject]) -> List[SceneObject]:
    """同じ言い方 (色 + 名前) の物が 1 つしか無いもの (どれのことか曖昧にならない)."""
    counts = Counter(o.phrase for o in objects)
    return [o for o in objects if counts[o.phrase] == 1]


def same_phrase(objects: Sequence[SceneObject], phrase: str) -> List[SceneObject]:
    return [o for o in objects if o.phrase == phrase]


def parse_instructions(spec: str) -> List[Tuple[str, str]]:
    """'left:turn left;right:turn right' -> [(kind, text), ...]."""
    out = []
    for item in (spec or "").split(";"):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"instruction must be 'kind:text' ({'/'.join(KINDS[:3])}): {item}")
        kind, text = item.split(":", 1)
        kind = kind.strip().lower()
        if kind not in KINDS[:3]:
            raise ValueError(f"unknown kind '{kind}' (use left / right / straight)")
        out.append((kind, text.strip()))
    return out


# ---------------------------------------------------------------------------
# スタート位置
# ---------------------------------------------------------------------------
def sample_open_start(planner, rng: np.random.Generator, component: int, min_clearance: float = 0.9,
                      ahead: float = 1.5, tries: int = 300) -> Optional[Pose]:
    """周りが開けていて、前に ahead [m] 進める姿勢 (左右・直進の指示用)."""
    for _ in range(tries):
        x, y = planner.sample_free(rng, component, min_clearance=min_clearance)
        yaw = float(rng.uniform(-math.pi, math.pi))
        q = (x + ahead * math.cos(yaw), y + ahead * math.sin(yaw))
        if planner.segment_free((x, y), q):
            return x, y, yaw
    return None


def sample_object_start(planner, obj: SceneObject, rng: np.random.Generator, component: int,
                        dmin: float = 1.8, dmax: float = 4.0, max_bearing_deg: float = 20.0,
                        tries: int = 300) -> Optional[Pose]:
    """物が前方 (±max_bearing_deg) の dmin〜dmax [m] 先に見え、間に障害物が無い姿勢."""
    c = obj.center
    for _ in range(tries):
        ang = float(rng.uniform(-math.pi, math.pi))
        r = float(rng.uniform(dmin, dmax))
        # 物の足元から r [m] 離れた点を、中心から ang の方向に探す
        lo, hi = 0.0, r + 6.0
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            p = c + mid * np.array([math.cos(ang), math.sin(ang)])
            if obj.distance(p[0], p[1]) < r:
                lo = mid
            else:
                hi = mid
        p = c + hi * np.array([math.cos(ang), math.sin(ang)])
        if not planner.is_free(p[0], p[1]) or planner.component(p[0], p[1]) != component:
            continue
        # 物の手前 0.5m までの見通し
        d = c - p
        n = float(np.linalg.norm(d))
        steps = np.linspace(0.0, 1.0, max(2, int(n / 0.05)))
        near = None
        for t in steps:
            q = p + t * d
            if obj.distance(q[0], q[1]) <= 0.5:
                near = q
                break
        if near is None or not planner.segment_free(p, near, extra_clearance=-0.12):
            continue
        yaw = math.atan2(d[1], d[0]) + math.radians(float(rng.uniform(-max_bearing_deg, max_bearing_deg)))
        return float(p[0]), float(p[1]), float(wrap_angle(yaw))
    return None


# ---------------------------------------------------------------------------
# 判定
# ---------------------------------------------------------------------------
@dataclass
class ProbeCriteria:
    turn_deg: float = 45.0         # left/right: これ以上曲がれば成功
    straight_tol_deg: float = 20.0  # straight: 向きの変化がこれ以内
    straight_min_m: float = 0.6     # straight: 前にこれ以上進む
    object_radius: float = 0.8      # object: 物の足元までこれ以内に近づく
    min_motion_m: float = 0.15      # これ未満しか動かなければ「動かなかった」


def accumulated_turn(poses: Sequence[Pose]) -> float:
    """向きの変化の合計 [rad] (左 +). 一周以上も数える."""
    total = 0.0
    for a, b in zip(poses[:-1], poses[1:]):
        total += wrap_angle(b[2] - a[2])
    return total


def score_probe(kind: str, poses: Sequence[Pose], targets: Optional[Sequence[SceneObject]] = None,
                crit: Optional[ProbeCriteria] = None) -> Dict:
    """targets: object の場合の対象 (同じ言い方の物が複数あれば全部. どれかに近づけば成功)."""
    crit = crit or ProbeCriteria()
    poses = [tuple(p) for p in poses]
    if len(poses) < 2:
        return {"success": 0, "moved": 0, "dyaw_deg": 0.0, "forward_m": 0.0, "left_m": 0.0, "path_m": 0.0}
    start, end = poses[0], poses[-1]
    dyaw = math.degrees(accumulated_turn(poses))
    loc = to_local(np.array(end[:2]), start[:2], start[2])
    path = float(sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(poses[:-1], poses[1:])))
    moved = int(path >= crit.min_motion_m or abs(dyaw) >= 10.0)
    r = {"dyaw_deg": dyaw, "forward_m": float(loc[0]), "left_m": float(loc[1]), "path_m": path, "moved": moved}
    if kind == "left":
        ok = dyaw >= crit.turn_deg
    elif kind == "right":
        ok = dyaw <= -crit.turn_deg
    elif kind == "straight":
        ok = abs(dyaw) <= crit.straight_tol_deg and loc[0] >= crit.straight_min_m
    elif kind == "object":
        if not targets:
            raise ValueError("object probe needs the target object(s)")

        def dist(p):
            return min(o.distance(p[0], p[1]) for o in targets)
        d0 = dist(start)
        dists = [dist(p) for p in poses]
        r.update({"start_dist": d0, "final_dist": dists[-1], "min_dist": min(dists)})
        ok = min(dists) <= crit.object_radius
    else:
        raise ValueError(f"unknown kind {kind}")
    r["success"] = int(bool(ok))
    return r


def summarize(rows: Sequence[Dict]) -> Dict:
    """kind ごと・言い方ごとの成功率など."""
    out: Dict = {"num_probes": len(rows), "by_kind": {}, "by_instruction": {}}

    def agg(items: Sequence[Dict]) -> Dict:
        d = {"n": len(items), "success_rate": float(np.mean([r["success"] for r in items])),
             "moved_rate": float(np.mean([r["moved"] for r in items])),
             "mean_dyaw_deg": float(np.mean([r["dyaw_deg"] for r in items])),
             "mean_forward_m": float(np.mean([r["forward_m"] for r in items]))}
        fd = [r["final_dist"] for r in items if r.get("final_dist") is not None]
        if fd:
            d["mean_final_dist"] = float(np.mean(fd))
            d["mean_start_dist"] = float(np.mean([r["start_dist"] for r in items if r.get("start_dist") is not None]))
        return d

    for key, field_name in (("by_kind", "kind"), ("by_instruction", "instruction")):
        groups: Dict[str, List[Dict]] = {}
        for r in rows:
            groups.setdefault(str(r[field_name]), []).append(r)
        out[key] = {k: agg(v) for k, v in sorted(groups.items())}
    if rows:
        out["success_rate"] = float(np.mean([r["success"] for r in rows]))
    return out
