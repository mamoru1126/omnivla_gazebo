"""要所だけ指示して走らせる「ルート帳」(ROS 非依存).

人に道を教えるときと同じで、要所でだけ次の指示を出し、要所と要所の間は VLA に任せる:
    「この景色が見えるまで真っ直ぐ」→「右に曲がる」→「その先のソファへ」
topomap (1m ごとの画像) のように細かくたどらず、指示は要所の数 (曲がり角 + 1) くらいで済む。

ファイル形式 (<dir>/route.yaml, 画像は同じディレクトリ):
    start: {x: 0.0, y: 0.0, yaw: 0.0}     # 任意. teleport --goal_dir で戻すスタート姿勢 (Gazebo の真値)
    goal: [7.5, 4.0, 1.57]                 # 任意. 走行ログの最終距離の計算に使う
    steps:
      - instruction: go straight           # 言葉で指示 (modality language)
        until: {image: key_0.jpg, after_m: 2.0, distance: 6.0}
      - instruction: turn right
        until: {turn_deg: -75, image: key_1.jpg}
      - goal_image: key_2.jpg              # 景色 (画像) に向かう (modality image)
      - move: [2.0, -1.5]                  # 前 2m・右 1.5m の位置へ (オドメトリ, modality pose)
      - instruction: move toward red sofa  # 言葉 + 相対位置 (modality language_pose)
        move: [3.0, 0.0, 0]                #   [前 m, 左 m, 向き deg]
        until: {time: 20}
        timeout: 60                        # この秒数で終わらなければ失敗として止める

ステップの中身と OmniVLA に入れる modality:
    instruction          -> language        instruction + move -> language_pose
    goal_image           -> image           goal_image + move  -> image_pose
    move                 -> pose
  (OmniVLA には「言葉 + ゴール画像」を同時に入れる modality が無いので、両方を 1 ステップに書くとエラー)
  move の基準: 直前のステップにも move があれば、その move のゴール (= 予定どおりならそこにいる位置)。
  無ければこのステップを始めた位置。続けて move を書くと、手前で切り替わったずれが積み重ならない。

until (次のステップへ進む条件. 書いたもののどれか 1 つを満たせば進む):
    image: <画像>        今のカメラ画像との類似度が threshold (既定 0.9) 以上
    distance: <m>        このステップで進んだ道のり (オドメトリ)
    turn_deg: <deg>      このステップで曲がった角度 (左 +, 右 -). 符号の向きにこの角度以上
    time: <s>            このステップの経過時間
    reached: <m>         move のゴールまでの距離がこれ以下 (オドメトリ)
    pose: [x, y(, yaw_deg)]  真値 (Gazebo) / 自己位置で、その位置の radius [m] 以内 (+ 向きが yaw_tol_deg 以内)
    after_m: <m>         このステップでこの道のりを進むまでは image と turn_deg を判定しない
                         (始めから景色が似ている・曲がり始めで角度だけ先に足りる、での早すぎる切り替えを防ぐ)
  until を書かない場合: move があれば reached、goal_image だけなら image (同じ画像)、言葉だけなら終わらない。
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import yaml
from PIL import Image

from .geometry import relative_pose, to_world, wrap_angle

ROUTE_FILE = "route.yaml"
Pose = Tuple[float, float, float]
SimilarityFn = Callable[[str, Image.Image], float]

UNTIL_KEYS = {"image", "threshold", "after_m", "distance", "turn_deg", "time", "reached", "pose", "radius",
              "yaw_tol_deg"}
STEP_KEYS = {"instruction", "goal_image", "move", "until", "timeout", "note"}


# ---------------------------------------------------------------------------
# データ
# ---------------------------------------------------------------------------
@dataclass
class RouteStep:
    instruction: Optional[str] = None
    goal_image: Optional[str] = None                     # 画像ファイル名 (route.yaml からの相対)
    move: Optional[Tuple[float, float, float]] = None    # (前 m, 左 m, 向き rad) 直前の move のゴール (無ければ開始時の姿勢) から
    until: Dict = field(default_factory=dict)
    timeout: Optional[float] = None                      # [s] 超えたら失敗
    note: str = ""

    @property
    def modality(self) -> str:
        return step_modality(self.instruction, self.goal_image, self.move)

    def image_names(self) -> List[str]:
        names = [self.goal_image] if self.goal_image else []
        if self.until.get("image"):
            names.append(str(self.until["image"]))
        return names

    def to_dict(self) -> Dict:
        d: Dict = {}
        if self.instruction:
            d["instruction"] = self.instruction
        if self.goal_image:
            d["goal_image"] = self.goal_image
        if self.move is not None:
            d["move"] = [round(float(self.move[0]), 3), round(float(self.move[1]), 3),
                         round(math.degrees(float(self.move[2])), 1)]
        if self.until:
            d["until"] = dict(self.until)
        if self.timeout:
            d["timeout"] = float(self.timeout)
        if self.note:
            d["note"] = self.note
        return d

    def describe(self) -> str:
        parts = []
        if self.instruction:
            parts.append(f'"{self.instruction}"')
        if self.goal_image:
            parts.append(f"view {self.goal_image}")
        if self.move is not None:
            parts.append(f"move ({self.move[0]:+.1f}, {self.move[1]:+.1f})m")
        return " + ".join(parts) or "(empty)"


def step_modality(instruction: Optional[str], goal_image: Optional[str], move) -> str:
    if instruction and goal_image:
        raise ValueError("OmniVLA has no 'language + goal image' modality: split it into two steps "
                         "(or use the image only in until.image)")
    if instruction:
        return "language_pose" if move is not None else "language"
    if goal_image:
        return "image_pose" if move is not None else "image"
    if move is not None:
        return "pose"
    raise ValueError("a step needs instruction, goal_image or move")


def _parse_move(v) -> Tuple[float, float, float]:
    if isinstance(v, dict):
        return float(v.get("x", 0.0)), float(v.get("y", 0.0)), math.radians(float(v.get("yaw_deg", 0.0)))
    vals = [float(a) for a in v]
    if len(vals) not in (2, 3):
        raise ValueError(f"move must be [x, y] or [x, y, yaw_deg]: {v}")
    return vals[0], vals[1], math.radians(vals[2]) if len(vals) == 3 else 0.0


def _parse_pose(v) -> Optional[Pose]:
    if v is None:
        return None
    if isinstance(v, dict):
        return float(v["x"]), float(v["y"]), float(v.get("yaw", 0.0))
    vals = [float(a) for a in v]
    return vals[0], vals[1], vals[2] if len(vals) > 2 else 0.0


def parse_step(d: Dict) -> RouteStep:
    if not isinstance(d, dict):
        raise ValueError(f"step must be a mapping: {d!r}")
    unknown = set(d) - STEP_KEYS
    if unknown:
        raise ValueError(f"unknown step keys {sorted(unknown)} (use {sorted(STEP_KEYS)})")
    until = dict(d.get("until") or {})
    bad = set(until) - UNTIL_KEYS
    if bad:
        raise ValueError(f"unknown until keys {sorted(bad)} (use {sorted(UNTIL_KEYS)})")
    step = RouteStep(instruction=(str(d["instruction"]).strip() or None) if d.get("instruction") else None,
                     goal_image=str(d["goal_image"]) if d.get("goal_image") else None,
                     move=_parse_move(d["move"]) if d.get("move") is not None else None,
                     until=until, timeout=float(d["timeout"]) if d.get("timeout") else None,
                     note=str(d.get("note") or ""))
    step_modality(step.instruction, step.goal_image, step.move)  # 組み合わせの確認
    return step


@dataclass
class RouteBook:
    steps: List[RouteStep]
    start: Optional[Pose] = None
    goal: Optional[Pose] = None
    root: str = ""                                     # 画像のあるディレクトリ
    meta: Dict = field(default_factory=dict)           # 作った条件 (style, switch, world …)
    images: Dict[str, Image.Image] = field(default_factory=dict, repr=False)

    def image(self, name: Optional[str]) -> Optional[Image.Image]:
        if not name:
            return None
        if name not in self.images:
            path = name if os.path.isabs(name) else os.path.join(self.root, name)
            self.images[name] = Image.open(path).convert("RGB")
        return self.images[name]

    def to_dict(self) -> Dict:
        d: Dict = {}
        if self.start is not None:
            d["start"] = {"x": float(self.start[0]), "y": float(self.start[1]), "yaw": float(self.start[2])}
        if self.goal is not None:
            d["goal"] = [float(v) for v in self.goal]
        if self.meta:
            d["meta"] = self.meta
        d["steps"] = [s.to_dict() for s in self.steps]
        return d

    def save(self, out_dir: str) -> str:
        """route.yaml と、メモリ上の画像 (self.images) を out_dir に書く."""
        os.makedirs(out_dir, exist_ok=True)
        for name, img in self.images.items():
            if not os.path.isabs(name):
                img.convert("RGB").save(os.path.join(out_dir, name), quality=95)
        path = os.path.join(out_dir, ROUTE_FILE)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, allow_unicode=True)
        self.root = out_dir
        return path


def route_file(path: str) -> Optional[str]:
    """path がルート帳 (route.yaml / それを含むディレクトリ / steps を持つ yaml) ならそのファイル."""
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.isdir(path):
        f = os.path.join(path, ROUTE_FILE)
        return f if os.path.exists(f) else None
    if os.path.isfile(path) and path.lower().endswith((".yaml", ".yml")):
        try:
            with open(path) as fh:
                data = yaml.safe_load(fh) or {}
        except yaml.YAMLError:
            return None
        return path if isinstance(data, dict) and "steps" in data else None
    return None


def is_route_book(path: str) -> bool:
    return route_file(path) is not None


def load_route_book(path: str, load_images: bool = True) -> RouteBook:
    f = route_file(path)
    if f is None:
        raise FileNotFoundError(f"no route book ({ROUTE_FILE}) at {path}")
    with open(f) as fh:
        data = yaml.safe_load(fh) or {}
    steps = [parse_step(s) for s in data.get("steps") or []]
    if not steps:
        raise ValueError(f"{f}: no steps")
    book = RouteBook(steps, start=_parse_pose(data.get("start")), goal=_parse_pose(data.get("goal")),
                     root=os.path.dirname(f), meta=dict(data.get("meta") or {}))
    if load_images:
        for s in steps:
            for name in s.image_names():
                book.image(name)
    return book


# ---------------------------------------------------------------------------
# 実行 (どのステップか・次へ進むか・モデルに何を入れるか)
# ---------------------------------------------------------------------------
@dataclass
class RunnerConfig:
    image_threshold: float = 0.9      # until.image の既定しきい値
    reach_radius: float = 0.4         # until.reached の既定 [m]
    pose_radius: float = 0.5          # until.pose の既定 [m]
    yaw_tol_deg: float = 25.0         # until.pose に向きがある場合の既定 [deg]
    default_timeout: float = 0.0      # ステップの timeout の既定 [s] (0 = なし)


@dataclass
class StepInputs:
    modality: str
    instruction: Optional[str]
    goal_image: Optional[Image.Image]
    goal_pose: Optional[Pose]          # ロボット座標 (前 m, 左 m, dyaw rad)
    key_image: Optional[Image.Image]   # 表示用 (ゴール画像 or 終わりの景色)


@dataclass
class _StepState:
    t0: Optional[float] = None
    pose0: Optional[Pose] = None
    prev: Optional[Pose] = None
    traveled: float = 0.0
    turned: float = 0.0               # [rad] 左 +
    move_goal: Optional[Pose] = None  # ワールド (オドメトリ) 座標
    similarity: Optional[float] = None
    best_similarity: Optional[float] = None
    move_dist: Optional[float] = None


class RouteRunner:
    """ルート帳を先頭から順に実行する. update() で終わり条件を見て次のステップへ進める.

    pose はオドメトリ (道のり・曲がった角度・move のゴール). world_pose は until.pose の判定用
    (Gazebo の真値 / 自己位置. 省略すると pose を使う)。
    """

    def __init__(self, book: RouteBook, cfg: Optional[RunnerConfig] = None):
        if not book.steps:
            raise ValueError("empty route book")
        self.book = book
        self.cfg = cfg or RunnerConfig()
        self.index = 0
        self.done = False
        self.failed = False
        self.last_reason: Optional[str] = None
        self.history: List[Dict] = []        # 終わったステップ: index, reason, 時間, 道のり, 曲がった角度
        self._st = _StepState()
        self._anchor: Optional[Pose] = None  # 次の move の基準 (直前の move のゴール)

    # ------------------------------------------------------------------
    @property
    def current(self) -> RouteStep:
        return self.book.steps[self.index]

    @property
    def num_steps(self) -> int:
        return len(self.book.steps)

    @property
    def state(self) -> _StepState:
        return self._st

    def until_of(self, step: RouteStep) -> Dict:
        if step.until:
            return step.until
        if step.move is not None:
            return {"reached": self.cfg.reach_radius}
        if step.goal_image:
            return {"image": step.goal_image}
        return {}

    def _begin(self, t: float, pose: Optional[Pose]) -> None:
        st = _StepState(t0=t, pose0=tuple(pose) if pose is not None else None,
                        prev=tuple(pose) if pose is not None else None)
        step = self.current
        base = self._anchor if self._anchor is not None else pose
        if step.move is not None and base is not None:
            gx, gy = to_world(np.array(step.move[:2]), base[:2], base[2])
            st.move_goal = (float(gx), float(gy), float(wrap_angle(base[2] + step.move[2])))
        self._st = st

    def start(self, t: float, pose: Optional[Pose]) -> None:
        self.index, self.done, self.failed, self.last_reason, self.history = 0, False, False, None, []
        self._anchor = None
        self._begin(t, pose)

    # ------------------------------------------------------------------
    def _check(self, step: RouteStep, t: float, pose: Optional[Pose], world_pose: Optional[Pose],
               similarity: Optional[SimilarityFn]) -> Optional[str]:
        st, u, cfg = self._st, self.until_of(step), self.cfg
        hits = []
        if "time" in u and t - st.t0 >= float(u["time"]):
            hits.append(f"time {t - st.t0:.1f}s")
        if "distance" in u and st.traveled >= float(u["distance"]):
            hits.append(f"distance {st.traveled:.2f}m")
        gate = st.traveled >= float(u.get("after_m", 0.0))
        if "turn_deg" in u and gate:
            target = math.radians(float(u["turn_deg"]))
            if (target >= 0 and st.turned >= target) or (target < 0 and st.turned <= target):
                hits.append(f"turned {math.degrees(st.turned):+.0f}deg")
        if "reached" in u and st.move_goal is not None and pose is not None:
            d = math.hypot(pose[0] - st.move_goal[0], pose[1] - st.move_goal[1])
            st.move_dist = d
            if d <= float(u["reached"]):
                hits.append(f"reached move goal ({d:.2f}m)")
        if "pose" in u and world_pose is not None:
            target = [float(v) for v in u["pose"]]
            d = math.hypot(world_pose[0] - target[0], world_pose[1] - target[1])
            ok = d <= float(u.get("radius", cfg.pose_radius))
            if ok and len(target) > 2:
                err = wrap_angle(world_pose[2] - math.radians(target[2]))
                ok = abs(err) <= math.radians(float(u.get("yaw_tol_deg", cfg.yaw_tol_deg)))
            if ok:
                hits.append(f"pose ({d:.2f}m)")
        if u.get("image") and similarity is not None:
            img = self.book.image(str(u["image"]))
            s = float(similarity(str(u["image"]), img))
            st.similarity = s
            st.best_similarity = s if st.best_similarity is None else max(st.best_similarity, s)
            if gate and s >= float(u.get("threshold", cfg.image_threshold)):
                hits.append(f"image {s:.3f}")
        return ", ".join(hits) if hits else None

    def update(self, t: float, pose: Optional[Pose], similarity: Optional[SimilarityFn] = None,
               world_pose: Optional[Pose] = None) -> Optional[str]:
        """今の観測で終わり条件を見る. 次へ進んだ / 完了 / 失敗 のときはその説明を返す."""
        if self.done or self.failed:
            return None
        st = self._st
        if st.t0 is None:
            self._begin(t, pose)
            st = self._st
        if pose is not None:
            if st.prev is not None:
                st.traveled += math.hypot(pose[0] - st.prev[0], pose[1] - st.prev[1])
                st.turned += wrap_angle(pose[2] - st.prev[2])
            elif st.pose0 is None:  # オドメトリが後から来た
                self._begin(st.t0, pose)
                st = self._st
            st.prev = tuple(pose)
        step = self.current
        reason = self._check(step, t, pose, world_pose if world_pose is not None else pose, similarity)
        if reason is not None:
            self.history.append({"index": self.index, "reason": reason, "time": t - st.t0,
                                 "distance": st.traveled, "turned_deg": math.degrees(st.turned)})
            self.last_reason = reason
            if self.index == self.num_steps - 1:
                self.done = True
                return f"route finished: step {self.index + 1}/{self.num_steps} [{reason}]"
            self._anchor = st.move_goal  # move のステップの後は、その予定の位置から次の move を測る
            self.index += 1
            self._begin(t, pose)
            return f"route step {self.index}/{self.num_steps} -> {self.index + 1}/{self.num_steps} [{reason}]: " \
                   f"{self.current.describe()}"
        timeout = step.timeout or self.cfg.default_timeout
        if timeout and t - st.t0 > timeout:
            self.failed = True
            self.last_reason = f"timeout {timeout:.0f}s"
            return f"route failed: step {self.index + 1}/{self.num_steps} timed out after {timeout:.0f}s"
        return None

    # ------------------------------------------------------------------
    def inputs(self, pose: Optional[Pose]) -> StepInputs:
        step = self.current
        goal_pose = None
        if step.move is not None:
            if pose is None or self._st.move_goal is None:
                raise ValueError("move step needs odometry")
            goal_pose = relative_pose(pose, self._st.move_goal)
        img = self.book.image(step.goal_image)
        key = img if img is not None else self.book.image(self.until_of(step).get("image"))
        return StepInputs(step.modality, step.instruction, img, goal_pose, key)

    def lines(self) -> List[str]:
        """デバッグ画面に出す説明."""
        if self.done:
            return ["route finished"]
        step, st = self.current, self._st
        u = self.until_of(step)
        out = [f"route step {self.index + 1}/{self.num_steps} [{step.modality}]", step.describe()[:44]]
        conds = []
        if "time" in u:
            conds.append(f"time {u['time']}s")
        if "distance" in u:
            conds.append(f"dist {float(u['distance']):.1f}m")
        if "turn_deg" in u:
            conds.append(f"turn {float(u['turn_deg']):+.0f}")
        if "reached" in u:
            conds.append("reached")
        if "pose" in u:
            conds.append("pose")
        if u.get("image"):
            conds.append(f"view>={float(u.get('threshold', self.cfg.image_threshold)):.2f}")
        out.append("until " + (" | ".join(conds) if conds else "(manual)"))
        now = f"dist {st.traveled:.1f}m turn {math.degrees(st.turned):+.0f}"
        if st.similarity is not None:
            now += f" view {st.similarity:.2f}"
        if st.move_dist is not None:
            now += f" goal {st.move_dist:.1f}m"
        out.append(now)
        return out


def single_instruction_book(instruction: str, seconds: Optional[float] = None) -> RouteBook:
    """言葉だけの 1 ステップ (seconds 経ったら終わり. None なら終わらない)."""
    until = {"time": float(seconds)} if seconds else {}
    return RouteBook([RouteStep(instruction=instruction, until=until)])


# ---------------------------------------------------------------------------
# 経路からルート帳を作る (要所 = 曲がり角)
# ---------------------------------------------------------------------------
def _arclength(path: np.ndarray) -> np.ndarray:
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def rdp_indices(path: np.ndarray, tol: float) -> np.ndarray:
    """Ramer-Douglas-Peucker で折れ線を単純化し、残す点の番号を返す."""
    pts = np.asarray(path, dtype=np.float64)
    n = len(pts)
    if n <= 2:
        return np.arange(n)
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = pts[i], pts[j]
        ab = b - a
        length = float(np.hypot(ab[0], ab[1]))
        rel = pts[i + 1:j] - a
        if length < 1e-9:
            d = np.hypot(rel[:, 0], rel[:, 1])
        else:
            d = np.abs(ab[0] * rel[:, 1] - ab[1] * rel[:, 0]) / length
        k = int(np.argmax(d))
        if d[k] > tol:
            keep[i + 1 + k] = True
            stack += [(i, i + 1 + k), (i + 1 + k, j)]
    return np.nonzero(keep)[0]


@dataclass
class Turn:
    s_start: float      # 曲がり始め (道のり [m])
    s_end: float        # 曲がり終わり
    angle: float        # [rad] 左 +


def find_turns(path: np.ndarray, min_turn_deg: float = 40.0, rdp_tol: float = 0.2, merge_dist: float = 1.2,
               turn_radius: float = 0.5) -> List[Turn]:
    """経路の曲がり角を探す. 単純化した折れ線の角で曲がる角度を測り、近い角はまとめる."""
    path = np.asarray(path, dtype=np.float64)
    if len(path) < 3:
        return []
    s = _arclength(path)
    idx = rdp_indices(path, rdp_tol)
    verts = []
    for k in range(1, len(idx) - 1):
        a, b, c = path[idx[k - 1]], path[idx[k]], path[idx[k + 1]]
        h1 = math.atan2(b[1] - a[1], b[0] - a[0])
        h2 = math.atan2(c[1] - b[1], c[0] - b[0])
        verts.append((float(s[idx[k]]), float(wrap_angle(h2 - h1))))
    groups: List[List[Tuple[float, float]]] = []
    for v in verts:
        if groups and v[0] - groups[-1][-1][0] < merge_dist:
            groups[-1].append(v)
        else:
            groups.append([v])
    turns = []
    for g in groups:
        ang = sum(a for _, a in g)
        if abs(math.degrees(ang)) < min_turn_deg:
            continue
        turns.append(Turn(max(0.0, g[0][0] - turn_radius), min(float(s[-1]), g[-1][0] + turn_radius), ang))
    merged: List[Turn] = []
    for t in turns:  # 曲がり区間が重なったらまとめる
        if merged and t.s_start <= merged[-1].s_end:
            m = merged[-1]
            merged[-1] = Turn(m.s_start, max(m.s_end, t.s_end), m.angle + t.angle)
        else:
            merged.append(t)
    return merged


def _point_at(path: np.ndarray, s_all: np.ndarray, s: float) -> np.ndarray:
    s = float(np.clip(s, 0.0, s_all[-1]))
    return np.array([np.interp(s, s_all, path[:, 0]), np.interp(s, s_all, path[:, 1])])


def _heading_at(path: np.ndarray, s_all: np.ndarray, s: float, mode: str, window: float = 0.5) -> float:
    """mode: in = s に入ってくる向き / out = s から出ていく向き."""
    total = float(s_all[-1])
    if mode == "in":
        a, b = max(0.0, s - window), s
        if b - a < 1e-3:
            a, b = 0.0, min(total, window)
    else:
        a, b = s, min(total, s + window)
        if b - a < 1e-3:
            a, b = max(0.0, total - window), total
    p, q = _point_at(path, s_all, a), _point_at(path, s_all, b)
    return math.atan2(q[1] - p[1], q[0] - p[0])


@dataclass
class Leg:
    kind: str           # straight | turn
    s_start: float
    s_end: float
    start: Pose         # 区間の始めの姿勢 (経路上)
    end: Pose           # 区間の終わりの姿勢 (= 次の要所)
    angle: float = 0.0  # turn: 曲がる角度 [rad] 左 +

    @property
    def length(self) -> float:
        return self.s_end - self.s_start


def route_legs(path: np.ndarray, start_yaw: Optional[float] = None, goal_yaw: Optional[float] = None,
               min_turn_deg: float = 40.0, min_straight: float = 0.8, align_deg: float = 30.0,
               rdp_tol: float = 0.2, merge_dist: float = 1.2, turn_radius: float = 0.5) -> List[Leg]:
    """経路を「直進」と「曲がる」の区間に分ける. 区間の終わりが要所 (次の指示を出す場所)."""
    path = np.asarray(path, dtype=np.float64)
    s_all = _arclength(path)
    total = float(s_all[-1])
    legs: List[Leg] = []

    def pose_at(s: float, mode: str) -> Pose:
        p = _point_at(path, s_all, s)
        return float(p[0]), float(p[1]), float(_heading_at(path, s_all, s, mode))

    # スタートの向きが経路と大きく違えば、まずその場で向きを合わせる
    h0 = _heading_at(path, s_all, 0.0, "out")
    if start_yaw is not None:
        d0 = float(wrap_angle(h0 - start_yaw))
        if abs(math.degrees(d0)) >= align_deg:
            p0 = _point_at(path, s_all, 0.0)
            legs.append(Leg("turn", 0.0, 0.0, (float(p0[0]), float(p0[1]), float(start_yaw)),
                            (float(p0[0]), float(p0[1]), h0), d0))
    cur = 0.0
    for t in find_turns(path, min_turn_deg, rdp_tol, merge_dist, turn_radius):
        a = t.s_start
        if a - cur >= min_straight:
            legs.append(Leg("straight", cur, a, pose_at(cur, "out"), pose_at(a, "in")))
        else:
            a = cur  # 短い直進は曲がりに含める
        legs.append(Leg("turn", a, t.s_end, pose_at(a, "in"), pose_at(t.s_end, "out"), t.angle))
        cur = t.s_end
    if total - cur >= min_straight or not legs or legs[-1].s_end < total - 1e-6:
        if legs and legs[-1].kind == "turn" and total - cur < min_straight and legs[-1].s_end > 0.0:
            last = legs[-1]  # 残りが短い: 最後の曲がりを経路の終わりまで伸ばす
            legs[-1] = Leg("turn", last.s_start, total, last.start, pose_at(total, "in"), last.angle)
        else:
            legs.append(Leg("straight", cur, total, pose_at(cur, "out"), pose_at(total, "in")))
    # 続けて曲がる区間 (間の直進が短いもの) は 1 つにまとめる。左右に振るだけ (合計が小さい) なら直進扱い
    merged: List[Leg] = []
    for lg in legs:
        if merged and lg.kind == "turn" and merged[-1].kind == "turn":
            prev = merged[-1]
            merged[-1] = Leg("turn", prev.s_start, lg.s_end, prev.start, lg.end, prev.angle + lg.angle)
        else:
            merged.append(lg)
    legs = [Leg("straight", lg.s_start, lg.s_end, lg.start, lg.end)
            if lg.kind == "turn" and abs(math.degrees(lg.angle)) < min_turn_deg and lg.length > 0 else lg
            for lg in merged]
    legs = [lg for lg in legs if not (lg.kind == "turn" and abs(math.degrees(lg.angle)) < align_deg)]
    out: List[Leg] = []  # 直進が続いたらつなぐ
    for lg in legs:
        if out and lg.kind == "straight" and out[-1].kind == "straight":
            out[-1] = Leg("straight", out[-1].s_start, lg.s_end, out[-1].start, lg.end)
        else:
            out.append(lg)
    legs = out
    if goal_yaw is not None and legs:
        last = legs[-1]
        legs[-1] = Leg(last.kind, last.s_start, last.s_end, last.start, (last.end[0], last.end[1], float(goal_yaw)),
                       last.angle)
    return legs


@dataclass
class Phrases:
    straight: str = "go straight"
    left: str = "turn left"
    right: str = "turn right"


STYLES = ("language", "image", "pose", "language_pose")
SWITCHES = ("image", "odom", "pose")


@dataclass
class RoutePlan:
    book: RouteBook                 # 画像はまだ無い (key_poses の位置で撮って book.images に入れる)
    legs: List[Leg]
    key_poses: Dict[str, Pose]      # 画像名 -> 撮る姿勢


def plan_route_book(path: np.ndarray, start: Pose, goal_yaw: Optional[float] = None, style: str = "language",
                    switch: str = "image", phrases: Optional[Phrases] = None, turn_margin_deg: float = 15.0,
                    reach_radius: float = 0.4, pose_radius: float = 0.5, after_ratio: float = 0.6,
                    distance_slack: float = 1.5, step_timeout: float = 0.0, **leg_kwargs) -> RoutePlan:
    """経路 (Gazebo の地図で計画したもの) から、要所ごとの指示を作る.

    style (要所と要所の間の指示):
      language      : 言葉 ("go straight" / "turn left" / "turn right")
      image         : 次の要所の景色 (画像ゴール. topomap を要所だけに間引いたもの)
      pose          : 次の要所の相対位置 (前の要所の予定位置から, オドメトリ)
      language_pose : 言葉 + 相対位置
    switch (次の要所に着いたことの判定):
      image : 次の要所の景色が見えたら (予定の道のりの after_ratio 以上進んでから) / その場の曲がりは角度でも可
      odom  : オドメトリだけ (その場の曲がりは角度, それ以外は予定の道のり)
      pose  : Gazebo の真値の位置 (切り替えが正しい場合の上限を見るため. 実機では自己位置が必要)
    """
    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}")
    if switch not in SWITCHES:
        raise ValueError(f"switch must be one of {SWITCHES}")
    phrases = phrases or Phrases()
    legs = route_legs(np.asarray(path, dtype=np.float64), start_yaw=start[2], goal_yaw=goal_yaw, **leg_kwargs)
    use_move = style in ("pose", "language_pose")
    if use_move and legs and legs[0].kind == "turn" and legs[0].length < 1e-6 and len(legs) > 1:
        # 相対位置で指示する場合、スタートでの向き合わせは不要 (次の要所の相対位置に向きのずれも入る)
        first = legs[1]
        legs = [Leg(first.kind, first.s_start, first.s_end, legs[0].start, first.end, first.angle)] + legs[2:]
    steps: List[RouteStep] = []
    keys: Dict[str, Pose] = {}
    for i, leg in enumerate(legs):
        key = f"key_{i}.jpg"
        if leg.kind == "turn":
            turn_target = math.copysign(max(abs(math.degrees(leg.angle)) - turn_margin_deg, 10.0), leg.angle)
            words = phrases.left if leg.angle > 0 else phrases.right
        else:
            turn_target = None
            words = phrases.straight
        # 相対位置は 1 つ前の要所 (最初はスタートの実際の姿勢) から測る
        rel = relative_pose(tuple(start) if i == 0 else legs[i - 1].end, leg.end)
        move = (rel[0], rel[1], rel[2]) if use_move else None
        if style == "image" or (switch == "image" and move is None):
            keys[key] = leg.end
        until: Dict = {}
        in_place = leg.kind == "turn" and leg.length < 0.3
        # 曲がりながら進む区間: 角度は途中で振れる (左右に振る経路など) ので、道のりと景色で判定する
        long_turn = leg.kind == "turn" and leg.length >= 1.0
        if switch == "pose":
            until["pose"] = [round(leg.end[0], 3), round(leg.end[1], 3)] + \
                ([round(math.degrees(leg.end[2]), 1)] if in_place else [])
            until["radius"] = pose_radius
        elif move is not None:
            until["reached"] = reach_radius
            if abs(rel[0]) + abs(rel[1]) < 0.3:  # その場で曲がる: 位置では判定できない
                until["turn_deg"] = round(turn_target, 1)
        elif switch == "odom":
            if leg.kind == "turn" and not long_turn:
                until["turn_deg"] = round(turn_target, 1)
            elif i < len(legs) - 1:
                until["distance"] = round(leg.length, 2)
            else:  # 最後: ゴールの手前で止まっても終われるように少し短く
                until["distance"] = round(max(leg.length - 0.3, 0.6 * leg.length), 2)
        else:  # image
            until["image"] = key
            if leg.kind == "turn" and not long_turn:
                until["turn_deg"] = round(turn_target, 1)
            else:
                until["after_m"] = round(after_ratio * leg.length, 2)
                until["distance"] = round(distance_slack * leg.length + 1.0, 2)
        note = f"{leg.kind} {leg.length:.1f}m" + (f" {math.degrees(leg.angle):+.0f}deg" if leg.kind == "turn" else "")
        step = RouteStep(
            instruction=words if style in ("language", "language_pose") else None,
            goal_image=key if style == "image" else None,
            move=move, until=until, timeout=step_timeout or None, note=note)
        steps.append(step)
    goal = legs[-1].end if legs else None
    meta = {"style": style, "switch": switch,
            "phrases": {"straight": phrases.straight, "left": phrases.left, "right": phrases.right},
            "legs": [{"kind": lg.kind, "length_m": round(lg.length, 2), "angle_deg": round(math.degrees(lg.angle), 1),
                      "end": [round(v, 3) for v in lg.end]} for lg in legs]}
    book = RouteBook(steps, start=tuple(start), goal=goal, meta=meta)
    return RoutePlan(book, legs, keys)


def describe_book(book: RouteBook) -> List[str]:
    return [f"{i + 1}. {s.describe()}  until {s.until or '(default)'}" for i, s in enumerate(book.steps)]
