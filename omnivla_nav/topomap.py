"""ゴール画像列 (topological map) の読み書きとサブゴール管理 (ROS 非依存).

ディレクトリ形式 (topomap_recorder が作成):
    <goal_dir>/
        0.jpg, 1.jpg, ..., N.jpg     # 経路に沿ったサブゴール画像 (N が最終ゴール)
        poses.yaml                    # 任意. 各画像を撮った位置 (Gazebo の真値)
単一画像ファイルを指定した場合は 1 ノードの topomap として扱う (同名 .yaml があれば姿勢も読む)。
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import yaml
from PIL import Image

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


@dataclass
class GoalNode:
    image: Image.Image
    pose: Optional[Tuple[float, float, float]] = None  # (x, y, yaw) ワールド座標
    path: str = ""


def _numeric_key(path: str):
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r"^(\d+)$", stem)
    return (0, int(m.group(1)), stem) if m else (1, 0, stem)


def _load_pose_entry(entry) -> Optional[Tuple[float, float, float]]:
    if entry is None:
        return None
    if isinstance(entry, dict):
        return float(entry["x"]), float(entry["y"]), float(entry.get("yaw", 0.0))
    vals = list(entry)
    return float(vals[0]), float(vals[1]), float(vals[2]) if len(vals) > 2 else 0.0


def load_goal_sequence(path: str) -> List[GoalNode]:
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.isfile(path):
        pose = None
        side = os.path.splitext(path)[0] + ".yaml"
        if os.path.exists(side):
            with open(side) as f:
                pose = _load_pose_entry(yaml.safe_load(f))
        return [GoalNode(Image.open(path).convert("RGB"), pose, path)]
    if not os.path.isdir(path):
        raise FileNotFoundError(f"goal path not found: {path}")
    images = sorted([os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(IMAGE_EXTS)],
                    key=_numeric_key)
    if not images:
        raise FileNotFoundError(f"no goal images in {path}")
    poses = {}
    pose_file = os.path.join(path, "poses.yaml")
    if os.path.exists(pose_file):
        with open(pose_file) as f:
            data = yaml.safe_load(f) or {}
        for item in data.get("nodes", []):
            poses[str(item["image"])] = _load_pose_entry(item)
    return [GoalNode(Image.open(p).convert("RGB"), poses.get(os.path.basename(p)), p) for p in images]


class TopomapWriter:
    """サブゴール画像と姿勢を逐次保存する."""

    def __init__(self, out_dir: str, overwrite: bool = False):
        self.out_dir = os.path.abspath(os.path.expanduser(out_dir))
        if os.path.isdir(self.out_dir) and os.listdir(self.out_dir) and not overwrite:
            raise FileExistsError(f"{self.out_dir} is not empty (use overwrite)")
        os.makedirs(self.out_dir, exist_ok=True)
        if overwrite:
            for f in os.listdir(self.out_dir):
                if f.lower().endswith(IMAGE_EXTS) or f == "poses.yaml":
                    os.remove(os.path.join(self.out_dir, f))
        self.nodes: List[dict] = []

    def add(self, image, pose: Optional[Tuple[float, float, float]] = None) -> str:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        name = f"{len(self.nodes)}.jpg"
        image.convert("RGB").save(os.path.join(self.out_dir, name), quality=95)
        entry = {"image": name}
        if pose is not None:
            entry.update({"x": float(pose[0]), "y": float(pose[1]), "yaw": float(pose[2])})
        self.nodes.append(entry)
        with open(os.path.join(self.out_dir, "poses.yaml"), "w") as f:
            yaml.safe_dump({"nodes": self.nodes}, f, sort_keys=False)
        return os.path.join(self.out_dir, name)


class GoalTracker:
    """サブゴール列を順にたどる. 到達判定は真値姿勢 (Gazebo) か画像類似度で行う.

    reach_check:
      "pose"  : ノード姿勢との距離 < radius (姿勢が無いノードでは使えない)
      "image" : 画像埋め込みのコサイン類似度 >= image_threshold
      "auto"  : 姿勢があれば pose、無ければ image
      "none"  : 自動では進めない (最終ゴールの判定もしない)
    """

    def __init__(self, nodes: Sequence[GoalNode], reach_check: str = "auto", subgoal_radius: float = 0.6,
                 goal_radius: float = 0.4, image_threshold: float = 0.92, lookahead_nodes: int = 2):
        if not nodes:
            raise ValueError("empty goal sequence")
        self.nodes = list(nodes)
        self.reach_check = reach_check
        self.subgoal_radius = subgoal_radius
        self.goal_radius = goal_radius
        self.image_threshold = image_threshold
        self.lookahead_nodes = max(0, int(lookahead_nodes))
        self.index = 0
        self.done = False
        self.last_similarity: Optional[float] = None
        self.last_distance: Optional[float] = None

    @property
    def current(self) -> GoalNode:
        return self.nodes[self.index]

    @property
    def is_final(self) -> bool:
        return self.index == len(self.nodes) - 1

    def _mode_for(self, node: GoalNode) -> str:
        if self.reach_check == "auto":
            return "pose" if node.pose is not None else "image"
        return self.reach_check

    def _reached(self, node: GoalNode, final: bool, robot_pose, similarity_fn) -> bool:
        mode = self._mode_for(node)
        if mode == "pose":
            if node.pose is None or robot_pose is None:
                return False
            d = math.hypot(node.pose[0] - robot_pose[0], node.pose[1] - robot_pose[1])
            self.last_distance = d
            return d < (self.goal_radius if final else self.subgoal_radius)
        if mode == "image":
            if similarity_fn is None:
                return False
            s = float(similarity_fn(node))
            self.last_similarity = s
            return s >= self.image_threshold
        return False

    def update(self, robot_pose: Optional[Tuple[float, float, float]] = None,
               similarity_fn: Optional[Callable[[GoalNode], float]] = None) -> bool:
        """到達判定をして index を進める. 最終ゴール到達で done=True. 進んだら True."""
        if self.done:
            return False
        advanced = False
        # 先のノードに既に近ければ飛ばす (ショートカット)
        last = min(len(self.nodes) - 1, self.index + self.lookahead_nodes)
        for j in range(last, self.index - 1, -1):
            final = j == len(self.nodes) - 1
            if self._reached(self.nodes[j], final, robot_pose, similarity_fn):
                if final:
                    self.index = j
                    self.done = True
                    return True
                self.index = j + 1
                advanced = True
                break
        return advanced
