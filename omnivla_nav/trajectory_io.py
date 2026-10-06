"""GNM (visualnav-transformer) 互換フォーマットで軌跡を保存/読み込みする (ROS 非依存).

ディレクトリ構成:
    <root>/<traj_name>/
        0.jpg, 1.jpg, ...          # 記録した RGB 画像
        traj_data.pkl              # {"position": (N,2) float64, "yaw": (N,) float64,
                                   #  "perturbed": (N,) bool  ... 外乱を入れていたフレーム (DART, 任意)}
        meta.json                  # 本リポジトリ独自のメタ情報 (記録レート, ワールド名, 時刻など)

GNM/ViNT/NoMaD の学習コード (vint_train) と同じ形式なので、そちらの学習にも流用できる。
"""
from __future__ import annotations

import json
import os
import pickle
import shutil
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

TRAJ_FILE = "traj_data.pkl"
META_FILE = "meta.json"


class TrajectoryWriter:
    """1 本の軌跡をフレーム単位で書き出す."""

    def __init__(self, root: str, name: str, jpeg_quality: int = 95,
                 resize: Optional[Tuple[int, int]] = None, metadata: Optional[dict] = None):
        self.root = root
        self.name = name
        self.dir = os.path.join(root, name)
        os.makedirs(self.dir, exist_ok=False)
        self.jpeg_quality = int(jpeg_quality)
        self.resize = tuple(resize) if resize and resize[0] > 0 and resize[1] > 0 else None
        self.metadata = dict(metadata or {})
        self.positions: List[Tuple[float, float]] = []
        self.yaws: List[float] = []
        self.stamps: List[float] = []
        self.perturbed: List[bool] = []
        self.closed = False

    def __len__(self) -> int:
        return len(self.positions)

    def add(self, image: Union[np.ndarray, Image.Image], x: float, y: float, yaw: float,
            stamp: Optional[float] = None, perturbed: bool = False) -> int:
        """perturbed=True: このフレームの間はお手本ではなく外乱で動いていた (正解ラベルに使わない)."""
        if self.closed:
            raise RuntimeError("writer already closed")
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        if self.resize is not None:
            image = image.resize(self.resize, Image.BILINEAR)
        idx = len(self.positions)
        image.save(os.path.join(self.dir, f"{idx}.jpg"), quality=self.jpeg_quality)
        self.positions.append((float(x), float(y)))
        self.yaws.append(float(yaw))
        self.stamps.append(float(stamp) if stamp is not None else float("nan"))
        self.perturbed.append(bool(perturbed))
        return idx

    def path_length(self) -> float:
        if len(self.positions) < 2:
            return 0.0
        p = np.asarray(self.positions)
        return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())

    def close(self, min_frames: int = 1, min_length: float = 0.0, extra_meta: Optional[dict] = None) -> bool:
        """保存を確定する. 短すぎる軌跡は削除して False を返す."""
        if self.closed:
            return os.path.isdir(self.dir)
        self.closed = True
        if len(self.positions) < min_frames or self.path_length() < min_length:
            shutil.rmtree(self.dir, ignore_errors=True)
            return False
        data = {
            "position": np.asarray(self.positions, dtype=np.float64).reshape(-1, 2),
            "yaw": np.asarray(self.yaws, dtype=np.float64).reshape(-1),
            "perturbed": np.asarray(self.perturbed, dtype=bool).reshape(-1),
        }
        with open(os.path.join(self.dir, TRAJ_FILE), "wb") as f:
            pickle.dump(data, f)
        meta = dict(self.metadata)
        meta.update(extra_meta or {})
        meta.update({
            "num_frames": len(self.positions),
            "path_length_m": self.path_length(),
            "perturbed_frames": int(sum(self.perturbed)),
            "stamps": self.stamps,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        with open(os.path.join(self.dir, META_FILE), "w") as f:
            json.dump(meta, f, indent=1)
        return True

    def discard(self) -> None:
        self.closed = True
        shutil.rmtree(self.dir, ignore_errors=True)


def load_trajectory(traj_dir: str) -> Dict[str, np.ndarray]:
    with open(os.path.join(traj_dir, TRAJ_FILE), "rb") as f:
        data = pickle.load(f)
    pos = np.asarray(data["position"], dtype=np.float64).reshape(-1, 2)
    yaw = np.asarray(data["yaw"], dtype=np.float64).reshape(-1)
    if len(pos) != len(yaw):
        raise ValueError(f"{traj_dir}: position/yaw length mismatch")
    pert = np.asarray(data.get("perturbed", np.zeros(len(pos), dtype=bool)), dtype=bool).reshape(-1)
    if len(pert) != len(pos):
        raise ValueError(f"{traj_dir}: perturbed length mismatch")
    return {"position": pos, "yaw": yaw, "perturbed": pert}


def load_meta(traj_dir: str) -> dict:
    path = os.path.join(traj_dir, META_FILE)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def image_path(traj_dir: str, t: int) -> str:
    return os.path.join(traj_dir, f"{t}.jpg")


def find_trajectories(roots: Union[str, Sequence[str]], max_depth: int = 3) -> List[str]:
    """roots 以下 (max_depth 階層まで) で traj_data.pkl を持つディレクトリを列挙 (ソート済み)."""
    if isinstance(roots, str):
        roots = [roots]
    found = []
    for root in roots:
        root = os.path.abspath(os.path.expanduser(root))
        if not os.path.isdir(root):
            raise FileNotFoundError(f"dataset directory not found: {root}")
        base_depth = root.rstrip(os.sep).count(os.sep)
        for dirpath, dirnames, filenames in os.walk(root):
            depth = dirpath.rstrip(os.sep).count(os.sep) - base_depth
            if TRAJ_FILE in filenames:
                found.append(dirpath)
                dirnames[:] = []  # 軌跡ディレクトリの中は探索しない
                continue
            if depth >= max_depth:
                dirnames[:] = []
            dirnames.sort()
    return sorted(set(found))


def unique_name(prefix: str) -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000:03d}"
