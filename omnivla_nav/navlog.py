"""navigator の走行ログ (ROS 非依存).

1 回の走行 (= ゴールを受け取ってから到達/中断/ゴール変更まで) ごとに 1 ディレクトリ:

    <root>/<YYYYmmdd_HHMMSS>/
        meta.json        起動パラメータ, モデル, ゴール列の姿勢, 開始姿勢
        steps.csv        1 推論ごとの行: 時刻, 真値姿勢, サブゴール, サブゴールの相対位置,
                         予測 8 点 (ロボット座標 [m] と向き [deg]), 指令 (v, w), レイテンシ など
        events.log       サブゴール切替・到達・停止・警告
        summary.json     終了時のまとめ (到達したか, 最終距離, 走行距離 …)
        goals/*.jpg      使ったゴール画像
        debug/*.jpg      毎ステップのデバッグ画像 (現在画像+予測軌跡, ゴール画像, 俯瞰図)
        raw/*.jpg        毎ステップのカメラ画像そのもの (モデル入力の再現用)

ファイルはコンテナ (root) から書かれるので、ホストのユーザーでも消せるよう権限を開けておく。
解析: python3 tools/plot_nav_log.py <run_dir>
"""
from __future__ import annotations

import csv
import json
import math
import os
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

NUM_WP = 8


def step_columns() -> List[str]:
    cols = ["step", "sim_time", "x", "y", "yaw", "subgoal", "num_nodes", "sg_x", "sg_y", "sg_yaw",
            "goal_local_x", "goal_local_y", "goal_bearing_deg", "pred_bearing_deg", "modality",
            "controller", "v", "w", "latency", "dist", "similarity", "state"]
    for i in range(NUM_WP):
        cols += [f"wp{i}_x", f"wp{i}_y", f"wp{i}_yaw_deg"]
    return cols


def _open_perm(path: str) -> None:
    try:
        os.chmod(path, 0o777 if os.path.isdir(path) else 0o666)
    except OSError:
        pass


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return "" if math.isnan(v) else f"{v:.5g}"
    return str(v)


class NavRunLogger:
    def __init__(self, root: str, meta: Dict, goal_images: Sequence[Image.Image] = (),
                 save_debug: bool = True, save_raw: bool = True, waypoint_index: int = 4):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        root = os.path.abspath(os.path.expanduser(root))
        os.makedirs(root, exist_ok=True)
        _open_perm(root)
        d = os.path.join(root, stamp)
        k = 1
        while os.path.exists(d):
            d = os.path.join(root, f"{stamp}_{k}")
            k += 1
        self.dir = d
        self.save_debug = save_debug
        self.save_raw = save_raw
        self.waypoint_index = waypoint_index
        for sub in ["", "goals"] + (["debug"] if save_debug else []) + (["raw"] if save_raw else []):
            p = os.path.join(d, sub)
            os.makedirs(p, exist_ok=True)
            _open_perm(p)
        self.meta = dict(meta)
        self.meta["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self._write_json("meta.json", self.meta)
        for i, img in enumerate(goal_images):
            self._save_img(img, os.path.join(d, "goals", f"{i}.jpg"), 90)
        self.csv_path = os.path.join(d, "steps.csv")
        self.csv_file = open(self.csv_path, "w", newline="")
        _open_perm(self.csv_path)
        self.csv = csv.writer(self.csv_file)
        self.csv.writerow(step_columns())
        self.events_path = os.path.join(d, "events.log")
        self.closed = False
        self.n_steps = 0
        self.first_pose = None
        self.last_pose = None
        self.last_time = None
        self.first_time = None
        self.path_len = 0.0
        self.min_final_dist = float("inf")
        self.final_goal = meta.get("final_goal_pose")

    # ------------------------------------------------------------------
    def _write_json(self, name: str, data) -> None:
        p = os.path.join(self.dir, name)
        with open(p, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        _open_perm(p)

    @staticmethod
    def _save_img(img, path: str, quality: int) -> None:
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype(np.uint8))
        img.convert("RGB").save(path, quality=quality)
        _open_perm(path)

    def event(self, sim_time: Optional[float], text: str) -> None:
        if self.closed:
            return
        new = not os.path.exists(self.events_path)
        with open(self.events_path, "a") as f:
            t = f"{sim_time:.2f}" if sim_time is not None else "-"
            f.write(f"[{t}] {text}\n")
        if new:
            _open_perm(self.events_path)

    def track_pose(self, sim_time: float, pose) -> None:
        """推論しないステップ (停止中など) でも走行距離・最終距離を更新する."""
        if pose is None:
            return
        if self.first_pose is None:
            self.first_pose, self.first_time = tuple(pose), sim_time
        if self.last_pose is not None:
            self.path_len += math.hypot(pose[0] - self.last_pose[0], pose[1] - self.last_pose[1])
        self.last_pose, self.last_time = tuple(pose), sim_time
        if self.final_goal is not None:
            self.min_final_dist = min(self.min_final_dist, math.hypot(pose[0] - self.final_goal[0],
                                                                        pose[1] - self.final_goal[1]))

    def step(self, step: int, sim_time: float, pose, subgoal: int, num_nodes: int, subgoal_pose, goal_local,
             modality: int, controller: str, v: float, w: float, latency: float, waypoints: np.ndarray,
             dist=None, similarity=None, state: str = "", debug_img=None, raw_img=None) -> None:
        if self.closed:
            return
        self.n_steps += 1
        self.track_pose(sim_time, pose)
        wps = np.asarray(waypoints, dtype=np.float64)
        k = int(np.clip(self.waypoint_index, 0, len(wps) - 1))
        pred_bearing = math.degrees(math.atan2(wps[k, 1], wps[k, 0]))
        goal_bearing = math.degrees(math.atan2(goal_local[1], goal_local[0])) if goal_local is not None else None
        x, y, yaw = pose if pose is not None else (math.nan,) * 3
        sg = subgoal_pose if subgoal_pose is not None else (None, None, None)
        gl = goal_local if goal_local is not None else (None, None)
        row = [step, sim_time, x, y, yaw, subgoal, num_nodes, sg[0], sg[1], sg[2], gl[0], gl[1], goal_bearing,
               pred_bearing, modality, controller, v, w, latency, dist, similarity, state]
        for i in range(NUM_WP):
            yaw_i = math.degrees(math.atan2(wps[i, 3], wps[i, 2])) if wps.shape[1] >= 4 else None
            row += [wps[i, 0], wps[i, 1], yaw_i]
        self.csv.writerow([_fmt(float(c)) if isinstance(c, (np.floating, float)) else _fmt(c) for c in row])
        self.csv_file.flush()
        if self.save_debug and debug_img is not None:
            self._save_img(debug_img, os.path.join(self.dir, "debug", f"{step:06d}.jpg"), 80)
        if self.save_raw and raw_img is not None:
            self._save_img(raw_img, os.path.join(self.dir, "raw", f"{step:06d}.jpg"), 90)

    def close(self, reason: str, reached: bool = False, extra: Optional[Dict] = None) -> Dict:
        if self.closed:
            return {}
        self.event(self.last_time, f"run closed: {reason}")
        self.closed = True
        self.csv_file.close()
        summary = {
            "reason": reason,
            "reached": bool(reached),
            "steps": self.n_steps,
            "sim_duration": (self.last_time - self.first_time) if self.first_time is not None else 0.0,
            "path_length_m": self.path_len,
            "start_pose": self.first_pose,
            "final_pose": self.last_pose,
            "final_goal_pose": self.final_goal,
            "final_dist_to_goal": (math.hypot(self.last_pose[0] - self.final_goal[0],
                                              self.last_pose[1] - self.final_goal[1])
                                   if self.final_goal is not None and self.last_pose is not None else None),
            "min_dist_to_goal": self.min_final_dist if math.isfinite(self.min_final_dist) else None,
            "closed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        summary.update(extra or {})
        self._write_json("summary.json", summary)
        return summary
