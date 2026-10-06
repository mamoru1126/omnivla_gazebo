#!/usr/bin/env python3
"""収集したデータの統計と、学習サンプル (現在画像 / ゴール画像 / 正解軌跡) の可視化 (GPU・torch 不要).

  python3 training/inspect_dataset.py /data/raw --num_viz 16 --out /runs/inspect

確認ポイント:
  * mean step [m] が metric_waypoint_spacing (既定 0.1) に近いか (= 0.3m/s, 3Hz で記録できているか)
  * 正解軌跡 (緑) が画像に投影したときに走行方向と一致しているか (座標系の確認)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from omnivla_nav.data_utils import (AugmentConfig, GoalSamplingConfig, MODALITY_NAMES, augment_pair,  # noqa: E402
                                    build_sample_index, denormalize_actions, make_targets, modality_id,
                                    summarize_spacing)
from omnivla_nav.trajectory_io import find_trajectories, image_path, load_meta, load_trajectory  # noqa: E402
from omnivla_nav.viz import render_debug  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dirs", nargs="+")
    ap.add_argument("--metric_waypoint_spacing", type=float, default=0.1)
    ap.add_argument("--num_viz", type=int, default=12)
    ap.add_argument("--augment", action="store_true", help="学習時の augmentation をかけた状態を可視化")
    ap.add_argument("--modality", default="", help="固定する modality (既定: image/image_pose/pose をランダム)")
    ap.add_argument("--out", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    trajs = find_trajectories(args.data_dirs)
    if not trajs:
        raise SystemExit("no trajectories found")
    lengths, frames, spacings, worlds, n_pert, n_pert_traj = [], [], [], {}, 0, 0
    data = []
    for d in trajs:
        tr = load_trajectory(d)
        meta = load_meta(d)
        n = len(tr["position"])
        frames.append(n)
        lengths.append(float(np.linalg.norm(np.diff(tr["position"], axis=0), axis=1).sum()) if n > 1 else 0.0)
        s = summarize_spacing(tr["position"])
        if s is not None:
            spacings.append(s)
        n_pert += int(tr["perturbed"].sum())
        n_pert_traj += int(tr["perturbed"].any())
        w = meta.get("world", "?")
        worlds[w] = worlds.get(w, 0) + 1
        data.append((d, tr))
    stats = {
        "trajectories": len(trajs),
        "frames": int(sum(frames)),
        "total_length_m": float(sum(lengths)),
        "frames_per_traj": [int(np.min(frames)), float(np.mean(frames)), int(np.max(frames))],
        "mean_step_m": float(np.mean(spacings)) if spacings else None,
        "worlds": worlds,
        "samples": len(build_sample_index(frames)),
        # 外乱つき収集 (DART) のフレーム. 0 なら「経路から外れた状態から戻る」データが無い
        "perturbed_frames": n_pert,
        "trajectories_with_perturbation": n_pert_traj,
    }
    print(json.dumps(stats, indent=2))
    if stats["mean_step_m"] and abs(stats["mean_step_m"] - args.metric_waypoint_spacing) > 0.5 * \
            args.metric_waypoint_spacing:
        print(f"WARNING: mean step {stats['mean_step_m']:.3f} m is far from metric_waypoint_spacing "
              f"{args.metric_waypoint_spacing} m (speed / record_rate を見直すか spacing を合わせる)")
    if args.num_viz <= 0:
        return
    out = args.out or os.path.join(args.data_dirs[0], "_inspect")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    index = build_sample_index(frames)
    goal_cfg = GoalSamplingConfig()
    force = modality_id(args.modality) if args.modality else None
    for k in range(min(args.num_viz, len(index))):
        ti, t = index[int(rng.integers(len(index)))]
        d, tr = data[ti]
        mod, goal_t, actions, goal_pose = make_targets(rng, tr["position"], tr["yaw"], t, goal_cfg,
                                                       args.metric_waypoint_spacing, force_modality=force)
        cur = Image.open(image_path(d, t)).convert("RGB")
        goal = Image.open(image_path(d, goal_t)).convert("RGB")
        cur2, goal2, actions, goal_pose = augment_pair(rng, cur, goal, actions, goal_pose, AugmentConfig(),
                                                       train=args.augment)
        gt_m = denormalize_actions(actions, args.metric_waypoint_spacing)
        gl = (float(goal_pose[0] * args.metric_waypoint_spacing), float(goal_pose[1] * args.metric_waypoint_spacing))
        lines = [os.path.basename(d), f"t={t} goal_t={goal_t} ({goal_t - t} frames)",
                 f"modality {mod} ({MODALITY_NAMES[mod]})", f"goal local ({gl[0]:.2f}, {gl[1]:.2f}) m",
                 "green: ground-truth 8 waypoints"]
        img = render_debug(cur if not args.augment else cur2, goal if not args.augment else goal2, None,
                           goal_local=gl, lines=lines, gt_waypoints=gt_m)
        img.save(os.path.join(out, f"sample_{k:03d}.jpg"), quality=90)
    print(f"visualizations -> {out}")


if __name__ == "__main__":
    main()
