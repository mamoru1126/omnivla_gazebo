#!/usr/bin/env python3
"""記録済みデータ上で OmniVLA の軌跡予測誤差を測る (シミュレータ不要).

ゼロショット (公式チェックポイント) とファインチューニング後を同じサンプルで比較できる。

  # ゼロショット
  python3 training/eval_offline.py --vla_path /checkpoints/omnivla-original --split /runs/<run>/split.json
  # ファインチューニング後
  python3 training/eval_offline.py --finetuned_dir /runs/<run>/checkpoints/step_005000 --split /runs/<run>/split.json

指標: ADE (8 点平均の位置誤差 [m]), FDE (最終点誤差 [m]), yaw_err (最終点の向き誤差 [rad])  modality 別にも出力。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from omnivla_nav.data_utils import AugmentConfig, GoalSamplingConfig, modality_id  # noqa: E402
from omnivla_nav.omnivla_model import collate, load_omnivla  # noqa: E402
from omnivla_nav.trajectory_io import find_trajectories  # noqa: E402

from common import evaluate  # noqa: E402
from gazebo_dataset import GazeboDatasetConfig, GazeboNavDataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vla_path", default="",
                    help="base model (default: /checkpoints/omnivla-original, or base_vla_path of --finetuned_dir)")
    ap.add_argument("--step", type=int, default=-1)
    ap.add_argument("--finetuned_dir", default="")
    ap.add_argument("--split", default="", help="finetune の split.json (val を使う)")
    ap.add_argument("--data_dirs", nargs="*", default=[], help="split の代わりにディレクトリ全体を評価")
    ap.add_argument("--modalities", nargs="+", default=["image", "image_pose", "pose"])
    ap.add_argument("--metric_waypoint_spacing", type=float, default=0.0, help="0: 自動")
    ap.add_argument("--image_goal_offset", nargs=2, type=int, default=[2, 30])
    ap.add_argument("--pose_goal_offset", nargs=2, type=int, default=[2, 300])
    ap.add_argument("--max_samples", type=int, default=400)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--num_viz", type=int, default=8)
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.split:
        with open(args.split) as f:
            dirs = json.load(f)["val"]
    elif args.data_dirs:
        dirs = find_trajectories(args.data_dirs)
    else:
        raise SystemExit("--split or --data_dirs is required")
    if not args.vla_path and not args.finetuned_dir:
        args.vla_path = "/checkpoints/omnivla-original"
    c = load_omnivla(args.vla_path or None, args.step if args.step >= 0 else None, args.finetuned_dir or None)
    spacing = args.metric_waypoint_spacing or float(c.meta.get("metric_waypoint_spacing", 0.1))
    tag = "finetuned" if args.finetuned_dir else "base"
    out_dir = args.out_dir or os.path.join(os.path.dirname(args.split) if args.split else "/runs",
                                           f"eval_offline_{tag}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)
    tok = c.processor.tokenizer
    results = {"vla_path": args.vla_path, "finetuned_dir": args.finetuned_dir, "metric_waypoint_spacing": spacing,
               "num_trajectories": len(dirs)}
    for mod_name in args.modalities:
        mid = modality_id(mod_name)
        ds_cfg = GazeboDatasetConfig(
            metric_waypoint_spacing=spacing,
            goal=GoalSamplingConfig(tuple(args.image_goal_offset), tuple(args.pose_goal_offset), {mid: 1.0}),
            aug=AugmentConfig(enabled=False))
        ds = GazeboNavDataset(dirs, c.processor, c.action_tokenizer, ds_cfg, train=False, force_modality=mid,
                              max_samples=args.max_samples, seed=args.seed)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            collate_fn=lambda b: collate(b, tok.pad_token_id, tok.model_max_length))
        m = evaluate(c.vla, c.action_head.predict_action, c.pose_projector, loader, c.num_patches, c.device,
                     spacing, viz_dir=os.path.join(out_dir, mod_name), num_viz=args.num_viz)
        results[mod_name] = m
        print(f"[{mod_name}] n={m['num_samples']} ADE={m['ade']:.3f}m FDE={m['fde']:.3f}m "
              f"yaw_err={m['yaw_err']:.3f}rad")
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
