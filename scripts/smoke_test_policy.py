#!/usr/bin/env python3
"""GPU・チェックポイント・推論ラッパの動作確認 (シミュレータ不要).

公式リポジトリ同梱のサンプル画像 (/opt/OmniVLA/inference/current_img.jpg, goal_img.jpg) で
公式 run_omnivla.py と同じ条件 (image goal / language) の推論を行い、軌跡を表示・保存する。

  python3 scripts/smoke_test_policy.py
  python3 scripts/smoke_test_policy.py --finetuned_dir /runs/<run>/checkpoints/step_005000 \
      --current my_cur.jpg --goal my_goal.jpg
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from omnivla_nav.controller import ControllerConfig, compute_command  # noqa: E402
from omnivla_nav.viz import render_debug  # noqa: E402

OMNIVLA_ROOT = os.environ.get("OMNIVLA_ROOT", "/opt/OmniVLA")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vla_path", default="/checkpoints/omnivla-original")
    ap.add_argument("--finetuned_dir", default="")
    ap.add_argument("--current", default=os.path.join(OMNIVLA_ROOT, "inference", "current_img.jpg"))
    ap.add_argument("--goal", default=os.path.join(OMNIVLA_ROOT, "inference", "goal_img.jpg"))
    ap.add_argument("--instruction", default="move toward blue trash bin")
    ap.add_argument("--goal_pose", nargs=3, type=float, default=[1.0, -10.0, -1.5708],
                    help="pose modality 用の相対ゴール (x前[m], y左[m], yaw[rad])")
    ap.add_argument("--out", default="/runs/smoke_test")
    ap.add_argument("--repeat", type=int, default=3, help="レイテンシ計測の繰り返し回数")
    args = ap.parse_args()

    from omnivla_nav.policy import OmniVLAPolicy, PolicyConfig

    policy = OmniVLAPolicy(PolicyConfig(vla_path=args.vla_path, finetuned_dir=args.finetuned_dir or None))
    cur = Image.open(args.current).convert("RGB")
    goal = Image.open(args.goal).convert("RGB")
    os.makedirs(args.out, exist_ok=True)
    ctrl = ControllerConfig()
    cases = [
        ("image", dict(goal_image=goal)),
        ("language", dict(instruction=args.instruction)),
        ("pose", dict(goal_pose=args.goal_pose)),
        ("image_pose", dict(goal_image=goal, goal_pose=args.goal_pose)),
    ]
    for name, kw in cases:
        lat = []
        for _ in range(max(1, args.repeat)):
            out = policy.predict(cur, modality=name, **kw)
            lat.append(out.latency)
        v, w = compute_command(out.waypoints, ctrl)
        np.set_printoptions(precision=3, suppress=True)
        print(f"\n=== modality: {name} (latency {np.median(lat) * 1000:.0f} ms) ===")
        print("waypoints [x, y, cos, sin] (m, robot frame):\n", out.waypoints)
        print(f"command: v={v:.3f} m/s, w={w:.3f} rad/s  (controller={ctrl.mode})")
        gl = tuple(args.goal_pose[:2]) if "pose" in name else None
        render_debug(cur, goal if "image" in name else None, out.waypoints, goal_local=gl,
                     lines=[f"modality {name}", f"v={v:.2f} w={w:.2f}"]).save(os.path.join(args.out, f"{name}.jpg"))
    emb_c, emb_g = policy.embed(cur), policy.embed(goal)
    print(f"\nimage similarity current-goal: {policy.similarity(emb_c, emb_g):.3f}")
    print(f"visualizations saved to {args.out}")


if __name__ == "__main__":
    main()
