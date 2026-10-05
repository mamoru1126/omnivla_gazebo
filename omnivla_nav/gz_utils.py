"""Gazebo (gz-sim Harmonic) をコマンドラインから操作するユーティリティ.

`gz service` CLI を subprocess で呼ぶため、gz-transport の Python バインディングは不要。
コンテナ間で通信する場合は GZ_PARTITION を揃えること (docker-compose.yml で設定済み)。
"""
from __future__ import annotations

import math
import shutil
import subprocess
from typing import Optional, Tuple


def gz_available() -> bool:
    return shutil.which("gz") is not None


def set_model_pose(world: str, model: str, x: float, y: float, yaw: float, z: float = 0.02,
                   timeout_ms: int = 3000) -> Tuple[bool, str]:
    """UserCommands system の /world/<world>/set_pose サービスでモデルを瞬間移動させる."""
    qz, qw = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
    req = (f'name: "{model}" position: {{x: {x:.4f} y: {y:.4f} z: {z:.4f}}} '
           f"orientation: {{x: 0 y: 0 z: {qz:.6f} w: {qw:.6f}}}")
    cmd = ["gz", "service", "-s", f"/world/{world}/set_pose",
           "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean",
           "--timeout", str(int(timeout_ms)), "--req", req]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_ms / 1000.0 + 5.0)
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return False, str(e)
    text = (out.stdout or "") + (out.stderr or "")
    return ("data: true" in text), text.strip()


def pause_world(world: str, pause: bool, timeout_ms: int = 3000) -> bool:
    req = f"pause: {'true' if pause else 'false'}"
    cmd = ["gz", "service", "-s", f"/world/{world}/control", "--reqtype", "gz.msgs.WorldControl",
           "--reptype", "gz.msgs.Boolean", "--timeout", str(int(timeout_ms)), "--req", req]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_ms / 1000.0 + 5.0)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
    return "data: true" in (out.stdout or "")


def main(argv: Optional[list] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Teleport a model in a running Gazebo world")
    ap.add_argument("--world", required=True)
    ap.add_argument("--model", default="omnivla_robot")
    ap.add_argument("--x", type=float)
    ap.add_argument("--y", type=float)
    ap.add_argument("--yaw", type=float, default=0.0, help="[rad]")
    ap.add_argument("--goal_dir", default="",
                    help="topomap ディレクトリ: poses.yaml に記録された走行開始姿勢に戻す")
    args = ap.parse_args(argv)
    if args.goal_dir:
        from .topomap import load_start_pose

        start = load_start_pose(args.goal_dir)
        if start is None:
            print(f"no start pose in {args.goal_dir}/poses.yaml (route モードで作り直してください)")
            return 1
        args.x, args.y, args.yaw = start
        print(f"start pose from {args.goal_dir}: x={args.x:.2f} y={args.y:.2f} yaw={args.yaw:.2f}")
    elif args.x is None or args.y is None:
        ap.error("--x/--y or --goal_dir is required")
    ok, msg = set_model_pose(args.world, args.model, args.x, args.y, args.yaw)
    print(msg)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
