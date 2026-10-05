#!/usr/bin/env python3
"""navigator の走行ログ (<log_dir>/<時刻>/) を解析して図とレポートを作る (GPU・ROS 不要).

  python3 tools/plot_nav_log.py log/nav/20261005_184500
  python3 tools/plot_nav_log.py log/nav/latest           # 一番新しい走行

出力 (ログディレクトリ内):
  overview.png   地図 + サブゴール (番号と向き) + 走行軌跡 + 予測軌跡 (左旋回=青 / 右旋回=赤)
  timeline.png   時間ごとの「サブゴールの方向」「予測軌跡の方向」「旋回指令 w」「サブゴール番号」
  report.txt     サブゴールごとの区間, 予測がサブゴールと逆を向いたステップ, 指令が予測と逆のステップ
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from omnivla_nav.geometry import to_world  # noqa: E402

WORLDS = os.path.join(REPO, "ros2_ws", "src", "omnivla_gazebo", "worlds")
OPPOSITE_MIN_DEG = 15.0   # これより大きく横にあるサブゴールだけ「逆向き」を判定


def resolve_run_dir(path: str) -> str:
    path = os.path.abspath(path)
    if os.path.basename(path) == "latest":
        runs = sorted(d for d in glob.glob(os.path.join(os.path.dirname(path), "*"))
                      if os.path.isfile(os.path.join(d, "steps.csv")))
        if not runs:
            raise SystemExit(f"no runs in {os.path.dirname(path)}")
        return runs[-1]
    if not os.path.isfile(os.path.join(path, "steps.csv")):
        raise SystemExit(f"{path} has no steps.csv")
    return path


def load_steps(run_dir: str):
    rows = []
    with open(os.path.join(run_dir, "steps.csv")) as f:
        for r in csv.DictReader(f):
            rows.append({k: (float(v) if v not in ("", None) and k not in ("controller", "state") else
                             (v if k in ("controller", "state") else math.nan)) for k, v in r.items()})
    return rows


def waypoints_of(r) -> np.ndarray:
    return np.array([[r[f"wp{i}_x"], r[f"wp{i}_y"]] for i in range(8)])


def sign(x: float, eps: float = 1e-6) -> int:
    return 0 if abs(x) < eps else (1 if x > 0 else -1)


def analyze(rows, meta):
    """逆向きの予測・指令を検出する."""
    pred_opposite, cmd_opposite = [], []
    for r in rows:
        gb, pb, w = r["goal_bearing_deg"], r["pred_bearing_deg"], r["w"]
        if not math.isnan(gb) and abs(gb) > OPPOSITE_MIN_DEG and sign(pb) == -sign(gb) and abs(pb) > 3:
            pred_opposite.append(r)
        if abs(w) > 1e-3 and abs(pb) > 3 and sign(w) == -sign(pb):
            cmd_opposite.append(r)
    return pred_opposite, cmd_opposite


def write_report(run_dir, rows, meta, summary, pred_opp, cmd_opp) -> str:
    lines = []
    lines.append(f"run: {run_dir}")
    lines.append(f"goal: {meta.get('goal_source')}  world: {meta.get('world')}  modality: {meta.get('modality')}  "
                 f"controller: {meta.get('controller')}")
    lines.append(f"model: {meta.get('model')}")
    if summary:
        lines.append("summary: " + ", ".join(f"{k}={summary[k]}" for k in
                                              ("reason", "reached", "steps", "final_dist_to_goal",
                                               "min_dist_to_goal", "path_length_m", "subgoal_index")
                                              if k in summary))
    lines.append("")
    lines.append("== サブゴールごとの区間 ==")
    seg = {}
    for r in rows:
        seg.setdefault(int(r["subgoal"]), []).append(r)
    for k, rs in sorted(seg.items()):
        lines.append(f"subgoal {k:2d}: steps {int(rs[0]['step'])}-{int(rs[-1]['step'])} ({len(rs)} steps)  "
                     f"goal_bearing {rs[0]['goal_bearing_deg']:+.0f} -> {rs[-1]['goal_bearing_deg']:+.0f} deg  "
                     f"dist {rs[0]['dist']:.2f} -> {rs[-1]['dist']:.2f} m")
    lines.append("")
    lines.append(f"== 予測軌跡がサブゴールと逆側を向いたステップ ({len(pred_opp)}/{len(rows)}) ==")
    lines.append("   (サブゴールが |bearing|>15deg の横にあるのに, 予測の 5 点目が反対側) -> モデル側の問題")
    for r in pred_opp[:40]:
        lines.append(f"step {int(r['step']):4d} subgoal {int(r['subgoal']):2d}  goal {r['goal_bearing_deg']:+6.0f}deg  "
                     f"pred {r['pred_bearing_deg']:+6.0f}deg  v={r['v']:.2f} w={r['w']:+.2f}")
    lines.append("")
    lines.append(f"== 旋回指令が予測軌跡と逆のステップ ({len(cmd_opp)}/{len(rows)}) ==")
    lines.append("   -> 制御 (軌跡 -> v, w の変換) 側の問題。0 であるべき")
    for r in cmd_opp[:40]:
        lines.append(f"step {int(r['step']):4d}  pred {r['pred_bearing_deg']:+6.0f}deg  w={r['w']:+.2f}")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(run_dir, "report.txt"), "w") as f:
        f.write(text)
    return text


def plot_overview(run_dir, rows, meta, world_sdf, every: int):
    fig, ax = plt.subplots(figsize=(12, 9), dpi=110)
    nodes = [n for n in meta.get("nodes", []) if n.get("pose")]
    pts = [(r["x"], r["y"]) for r in rows if not math.isnan(r["x"])]
    if world_sdf and os.path.exists(world_sdf):
        from omnivla_nav.sim_map import build_occupancy_from_sdf

        grid = build_occupancy_from_sdf(world_sdf, resolution=0.05)
        h, w = grid.shape
        ext = [grid.origin[0], grid.origin[0] + w * grid.resolution, grid.origin[1], grid.origin[1] + h * grid.resolution]
        ax.imshow(np.where(grid.occupied, 0.15, 1.0), cmap="gray", vmin=0, vmax=1, origin="lower", extent=ext,
                  interpolation="nearest")
    # subgoals
    for n in nodes:
        x, y, yaw = n["pose"]
        ax.plot(x, y, "s", color="#b7791f", ms=7)
        ax.arrow(x, y, 0.45 * math.cos(yaw), 0.45 * math.sin(yaw), head_width=0.12, color="#b7791f", lw=1.2)
        ax.annotate(str(n["index"]), (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9,
                    color="#8a5a12", weight="bold")
    # predicted trajectories
    for r in rows[::max(1, every)]:
        if math.isnan(r["x"]):
            continue
        wp = to_world(waypoints_of(r), (r["x"], r["y"]), r["yaw"])
        color = "#2f6fdb" if r["w"] > 0.02 else ("#d0302f" if r["w"] < -0.02 else "#777777")
        ax.plot(np.r_[r["x"], wp[:, 0]], np.r_[r["y"], wp[:, 1]], "-", color=color, lw=0.9, alpha=0.75)
    # robot path
    if pts:
        p = np.array(pts)
        sc = ax.scatter(p[:, 0], p[:, 1], c=np.arange(len(p)), cmap="viridis", s=9, zorder=5)
        fig.colorbar(sc, ax=ax, fraction=0.03, label="step")
        ax.plot(p[0, 0], p[0, 1], "o", color="green", ms=10, label="start", zorder=6)
        ax.plot(p[-1, 0], p[-1, 1], "X", color="black", ms=11, label="end", zorder=6)
        allx = np.r_[p[:, 0], [n["pose"][0] for n in nodes]]
        ally = np.r_[p[:, 1], [n["pose"][1] for n in nodes]]
        m = 1.5
        ax.set_xlim(allx.min() - m, allx.max() + m)
        ax.set_ylim(ally.min() - m, ally.max() + m)
    ax.plot([], [], "-", color="#2f6fdb", label="predicted (w>0, left)")
    ax.plot([], [], "-", color="#d0302f", label="predicted (w<0, right)")
    ax.plot([], [], "s", color="#b7791f", label="subgoal (pose, heading)")
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=8)
    ax.set_title(f"{os.path.basename(run_dir)}  world={meta.get('world') or '?'}  modality={meta.get('modality')}")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    fig.tight_layout()
    out = os.path.join(run_dir, "overview.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_timeline(run_dir, rows):
    t = np.array([r["step"] for r in rows])
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), dpi=110, sharex=True)
    axs[0].plot(t, [r["goal_bearing_deg"] for r in rows], label="subgoal bearing", color="#b7791f")
    axs[0].plot(t, [r["pred_bearing_deg"] for r in rows], label="predicted (wp5) bearing", color="#2f6fdb")
    axs[0].axhline(0, color="#999", lw=0.8)
    axs[0].set_ylabel("deg (+ = left)")
    axs[0].legend(fontsize=8)
    axs[1].plot(t, [r["w"] for r in rows], label="w [rad/s]", color="#d0602f")
    axs[1].plot(t, [r["v"] for r in rows], label="v [m/s]", color="#2e8b57")
    axs[1].axhline(0, color="#999", lw=0.8)
    axs[1].legend(fontsize=8)
    axs[2].step(t, [r["subgoal"] for r in rows], where="post", color="#7a4fd0", label="subgoal index")
    axs[2].plot(t, [r["dist"] for r in rows], color="#555", label="dist to subgoal [m]")
    axs[2].legend(fontsize=8)
    axs[2].set_xlabel("step (3Hz)")
    fig.tight_layout()
    out = os.path.join(run_dir, "timeline.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--world", default="", help="ワールド名 or SDF パス (既定: meta.json の world)")
    ap.add_argument("--every", type=int, default=3, help="予測軌跡を何ステップごとに描くか")
    args = ap.parse_args(argv)
    run_dir = resolve_run_dir(args.run_dir)
    with open(os.path.join(run_dir, "meta.json")) as f:
        meta = json.load(f)
    summary = {}
    if os.path.exists(os.path.join(run_dir, "summary.json")):
        with open(os.path.join(run_dir, "summary.json")) as f:
            summary = json.load(f)
    rows = load_steps(run_dir)
    if not rows:
        raise SystemExit("steps.csv is empty (navigator が推論する前に終了した)")
    world = args.world or meta.get("world") or ""
    sdf = world if world.endswith(".sdf") else (os.path.join(WORLDS, f"{world}.sdf") if world else "")
    pred_opp, cmd_opp = analyze(rows, meta)
    print(write_report(run_dir, rows, meta, summary, pred_opp, cmd_opp))
    print("->", plot_overview(run_dir, rows, meta, sdf, args.every))
    print("->", plot_timeline(run_dir, rows))


if __name__ == "__main__":
    main()
