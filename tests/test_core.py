"""omnivla_nav の ROS / torch 非依存部分の単体テスト.

  python3 -m pytest -q tests          (コンテナ内)
  python3 tests/test_core.py          (pytest が無い環境でも実行可)
"""
from __future__ import annotations

import math
import os
import shutil
import sys
import tempfile

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from omnivla_nav import controller, data_utils, expert, geometry, navlog, sim_map, topomap, trajectory_io, viz  # noqa: E402
from omnivla_nav import langprobe, routebook  # noqa: E402

WORLDS = os.path.join(REPO, "ros2_ws", "src", "omnivla_gazebo", "worlds")


# ---------------------------------------------------------------- geometry
def test_local_world_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        o = rng.uniform(-5, 5, 2)
        yaw = rng.uniform(-math.pi, math.pi)
        pts = rng.uniform(-10, 10, (7, 2))
        back = geometry.to_world(geometry.to_local(pts, o, yaw), o, yaw)
        assert np.allclose(back, pts)


def test_local_axes():
    # 北 (+y) を向いたロボットにとって、北は前 (+x)、西 (-x) は左 (+y)
    assert np.allclose(geometry.to_local([0.0, 2.0], [0, 0], math.pi / 2), [2.0, 0.0])
    assert np.allclose(geometry.to_local([-1.0, 0.0], [0, 0], math.pi / 2), [0.0, 1.0])


def _upstream_goal_pose(cur_xy, cur_yaw, goal_xy, goal_yaw, spacing=0.1, thres=30.0):
    """公式 run_omnivla.py の GPS(UTM)->goal_pose 計算をそのまま再現 (コンパスは北から時計回り)."""
    cur_compass = 90.0 - math.degrees(cur_yaw)
    goal_compass = 90.0 - math.degrees(goal_yaw)
    cc = -cur_compass / 180.0 * math.pi
    gc = -goal_compass / 180.0 * math.pi
    dx, dy = goal_xy[0] - cur_xy[0], goal_xy[1] - cur_xy[1]
    rel_x = dx * math.cos(cc) + dy * math.sin(cc)
    rel_y = -dx * math.sin(cc) + dy * math.cos(cc)
    r = math.hypot(rel_x, rel_y)
    if r > thres:
        rel_x *= thres / r
        rel_y *= thres / r
    return np.array([rel_y / spacing, -rel_x / spacing, math.cos(gc - cc), math.sin(gc - cc)])


def test_goal_pose_matches_upstream_convention():
    rng = np.random.default_rng(1)
    for _ in range(100):
        cur = rng.uniform(-20, 20, 2)
        goal = cur + rng.uniform(-40, 40, 2)
        cy, gy = rng.uniform(-math.pi, math.pi, 2)
        rel = geometry.relative_pose((cur[0], cur[1], cy), (goal[0], goal[1], gy))
        ours = data_utils.normalize_goal_pose(*rel, metric_spacing=0.1, max_goal_dist=30.0)
        ref = _upstream_goal_pose(cur, cy, goal, gy)
        assert np.allclose(ours, ref, atol=1e-5), (ours, ref)


# ---------------------------------------------------------------- data_utils
def test_action_targets_straight_and_turn():
    n = 20
    # +y 方向へ 0.1m/frame で直進 (yaw = 90deg)
    pos = np.stack([np.zeros(n), 0.1 * np.arange(n)], 1)
    yaw = np.full(n, math.pi / 2)
    a = data_utils.compute_action_targets(pos, yaw, 3, metric_spacing=0.1)
    assert a.shape == (8, 4)
    assert np.allclose(a[:, 0], np.arange(1, 9), atol=1e-6)
    assert np.allclose(a[:, 1], 0, atol=1e-6)
    assert np.allclose(a[:, 2], 1) and np.allclose(a[:, 3], 0, atol=1e-6)
    # 末尾はパディング (止まる)
    a_end = data_utils.compute_action_targets(pos, yaw, n - 3, metric_spacing=0.1)
    assert np.allclose(a_end[2:, 0], 2.0)
    # 左旋回 (反時計回りの円弧) -> y > 0, sin > 0
    th = np.linspace(0, math.pi / 2, n)
    pos_c = np.stack([np.sin(th), 1 - np.cos(th)], 1)
    a_c = data_utils.compute_action_targets(pos_c, th, 0, metric_spacing=0.1)
    assert np.all(a_c[1:, 1] > 0) and np.all(a_c[:, 3] > 0)


def test_flip_and_sampling():
    a = np.arange(32, dtype=np.float32).reshape(8, 4)
    g = np.array([1, 2, 3, 4], np.float32)
    fa, fg = data_utils.flip_left_right(a, g)
    assert np.allclose(fa[:, 1], -a[:, 1]) and np.allclose(fa[:, 3], -a[:, 3]) and np.allclose(fa[:, 0], a[:, 0])
    assert fg.tolist() == [1, -2, 3, -4]
    rng = np.random.default_rng(0)
    cfg = data_utils.GoalSamplingConfig()
    for t, n in [(0, 100), (95, 100), (98, 100), (10, 12)]:
        for m in (4, 5, 6):
            off = data_utils.sample_goal_offset(rng, t, n, m, cfg)
            assert 1 <= off <= n - 1 - t
            if m == 6:
                assert off <= cfg.image_goal_offset[1]
    counts = {4: 0, 5: 0, 6: 0}
    for _ in range(2000):
        counts[data_utils.choose_modality(rng, cfg.modality_weights)] += 1
    assert 850 < counts[6] < 1150


def test_make_targets_and_augment():
    rng = np.random.default_rng(3)
    n = 40
    pos = np.stack([0.1 * np.arange(n), np.zeros(n)], 1)
    yaw = np.zeros(n)
    cfg = data_utils.GoalSamplingConfig()
    mod, gt, act, gp = data_utils.make_targets(rng, pos, yaw, 5, cfg, 0.1, force_modality=6)
    assert mod == 6 and 5 < gt <= 35 and act.shape == (8, 4) and gp.shape == (4,)
    assert abs(gp[0] - (gt - 5)) < 1e-4  # 0.1m/frame -> 正規化で 1/frame
    cur = Image.new("RGB", (640, 480), (200, 10, 10))
    goal = Image.new("RGB", (640, 480), (10, 200, 10))
    c2, g2, a2, gp2 = data_utils.augment_pair(rng, cur, goal, act, gp, data_utils.AugmentConfig(), train=True)
    assert c2.size == (224, 224) and g2.size == (224, 224)
    c3, _, a3, _ = data_utils.augment_pair(rng, cur, goal, act, gp, data_utils.AugmentConfig(), train=False)
    assert np.allclose(a3, act) and c3.size == (224, 224)
    box = data_utils.random_crop_box(rng, 640, 480, 0.2, 0.1)
    assert 0 <= box[0] <= 64 and 0 <= box[1] <= 96 and box[2] - box[0] >= 512 and box[3] - box[1] >= 288


def test_turn_flags_and_balanced_weights():
    n = 60
    # 30 フレーム直進 -> 左へ 90 度旋回 (10 フレーム) -> 直進
    yaw = np.concatenate([np.zeros(30), np.linspace(0, math.pi / 2, 10), np.full(20, math.pi / 2)])
    pos = np.zeros((n, 2))
    for t in range(1, n):
        pos[t] = pos[t - 1] + 0.1 * np.array([math.cos(yaw[t - 1]), math.sin(yaw[t - 1])])
    f = data_utils.turn_flags(pos, yaw, horizon=10, threshold_deg=20)
    assert not f[:15].any()                # 曲がり角のずっと手前は直進
    assert f[25:38].all()                  # 曲がり角の直前〜旋回中は「曲がる」
    assert not f[45:].any()                # 曲がり終わった後は直進
    w = data_utils.balanced_weights(f, 0.5)
    assert abs(w.sum() - 1) < 1e-9 and abs(w[f].sum() - 0.5) < 1e-9
    assert np.allclose(data_utils.balanced_weights(f, 0.0), 1 / n)
    assert np.allclose(data_utils.balanced_weights(np.zeros(5, bool), 0.5), 0.2)


def test_modality_ids():
    assert data_utils.modality_id("image") == 6 and data_utils.modality_id("pose") == 4
    assert data_utils.modality_id("language") == 7 and data_utils.modality_id(5) == 5
    assert data_utils.modality_id("8") == 8


# ---------------------------------------------------------------- controller
def test_upstream_controller():
    cfg = controller.ControllerConfig(mode="upstream")
    v, w = controller.upstream_command(np.array([0.5, 0.0, 1.0, 0.0]), cfg)
    assert math.isclose(v, 0.3) and math.isclose(w, 0.0, abs_tol=1e-9)
    v, w = controller.upstream_command(np.array([0.5, 0.2, 1.0, 0.0]), cfg)
    # v=1.5->0.5, w=atan(0.4)*3->1.0 ; rd=0.5 < maxv/maxw=1 -> v=0.3*0.5, w=0.3
    assert math.isclose(v, 0.15, rel_tol=1e-6) and math.isclose(w, 0.3, rel_tol=1e-6)
    # 目標点が後方: その場で「目標のある側」に旋回する (公式の atan(dy/dx) では逆側に回っていた)
    v, w = controller.upstream_command(np.array([-0.2, 0.4, 1.0, 0.0]), cfg)
    assert v == 0.0 and w > 0
    v, w = controller.upstream_command(np.array([-0.2, -0.4, 1.0, 0.0]), cfg)
    assert v == 0.0 and w < 0
    v, w = controller.upstream_command(np.array([0.0, 0.0, 0.0, 1.0]), cfg)  # 公式では NameError だった分岐
    assert v == 0.0 and math.isclose(w, 0.3)
    wps = np.zeros((8, 4))
    wps[:, 0] = np.linspace(0.1, 0.8, 8)
    wps[:, 2] = 1.0
    v, w = controller.compute_command(wps, cfg)
    assert v > 0 and abs(w) < 1e-9
    v2, w2 = controller.compute_command(wps, controller.ControllerConfig(mode="pure_pursuit"))
    assert v2 > 0 and abs(w2) < 1e-9


def test_respect_predicted_speed_and_stuck():
    cfg = controller.ControllerConfig(mode="upstream")
    wps = np.zeros((8, 4))
    wps[:, 2] = 1.0
    wps[:, 0] = 0.1 * np.arange(1, 9)            # 0.1m/step = 0.3m/s (学習データの通常速度)
    v, w = controller.compute_command(wps, cfg)
    assert math.isclose(v, 0.3, rel_tol=1e-6)
    wps[:, 0] = 0.025 * np.arange(1, 9)          # 予測が短い = 減速の予測
    wps[:, 1] = 0.002 * np.arange(1, 9)
    v, w = controller.compute_command(wps, cfg)
    v0, w0 = controller.compute_command(wps, controller.ControllerConfig(mode="upstream", respect_predicted_speed=False))
    assert v0 >= 0.29 and v < 0.1 and math.isclose(v / w, v0 / w0, rel_tol=1e-6)  # 公式は全速, 曲率は同じ
    st = controller.StuckDetector(timeout=4.0)
    assert not st.update(0.0, (0, 0, 0), 0.3)
    assert not st.update(3.0, (0.01, 0, 0), 0.3)
    assert st.update(4.5, (0.02, 0, 0.01), 0.3)   # 4 秒以上ほぼ動かない -> stuck
    st.reset()
    for k in range(20):                            # 動いていれば stuck にならない
        assert not st.update(k * 0.5, (0.1 * k, 0, 0), 0.3)
    st.reset()
    for k in range(20):                            # 止まる指令中は判定しない
        assert not st.update(k * 0.5, (0, 0, 0), 0.0)


def _arc_waypoints(v, w, dt=1.0 / 3.0, n=8):
    """一定の (v, w) で走ったときの 8 waypoint (k 番目は (k+1)*dt 秒後) = 学習ラベルと同じ作り方."""
    t = (np.arange(n) + 1.0) * dt
    yaw = w * t
    if abs(w) < 1e-9:
        x, y = v * t, np.zeros(n)
    else:
        x, y = v / w * np.sin(yaw), v / w * (1.0 - np.cos(yaw))
    return np.stack([x, y, np.cos(yaw), np.sin(yaw)], axis=1)


def test_trajectory_controller_reproduces_prediction():
    cfg = controller.ControllerConfig()
    assert cfg.mode == "trajectory"   # 既定
    for v, w in [(0.3, 0.0), (0.3, 0.4), (0.2, -0.8), (0.1, 0.6), (0.25, 0.3), (0.0, 0.5), (0.0, -0.7), (0.0, 0.0)]:
        vc, wc = controller.compute_command(_arc_waypoints(v, w), cfg)
        assert abs(vc - v) < 0.01 and abs(wc - w) < 0.01, (v, w, vc, wc)
    # 公式の式: 角速度は 0.3rad/s 止まり (お手本は 0.8rad/s で曲がる), 旋回半径 0.5m 未満の予測は 1.4 倍以上大回り
    # -> 曲がり角で外側に膨らむ原因
    for v, w in [(0.2, 0.6), (0.15, 0.8), (0.1, 0.3)]:
        vu, wu = controller.compute_command(_arc_waypoints(v, w), controller.ControllerConfig(mode="upstream"))
        assert (vu / wu) > 1.4 * (v / w) and abs(wu) <= 0.3 + 1e-9, (v, w, vu / wu)
    # 上限を超える予測は曲率を保ったまま遅くする
    vc, wc = controller.compute_command(_arc_waypoints(0.3, 1.5), cfg)
    assert abs(wc) <= cfg.track_max_w + 1e-9 and math.isclose(vc / wc, 0.3 / 1.5, rel_tol=0.02)
    # 位置・向きの予測にばらつきがあっても大きく外れない
    rng = np.random.default_rng(0)
    for _ in range(50):
        v, w = rng.uniform(0.05, 0.3), rng.uniform(-0.8, 0.8)
        wps = _arc_waypoints(v, w)
        sc = np.linspace(0.3, 1.0, 8)                # 先の点ほど誤差が大きい (学習後の val 誤差 ~3cm, ~6deg)
        wps[:, :2] += rng.normal(0, 0.03, (8, 2)) * sc[:, None]
        ang = np.arctan2(wps[:, 3], wps[:, 2]) + rng.normal(0, math.radians(6), 8) * sc
        wps[:, 2], wps[:, 3] = np.cos(ang), np.sin(ang)
        vc, wc = controller.compute_command(wps, cfg)
        assert abs(vc - v) < 0.06 and abs(wc - w) < 0.12, (v, w, vc, wc)


def _office_setup():
    path = os.path.join(WORLDS, "office_0.sdf")
    if not os.path.exists(path):
        return None
    grid = sim_map.build_occupancy_from_sdf(path, resolution=0.05)
    planner = sim_map.PathPlanner(grid, inflate=0.4)
    spawn = sim_map.robot_spawn_pose(path)
    return grid, planner, planner.component(*spawn[:2])


def _sample_task(planner, comp, rng, dmin=4.0, dmax=12.0):
    for _ in range(100):
        sx, sy = planner.sample_free(rng, comp, min_clearance=0.6)
        gx, gy = planner.sample_free(rng, comp, min_clearance=0.55)
        if dmin <= math.hypot(gx - sx, gy - sy) <= dmax:
            route = planner.plan((sx, sy), (gx, gy), rng=rng, noise=0.6)
            if route is not None:
                return route
    raise AssertionError("no task")


def _expert_prediction(route, pose):
    """お手本が今の姿勢から 8 フレーム (3Hz) でどう動くか (経路から外れていれば戻る) = 学習ラベルと同じ."""
    f = expert.PathFollower(route, expert.FollowerConfig(speed=0.3))
    f.idx = int(np.argmin(np.linalg.norm(route - np.asarray(pose[:2]), axis=1)))
    p, out = tuple(pose), []
    for _ in range(8):
        for _ in range(17):
            v, w, _ = f.step(p)
            p = expert.unicycle_step(p, v, w, 1.0 / 51.0)
        loc = geometry.to_local(np.asarray(p[:2]), pose[:2], pose[2])
        out.append([loc[0], loc[1], math.cos(p[2] - pose[2]), math.sin(p[2] - pose[2])])
    return np.asarray(out)


def test_trajectory_controller_closed_loop():
    """予測が正しければ (= 経路から外れても戻る予測を出せれば) trajectory 制御で衝突せずゴールまで行けるか.

    DiffDrive の加速度制限 (1 m/s², 3 rad/s²) と 3Hz の推論周期を入れた運動学シミュレーション。
    """
    setup = _office_setup()
    if setup is None:
        return
    grid, planner, comp = setup
    rng = np.random.default_rng(5)
    cfg = controller.ControllerConfig()
    for ep in range(6):
        route = _sample_task(planner, comp, rng)
        d = np.diff(route[:4], axis=0).sum(axis=0)
        pose, v_cur, w_cur, ok = (route[0][0], route[0][1], math.atan2(d[1], d[0])), 0.0, 0.0, False
        for _ in range(600):
            if math.hypot(pose[0] - route[-1][0], pose[1] - route[-1][1]) < 0.4:
                ok = True
                break
            v_cmd, w_cmd = controller.compute_command(_expert_prediction(route, pose), cfg)
            for _ in range(33):
                v_cur += float(np.clip(v_cmd - v_cur, -0.01, 0.01))
                w_cur += float(np.clip(w_cmd - w_cur, -0.03, 0.03))
                pose = expert.unicycle_step(pose, v_cur, w_cur, 0.01)
                assert grid.clearance(pose[0], pose[1]) > 0.2, f"collision in episode {ep} at {pose}"
        assert ok, f"episode {ep} did not reach the goal"


# ---------------------------------------------------------------- DART (外乱つきデータ収集)
def test_label_valid_mask():
    p = np.zeros(30, dtype=bool)
    p[10:13] = True
    valid = data_utils.label_valid_mask(p, 8, 1)
    assert valid[1] and not valid[2]                 # t=2 は 3..10 に外乱フレームを含む
    assert not valid[10] and not valid[11]
    assert valid[12] and valid[13]                   # 外乱の最後のフレーム = 立て直しの開始は有効
    assert valid[:2].all() and valid[12:].all() and not valid[2:12].any()
    assert data_utils.label_valid_mask(np.zeros(5, dtype=bool)).all()
    # 立て直しサンプルを少なくとも 3 割引く重み (曲がるサンプルの重み付けと両立)
    turn = np.arange(100) % 10 == 0
    rec = np.arange(100) % 20 < 2
    w = data_utils.reweight(data_utils.balanced_weights(turn, 0.5), rec, 0.3)
    assert math.isclose(w.sum(), 1.0) and math.isclose(w[rec].sum(), 0.3)
    assert np.allclose(data_utils.reweight(w, np.zeros(100, bool), 0.3), w)   # 外乱データが無ければそのまま
    assert np.allclose(data_utils.reweight(w, rec, 0.1), w)                   # 既に多ければそのまま


def test_perturbed_expert_kinematic_sim():
    """外乱つきのお手本走行: 衝突せず、経路から外れた状態と、そこからの立て直しが記録されるか."""
    setup = _office_setup()
    if setup is None:
        return
    grid, planner, comp = setup
    rng = np.random.default_rng(3)
    dt, rate = 0.05, 3.0
    flags_all, offs, n_pert, successes, n_ep = [], [], 0, 0, 12
    for ep in range(n_ep):
        route = _sample_task(planner, comp, rng, 5.0, 12.0)
        pose = (route[0][0], route[0][1], float(rng.uniform(-math.pi, math.pi)))
        f = expert.PathFollower(route, expert.FollowerConfig(speed=0.3))
        pert = expert.Perturber(expert.PerturbConfig(), rng)
        pert.reset(0.0)
        t, last_rec, flags = 0.0, -1e9, []
        for _ in range(int(240 / dt)):
            v, w, done = f.step(pose)
            if done and not pert.active:
                successes += 1
                break
            v, w = pert.step(t, grid.clearance(pose[0], pose[1]),
                             math.hypot(route[-1][0] - pose[0], route[-1][1] - pose[1]), (v, w))
            if t - last_rec >= 1.0 / rate - 1e-9:
                flags.append(pert.flagged(t))
                offs.append(float(np.min(np.linalg.norm(route - np.asarray(pose[:2]), axis=1))))
                last_rec = t
            pose = expert.unicycle_step(pose, v, w, dt)
            t += dt
            assert grid.clearance(pose[0], pose[1]) > 0.2, f"collision in episode {ep}"
        n_pert += pert.count
        flags_all.append(np.asarray(flags, dtype=bool))
    assert successes == n_ep
    allf = np.concatenate(flags_all)
    valid = np.concatenate([data_utils.label_valid_mask(fl) for fl in flags_all])
    assert n_pert >= n_ep and 0.05 < allf.mean() < 0.3, (n_pert, allf.mean())
    assert 0.5 < valid.mean() < 0.95, valid.mean()   # 外乱区間のラベルは除外, 大半は使える
    assert np.max(offs) > 0.4                         # 経路から外れた状態が記録されている


def test_goal_tracker_pass_detection():
    img = Image.new("RGB", (8, 8))
    nodes = [topomap.GoalNode(img, (1.0, 1.0, 0.0)), topomap.GoalNode(img, (3.0, 1.0, 0.0))]
    tr = topomap.GoalTracker(nodes, subgoal_radius=0.6, lookahead_nodes=0)
    assert not tr.update((0.3, 1.0, 0.0))          # 前方 0.7m: まだ
    assert not tr.update((1.0, 0.25, math.pi / 2))  # 0.75m 離れて真横: まだ
    assert tr.update((1.6, 0.5, 0.0)) and tr.index == 1 and "passed" in tr.last_reason  # 0.78m, 左後ろ 140deg
    fin = topomap.GoalTracker(nodes[1:], goal_radius=0.4)
    assert not fin.update((3.5, 1.5, 0.0))          # 最終ゴールは通過扱いにしない
    # 向きの条件: 半径内でも向きが 45deg 以上ずれていれば (曲がり角) まだ到達にしない
    corner = [topomap.GoalNode(img, (1.0, 1.0, math.pi / 2)), topomap.GoalNode(img, (1.0, 2.0, math.pi / 2))]
    tr = topomap.GoalTracker(corner, subgoal_radius=0.6, lookahead_nodes=0)
    assert not tr.update((0.6, 0.8, 0.0))           # 0.45m だが 90deg ずれ
    assert not tr.update((0.7, 0.85, math.radians(30)))  # 60deg ずれ
    assert not tr.update((0.72, 0.88, math.radians(60)))  # 30deg ずれ (既定 25deg) -> まだ
    assert tr.update((0.75, 0.9, math.radians(70))) and "heading" in tr.last_reason  # 20deg -> 到達
    off = topomap.GoalTracker(corner, subgoal_radius=0.6, lookahead_nodes=0, reach_angle_deg=0)
    assert off.update((0.6, 0.8, 0.0))              # 0 で距離だけ (従来どおり)
    fin = topomap.GoalTracker(corner[1:], goal_radius=0.4)
    assert fin.update((1.0, 1.8, 0.0)) and fin.done  # 最終ゴールは向きを問わない


# ---------------------------------------------------------------- sim_map
SDF = """<?xml version="1.0"?>
<sdf version="1.9"><world name="test_world">
  <model name="ground"><static>true</static><link name="l"><collision name="c"><geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry></collision></link></model>
  <model name="wall"><static>true</static><pose>0 0 0 0 0 0</pose><link name="l">
    <collision name="c1"><pose>0 -3 1 0 0 0</pose><geometry><box><size>8 0.2 2</size></box></geometry></collision>
    <collision name="c2"><pose>0 3 1 0 0 0</pose><geometry><box><size>8 0.2 2</size></box></geometry></collision>
    <collision name="c3"><pose>-4 0 1 0 0 0</pose><geometry><box><size>0.2 6 2</size></box></geometry></collision>
    <collision name="c4"><pose>4 0 1 0 0 0</pose><geometry><box><size>0.2 6 2</size></box></geometry></collision>
    <collision name="mid"><pose>0 0.5 0.5 0 0 0</pose><geometry><box><size>0.4 5 1</size></box></geometry></collision>
  </link></model>
  <model name="rot"><static>true</static><pose>2 -1.5 0 0 0 0.785398</pose><link name="l"><collision name="c"><pose>0 0 0.3 0 0 0</pose><geometry><box><size>1 0.3 0.6</size></box></geometry></collision></link></model>
  <model name="pole"><static>true</static><pose>-2 1 0 0 0 0</pose><link name="l"><collision name="c"><pose>0 0 1 0 0 0</pose><geometry><cylinder><radius>0.2</radius><length>2</length></cylinder></geometry></collision></link></model>
  <model name="high"><static>true</static><pose>-2 -1.5 0 0 0 0</pose><link name="l"><collision name="c"><pose>0 0 2.5 0 0 0</pose><geometry><box><size>1 1 0.2</size></box></geometry></collision></link></model>
  <include><uri>model://omnivla_robot</uri><name>omnivla_robot</name><pose>-3 -2 0.02 0 0 1.5708</pose></include>
</world></sdf>"""


def test_sdf_map_and_planner():
    obs = sim_map.extract_obstacles(SDF)
    names = {o.name.split("/")[0] for o in obs}
    assert "high" not in names and "ground" not in names and {"wall", "rot", "pole"} <= names
    grid = sim_map.rasterize(obs, resolution=0.05)
    occ = lambda x, y: bool(grid.occupied[grid.world_to_cell(x, y)[1], grid.world_to_cell(x, y)[0]])  # noqa: E731
    assert occ(0, 0.5) and occ(-2, 1.0) and occ(2, -1.5) and not occ(-2, -1.5) and not occ(-3, -2)
    # 45 度回転した箱: 長軸方向は占有, 短軸方向の外側は空き
    assert occ(2 + 0.4 * math.cos(math.pi / 4), -1.5 + 0.4 * math.sin(math.pi / 4))
    assert not occ(2 - 0.4 * math.sin(math.pi / 4), -1.5 + 0.4 * math.cos(math.pi / 4))
    spawn = sim_map.robot_spawn_pose(SDF)
    assert spawn is not None and np.allclose(spawn, (-3, -2, 1.5708), atol=1e-4)
    assert sim_map.world_name(SDF) == "test_world"
    planner = sim_map.PathPlanner(grid, inflate=0.35)
    path = planner.plan((-3, 0), (3, 0), rng=np.random.default_rng(0), noise=0.5)
    assert path is not None and np.allclose(path[0], (-3, 0), atol=0.06) and np.allclose(path[-1], (3, 0), atol=0.06)
    d = grid.distance_map()
    ix, iy = grid.world_to_cell(path[:, 0], path[:, 1])
    assert d[iy, ix].min() > 0.3  # 膨張半径 (0.35) - 1 セル程度の余裕
    assert min(path[:, 1]) < -1.9  # 中央の壁 (y in [-2, 3]) の下側を回り込む
    assert planner.plan((-3, 0), (10, 10)) is None  # 囲いの外は到達不能


def test_generated_worlds_are_navigable():
    rng = np.random.default_rng(0)
    for name in ("office_0", "park_0"):
        path = os.path.join(WORLDS, f"{name}.sdf")
        if not os.path.exists(path):
            continue
        grid = sim_map.build_occupancy_from_sdf(path, resolution=0.1)
        planner = sim_map.PathPlanner(grid, inflate=0.4)
        spawn = sim_map.robot_spawn_pose(path)
        assert spawn is not None and planner.is_free(*spawn[:2]), name
        comp = planner.component(*spawn[:2])
        assert (planner.labels == comp).sum() * 0.01 > 60, name  # 60 m^2 以上走れる
        ok = 0
        for _ in range(5):
            goal = planner.sample_free(rng, component=comp)
            if planner.plan(spawn[:2], goal) is not None:
                ok += 1
        assert ok >= 4, name


def test_expert_follower_kinematic_sim():
    """エキスパート (計画 + pure pursuit) を運動学シミュレーションで走らせ、衝突せずゴールできるか."""
    path = os.path.join(WORLDS, "office_0.sdf")
    if not os.path.exists(path):
        return
    grid = sim_map.build_occupancy_from_sdf(path, resolution=0.05)
    planner = sim_map.PathPlanner(grid, inflate=0.4)
    spawn = sim_map.robot_spawn_pose(path)
    comp = planner.component(*spawn[:2])
    rng = np.random.default_rng(2)
    dt, rate = 0.05, 3.0
    successes = 0
    for ep in range(6):
        sx, sy = planner.sample_free(rng, comp, min_clearance=0.6)
        pose = (sx, sy, float(rng.uniform(-math.pi, math.pi)))
        for _ in range(30):
            gx, gy = planner.sample_free(rng, comp, min_clearance=0.55)
            if 3.0 <= math.hypot(gx - sx, gy - sy) <= 12.0:
                break
        route = planner.plan(pose[:2], (gx, gy), rng=rng, noise=0.6)
        assert route is not None
        f = expert.PathFollower(route, expert.FollowerConfig(speed=0.3))
        rec, t, last_rec = [], 0.0, -1e9
        for _ in range(int(240 / dt)):
            v, w, done = f.step(pose)
            if t - last_rec >= 1.0 / rate - 1e-9:
                rec.append(pose)
                last_rec = t
            if done:
                successes += 1
                break
            pose = expert.unicycle_step(pose, v, w, dt)
            t += dt
            assert grid.clearance(pose[0], pose[1]) > 0.25 * 0.6, f"collision in episode {ep}"
        rec = np.array(rec)
        step = data_utils.summarize_spacing(rec[:, :2])
        assert step is not None and 0.06 < step < 0.11, step  # 0.3m/s, 3Hz -> ~0.1m/frame (旋回時は小さい)
    assert successes == 6


# ---------------------------------------------------------------- io / topomap / viz
def test_trajectory_io_roundtrip():
    tmp = tempfile.mkdtemp()
    try:
        w = trajectory_io.TrajectoryWriter(tmp, "traj_a", resize=(64, 48), metadata={"world": "x"})
        for k in range(5):
            w.add(np.full((48, 64, 3), k * 40, np.uint8), 0.1 * k, 0.0, 0.0, stamp=k / 3)
        assert w.close(min_frames=3)
        w2 = trajectory_io.TrajectoryWriter(tmp, "traj_short")
        w2.add(Image.new("RGB", (10, 10)), 0, 0, 0)
        assert not w2.close(min_frames=3) and not os.path.exists(os.path.join(tmp, "traj_short"))
        found = trajectory_io.find_trajectories(tmp)
        assert len(found) == 1
        data = trajectory_io.load_trajectory(found[0])
        assert data["position"].shape == (5, 2) and math.isclose(data["position"][-1, 0], 0.4)
        assert Image.open(trajectory_io.image_path(found[0], 4)).size == (64, 48)
        assert trajectory_io.load_meta(found[0])["world"] == "x"
    finally:
        shutil.rmtree(tmp)


def test_topomap_and_tracker():
    tmp = tempfile.mkdtemp()
    try:
        w = topomap.TopomapWriter(os.path.join(tmp, "g"))
        for k in range(3):
            w.add(Image.new("RGB", (32, 32), (k * 80, 0, 0)), (float(k), 0.0, 0.0))
        nodes = topomap.load_goal_sequence(os.path.join(tmp, "g"))
        assert [n.pose[0] for n in nodes] == [0.0, 1.0, 2.0]
        tr = topomap.GoalTracker(nodes[1:], subgoal_radius=0.5, goal_radius=0.3, lookahead_nodes=1)
        assert not tr.update((0.0, 0.0, 0.0)) and tr.index == 0
        assert tr.update((0.7, 0.0, 0.0)) and tr.index == 1 and not tr.done
        tr.update((1.9, 0.0, 0.0))
        assert tr.done
        img = Image.new("RGB", (32, 32))
        img.save(os.path.join(tmp, "single.png"))
        assert len(topomap.load_goal_sequence(os.path.join(tmp, "single.png"))) == 1
    finally:
        shutil.rmtree(tmp)


def test_navlog_and_analyzer():
    import csv
    import json
    import subprocess

    tmp = tempfile.mkdtemp()
    try:
        meta = {"goal_source": "x", "world": "office_0", "modality": 6, "controller": "upstream",
                "final_goal_pose": (2.0, 1.0, 0.0), "nodes": [{"index": 0, "path": "a", "pose": (2.0, 1.0, 0.0)}]}
        lg = navlog.NavRunLogger(tmp, meta, [Image.new("RGB", (16, 16))])
        wps = np.zeros((8, 4))
        wps[:, 0] = np.linspace(0.1, 0.8, 8)
        wps[:, 1] = -np.linspace(0.0, 0.3, 8)   # 右へ曲がる予測
        wps[:, 2] = 1.0
        for k in range(5):
            lg.step(k + 1, k / 3, (0.1 * k, 0.0, 0.0), 0, 1, (2.0, 1.0, 0.0), (2.0 - 0.1 * k, 1.0, 0.0), 6,
                    "upstream", 0.2, -0.3, 0.25, wps, 1.5, None, "running",
                    Image.new("RGB", (64, 48)), np.zeros((48, 64, 3), np.uint8))
        lg.event(1.0, "test")
        s = lg.close("test")
        assert s["steps"] == 5 and abs(s["path_length_m"] - 0.4) < 1e-6 and not s["reached"]
        run = lg.dir
        rows = list(csv.DictReader(open(os.path.join(run, "steps.csv"))))
        assert len(rows) == 5 and float(rows[0]["wp7_x"]) == 0.8 and float(rows[0]["goal_bearing_deg"]) > 0
        assert len(os.listdir(os.path.join(run, "debug"))) == 5 and len(os.listdir(os.path.join(run, "raw"))) == 5
        out = subprocess.run([sys.executable, os.path.join(REPO, "tools", "plot_nav_log.py"), run],
                             capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        rep = open(os.path.join(run, "report.txt")).read()
        assert "(5/5)" in rep  # サブゴールは左 (+27deg) なのに予測は右 -> 全ステップ検出
        assert os.path.exists(os.path.join(run, "overview.png")) and os.path.exists(os.path.join(run, "timeline.png"))
        assert json.load(open(os.path.join(run, "meta.json")))["world"] == "office_0"
        # その場で左に曲がる予測 (位置はほぼ動かず向きだけ +60deg) を w≈0 で実行 -> 「実行できていない区間」に出る
        sys.path.insert(0, os.path.join(REPO, "tools"))
        import plot_nav_log
        rows = []
        for k in range(6):
            r = {"step": k + 1, "sim_time": k / 3, "subgoal": 0, "goal_bearing_deg": 90.0, "v": 0.05, "w": 0.01}
            for i in range(8):
                yaw = 60.0 * (i + 1) / 8
                r.update({f"wp{i}_x": 0.02 * (i + 1), f"wp{i}_y": 0.0, f"wp{i}_yaw_deg": yaw})
            r["pred_w"] = plot_nav_log.predicted_turn_rate(r)
            rows.append(r)
        assert rows[0]["pred_w"] > 0.3
        segs = plot_nav_log.under_turn_segments(rows)
        assert len(segs) == 1 and len(segs[0]) == 6
        for r in rows:
            r["w"] = r["pred_w"]                      # trajectory 制御なら予測どおり曲がる
        assert not plot_nav_log.under_turn_segments(rows)
    finally:
        shutil.rmtree(tmp)


def test_viz():
    wps = np.zeros((8, 4))
    wps[:, 0] = np.linspace(0.1, 0.8, 8)
    wps[:, 1] = np.linspace(0, 0.2, 8)
    img = viz.render_debug(np.zeros((480, 640, 3), np.uint8), Image.new("RGB", (100, 80)), wps,
                           goal_local=(3.0, 1.0), lines=["a", "b"], gt_waypoints=wps * 0.9)
    assert img.size == (640, 480)



# ---------------------------------------------------------------- route book (要所の指示)
def test_routebook_format_and_modalities():
    tmp = tempfile.mkdtemp()
    try:
        steps = [routebook.RouteStep(instruction="go straight", until={"image": "key_0.jpg", "after_m": 1.0}),
                 routebook.RouteStep(goal_image="key_0.jpg"),
                 routebook.RouteStep(move=(2.0, -1.0, math.radians(-90))),
                 routebook.RouteStep(instruction="turn left", move=(1.0, 1.0, 0.0), until={"time": 5}, timeout=30)]
        book = routebook.RouteBook(steps, start=(1.0, 2.0, 0.5), goal=(3.0, 4.0, 0.0), meta={"style": "x"})
        book.images["key_0.jpg"] = Image.new("RGB", (32, 24), (200, 10, 10))
        out = os.path.join(tmp, "r")
        book.save(out)
        assert routebook.is_route_book(out) and routebook.is_route_book(os.path.join(out, "route.yaml"))
        assert not routebook.is_route_book(tmp)
        loaded = routebook.load_route_book(out)
        assert [s.modality for s in loaded.steps] == ["language", "image", "pose", "language_pose"]
        assert loaded.start == (1.0, 2.0, 0.5) and loaded.goal == (3.0, 4.0, 0.0)
        assert math.isclose(math.degrees(loaded.steps[2].move[2]), -90.0, abs_tol=1e-6)
        assert loaded.steps[3].timeout == 30.0 and loaded.image("key_0.jpg").size == (32, 24)
        runner = routebook.RouteRunner(loaded)
        assert runner.until_of(loaded.steps[1]) == {"image": "key_0.jpg"}       # 景色だけ: その景色で終わり
        assert runner.until_of(loaded.steps[2]) == {"reached": 0.4}             # 相対位置だけ: 着いたら終わり
        # topomap はルート帳ではない
        tw = topomap.TopomapWriter(os.path.join(tmp, "topo"))
        tw.add(Image.new("RGB", (8, 8)), (0.0, 0.0, 0.0))
        assert not routebook.is_route_book(os.path.join(tmp, "topo"))
        for bad in ({"instruction": "a", "goal_image": "b.jpg"}, {"until": {"time": 1}}, {"move": [1]},
                    {"instruction": "a", "until": {"when": 1}}, {"instruction": "a", "speed": 1}):
            try:
                routebook.parse_step(bad)
            except ValueError:
                continue
            raise AssertionError(f"accepted invalid step {bad}")
    finally:
        shutil.rmtree(tmp)


def test_route_runner_conditions():
    img = Image.new("RGB", (8, 8))
    steps = [routebook.RouteStep(instruction="go straight", until={"distance": 1.0}),
             routebook.RouteStep(instruction="turn right", until={"turn_deg": -60}),
             routebook.RouteStep(instruction="go straight", until={"image": "k.jpg", "after_m": 0.5}),
             routebook.RouteStep(move=(1.0, 0.0, 0.0)),
             routebook.RouteStep(instruction="wait", until={"time": 2.0}),
             routebook.RouteStep(instruction="go", until={"pose": [5.0, 5.0, 90.0], "radius": 0.3})]
    book = routebook.RouteBook(steps)
    book.images["k.jpg"] = img
    r = routebook.RouteRunner(book)
    sims = {"value": 0.95}
    sim = lambda name, image: sims["value"]  # noqa: E731
    t, pose = 0.0, (0.0, 0.0, 0.0)
    assert r.update(t, pose) is None and r.index == 0
    assert r.update(1.0, (0.6, 0.0, 0.0)) is None
    ev = r.update(2.0, (1.05, 0.0, 0.0))
    assert ev and "distance" in ev and r.index == 1
    # 左に回っても右の条件は満たさない
    assert r.update(3.0, (1.05, 0.0, math.radians(70))) is None
    r.update(4.0, (1.05, 0.0, math.radians(0)))
    ev = r.update(5.0, (1.05, 0.0, math.radians(-65)))
    assert ev and "turned" in ev and r.index == 2
    # 景色が似ていても after_m 進むまでは判定しない
    assert r.update(6.0, (1.25, -0.0, math.radians(-65)), sim) is None and r.state.similarity == 0.95
    ev = r.update(7.0, (1.45, -0.6, math.radians(-65)), sim)
    assert ev and "image" in ev and r.index == 3
    inp = r.inputs((1.45, -0.6, math.radians(-65)))
    assert inp.modality == "pose" and math.isclose(inp.goal_pose[0], 1.0, abs_tol=1e-6)
    gx = 1.45 + math.cos(math.radians(-65))
    gy = -0.6 + math.sin(math.radians(-65))
    assert r.update(8.0, (gx - 0.5, gy, 0.0)) is None
    ev = r.update(9.0, (gx - 0.1, gy + 0.1, 0.0))
    assert ev and "reached" in ev and r.index == 4
    assert r.update(10.0, (gx, gy, 0.0)) is None
    ev = r.update(11.5, (gx, gy, 0.0))
    assert ev and "time" in ev and r.index == 5
    assert r.update(12.0, (5.1, 5.0, 0.0)) is None             # 位置は合っているが向きが違う
    ev = r.update(13.0, (5.1, 5.0, math.radians(80)))
    assert r.done and ev.startswith("route finished") and len(r.history) == 6
    assert r.update(14.0, (5.1, 5.0, 0.0)) is None
    # timeout で失敗
    r2 = routebook.RouteRunner(routebook.RouteBook([routebook.RouteStep(instruction="x", until={"distance": 5},
                                                                        timeout=3.0)]))
    r2.update(0.0, (0.0, 0.0, 0.0))
    assert r2.update(2.0, (0.1, 0.0, 0.0)) is None
    assert r2.update(3.5, (0.2, 0.0, 0.0)).startswith("route failed") and r2.failed and not r2.done
    # move はオドメトリが無いと入力を作れない
    r3 = routebook.RouteRunner(routebook.RouteBook([routebook.RouteStep(move=(1.0, 0.0, 0.0))]))
    r3.update(0.0, None)
    try:
        r3.inputs(None)
        raise AssertionError("move without odometry must fail")
    except ValueError:
        pass
    assert r3.lines()[0].startswith("route step 1/1")


def _l_path(east=4.0, north=3.0, ds=0.05):
    a = np.stack([np.arange(0.0, east, ds), np.zeros(int(round(east / ds)))], axis=1)
    b = np.stack([np.full(int(round(north / ds)) + 1, east), np.arange(0.0, north + 1e-9, ds)], axis=1)
    return np.concatenate([a, b])


def test_route_legs_on_simple_paths():
    path = _l_path()
    turns = routebook.find_turns(path)
    assert len(turns) == 1 and math.isclose(math.degrees(turns[0].angle), 90.0, abs_tol=3.0)
    legs = routebook.route_legs(path, start_yaw=0.0)
    assert [lg.kind for lg in legs] == ["straight", "turn", "straight"]
    assert math.isclose(legs[0].end[0], 3.5, abs_tol=0.1) and math.isclose(legs[2].end[1], 3.0, abs_tol=0.05)
    assert math.isclose(math.degrees(legs[2].end[2]), 90.0, abs_tol=3.0)
    # 逆向きに置いたら、まずその場で向きを合わせる
    legs = routebook.route_legs(path, start_yaw=math.pi)
    assert legs[0].kind == "turn" and legs[0].length == 0.0 and abs(math.degrees(legs[0].angle)) > 170
    # 左右に振るだけの経路は直進 1 本
    xs = np.arange(0.0, 8.0, 0.05)
    wiggle = np.stack([xs, 0.25 * np.sin(xs * 2.0)], axis=1)
    legs = routebook.route_legs(wiggle, start_yaw=0.0)
    assert [lg.kind for lg in legs] == ["straight"], [(lg.kind, lg.angle) for lg in legs]
    # 文言と判定の組み合わせ
    for style in routebook.STYLES:
        for switch in routebook.SWITCHES:
            plan = routebook.plan_route_book(path, (0.0, 0.0, 0.0), style=style, switch=switch)
            mods = [s.modality for s in plan.book.steps]
            assert len(mods) == 3, (style, switch, mods)
            assert set(mods) == {{"language": "language", "image": "image", "pose": "pose",
                                  "language_pose": "language_pose"}[style]}
            needs_images = style == "image" or (switch == "image" and style in ("language",))
            assert bool(plan.key_poses) == needs_images, (style, switch)
            for s in plan.book.steps:
                assert routebook.RouteRunner(plan.book).until_of(s), (style, switch, s)
            words = [s.instruction for s in plan.book.steps]
            if style.startswith("language"):
                assert words == ["go straight", "turn left", "go straight"]


def _fake_similarity(pose_fn, key_poses):
    def sim(name, image):
        x, y, yaw = pose_fn()
        kx, ky, kyaw = key_poses[name]
        d = math.hypot(x - kx, y - ky)
        return math.exp(-d * d / (2 * 0.5 ** 2)) * max(0.0, math.cos(geometry.wrap_angle(yaw - kyaw)))
    return sim


def _drive_route(plan, path, steps_limit=900, use_move_goal=True):
    """要所の切り替えだけを確かめる運動学シミュレーション (3Hz で判定・指令).

    指示どおり動く理想の VLA の代わりに: 相対位置 (move) があればそこへ直進, 無ければ計画経路を pure pursuit。
    """
    for name in plan.key_poses:  # 本来は要所で撮った画像 (ここでは類似度を位置から作るので中身は使わない)
        plan.book.images[name] = Image.new("RGB", (8, 8))
    runner = routebook.RouteRunner(plan.book)
    follower = expert.PathFollower(path, expert.FollowerConfig(speed=0.3, goal_tolerance=0.08))
    pose = tuple(plan.book.start)
    state = {"pose": pose}
    sim = _fake_similarity(lambda: state["pose"], plan.key_poses)
    t = 0.0
    for _ in range(steps_limit):
        runner.update(t, pose, sim)
        if runner.done or runner.failed:
            break
        inp = runner.inputs(pose)
        for _ in range(10):  # 1/3 秒
            if use_move_goal and inp.goal_pose is not None:
                gx, gy = geometry.to_world(np.array(inp.goal_pose[:2]), pose[:2], pose[2])
                loc = geometry.to_local(np.array([gx, gy]), pose[:2], pose[2])
                ang = math.atan2(loc[1], loc[0])
                v = 0.0 if abs(ang) > 0.6 else 0.3 * min(1.0, math.hypot(*loc) / 0.3)
                w = float(np.clip(1.5 * ang, -0.8, 0.8))
            else:
                v, w, done = follower.step(pose)
                if done:  # 言葉の指示ではゴールが分からないので、そのまま進み続ける
                    v, w = 0.3, 0.0
            pose = expert.unicycle_step(pose, v, w, 1.0 / 30.0)
            state["pose"] = pose
        t += 1.0 / 3.0
    return runner, pose


def test_routebook_closed_loop_switching():
    """経路から作ったルート帳を、理想の VLA で走らせて要所の切り替えが正しく進み、ゴールで終わるか."""
    setup = _office_setup()
    if setup is None:
        return
    grid, planner, comp = setup
    rng = np.random.default_rng(11)
    combos = [("language", "image"), ("language", "odom"), ("language", "pose"), ("image", "image"),
              ("pose", "odom"), ("language_pose", "pose")]
    spawn = sim_map.robot_spawn_pose(os.path.join(WORLDS, "office_0.sdf"))
    routes = [planner.plan(spawn[:2], (7.5, 4.0))]  # README のデモ (曲がり角が続く)
    while len(routes) < 4:  # 曲がり角のある経路
        route = _sample_task(planner, comp, rng, dmin=4.0, dmax=12.0)
        if routebook.find_turns(route):
            routes.append(route)
    for ep, route in enumerate(routes):
        start = (route[0][0], route[0][1], float(rng.uniform(-math.pi, math.pi)))
        assert any(lg.kind == "turn" and lg.length > 0 for lg in routebook.route_legs(route, start_yaw=start[2]))
        for style, switch in combos:
            plan = routebook.plan_route_book(route, start, style=style, switch=switch)
            runner, pose = _drive_route(plan, route)
            d = math.hypot(pose[0] - route[-1][0], pose[1] - route[-1][1])
            assert runner.done, (ep, style, switch, runner.index, runner.num_steps, runner.history[-1:])
            assert d < 0.7, (ep, style, switch, d)
            assert [h["index"] for h in runner.history] == list(range(runner.num_steps))


def test_routebook_detects_ignored_instruction():
    """曲がる指示を無視して止まる VLA なら、ステップの制限時間で失敗として止まる."""
    path = _l_path()
    plan = routebook.plan_route_book(path, (0.0, 0.0, 0.0), style="language", switch="odom", step_timeout=20.0)
    runner = routebook.RouteRunner(plan.book)
    pose, t = (0.0, 0.0, 0.0), 0.0
    while not (runner.done or runner.failed) and t < 200:
        runner.update(t, pose)
        if runner.current.instruction == "go straight" and pose[0] < 3.9:
            pose = (pose[0] + 0.1, pose[1], pose[2])
        t += 1.0 / 3.0
    assert runner.failed and runner.index == 1 and "timeout" in runner.last_reason


# ---------------------------------------------------------------- language probe (言葉の指示の評価)
def test_langprobe_objects_and_scoring():
    office = os.path.join(WORLDS, "office_0.sdf")
    if not os.path.exists(office):
        return
    objs = langprobe.scene_objects(office)
    phrases = {o.phrase for o in objs}
    assert {"blue sofa", "red sofa", "gray cabinet"} <= phrases, phrases
    sofa = [o for o in objs if o.phrase == "blue sofa"][0]
    c = sofa.center
    assert sofa.distance(c[0], c[1]) == 0.0 and sofa.distance(c[0] + 5.0, c[1]) > 3.0
    grid = sim_map.build_occupancy_from_sdf(office, resolution=0.05)
    planner = sim_map.PathPlanner(grid, inflate=0.4)
    comp = planner.component(*sim_map.robot_spawn_pose(office)[:2])
    rng = np.random.default_rng(3)
    found = 0
    for o in objs[:12]:
        st = langprobe.sample_object_start(planner, o, rng, comp, tries=200)
        if st is None:
            continue
        found += 1
        d = o.distance(st[0], st[1])
        assert 1.7 <= d <= 4.1, d
        oc = o.center
        bearing = geometry.wrap_angle(math.atan2(oc[1] - st[1], oc[0] - st[0]) - st[2])
        assert abs(math.degrees(bearing)) <= 21.0, math.degrees(bearing)
    assert found >= 6, found
    st = langprobe.sample_open_start(planner, rng, comp)
    assert st is not None and grid.clearance(st[0], st[1]) > 0.9
    # 判定
    left = [(0.0, 0.0, math.radians(a)) for a in range(0, 100, 10)]
    assert langprobe.score_probe("left", left)["success"] == 1
    assert langprobe.score_probe("right", left)["success"] == 0
    straight = [(0.1 * k, 0.0, 0.05) for k in range(10)]
    assert langprobe.score_probe("straight", straight)["success"] == 1
    r = langprobe.score_probe("straight", [(0.0, 0.0, 0.0)] * 5)
    assert r["success"] == 0 and r["moved"] == 0
    toward = [(c[0] - 3.0 + 0.25 * k, c[1], 0.0) for k in range(11)]
    rr = langprobe.score_probe("object", toward, [sofa])
    assert rr["success"] == 1 and rr["final_dist"] < rr["start_dist"]
    assert langprobe.parse_instructions("left:turn left; straight : go") == [("left", "turn left"),
                                                                          ("straight", "go")]
    rows = [dict(kind="left", instruction="turn left", **langprobe.score_probe("left", left)),
            dict(kind="object", instruction="move toward blue sofa", **rr)]
    s = langprobe.summarize(rows)
    assert s["by_kind"]["left"]["success_rate"] == 1.0 and "mean_final_dist" in s["by_kind"]["object"]

if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                import traceback

                traceback.print_exc()
                print(f"FAIL {name}: {e}")
    sys.exit(1 if failed else 0)
