"""Gazebo 上で OmniVLA navigator の成功率を測る ROS 2 ノード (navigator を別に起動しておくこと).

各タスク:
  1. ロボットをゴール姿勢にテレポートしてゴール画像を撮る
     (mode=route のときは経路沿いに spacing ごとにサブゴール画像を撮って topomap を作る)
  2. スタート姿勢にテレポート
  3. /omnivla/goal_pose + /omnivla/goal_image (または /omnivla/goal_dir) を送り /omnivla/enable=true
  4. 真値オドメトリで監視: ゴール半径内到達 / タイムアウト / 衝突
結果は <out_dir>/results.csv と summary.json に保存。ゼロショットとファインチューニング後の比較に使う。

タスクはランダム生成 (seed 固定で再現可能) か tasks_file (YAML) で与える:
  tasks:
    - {start: [x, y, yaw], goal: [x, y, yaw]}
"""
from __future__ import annotations

import csv
import json
import math
import os
import threading
import time

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image as ImageMsg
from std_msgs.msg import Bool, String

from .ros_utils import Latest, image_msg_to_rgb, make_twist, odom_to_pose2d, sleep_sim, stamp_to_sec
from .sim_common import load_sim_map

from omnivla_nav.geometry import quaternion_from_yaw  # noqa: E402
from omnivla_nav.gz_utils import set_model_pose  # noqa: E402
from omnivla_nav.sim_map import path_headings, resample_path  # noqa: E402
from omnivla_nav.topomap import TopomapWriter  # noqa: E402


class EvalRunnerNode(Node):
    def __init__(self):
        super().__init__("omnivla_eval_runner")
        p = self.declare_parameter
        p("world", "office_0")
        p("world_sdf", "")
        p("robot_name", "omnivla_robot")
        p("robot_radius", 0.25)
        p("safety_margin", 0.15)
        p("tasks_file", "")
        p("num_tasks", 20)
        p("seed", 0)
        p("mode", "single")          # single: ゴール画像 1 枚 / route: サブゴール列
        p("min_dist", 2.0)
        p("max_dist", 5.0)
        p("route_spacing", 1.0)
        p("start_yaw_noise", 0.4)    # [rad] スタートの向きを経路方向からずらす量
        p("timeout", 90.0)           # [s] (シミュレーション時間)
        p("success_radius", 0.5)
        p("settle_time", 1.0)
        p("out_dir", "/runs/eval")
        p("label", "")
        p("image_topic", "/camera/image_raw")
        p("odom_topic", "/odom")
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.g = g
        self.map = load_sim_map(g("world"), g("world_sdf"), 0.05, float(g("robot_radius")), float(g("safety_margin")),
                                g("robot_name"))
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.out_dir = os.path.join(g("out_dir"), f"{g('label') + '_' if g('label') else ''}{self.map.world}_{stamp}")
        os.makedirs(self.out_dir, exist_ok=True)
        self.latest_image = Latest()
        self.latest_odom = Latest()
        self.nav_status = Latest()
        self.create_subscription(ImageMsg, g("image_topic"), lambda m: self.latest_image.set(
            m, stamp_to_sec(m.header.stamp)), qos_profile_sensor_data)
        self.create_subscription(Odometry, g("odom_topic"), lambda m: self.latest_odom.set(
            odom_to_pose2d(m), stamp_to_sec(m.header.stamp)), 20)
        self.create_subscription(String, "/omnivla/status", self._on_status, 10)
        self.goal_pose_pub = self.create_publisher(PoseStamped, "/omnivla/goal_pose", 2)
        self.goal_img_pub = self.create_publisher(ImageMsg, "/omnivla/goal_image", 2)
        self.goal_dir_pub = self.create_publisher(String, "/omnivla/goal_dir", 2)
        self.enable_pub = self.create_publisher(Bool, "/omnivla/enable", 10)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.done = False
        threading.Thread(target=self._run, daemon=True).start()

    def _on_status(self, msg: String):
        try:
            self.nav_status.set(json.loads(msg.data), time.time())
        except json.JSONDecodeError:
            pass

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # ------------------------------------------------------------------ tasks
    def _make_tasks(self):
        g = self.g
        if g("tasks_file"):
            with open(g("tasks_file")) as f:
                data = yaml.safe_load(f)
            return [(tuple(t["start"]), tuple(t["goal"])) for t in data["tasks"]]
        rng = np.random.default_rng(int(g("seed")))
        rr = float(g("robot_radius"))
        tasks = []
        tries = 0
        while len(tasks) < int(g("num_tasks")) and tries < 5000:
            tries += 1
            s = self.map.planner.sample_free(rng, self.map.main_component, min_clearance=rr + 0.35)
            goal = self.map.planner.sample_free(rng, self.map.main_component, min_clearance=rr + 0.35)
            path = self.map.planner.plan(s, goal)
            if path is None:
                continue
            length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
            if not (float(g("min_dist")) <= length <= float(g("max_dist"))):
                continue
            heads = path_headings(path)
            start_yaw = float(heads[0] + rng.uniform(-1, 1) * float(g("start_yaw_noise")))
            tasks.append(((s[0], s[1], start_yaw), (goal[0], goal[1], float(heads[-1]))))
        with open(os.path.join(self.out_dir, "tasks.yaml"), "w") as f:
            yaml.safe_dump({"tasks": [{"start": list(map(float, s)), "goal": list(map(float, gl))}
                                      for s, gl in tasks]}, f)
        return tasks

    def _teleport(self, x, y, yaw):
        self.cmd_pub.publish(make_twist(0.0, 0.0))
        ok, msg = set_model_pose(self.map.world, self.g("robot_name"), float(x), float(y), float(yaw))
        if not ok:
            raise RuntimeError(f"teleport failed: {msg}")
        sleep_sim(self, float(self.g("settle_time")))

    def _capture(self):
        after = self._now()
        t0 = time.time()
        while time.time() - t0 < 10.0:
            msg, t = self.latest_image.get()
            if msg is not None and t >= after:
                return msg
            time.sleep(0.02)
        raise RuntimeError("no camera image")

    # ------------------------------------------------------------------ run
    def _run(self):
        log = self.get_logger()
        while rclpy.ok() and (self.latest_image.get()[0] is None or self.latest_odom.get()[0] is None):
            log.info("waiting for camera and /odom...", throttle_duration_sec=5.0)
            time.sleep(0.2)
        while rclpy.ok() and self.nav_status.get()[0] is None:
            log.info("waiting for navigator (/omnivla/status)...", throttle_duration_sec=5.0)
            time.sleep(0.5)
        tasks = self._make_tasks()
        log.info(f"{len(tasks)} tasks -> {self.out_dir}")
        rows = []
        with open(os.path.join(self.out_dir, "results.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            header = ["task", "success", "stopped_in_radius", "final_dist", "min_dist", "time", "path_len",
                      "shortest", "spl", "collisions", "nav_reached", "reason"]
            writer.writerow(header)
            for i, (start, goal) in enumerate(tasks):
                if not rclpy.ok():
                    break
                try:
                    r = self._run_task(i, start, goal)
                except Exception as e:  # noqa: BLE001
                    log.error(f"task {i} failed: {e}")
                    r = dict(task=i, success=0, stopped_in_radius=0, final_dist=float("nan"),
                             min_dist=float("nan"), time=0, path_len=0, shortest=0, spl=0, collisions=0,
                             nav_reached=0, reason=f"error: {e}")
                rows.append(r)
                writer.writerow([r[k] for k in header])
                f.flush()
                log.info(f"task {i}: success={r['success']} final_dist={r['final_dist']:.2f} "
                         f"reason={r['reason']}")
        self.enable_pub.publish(Bool(data=False))
        if rows:
            summary = {
                "world": self.map.world, "mode": self.g("mode"), "num_tasks": len(rows),
                "success_rate": float(np.mean([r["success"] for r in rows])),
                "stop_success_rate": float(np.mean([r["stopped_in_radius"] for r in rows])),
                "spl": float(np.mean([r["spl"] for r in rows])),
                "mean_collisions": float(np.mean([r["collisions"] for r in rows])),
                "mean_final_dist": float(np.nanmean([r["final_dist"] for r in rows])),
            }
            with open(os.path.join(self.out_dir, "summary.json"), "w") as f:
                json.dump(summary, f, indent=2)
            log.info(f"summary: {summary}")
        self.done = True

    def _run_task(self, i, start, goal):
        g = self.g
        self.enable_pub.publish(Bool(data=False))
        shortest_path = self.map.planner.plan(start[:2], goal[:2])
        shortest = float(np.linalg.norm(np.diff(shortest_path, axis=0), axis=1).sum()) if shortest_path is not None \
            else float("nan")
        task_dir = os.path.join(self.out_dir, f"task_{i:03d}")
        # goal images
        if g("mode") == "route" and shortest_path is not None:
            writer = TopomapWriter(os.path.join(task_dir, "goals"), overwrite=True)
            n = max(1, int(round(shortest / float(g("route_spacing")))))
            sample = resample_path(shortest_path, shortest / n)
            heads = path_headings(sample)
            for k in range(1, len(sample)):
                yaw = goal[2] if k == len(sample) - 1 else float(heads[k])
                self._teleport(sample[k][0], sample[k][1], yaw)
                pose, _ = self.latest_odom.get()
                writer.add(image_msg_to_rgb(self._capture()), pose)
            goal_msg = None
        else:
            self._teleport(*goal)
            goal_msg = self._capture()
            os.makedirs(task_dir, exist_ok=True)
            from PIL import Image

            Image.fromarray(image_msg_to_rgb(goal_msg)).save(os.path.join(task_dir, "goal.jpg"))
        self._teleport(*start)
        # send goal
        ps = PoseStamped()
        ps.header.frame_id = "odom"
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x, ps.pose.position.y = float(goal[0]), float(goal[1])
        qx, qy, qz, qw = quaternion_from_yaw(float(goal[2]))
        ps.pose.orientation.x, ps.pose.orientation.y = qx, qy
        ps.pose.orientation.z, ps.pose.orientation.w = qz, qw
        if goal_msg is not None:
            self.goal_pose_pub.publish(ps)
            time.sleep(0.3)
            self.goal_img_pub.publish(goal_msg)
        else:
            self.goal_dir_pub.publish(String(data=os.path.join(task_dir, "goals")))
        time.sleep(0.5)
        self.enable_pub.publish(Bool(data=True))
        # monitor
        t0 = self._now()
        traveled, min_dist, collisions, in_collision = 0.0, float("inf"), 0, False
        last = None
        reason, nav_reached = "timeout", 0
        status_t0 = time.time()
        while rclpy.ok():
            now = self._now()
            pose, _ = self.latest_odom.get()
            d = math.hypot(pose[0] - goal[0], pose[1] - goal[1])
            min_dist = min(min_dist, d)
            if last is not None:
                traveled += math.hypot(pose[0] - last[0], pose[1] - last[1])
            last = pose
            hit = self.map.in_collision(pose[0], pose[1], factor=0.9)
            if hit and not in_collision:
                collisions += 1
            in_collision = hit
            st, st_t = self.nav_status.get()
            if st is not None and st_t > status_t0 + 0.6 and st.get("state") == "reached":
                reason, nav_reached = "nav_reached", 1
                break
            if now - t0 > float(g("timeout")):
                reason = "timeout"
                break
            time.sleep(0.05)
        self.enable_pub.publish(Bool(data=False))
        self.cmd_pub.publish(make_twist(0.0, 0.0))
        sleep_sim(self, 0.5)
        pose, _ = self.latest_odom.get()
        final = math.hypot(pose[0] - goal[0], pose[1] - goal[1])
        radius = float(g("success_radius"))
        success = int(min_dist < radius)
        stopped = int(final < radius)
        spl = success * shortest / max(shortest, traveled) if shortest == shortest and traveled > 0 else 0.0
        return dict(task=i, success=success, stopped_in_radius=stopped, final_dist=final, min_dist=min_dist,
                    time=self._now() - t0, path_len=traveled, shortest=shortest, spl=spl, collisions=collisions,
                    nav_reached=nav_reached, reason=reason)


def main(args=None):
    rclpy.init(args=args)
    node = EvalRunnerNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        while rclpy.ok() and not node.done:
            executor.spin_once(timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
