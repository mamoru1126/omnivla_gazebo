"""ファインチューニング用データを自動収集する ROS 2 ノード (エキスパート走行 + 記録).

1 エピソード:
  1. (任意) ロボットをランダムな空き位置・向きにテレポート
  2. ワールド SDF から作った占有格子上でランダムなゴールまでの経路を計画
     (コストにランダム場を混ぜて毎回少し違う経路にする)
  3. 真値オドメトリで pure pursuit 追従しながら、カメラ画像 + 真値姿勢を record_rate で記録
  4. ゴール到達で停止 (停止フレームも少し記録) -> 軌跡を保存。スタック/衝突なら破棄
を num_episodes 回繰り返す。出力は GNM 互換フォーマット (omnivla_nav/trajectory_io.py)。

記録された軌跡はゴール到達で終わる「目的地に向かう走行」なので、
学習時に未来フレームをゴール画像としてサンプル (hindsight relabeling) すれば
OmniVLA の (現在画像, ゴール画像) -> 軌跡 の教師データになる。
"""
from __future__ import annotations

import math
import threading
import time
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image as ImageMsg

from .data_collector_node import OdomBuffer, next_due_time
from .ros_utils import Latest, image_msg_to_rgb, make_twist, odom_to_pose2d, sleep_sim, stamp_to_sec
from .sim_common import load_sim_map

from omnivla_nav.expert import FollowerConfig, PathFollower, PerturbConfig, Perturber  # noqa: E402
from omnivla_nav.gz_utils import set_model_pose  # noqa: E402
from omnivla_nav.trajectory_io import TrajectoryWriter, unique_name  # noqa: E402


class AutoExplorerNode(Node):
    def __init__(self):
        super().__init__("omnivla_auto_explorer")
        p = self.declare_parameter
        p("world", "office_0")
        p("world_sdf", "")
        p("robot_name", "omnivla_robot")
        p("map_resolution", 0.05)
        p("robot_radius", 0.25)
        p("safety_margin", 0.15)
        p("num_episodes", 50)             # 0 以下で無限
        p("seed", 0)
        p("teleport", True)               # 各エピソードの開始位置をランダムにする
        p("min_goal_dist", 3.0)
        p("max_goal_dist", 12.0)
        p("path_noise", 0.6)
        p("linear_speed", 0.3)            # [m/s] 0.3m/s x 3Hz = 0.1m/frame (= metric_waypoint_spacing)
        p("speed_jitter", 0.03)
        p("max_angular", 0.8)
        p("lookahead", 0.6)
        p("goal_tolerance", 0.25)
        p("stuck_timeout", 15.0)
        p("max_episode_time", 180.0)
        p("hold_time", 1.5)               # ゴール到着後に止まったまま記録する時間 [s]
        # DART: お手本の走行をわざと乱し、経路から外れた状態から戻る様子を記録する (外乱中のフレームはラベルに使わない)
        p("perturb", True)
        p("perturb_interval_min", 3.0)
        p("perturb_interval_max", 8.0)
        p("perturb_duration_min", 0.8)
        p("perturb_duration_max", 2.5)
        p("record", True)
        p("out_dir", "/data/raw/auto")
        p("record_rate", 3.0)
        p("min_frames", 15)
        p("jpeg_quality", 95)
        p("image_topic", "/camera/image_raw")
        p("odom_topic", "/odom")
        p("cmd_vel_topic", "/cmd_vel")
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.cfg = {k: g(k) for k in ["world", "robot_name", "num_episodes", "teleport", "min_goal_dist",
                                       "max_goal_dist", "path_noise", "linear_speed", "speed_jitter", "max_angular",
                                       "lookahead", "goal_tolerance", "stuck_timeout", "max_episode_time",
                                       "hold_time", "record", "out_dir", "record_rate", "min_frames",
                                       "jpeg_quality", "robot_radius"]}
        self.rng = np.random.default_rng(int(g("seed")))
        self.perturber = Perturber(PerturbConfig(
            enabled=bool(g("perturb")),
            interval=(float(g("perturb_interval_min")), float(g("perturb_interval_max"))),
            duration=(float(g("perturb_duration_min")), float(g("perturb_duration_max")))), self.rng)
        self.get_logger().info("building occupancy map from world SDF...")
        self.map = load_sim_map(g("world"), g("world_sdf"), float(g("map_resolution")), float(g("robot_radius")),
                                float(g("safety_margin")), g("robot_name"))
        self.get_logger().info(f"map ready: world={self.map.world}, grid={self.map.grid.shape}, "
                               f"component={self.map.main_component}")
        self.cmd_pub = self.create_publisher(Twist, g("cmd_vel_topic"), 10)
        self.latest_odom = Latest()
        self.odom_buf = OdomBuffer()
        self.latest_image = Latest()
        self.writer: Optional[TrajectoryWriter] = None
        self.writer_lock = threading.Lock()
        self.next_due = -1e9
        self.period = 1.0 / float(self.cfg["record_rate"])
        self.create_subscription(ImageMsg, g("image_topic"), self._on_image, qos_profile_sensor_data)
        self.create_subscription(Odometry, g("odom_topic"), self._on_odom, 50)
        self.stats = {"success": 0, "stuck": 0, "collision": 0, "timeout": 0, "plan_fail": 0, "frames": 0,
                      "perturbations": 0}
        self.done = False
        threading.Thread(target=self._run, daemon=True).start()

    # ------------------------------------------------------------------ io
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_odom(self, msg: Odometry):
        t = stamp_to_sec(msg.header.stamp)
        pose = odom_to_pose2d(msg)
        self.latest_odom.set(pose, t)
        tw = msg.twist.twist
        self.odom_buf.add(t, pose, float(np.hypot(tw.linear.x, tw.linear.y)), float(tw.angular.z))

    def _on_image(self, msg: ImageMsg):
        t = stamp_to_sec(msg.header.stamp)
        self.latest_image.set(True, t)
        with self.writer_lock:
            if self.writer is None or t + 1e-3 < self.next_due:
                return
            od = self.odom_buf.closest(t)
            if od is None:
                return
            pose = od[1]
            self.writer.add(image_msg_to_rgb(msg), pose[0], pose[1], pose[2], stamp=t,
                            perturbed=self.perturber.flagged(t))
            self.next_due = next_due_time(self.next_due, t, self.period)

    def _cmd(self, v: float, w: float):
        self.cmd_pub.publish(make_twist(v, w))

    # ------------------------------------------------------------------ episodes
    def _run(self):
        log = self.get_logger()
        while rclpy.ok() and (self.latest_odom.get()[0] is None or self.latest_image.get()[0] is None):
            log.info("waiting for /odom and camera...", throttle_duration_sec=5.0)
            time.sleep(0.5)
        ep = 0
        n_eps = int(self.cfg["num_episodes"])
        while rclpy.ok() and (n_eps <= 0 or ep < n_eps):
            try:
                result = self._episode(ep)
            except Exception as e:  # noqa: BLE001
                log.error(f"episode failed: {e}")
                result = "error"
                self._stop_recording(keep=False)
            self.stats[result] = self.stats.get(result, 0) + 1
            ep += 1
            log.info(f"episode {ep}/{n_eps if n_eps > 0 else 'inf'}: {result}  stats={self.stats}")
        self._cmd(0.0, 0.0)
        self.done = True
        log.info(f"finished. {self.stats}")

    def _teleport(self, x, y, yaw) -> bool:
        self._cmd(0.0, 0.0)
        ok, msg = set_model_pose(self.map.world, self.cfg["robot_name"], x, y, yaw)
        if not ok:
            self.get_logger().warn(f"teleport failed: {msg}")
        sleep_sim(self, 1.0)
        return ok

    def _episode(self, ep: int) -> str:
        rr = float(self.cfg["robot_radius"])
        if self.cfg["teleport"]:
            x, y, yaw = self.map.sample_pose(self.rng, min_clearance=rr + 0.35)
            self._teleport(x, y, yaw)
        pose, _ = self.latest_odom.get()
        path = None
        for _ in range(30):
            gx, gy = self.map.planner.sample_free(self.rng, component=self.map.main_component,
                                                  min_clearance=rr + 0.3)
            d = math.hypot(gx - pose[0], gy - pose[1])
            if not (self.cfg["min_goal_dist"] <= d <= self.cfg["max_goal_dist"]):
                continue
            path = self.map.planner.plan(pose[:2], (gx, gy), rng=self.rng, noise=float(self.cfg["path_noise"]))
            if path is not None and len(path) >= 4:
                break
            path = None
        if path is None:
            return "plan_fail"
        speed = float(self.cfg["linear_speed"]) + float(self.rng.uniform(-1, 1)) * float(self.cfg["speed_jitter"])
        if self.cfg["record"]:
            name = unique_name(f"auto_{self.map.world}_ep{ep:04d}")
            with self.writer_lock:
                self.writer = TrajectoryWriter(self.cfg["out_dir"], name, int(self.cfg["jpeg_quality"]),
                                               metadata={"source": "auto_explorer", "world": self.map.world,
                                                         "record_rate": float(self.cfg["record_rate"]),
                                                         "speed": speed, "goal": [float(path[-1][0]),
                                                                                  float(path[-1][1])]})
                self.next_due = -1e9
        result = self._follow(path, speed)
        self._cmd(0.0, 0.0)
        if result == "success":
            sleep_sim(self, float(self.cfg["hold_time"]))
        self._stop_recording(keep=(result == "success"))
        return result

    def _stop_recording(self, keep: bool):
        with self.writer_lock:
            w, self.writer = self.writer, None
        if w is None:
            return
        if keep:
            n = len(w)
            if w.close(min_frames=int(self.cfg["min_frames"])):
                self.stats["frames"] += n
        else:
            w.discard()

    def _follow(self, path: np.ndarray, speed: float) -> str:
        follower = PathFollower(path, FollowerConfig(speed=speed, lookahead=float(self.cfg["lookahead"]),
                                                     max_angular=float(self.cfg["max_angular"]),
                                                     goal_tolerance=float(self.cfg["goal_tolerance"])))
        t0 = self._now()
        best_rem, best_t = float("inf"), t0
        self.perturber.reset(t0)
        n_pert = self.perturber.count
        goal = path[-1]
        while rclpy.ok():
            now = self._now()
            pose, _ = self.latest_odom.get()
            v, w, reached = follower.step(pose)
            if reached and not self.perturber.active:
                self.stats["perturbations"] += self.perturber.count - n_pert
                return "success"
            # 外乱 (DART): 一定間隔でお手本の指令を乱す。外乱中のフレームは perturbed として記録
            v, w = self.perturber.step(now, self.map.clearance(pose[0], pose[1]),
                                       math.hypot(goal[0] - pose[0], goal[1] - pose[1]), (v, w))
            rem = follower.remaining(pose[:2])
            if rem < best_rem - 0.05:
                best_rem, best_t = rem, now
            if now - best_t > float(self.cfg["stuck_timeout"]):
                return "stuck"
            if now - t0 > float(self.cfg["max_episode_time"]):
                return "timeout"
            if self.map.in_collision(pose[0], pose[1], factor=0.6):
                return "collision"
            self._cmd(v, w)
            sleep_sim(self, 0.05)
        return "aborted"


def main(args=None):
    rclpy.init(args=args)
    node = AutoExplorerNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        while rclpy.ok() and not node.done:
            executor.spin_once(timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node._cmd(0.0, 0.0)
        node._stop_recording(keep=False)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
