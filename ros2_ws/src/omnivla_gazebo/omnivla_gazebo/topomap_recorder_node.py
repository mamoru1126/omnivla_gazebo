"""ゴール画像列 (topomap) を作る ROS 2 ノード.

mode:
  route    : (Gazebo 専用・おすすめ) 現在位置 (または start_x/start_y) から goal_x/goal_y までの経路を地図上で計画し、
             spacing [m] ごとにロボットをテレポートさせてその場の画像と真値姿勢を保存する。
             最後にロボットをスタート位置に戻すので、そのまま navigator を起動すればゴールを目指す。
  distance : 手動操縦 (teleop) 中、spacing [m] 進むごとに画像を保存 (ViNT/NoMaD の create_topomap と同様)
  manual   : 端末で Enter を押すたびに画像を保存 ('q' + Enter で終了)

出力: <out_dir>/0.jpg, 1.jpg, ..., poses.yaml  (navigator の goal_path にそのまま渡せる)
"""
from __future__ import annotations

import math
import sys
import threading
import time

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image as ImageMsg

from .ros_utils import Latest, image_msg_to_rgb, odom_to_pose2d, sleep_sim, stamp_to_sec

from omnivla_nav.gz_utils import set_model_pose  # noqa: E402
from omnivla_nav.sim_map import path_headings, resample_path  # noqa: E402
from omnivla_nav.topomap import TopomapWriter  # noqa: E402


class TopomapRecorderNode(Node):
    def __init__(self):
        super().__init__("omnivla_topomap_recorder")
        p = self.declare_parameter
        p("out_dir", "/data/goals/route")
        p("overwrite", False)
        p("mode", "route")             # route | distance | manual
        p("spacing", 1.0)              # [m]
        p("goal_x", float("nan"))
        p("goal_y", float("nan"))
        p("goal_yaw", float("nan"))    # nan: 経路の向き
        p("start_x", float("nan"))     # nan: 現在位置
        p("start_y", float("nan"))
        p("world", "office_0")
        p("world_sdf", "")
        p("robot_name", "omnivla_robot")
        p("robot_radius", 0.25)
        p("safety_margin", 0.15)
        p("settle_time", 0.8)
        p("image_topic", "/camera/image_raw")
        p("odom_topic", "/odom")
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.g = g
        self.writer = TopomapWriter(g("out_dir"), overwrite=bool(g("overwrite")))
        self.latest_image = Latest()
        self.latest_odom = Latest()
        self.create_subscription(ImageMsg, g("image_topic"), self._on_image, qos_profile_sensor_data)
        self.create_subscription(Odometry, g("odom_topic"), self._on_odom, 20)
        self.done = False
        threading.Thread(target=self._run, daemon=True).start()

    def _on_image(self, msg):
        self.latest_image.set(msg, stamp_to_sec(msg.header.stamp))

    def _on_odom(self, msg):
        self.latest_odom.set(odom_to_pose2d(msg), stamp_to_sec(msg.header.stamp))

    def _wait_inputs(self):
        while rclpy.ok() and (self.latest_image.get()[0] is None or self.latest_odom.get()[0] is None):
            self.get_logger().info("waiting for camera and /odom...", throttle_duration_sec=5.0)
            time.sleep(0.2)

    def _fresh_image(self, after: float):
        """シミュレーション時刻 after 以降に撮られた画像を待つ."""
        t0 = time.time()
        while rclpy.ok() and time.time() - t0 < 10.0:
            msg, t = self.latest_image.get()
            if msg is not None and t >= after:
                return image_msg_to_rgb(msg)
            time.sleep(0.02)
        raise RuntimeError("no fresh camera image")

    def _save(self, img=None):
        pose, _ = self.latest_odom.get()
        if img is None:
            img = image_msg_to_rgb(self.latest_image.get()[0])
        path = self.writer.add(img, pose)
        self.get_logger().info(f"saved node {len(self.writer.nodes) - 1}: {path} at "
                               f"({pose[0]:.2f}, {pose[1]:.2f}, {math.degrees(pose[2]):.0f}deg)")

    def _run(self):
        try:
            self._wait_inputs()
            mode = self.g("mode")
            if mode == "route":
                self._route()
            elif mode == "distance":
                self._distance()
            elif mode == "manual":
                self._manual()
            else:
                raise ValueError(f"unknown mode {mode}")
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(str(e))
        self.get_logger().info(f"topomap: {len(self.writer.nodes)} nodes in {self.writer.out_dir}")
        self.done = True

    def _route(self):
        from .sim_common import load_sim_map

        g = self.g
        if math.isnan(g("goal_x")) or math.isnan(g("goal_y")):
            raise ValueError("route mode requires goal_x and goal_y")
        sim = load_sim_map(g("world"), g("world_sdf"), 0.05, float(g("robot_radius")), float(g("safety_margin")),
                           g("robot_name"))
        pose, _ = self.latest_odom.get()
        start = (pose[0], pose[1]) if math.isnan(g("start_x")) else (g("start_x"), g("start_y"))
        path = sim.planner.plan(start, (g("goal_x"), g("goal_y")))
        if path is None:
            raise RuntimeError(f"no path from {start} to ({g('goal_x')}, {g('goal_y')})")
        length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
        n = max(1, int(round(length / float(g("spacing")))))
        sample = resample_path(path, length / n)  # n+1 点 (先頭はスタート)
        # 各ノードの向き = そこから先の経路の向き (最後のノードは到着方向)
        sample_heads = path_headings(sample)
        self.get_logger().info(f"route length {length:.1f} m -> {n} nodes")
        for i in range(1, len(sample)):
            yaw = float(sample_heads[i])
            if i == len(sample) - 1 and not math.isnan(g("goal_yaw")):
                yaw = float(g("goal_yaw"))
            self._teleport(sample[i][0], sample[i][1], yaw, sim.world)
            self._save(self._fresh_image(self.get_clock().now().nanoseconds * 1e-9))
        start_yaw = float(path_headings(path)[0])
        self._teleport(start[0], start[1], start_yaw, sim.world)
        self.writer.set_start((start[0], start[1], start_yaw))
        self.get_logger().info(f"robot returned to start ({start[0]:.2f}, {start[1]:.2f}, yaw {start_yaw:.2f})")

    def _teleport(self, x, y, yaw, world):
        ok, msg = set_model_pose(world, self.g("robot_name"), float(x), float(y), float(yaw))
        if not ok:
            raise RuntimeError(f"teleport failed: {msg}")
        sleep_sim(self, float(self.g("settle_time")))

    def _distance(self):
        spacing = float(self.g("spacing"))
        last = None
        self.get_logger().info(f"drive the robot (teleop). a node is saved every {spacing} m. Ctrl-C to finish")
        while rclpy.ok():
            pose, _ = self.latest_odom.get()
            if last is None or math.hypot(pose[0] - last[0], pose[1] - last[1]) >= spacing:
                self._save()
                last = pose
            time.sleep(0.05)

    def _manual(self):
        self.get_logger().info("press Enter to save a node, 'q'+Enter to quit")
        for line in sys.stdin:
            if line.strip().lower() == "q":
                break
            self._save()


def main(args=None):
    rclpy.init(args=args)
    node = TopomapRecorderNode()
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
