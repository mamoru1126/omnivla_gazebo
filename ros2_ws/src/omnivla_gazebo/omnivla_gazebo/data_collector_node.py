"""手動操縦 (teleop) の走行を GNM 互換フォーマットで記録する ROS 2 ノード.

  ros2 run teleop_twist_keyboard teleop_twist_keyboard   # 別端末で操縦
  ros2 run omnivla_gazebo data_collector --ros-args -p out_dir:=/data/raw/teleop_office

* 画像 (/camera/image_raw) を record_rate [Hz] に間引き、その時刻に最も近い真値オドメトリ (/odom) と組で保存
* auto_segment=true: 動き出したら新しい軌跡を開始し、stop_timeout 秒止まったら軌跡を閉じる
* auto_segment=false: サービス ~/start_episode, ~/end_episode (std_srvs/Trigger) で区切る
学習時の waypoint 間隔は metric_waypoint_spacing=0.1m を想定しているので、
0.3m/s 前後で走らせ record_rate=3Hz で記録すると公式推論と同じスケールになる。
"""
from __future__ import annotations

import collections
import os
from typing import Optional

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image as ImageMsg
from std_srvs.srv import Trigger

from .ros_utils import image_msg_to_rgb, odom_to_pose2d, stamp_to_sec

from omnivla_nav.trajectory_io import TrajectoryWriter, unique_name  # noqa: E402


def next_due_time(prev_due: float, t: float, period: float) -> float:
    """記録レートを平均で正確に保つための次回保存時刻 (カメラ周期のゆらぎで間隔が伸びないようにする)."""
    if t - prev_due < period:
        return prev_due + period
    return t + period


class OdomBuffer:
    def __init__(self, maxlen: int = 400):
        self.buf = collections.deque(maxlen=maxlen)

    def add(self, t: float, pose, speed: float, yaw_rate: float):
        self.buf.append((t, pose, speed, yaw_rate))

    def closest(self, t: float, max_dt: float = 0.1):
        if not self.buf:
            return None
        best = min(self.buf, key=lambda e: abs(e[0] - t))
        return best if abs(best[0] - t) <= max_dt else None


class DataCollectorNode(Node):
    def __init__(self):
        super().__init__("omnivla_data_collector")
        p = self.declare_parameter
        p("out_dir", "/data/raw/teleop")
        p("prefix", "teleop")
        p("world", "")
        p("record_rate", 3.0)
        p("auto_segment", True)
        p("min_speed", 0.03)
        p("min_yaw_rate", 0.1)
        p("stop_timeout", 2.0)
        p("min_frames", 15)
        p("min_length", 1.0)
        p("resize_width", 0)
        p("resize_height", 0)
        p("jpeg_quality", 95)
        p("image_topic", "/camera/image_raw")
        p("odom_topic", "/odom")
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.out_dir = os.path.abspath(g("out_dir"))
        os.makedirs(self.out_dir, exist_ok=True)
        self.prefix = g("prefix")
        self.world = g("world")
        self.period = 1.0 / float(g("record_rate"))
        self.auto = bool(g("auto_segment"))
        self.min_speed = float(g("min_speed"))
        self.min_yaw_rate = float(g("min_yaw_rate"))
        self.stop_timeout = float(g("stop_timeout"))
        self.min_frames = int(g("min_frames"))
        self.min_length = float(g("min_length"))
        self.resize = (int(g("resize_width")), int(g("resize_height")))
        self.jpeg_quality = int(g("jpeg_quality"))
        self.writer: Optional[TrajectoryWriter] = None
        self.next_due = -1e9
        self.last_moving = -1e9
        self.saved_count = 0
        self.odom = OdomBuffer()
        self.create_subscription(ImageMsg, g("image_topic"), self._on_image, qos_profile_sensor_data)
        self.create_subscription(Odometry, g("odom_topic"), self._on_odom, 50)
        self.create_service(Trigger, "~/start_episode", self._srv_start)
        self.create_service(Trigger, "~/end_episode", self._srv_end)
        self.get_logger().info(f"recording to {self.out_dir} (auto_segment={self.auto}, rate={1 / self.period:.1f}Hz)")

    def _on_odom(self, msg: Odometry):
        tw = msg.twist.twist
        self.odom.add(stamp_to_sec(msg.header.stamp), odom_to_pose2d(msg), float(np.hypot(tw.linear.x, tw.linear.y)),
                      float(tw.angular.z))

    def _start(self):
        if self.writer is not None:
            return
        name = unique_name(f"{self.prefix}_{self.world}" if self.world else self.prefix)
        self.writer = TrajectoryWriter(self.out_dir, name, self.jpeg_quality, self.resize,
                                       metadata={"source": "teleop", "world": self.world,
                                                 "record_rate": 1.0 / self.period})
        self.next_due = -1e9
        self.get_logger().info(f"start trajectory {name}")

    def _end(self):
        if self.writer is None:
            return
        n, length = len(self.writer), self.writer.path_length()
        ok = self.writer.close(min_frames=self.min_frames, min_length=self.min_length)
        if ok:
            self.saved_count += 1
            self.get_logger().info(f"saved trajectory {self.writer.name}: {n} frames, {length:.1f} m "
                                   f"(total {self.saved_count})")
        else:
            self.get_logger().info(f"discarded short trajectory ({n} frames, {length:.1f} m)")
        self.writer = None

    def _srv_start(self, req, res):
        self._start()
        res.success, res.message = True, "started"
        return res

    def _srv_end(self, req, res):
        self._end()
        res.success, res.message = True, "ended"
        return res

    def _on_image(self, msg: ImageMsg):
        t = stamp_to_sec(msg.header.stamp)
        od = self.odom.closest(t)
        if od is None:
            return
        _, pose, speed, yaw_rate = od
        moving = speed > self.min_speed or abs(yaw_rate) > self.min_yaw_rate
        if moving:
            self.last_moving = t
        if self.auto:
            if self.writer is None and moving:
                self._start()
            elif self.writer is not None and not moving and t - self.last_moving > self.stop_timeout:
                self._end()
                return
        if self.writer is None or t + 1e-3 < self.next_due:
            return
        self.writer.add(image_msg_to_rgb(msg), pose[0], pose[1], pose[2], stamp=t)
        self.next_due = next_due_time(self.next_due, t, self.period)

    def destroy_node(self):
        self._end()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DataCollectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
