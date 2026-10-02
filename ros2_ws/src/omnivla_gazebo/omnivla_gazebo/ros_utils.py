"""ROS メッセージと numpy の相互変換など (cv_bridge 非依存)."""
from __future__ import annotations

import os
import sys
import threading
import time
from typing import Optional, Tuple

import numpy as np
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image as ImageMsg

# omnivla_nav (リポジトリ直下のライブラリ) を import できるようにする
_REPO_CANDIDATES = [os.environ.get("OMNIVLA_GAZEBO_ROOT", ""), "/workspace"]
for _p in _REPO_CANDIDATES:
    if _p and os.path.isdir(os.path.join(_p, "omnivla_nav")) and _p not in sys.path:
        sys.path.insert(0, _p)

from omnivla_nav.geometry import yaw_from_quaternion  # noqa: E402


def image_msg_to_rgb(msg: ImageMsg) -> np.ndarray:
    enc = msg.encoding.lower()
    channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1, "8uc3": 3, "8uc1": 1}.get(enc)
    if channels is None:
        raise ValueError(f"unsupported image encoding: {msg.encoding}")
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    arr = buf.reshape(msg.height, msg.step)[:, : msg.width * channels].reshape(msg.height, msg.width, channels)
    if enc in ("bgr8", "bgra8"):
        arr = arr[..., [2, 1, 0]]
    elif enc == "rgba8":
        arr = arr[..., :3]
    elif channels == 1:
        arr = np.repeat(arr, 3, axis=2)
    return np.ascontiguousarray(arr)


def rgb_to_image_msg(arr: np.ndarray, stamp: Optional[TimeMsg] = None, frame_id: str = "") -> ImageMsg:
    arr = np.ascontiguousarray(arr.astype(np.uint8))
    msg = ImageMsg()
    if stamp is not None:
        msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height, msg.width = int(arr.shape[0]), int(arr.shape[1])
    msg.encoding = "rgb8"
    msg.is_bigendian = 0
    msg.step = int(arr.shape[1] * 3)
    msg.data = arr.tobytes()
    return msg


def odom_to_pose2d(msg: Odometry) -> Tuple[float, float, float]:
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    return float(p.x), float(p.y), yaw_from_quaternion(q.x, q.y, q.z, q.w)


def odom_speed(msg: Odometry) -> float:
    v = msg.twist.twist.linear
    return float(np.hypot(v.x, v.y))


def make_twist(v: float, w: float) -> Twist:
    t = Twist()
    t.linear.x = float(v)
    t.angular.z = float(w)
    return t


def stamp_to_sec(stamp: TimeMsg) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class Latest:
    """スレッドセーフに最新値を保持する小さなコンテナ."""

    def __init__(self):
        self._lock = threading.Lock()
        self._value = None
        self._t = 0.0

    def set(self, value, t: float):
        with self._lock:
            self._value, self._t = value, t

    def get(self):
        with self._lock:
            return self._value, self._t


def sleep_sim(node, seconds: float, poll: float = 0.01, timeout_wall: Optional[float] = None) -> None:
    """シミュレーション時刻 (use_sim_time) で seconds 待つ. 別スレッドで spin していること."""
    clock = node.get_clock()
    start = clock.now().nanoseconds * 1e-9
    wall0 = time.time()
    limit = timeout_wall if timeout_wall is not None else max(10.0, seconds * 20.0)
    while clock.now().nanoseconds * 1e-9 - start < seconds:
        if time.time() - wall0 > limit:
            break
        time.sleep(poll)
