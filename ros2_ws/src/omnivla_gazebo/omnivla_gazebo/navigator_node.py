"""OmniVLA で Gazebo のロボットを自律移動させる ROS 2 ノード.

ループ (既定 3Hz, 公式 inference と同じ):
  1. 最新のカメラ画像 (/camera/image_raw) と真値オドメトリ (/odom) を取得
  2. 現在のサブゴール (ゴール画像列 = topomap の i 番目) を決める
  3. OmniVLA(現在画像, サブゴール画像[, 相対ゴール姿勢 / 言語]) -> 8 点の waypoint
  4. waypoint -> (v, w) を /cmd_vel に出す
  5. サブゴールに到達したら次へ。最終ゴールに着いたら停止
を繰り返す。

ゴールの与え方:
  * パラメータ goal_path : 画像 1 枚、topomap ディレクトリ (0.jpg, 1.jpg, ..., poses.yaml)、
                           またはルート帳 (route.yaml を含むディレクトリ. omnivla_nav/routebook.py)
  * トピック /omnivla/goal_dir (std_msgs/String)       : topomap / ルート帳を実行中に切り替え
  * トピック /omnivla/goal_image (sensor_msgs/Image)   : 単一ゴール画像
    (+ 直前に /omnivla/goal_pose (PoseStamped, odom 座標) を送ると到達判定・pose modality に使う)
  * modality:=language + instruction (ゴール無し): 言葉だけで走り続ける。
    /omnivla/instruction (std_msgs/String) で走行中に指示を変えられる
  * /omnivla/enable (std_msgs/Bool) で開始/停止
ルート帳: 要所ごとの指示 (言葉 / 景色 / 相対位置) を順に実行し、終わり条件 (景色が見えた・道のり・
  曲がった角度・時間 …) を満たしたら次の指示へ自動で切り替える。modality はステップごとに決まる。
出力:
  /cmd_vel, /omnivla/path (予測軌跡, odom 座標), /omnivla/debug_image, /omnivla/status (JSON)
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
import traceback
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image as ImageMsg
from std_msgs.msg import Bool, String

from .ros_utils import Latest, image_msg_to_rgb, make_twist, odom_to_pose2d, rgb_to_image_msg, stamp_to_sec

from omnivla_nav.controller import ControllerConfig, StuckDetector, compute_command  # noqa: E402
from omnivla_nav.data_utils import IMAGE_MODALITIES, LANGUAGE_MODALITIES, POSE_MODALITIES, modality_id  # noqa: E402
from omnivla_nav.navlog import NavRunLogger  # noqa: E402
from omnivla_nav.geometry import quaternion_from_yaw, relative_pose, to_world, yaw_from_quaternion  # noqa: E402
from omnivla_nav.routebook import RouteBook, RouteRunner, RunnerConfig, is_route_book, load_route_book  # noqa: E402
from omnivla_nav.topomap import GoalNode, GoalTracker, load_goal_sequence  # noqa: E402
from omnivla_nav.viz import CameraModel, render_debug  # noqa: E402
from PIL import Image  # noqa: E402


class NavigatorNode(Node):
    def __init__(self):
        super().__init__("omnivla_navigator")
        self._param_names = []

        def p(name, default):
            self._param_names.append(name)
            return self.declare_parameter(name, default)
        # model
        p("vla_path", "/checkpoints/omnivla-original")
        p("checkpoint_step", -1)
        p("finetuned_dir", "")
        p("device", "cuda:0")
        p("metric_waypoint_spacing", 0.0)  # 0: auto (finetune_meta.json or 0.1)
        p("merge_lora", True)
        # task
        p("modality", "image")            # image | image_pose | pose | language | language_pose
        p("instruction", "")              # 例: "move toward blue trash bin"
        p("goal_path", "")
        p("reach_check", "auto")          # auto | pose | image | none
        p("subgoal_radius", 0.6)
        p("goal_radius", 0.4)
        p("image_reach_threshold", 0.92)
        p("lookahead_nodes", 1)
        p("pass_radius", 1.0)             # この距離以内でサブゴールが真横より後ろなら通過扱い
        p("pass_angle_deg", 90.0)
        p("reach_angle_deg", 25.0)        # 途中のサブゴールは向きの差もこれ以内で到達 (0 で距離だけ)
        p("stop_at_goal", True)
        p("autostart", True)
        # ルート帳 (route.yaml) の既定値 (ステップの until に書けばそちらが優先)
        p("route_image_threshold", 0.9)   # until.image: 景色の類似度のしきい値
        p("route_reach_radius", 0.4)      # until.reached: move のゴールに着いたとみなす距離 [m]
        p("route_pose_radius", 0.5)       # until.pose: 真値の位置の判定半径 [m]
        p("route_step_timeout", 0.0)      # 1 ステップの制限時間 [s] (0 = なし. 超えたら失敗で停止)
        # control
        p("control_rate", 3.0)
        p("controller", "trajectory")     # trajectory | upstream | pure_pursuit
        p("track_horizon", 4)
        p("track_max_v", 0.4)
        p("track_max_w", 1.0)
        p("waypoint_index", 4)
        p("dt", 1.0 / 3.0)
        p("max_v", 0.3)
        p("max_w", 0.3)
        p("lookahead", 0.5)
        p("max_image_age", 1.0)
        p("cmd_timeout", 1.5)
        p("respect_predicted_speed", True)  # 予測軌跡が短い (減速の予測) ときは速度を落とす
        p("stuck_timeout", 4.0)           # 前進指令中に この秒数 動かなければ停止 (0 で無効)
        # io
        p("image_topic", "/camera/image_raw")
        p("odom_topic", "/odom")
        p("cmd_vel_topic", "/cmd_vel")
        p("publish_debug_image", True)
        # 走行ログ: 1 走行ごとに <log_dir>/<時刻>/ を作る (空文字で無効)
        p("log_dir", "/workspace/log/nav")
        p("log_debug_images", True)
        p("log_raw_images", True)
        p("world", "")                    # ログの解析用 (地図の描画に使う)
        p("camera_hfov", 1.75)
        p("camera_height", 0.35)
        p("camera_x", 0.17)

        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.modality = modality_id(g("modality"))
        self.instruction = g("instruction") or None
        self.ctrl = ControllerConfig(mode=g("controller"), waypoint_index=int(g("waypoint_index")), dt=float(g("dt")),
                                     max_v=float(g("max_v")), max_w=float(g("max_w")),
                                     lookahead=float(g("lookahead")), pp_speed=float(g("max_v")),
                                     respect_predicted_speed=bool(g("respect_predicted_speed")),
                                     track_horizon=int(g("track_horizon")), track_max_v=float(g("track_max_v")),
                                     track_max_w=float(g("track_max_w")))
        self.stuck = StuckDetector(timeout=float(g("stuck_timeout")))
        self.cam = CameraModel(hfov=float(g("camera_hfov")), height=float(g("camera_height")),
                               x_offset=float(g("camera_x")))
        self.rate = float(g("control_rate"))
        self.max_image_age = float(g("max_image_age"))
        self.cmd_timeout = float(g("cmd_timeout"))
        self.stop_at_goal = bool(g("stop_at_goal"))
        self.publish_debug = bool(g("publish_debug_image"))
        self.log_dir = g("log_dir")
        self.log_debug = bool(g("log_debug_images"))
        self.log_raw = bool(g("log_raw_images"))
        self.runlog: Optional[NavRunLogger] = None
        self.params_snapshot = {n: g(n) for n in self._param_names}
        self._tracker_kwargs = dict(reach_check=g("reach_check"), subgoal_radius=float(g("subgoal_radius")),
                                    goal_radius=float(g("goal_radius")),
                                    image_threshold=float(g("image_reach_threshold")),
                                    lookahead_nodes=int(g("lookahead_nodes")),
                                    pass_radius=float(g("pass_radius")),
                                    pass_angle_deg=float(g("pass_angle_deg")),
                                    reach_angle_deg=float(g("reach_angle_deg")))
        self._route_cfg = RunnerConfig(image_threshold=float(g("route_image_threshold")),
                                       reach_radius=float(g("route_reach_radius")),
                                       pose_radius=float(g("route_pose_radius")),
                                       default_timeout=float(g("route_step_timeout")))

        self.cmd_pub = self.create_publisher(Twist, g("cmd_vel_topic"), 10)
        self.path_pub = self.create_publisher(Path, "/omnivla/path", 10)
        self.debug_pub = self.create_publisher(ImageMsg, "/omnivla/debug_image", 2)
        self.status_pub = self.create_publisher(String, "/omnivla/status", 10)

        self.latest_image = Latest()
        self.latest_odom = Latest()
        self.lock = threading.Lock()
        self.tracker: Optional[GoalTracker] = None
        self.route: Optional[RouteRunner] = None
        self.enabled = bool(g("autostart"))
        self.pending_goal_pose = None
        self.step_count = 0
        self.last_cmd_wall = time.time()
        self.moving = False
        self.state = "loading"
        self._stop = False

        self.create_subscription(ImageMsg, g("image_topic"), self._on_image, qos_profile_sensor_data)
        self.create_subscription(Odometry, g("odom_topic"), self._on_odom, 20)
        self.create_subscription(ImageMsg, "/omnivla/goal_image", self._on_goal_image, 2)
        self.create_subscription(PoseStamped, "/omnivla/goal_pose", self._on_goal_pose, 2)
        self.create_subscription(String, "/omnivla/goal_dir", self._on_goal_dir, 2)
        self.create_subscription(Bool, "/omnivla/enable", self._on_enable, 10)
        self.create_subscription(String, "/omnivla/instruction", self._on_instruction, 2)
        self.create_timer(0.2, self._watchdog)

        # --- model (重い: 数十秒) ---
        from omnivla_nav.policy import OmniVLAPolicy, PolicyConfig

        spacing = float(g("metric_waypoint_spacing"))
        self.policy = OmniVLAPolicy(PolicyConfig(
            vla_path=g("vla_path") or None,
            step=int(g("checkpoint_step")) if int(g("checkpoint_step")) >= 0 else None,
            finetuned_dir=g("finetuned_dir") or None,
            device=g("device"),
            metric_waypoint_spacing=spacing if spacing > 0 else None,
            merge_lora=bool(g("merge_lora")),
        ))
        self.model_info = {"vla_path": g("vla_path"), "finetuned_dir": g("finetuned_dir"),
                           "metric_waypoint_spacing": self.policy.metric_spacing}
        if g("goal_path"):
            self._load_goal(g("goal_path"))
        has_goal = self.tracker is not None or self.route is not None
        self.state = "idle" if not has_goal else ("running" if self.enabled else "paused")
        self.get_logger().info(f"OmniVLA navigator ready: modality={self.modality}, controller={self.ctrl.mode}")
        if self.ctrl.mode != "trajectory":
            self.get_logger().warn(
                f"controller={self.ctrl.mode}: 予測軌跡をそのまま実行しない制御則です"
                + (" (公式の式は予測した向きを使わず, その場で曲がる予測をしても曲がりません)" if self.ctrl.mode == "upstream" else "")
                + "。通常は controller:=trajectory (NAV_CONTROLLER=trajectory) を使ってください")
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    # ------------------------------------------------------------------ callbacks
    def _sim_now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_image(self, msg: ImageMsg):
        try:
            self.latest_image.set(image_msg_to_rgb(msg), stamp_to_sec(msg.header.stamp))
        except ValueError as e:
            self.get_logger().error(str(e), throttle_duration_sec=5.0)

    def _on_odom(self, msg: Odometry):
        self.latest_odom.set(odom_to_pose2d(msg), stamp_to_sec(msg.header.stamp))

    def _on_goal_pose(self, msg: PoseStamped):
        q = msg.pose.orientation
        self.pending_goal_pose = (msg.pose.position.x, msg.pose.position.y, yaw_from_quaternion(q.x, q.y, q.z, q.w))

    def _on_goal_image(self, msg: ImageMsg):
        img = Image.fromarray(image_msg_to_rgb(msg))
        node = GoalNode(img, self.pending_goal_pose, path=f"topic_goal_{time.time():.3f}")
        self.pending_goal_pose = None
        self._set_goals([node], source="/omnivla/goal_image")

    def _on_goal_dir(self, msg: String):
        try:
            self._load_goal(msg.data)
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"failed to load goal dir {msg.data}: {e}")

    def _on_enable(self, msg: Bool):
        self.enabled = bool(msg.data)
        self.get_logger().info(f"enabled={self.enabled}")
        self._event(f"enabled={self.enabled}")
        if self.enabled:
            self.stuck.reset()
            with self.lock:
                tracker, route = self.tracker, self.route
            if self.runlog is None and tracker is not None and not tracker.done:
                self._open_runlog(tracker.nodes, "resumed (/omnivla/enable)")
            if self.runlog is None and route is not None and not (route.done or route.failed):
                self._open_route_log(route.book, "resumed (/omnivla/enable)")
        if not self.enabled:
            self._publish_cmd(0.0, 0.0)

    def _on_instruction(self, msg: String):
        with self.lock:
            route = self.route
        if route is not None:
            self.get_logger().warn("a route book is running: /omnivla/instruction is ignored "
                                   "(the route book decides the instruction of each step)")
            return
        self.instruction = msg.data or None
        self.get_logger().info(f"instruction: {self.instruction}")
        self._event(f"instruction: {self.instruction}")

    def _load_goal(self, path: str):
        """topomap / 画像 1 枚 / ルート帳 のどれかを読んで走行を始める."""
        if is_route_book(path):
            self._set_route(load_route_book(path), source=path)
        else:
            self._set_goals(load_goal_sequence(path), source=path)

    def _set_goals(self, nodes, source: str):
        self._close_runlog("goal_changed")  # 前の走行のログは前のゴールの状態で閉じる
        with self.lock:
            self.tracker = GoalTracker(nodes, **self._tracker_kwargs)
            self.route = None
        self.stuck.reset()
        self._open_runlog(nodes, source)
        if self.modality in POSE_MODALITIES and any(n.pose is None for n in nodes):
            self.get_logger().warn("pose modality selected but some goal nodes have no pose")
        self.get_logger().info(f"new goal: {len(nodes)} node(s) from {source}")
        self.state = "running" if self.enabled else "paused"

    def _set_route(self, book: RouteBook, source: str):
        runner = RouteRunner(book, self._route_cfg)
        self._close_runlog("goal_changed")
        with self.lock:
            self.route = runner
            self.tracker = None
        self.stuck.reset()
        self._open_route_log(book, source)
        self.get_logger().info(f"new route book: {len(book.steps)} step(s) from {source}")
        for i, s in enumerate(book.steps):
            self.get_logger().info(f"  {i + 1}. [{s.modality}] {s.describe()}  until {runner.until_of(s) or '(none)'}")
        self.state = "running" if self.enabled else "paused"

    # ------------------------------------------------------------------ run log
    def _open_runlog(self, nodes, source: str):
        final = nodes[-1].pose if nodes and nodes[-1].pose is not None else None
        extra = {"nodes": [{"index": i, "path": n.path, "pose": n.pose} for i, n in enumerate(nodes)]}
        self._open_log([n.image for n in nodes], source, final, extra)

    def _open_route_log(self, book: RouteBook, source: str):
        names, images = [], []
        for s in book.steps:
            for name in s.image_names():
                if name not in names:
                    names.append(name)
                    images.append(book.image(name))
        extra = {"route": book.to_dict(), "route_images": names}
        self._open_log(images, source, book.goal, extra)

    def _open_log(self, images, source: str, final_goal, extra):
        self._close_runlog("goal_changed")
        if not self.log_dir:
            return
        pose, _ = self.latest_odom.get()
        meta = {
            "goal_source": source,
            "world": self.params_snapshot.get("world", ""),
            "modality": self.modality,
            "instruction": self.instruction,
            "controller": self.ctrl.mode,
            "model": getattr(self, "model_info", {}),
            "start_pose": pose,
            "final_goal_pose": final_goal,
            "params": self.params_snapshot,
        }
        meta.update(extra)
        try:
            self.runlog = NavRunLogger(self.log_dir, meta, images, self.log_debug, self.log_raw,
                                       self.ctrl.waypoint_index)
            self.get_logger().info(f"run log: {self.runlog.dir}")
        except OSError as e:
            self.runlog = None
            self.get_logger().error(f"cannot create run log in {self.log_dir}: {e}")

    def _close_runlog(self, reason: str, reached: bool = False):
        if self.runlog is not None:
            extra = {"subgoal_index": self.tracker.index if self.tracker else None}
            if self.route is not None:
                extra.update({"route_step": self.route.index, "route_steps": self.route.num_steps,
                              "route_done": self.route.done, "route_failed": self.route.failed,
                              "route_history": self.route.history})
            s = self.runlog.close(reason, reached, extra)
            self.get_logger().info(f"run log closed ({reason}): {self.runlog.dir} "
                                   f"final_dist={s.get('final_dist_to_goal')}")
            self.runlog = None

    def _event(self, text: str):
        if self.runlog is not None:
            self.runlog.event(self._sim_now(), text)

    def _watchdog(self):
        if self.moving and time.time() - self.last_cmd_wall > self.cmd_timeout:
            self.get_logger().warn("no new command in time (inference stalled?) -> stop", throttle_duration_sec=5.0)
            self._publish_cmd(0.0, 0.0)

    # ------------------------------------------------------------------ main loop
    def _publish_cmd(self, v: float, w: float):
        self.cmd_pub.publish(make_twist(v, w))
        self.moving = abs(v) > 1e-6 or abs(w) > 1e-6
        self.last_cmd_wall = time.time()

    def _publish_status(self, **extra):
        data = {"state": self.state, "enabled": self.enabled, "step": self.step_count}
        if self.tracker is not None:
            data.update({"subgoal": self.tracker.index, "num_nodes": len(self.tracker.nodes),
                         "done": self.tracker.done, "dist": self.tracker.last_distance,
                         "similarity": self.tracker.last_similarity})
        route = self.route
        if route is not None:
            cur = route.current
            data.update({"route_step": route.index, "route_steps": route.num_steps, "done": route.done,
                         "failed": route.failed, "instruction": cur.instruction, "step_modality": cur.modality,
                         "similarity": route.state.similarity, "reason": route.last_reason})
        elif self.modality in LANGUAGE_MODALITIES:
            data["instruction"] = self.instruction
        data.update(extra)
        self.status_pub.publish(String(data=json.dumps(data)))

    def _loop(self):
        period = 1.0 / max(self.rate, 1e-3)
        next_t = self._sim_now()
        while rclpy.ok() and not self._stop:
            now = self._sim_now()
            if now + 1.0 < next_t:  # シミュレーションのリセット等で時刻が戻った
                next_t = now
            if now < next_t:
                time.sleep(0.005)
                continue
            next_t = now + period
            try:
                self._step(now)
            except Exception as e:  # noqa: BLE001
                self.get_logger().error(f"step failed: {e}\n{traceback.format_exc()}")
                self._publish_cmd(0.0, 0.0)
                time.sleep(0.5)

    def _idle(self, state: str):
        """走らせない周期 (停止中・ゴール無し・到達後)."""
        if self.moving:
            self._publish_cmd(0.0, 0.0)
        self.state = state
        if self.step_count % 10 == 0:
            self._publish_status()
        self.step_count += 1

    def _observe(self, now: float):
        """推論に使う最新のカメラ画像と姿勢. 画像が無い / 古いときは (None, None)."""
        img, t_img = self.latest_image.get()
        if img is None:
            self.get_logger().warn("waiting for camera image...", throttle_duration_sec=5.0)
            return None, None
        if now - t_img > self.max_image_age:
            self.get_logger().warn(f"camera image is old ({now - t_img:.2f}s)", throttle_duration_sec=5.0)
            self._event(f"camera image is old ({now - t_img:.2f}s) -> stop")
            self._publish_cmd(0.0, 0.0)
            return None, None
        pose, _ = self.latest_odom.get()
        if self.runlog is not None:
            self.runlog.track_pose(now, pose)
        return img, pose

    def _step(self, now: float):
        with self.lock:
            tracker, route = self.tracker, self.route
        if route is not None:
            self._step_route(now, route)
            return
        if tracker is None and self.modality == 7 and self.instruction:
            # 言語のみ: ゴール画像/姿勢は不要。終了判定はしない (/omnivla/enable false で止める)
            with self.lock:
                if self.tracker is None:
                    dummy = Image.new("RGB", (224, 224))
                    self.tracker = GoalTracker([GoalNode(dummy, None, "language_only")], reach_check="none")
                    created = True
                else:
                    created = False
                tracker = self.tracker
            if created:
                self._open_runlog(tracker.nodes, f"language: {self.instruction}")
        if not self.enabled or tracker is None:
            self._idle("idle" if tracker is None else ("reached" if tracker.done else "paused"))
            return
        img, pose = self._observe(now)
        if img is None:
            return

        cache = {}

        def similarity(node: GoalNode) -> float:
            if "cur" not in cache:
                cache["cur"] = self.policy.embed(img)
            key = f"{node.path}:{id(node)}"
            return self.policy.similarity(cache["cur"], self.policy.embed(node.image, cache_key=key))

        if tracker.update(pose, similarity):
            msg = f"subgoal -> {tracker.index}/{len(tracker.nodes) - 1}" + \
                (" (final goal reached)" if tracker.done else "") + \
                (f" [{tracker.last_reason}]" if tracker.last_reason else "")
            self.get_logger().info(msg)
            self._event(msg + (f" at ({pose[0]:.2f}, {pose[1]:.2f}, {math.degrees(pose[2]):.0f}deg)"
                               if pose is not None else ""))
        if tracker.done:
            self._publish_cmd(0.0, 0.0)
            self.state = "reached"
            if self.stop_at_goal:
                self.enabled = False
            self._close_runlog("reached", reached=True)
            self._publish_status()
            return

        node = tracker.current
        goal_local = None
        if node.pose is not None and pose is not None:
            goal_local = relative_pose(pose, node.pose)
        if self.modality in POSE_MODALITIES and goal_local is None:
            self.get_logger().error("pose modality needs goal node pose and /odom", throttle_duration_sec=5.0)
            self._event("pose modality needs goal node pose and /odom -> stop")
            self._publish_cmd(0.0, 0.0)
            return
        out = self.policy.predict(img, goal_image=node.image if self.modality in IMAGE_MODALITIES else None,
                                  goal_pose=goal_local, instruction=self.instruction, modality=self.modality)
        info = [f"subgoal {tracker.index}/{len(tracker.nodes) - 1}"]
        if tracker.last_distance is not None:
            info.append(f"dist to subgoal {tracker.last_distance:.2f} m")
        if tracker.last_similarity is not None:
            info.append(f"similarity {tracker.last_similarity:.3f}")
        self._drive(now, img, pose, out, self.modality, node.image, goal_local, info,
                    tracker.index, len(tracker.nodes), node.pose, tracker.last_distance, tracker.last_similarity)

    def _step_route(self, now: float, route: RouteRunner):
        """ルート帳: 終わり条件を見てステップを進め、今のステップの指示でモデルを動かす."""
        if not self.enabled:
            self._idle("reached" if route.done else ("failed" if route.failed else "paused"))
            return
        if route.done or route.failed:
            self._idle("reached" if route.done else "failed")
            return
        img, pose = self._observe(now)
        if img is None:
            return
        cache = {}

        def similarity(name: str, image) -> float:
            if "cur" not in cache:
                cache["cur"] = self.policy.embed(img)
            return self.policy.similarity(cache["cur"],
                                          self.policy.embed(image, cache_key=f"route:{id(route.book)}:{name}"))

        ev = route.update(now, pose, similarity)
        if ev:
            self.get_logger().info(ev)
            self._event(ev + (f" at ({pose[0]:.2f}, {pose[1]:.2f}, {math.degrees(pose[2]):.0f}deg)"
                              if pose is not None else ""))
        if route.done or route.failed:
            self._publish_cmd(0.0, 0.0)
            self.state = "reached" if route.done else "failed"
            if route.failed or self.stop_at_goal:
                self.enabled = False
            self._close_runlog("reached" if route.done else f"failed: {route.last_reason}", reached=route.done)
            self._publish_status()
            return
        try:
            inp = route.inputs(pose)
        except ValueError as e:
            self.get_logger().error(f"route step {route.index + 1}: {e}", throttle_duration_sec=5.0)
            self._event(f"route step {route.index + 1}: {e} -> stop")
            self._publish_cmd(0.0, 0.0)
            return
        out = self.policy.predict(img, goal_image=inp.goal_image, goal_pose=inp.goal_pose,
                                  instruction=inp.instruction, modality=inp.modality)
        st = route.state
        self._drive(now, img, pose, out, modality_id(inp.modality), inp.key_image, inp.goal_pose, route.lines(),
                    route.index, route.num_steps, None, st.move_dist, st.similarity)

    def _drive(self, now: float, img, pose, out, modality: int, panel, goal_local, info, index: int, count: int,
               node_pose, dist, sim):
        """予測軌跡 -> 速度指令. 押し付け検知、予測軌跡・デバッグ画像・走行ログ・状態の出力."""
        v, w = compute_command(out.waypoints, self.ctrl)
        if self.stuck.update(now, pose, v):
            # 前進指令を出し続けているのに動いていない = 障害物に押し付けている。止めて終了
            self._publish_cmd(0.0, 0.0)
            self.enabled = False
            self.state = "stuck"
            msg = (f"stuck: no movement for {self.stuck.timeout:.0f}s while commanding v={v:.2f} "
                   f"at ({pose[0]:.2f}, {pose[1]:.2f}) -> stopped")
            self.get_logger().warn(msg)
            self._event(msg)
            self._close_runlog("stuck")
            self._publish_status()
            return
        self._publish_cmd(v, w)
        self.state = "running"
        self.step_count += 1

        stamp = self.get_clock().now().to_msg()
        if pose is not None:
            self._publish_path(out.waypoints, pose, stamp)
        dbg = None
        if self.publish_debug or (self.runlog is not None and self.log_debug):
            lines = [f"step {self.step_count}  modality {modality}"] + list(info) + \
                [f"v={v:.2f} m/s  w={w:.2f} rad/s", f"latency {out.latency * 1000:.0f} ms"]
            dbg = render_debug(img, panel, out.waypoints, self.cam,
                               goal_local[:2] if goal_local is not None else None, lines)
            if self.publish_debug:
                self.debug_pub.publish(rgb_to_image_msg(np.asarray(dbg), stamp))
        if self.runlog is not None:
            self.runlog.step(self.step_count, now, pose, index, count, node_pose, goal_local,
                             modality, self.ctrl.mode, v, w, out.latency, out.waypoints,
                             dist, sim, self.state, dbg, img)
        self._publish_status(v=v, w=w, latency=out.latency)

    def _publish_path(self, waypoints: np.ndarray, pose, stamp):
        path = Path()
        path.header.stamp = stamp
        path.header.frame_id = "odom"
        pts = np.concatenate([[[0.0, 0.0]], waypoints[:, :2]], axis=0)
        world = to_world(pts, pose[:2], pose[2])
        headings = np.concatenate([[0.0], np.arctan2(waypoints[:, 3], waypoints[:, 2])]) + pose[2]
        for (x, y), yaw in zip(world, headings):
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x, ps.pose.position.y = float(x), float(y)
            qx, qy, qz, qw = quaternion_from_yaw(float(yaw))
            ps.pose.orientation.z, ps.pose.orientation.w = qz, qw
            path.poses.append(ps)
        self.path_pub.publish(path)

    def destroy_node(self):
        self._stop = True
        try:
            self._close_runlog("shutdown")
        except Exception:  # noqa: BLE001
            pass
        try:
            self._publish_cmd(0.0, 0.0)
        except Exception:  # noqa: BLE001
            pass
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = NavigatorNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
