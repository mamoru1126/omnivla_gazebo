"""ゴール画像列 (topomap) を作る.

  # 経路を自動計画して 1m ごとにサブゴール画像を撮る (Gazebo 専用)
  ros2 launch omnivla_gazebo topomap.launch.py world:=office_0 out_dir:=/data/goals/demo goal_x:=7.5 goal_y:=4.0
  # 手動操縦しながら 1m ごとに保存
  ros2 launch omnivla_gazebo topomap.launch.py mode:=distance out_dir:=/data/goals/teleop
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    nan = ".nan"  # YAML の NaN 表記
    args = {
        "mode": ("route", str),
        "world": ("office_0", str),
        "out_dir": ("/data/goals/route", str),
        "overwrite": ("false", bool),
        "spacing": ("1.0", float),
        "goal_x": (nan, float),
        "goal_y": (nan, float),
        "goal_yaw": (nan, float),
        "start_x": (nan, float),
        "start_y": (nan, float),
    }
    decls = [DeclareLaunchArgument(k, default_value=v[0]) for k, v in args.items()]
    params = {k: ParameterValue(LaunchConfiguration(k), value_type=v[1]) for k, v in args.items()}
    params["use_sim_time"] = True
    return LaunchDescription(decls + [
        Node(package="omnivla_gazebo", executable="topomap_recorder", name="omnivla_topomap_recorder",
             parameters=[params], output="screen", emulate_tty=True),
    ])
