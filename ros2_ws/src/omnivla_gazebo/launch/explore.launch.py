"""自動データ収集 (エキスパート走行 + 記録). sim.launch.py を同じ world で起動しておくこと.

  ros2 launch omnivla_gazebo explore.launch.py world:=office_0 num_episodes:=100 out_dir:=/data/raw/office_0
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    args = {
        "world": ("office_0", str),
        "out_dir": ("/data/raw/auto", str),
        "num_episodes": ("50", int),
        "seed": ("0", int),
        "record": ("true", bool),
        "teleport": ("true", bool),
        "linear_speed": ("0.3", float),
        "record_rate": ("3.0", float),
        "min_goal_dist": ("3.0", float),
        "max_goal_dist": ("12.0", float),
        "path_noise": ("0.6", float),
    }
    decls = [DeclareLaunchArgument(k, default_value=v[0]) for k, v in args.items()]
    params = {k: ParameterValue(LaunchConfiguration(k), value_type=v[1]) for k, v in args.items()}
    params["use_sim_time"] = True
    return LaunchDescription(decls + [
        Node(package="omnivla_gazebo", executable="auto_explorer", name="omnivla_auto_explorer",
             parameters=[params], output="screen", emulate_tty=True),
    ])
