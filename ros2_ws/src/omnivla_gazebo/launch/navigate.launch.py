"""OmniVLA navigator を起動する (sim.launch.py を別に起動しておくこと).

  ros2 launch omnivla_gazebo navigate.launch.py goal_path:=/data/goals/route_demo
  ros2 launch omnivla_gazebo navigate.launch.py finetuned_dir:=/runs/<run>/checkpoints/step_005000 goal_path:=...

その他のパラメータは config/navigator.yaml を編集するか params_file:=<yaml> で差し替える。
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg = get_package_share_directory("omnivla_gazebo")
    str_args = {
        "vla_path": "/checkpoints/omnivla-original",
        "finetuned_dir": "",
        "goal_path": "",
        "modality": "image",
        "instruction": "",
        "controller": "upstream",
        "reach_check": "auto",
        "log_dir": "/workspace/log/nav",
        "world": "",
    }
    decls = [DeclareLaunchArgument("params_file", default_value=os.path.join(pkg, "config", "navigator.yaml")),
             DeclareLaunchArgument("autostart", default_value="true")]
    decls += [DeclareLaunchArgument(k, default_value=v) for k, v in str_args.items()]
    overrides = {k: ParameterValue(LaunchConfiguration(k), value_type=str) for k in str_args}
    overrides["autostart"] = ParameterValue(LaunchConfiguration("autostart"), value_type=bool)
    overrides["use_sim_time"] = True
    return LaunchDescription(decls + [
        Node(
            package="omnivla_gazebo",
            executable="navigator",
            name="omnivla_navigator",
            parameters=[LaunchConfiguration("params_file"), overrides],
            output="screen",
            emulate_tty=True,
        ),
    ])
