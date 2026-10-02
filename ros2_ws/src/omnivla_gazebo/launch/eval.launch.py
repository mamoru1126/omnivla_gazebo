"""シミュレーション評価 (navigate.launch.py を goal_path なしで起動しておくこと).

  ros2 launch omnivla_gazebo eval.launch.py world:=office_0 num_tasks:=20 mode:=single label:=zeroshot
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    args = {
        "world": ("office_0", str),
        "mode": ("single", str),
        "num_tasks": ("20", int),
        "seed": ("0", int),
        "min_dist": ("2.0", float),
        "max_dist": ("5.0", float),
        "route_spacing": ("1.0", float),
        "timeout": ("90.0", float),
        "success_radius": ("0.5", float),
        "tasks_file": ("", str),
        "out_dir": ("/runs/eval", str),
        "label": ("", str),
    }
    decls = [DeclareLaunchArgument(k, default_value=v[0]) for k, v in args.items()]
    params = {k: ParameterValue(LaunchConfiguration(k), value_type=v[1]) for k, v in args.items()}
    params["use_sim_time"] = True
    return LaunchDescription(decls + [
        Node(package="omnivla_gazebo", executable="eval_runner", name="omnivla_eval_runner",
             parameters=[params], output="screen", emulate_tty=True),
    ])
