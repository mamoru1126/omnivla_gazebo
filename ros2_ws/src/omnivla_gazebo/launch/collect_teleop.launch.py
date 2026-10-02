"""手動操縦の走行を記録する (teleop_twist_keyboard は別端末で起動).

  ros2 launch omnivla_gazebo collect_teleop.launch.py out_dir:=/data/raw/teleop world:=office_0
  ros2 run teleop_twist_keyboard teleop_twist_keyboard   # 速度は 0.3 m/s 前後が推奨
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    args = {
        "out_dir": ("/data/raw/teleop", str),
        "world": ("", str),
        "record_rate": ("3.0", float),
        "auto_segment": ("true", bool),
        "stop_timeout": ("2.0", float),
    }
    decls = [DeclareLaunchArgument(k, default_value=v[0]) for k, v in args.items()]
    params = {k: ParameterValue(LaunchConfiguration(k), value_type=v[1]) for k, v in args.items()}
    params["use_sim_time"] = True
    return LaunchDescription(decls + [
        Node(package="omnivla_gazebo", executable="data_collector", name="omnivla_data_collector",
             parameters=[params], output="screen", emulate_tty=True),
    ])
