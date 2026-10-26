"""シミュレーション評価 (navigate.launch.py を goal_path なしで起動しておくこと).

  ros2 launch omnivla_gazebo eval.launch.py world:=office_0 num_tasks:=20 mode:=single label:=zeroshot
  # 要所だけのルート帳 (style: language / image / pose / language_pose, switch: image / odom / pose)
  ros2 launch omnivla_gazebo eval.launch.py world:=office_0 num_tasks:=20 mode:=routebook style:=language switch:=image
  # 言葉の指示がどれくらい効くか (開けた場所 num_tasks か所 x 指示 + 物に向かう指示 object_probes 回)
  ros2 launch omnivla_gazebo eval.launch.py world:=office_0 mode:=language num_tasks:=5 object_probes:=10
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
        # routebook
        "style": ("language", str),
        "switch": ("image", str),
        "phrase_straight": ("go straight", str),
        "phrase_left": ("turn left", str),
        "phrase_right": ("turn right", str),
        "min_turn_deg": ("40.0", float),
        # language
        "instructions": ("left:turn left;left:go left;right:turn right;right:go right;"
                         "straight:go straight;straight:move forward", str),
        "object_probes": ("10", int),
        "object_template": ("move toward {target}", str),
        "probe_time": ("10.0", float),
    }
    decls = [DeclareLaunchArgument(k, default_value=v[0]) for k, v in args.items()]
    params = {k: ParameterValue(LaunchConfiguration(k), value_type=v[1]) for k, v in args.items()}
    params["use_sim_time"] = True
    return LaunchDescription(decls + [
        Node(package="omnivla_gazebo", executable="eval_runner", name="omnivla_eval_runner",
             parameters=[params], output="screen", emulate_tty=True),
    ])
