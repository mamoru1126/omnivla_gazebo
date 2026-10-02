"""Gazebo Harmonic + ros_gz_bridge を起動する.

  ros2 launch omnivla_gazebo sim.launch.py world:=office_0 headless:=false rviz:=false

world: worlds/<world>.sdf (tools/generate_worlds.py で生成) または SDF の絶対パス
headless:=true で GUI なし (カメラはヘッドレスレンダリング, EGL)
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _gazebo(context, *args, **kwargs):
    pkg = get_package_share_directory("omnivla_gazebo")
    world = LaunchConfiguration("world").perform(context)
    path = world if world.endswith(".sdf") else os.path.join(pkg, "worlds", f"{world}.sdf")
    if not os.path.exists(path):
        raise FileNotFoundError(f"world not found: {path}")
    cmd = ["gz", "sim", "-r", "-v", LaunchConfiguration("verbosity").perform(context), path]
    if LaunchConfiguration("headless").perform(context).lower() in ("true", "1"):
        cmd += ["-s", "--headless-rendering"]
    return [ExecuteProcess(cmd=cmd, output="screen", shell=False)]


def generate_launch_description():
    pkg = get_package_share_directory("omnivla_gazebo")
    models = os.path.join(pkg, "models")
    resource_path = models + (":" + os.environ["GZ_SIM_RESOURCE_PATH"] if os.environ.get("GZ_SIM_RESOURCE_PATH")
                              else "")
    return LaunchDescription([
        DeclareLaunchArgument("world", default_value="office_0"),
        DeclareLaunchArgument("headless", default_value="false"),
        DeclareLaunchArgument("verbosity", default_value="2"),
        DeclareLaunchArgument("rviz", default_value="false"),
        SetEnvironmentVariable("GZ_SIM_RESOURCE_PATH", resource_path),
        OpaqueFunction(function=_gazebo),
        Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            name="gz_bridge",
            parameters=[{"config_file": os.path.join(pkg, "config", "bridge.yaml"), "use_sim_time": True}],
            output="screen",
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            arguments=["-d", os.path.join(pkg, "rviz", "omnivla.rviz")],
            parameters=[{"use_sim_time": True}],
            condition=IfCondition(LaunchConfiguration("rviz")),
            output="screen",
        ),
    ])
