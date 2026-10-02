import os
from glob import glob

from setuptools import setup

package_name = "omnivla_gazebo"


def tree(directory):
    """directory 以下のファイルを share/<pkg>/<directory>/... にインストールする data_files を作る."""
    out = []
    for path, _, files in os.walk(directory):
        if files:
            out.append((os.path.join("share", package_name, path), [os.path.join(path, f) for f in files]))
    return out


setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "worlds"), glob("worlds/*.sdf")),
        (os.path.join("share", package_name, "rviz"), glob("rviz/*.rviz")),
    ] + tree("models") + tree("maps"),
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="mamoru1126",
    maintainer_email="sunottty1126@gmail.com",
    description="Run, evaluate and fine-tune OmniVLA in Gazebo Harmonic",
    license="MIT",
    entry_points={
        "console_scripts": [
            "navigator = omnivla_gazebo.navigator_node:main",
            "data_collector = omnivla_gazebo.data_collector_node:main",
            "auto_explorer = omnivla_gazebo.auto_explorer_node:main",
            "topomap_recorder = omnivla_gazebo.topomap_recorder_node:main",
            "eval_runner = omnivla_gazebo.eval_runner_node:main",
            "teleport = omnivla_gazebo.teleport:main",
        ],
    },
)
