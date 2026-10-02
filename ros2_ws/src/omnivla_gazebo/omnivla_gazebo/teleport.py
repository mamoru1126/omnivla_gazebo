"""ロボットを瞬間移動させる CLI:  ros2 run omnivla_gazebo teleport --world office_0 --x 1 --y 2 --yaw 0"""
import sys

from . import ros_utils  # noqa: F401  (sys.path 設定)
from omnivla_nav.gz_utils import main as _main


def main():
    sys.exit(_main(sys.argv[1:]))


if __name__ == "__main__":
    main()
