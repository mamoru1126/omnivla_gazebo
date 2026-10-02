#!/bin/bash
# ROS 2 / 本リポジトリの環境を読み込んでからコマンドを実行する
set -e
source /opt/ros/humble/setup.bash

export OMNIVLA_GAZEBO_ROOT="${OMNIVLA_GAZEBO_ROOT:-/workspace}"
export PYTHONPATH="${OMNIVLA_GAZEBO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
WS=/opt/omnivla_ws

# マウントされたソースで ROS パッケージを (再) ビルド. symlink-install なので Python/launch/world の編集は即反映される
if [ "${OMNIVLA_SKIP_BUILD:-0}" != "1" ] && [ -d "${OMNIVLA_GAZEBO_ROOT}/ros2_ws/src" ]; then
  mkdir -p "${WS}"
  if ! colcon --log-base "${WS}/log" build --symlink-install \
        --base-paths "${OMNIVLA_GAZEBO_ROOT}/ros2_ws/src" \
        --build-base "${WS}/build" --install-base "${WS}/install" > "${WS}/build.log" 2>&1; then
    echo "[entrypoint] colcon build failed:" >&2
    tail -n 40 "${WS}/build.log" >&2
  fi
fi
if [ -f "${WS}/install/setup.bash" ]; then
  source "${WS}/install/setup.bash"
fi

exec "$@"
