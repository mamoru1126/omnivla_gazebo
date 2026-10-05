# syntax=docker/dockerfile:1
# =============================================================================
# OmniVLA x Gazebo 環境
#   Ubuntu 22.04 / Python 3.10 (OmniVLA 公式の想定) / ROS 2 Humble / Gazebo Harmonic (gz-sim 8)
#   PyTorch 2.2.0 (CUDA 12.1 同梱 wheel) / OmniVLA (コミット固定) + 依存一式
# GPU ドライバはホストのものを NVIDIA Container Toolkit 経由で使う (CUDA ベースイメージは不要)
# =============================================================================
ARG BASE_IMAGE=ros:humble-ros-base-jammy
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG OMNIVLA_REPO=https://github.com/NHirose/OmniVLA.git
ARG OMNIVLA_COMMIT=5182600cb4a9ee07684e17cdd2a6cbafc56b8a68

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TOKENIZERS_PARALLELISM=false \
    OMNIVLA_ROOT=/opt/OmniVLA \
    OMNIVLA_GAZEBO_ROOT=/workspace

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# ---- base tools -------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs curl wget gnupg lsb-release ca-certificates build-essential \
        python3-pip python3-dev python3-colcon-common-extensions \
        libgl1 libegl1 libglib2.0-0 libopenexr-dev zlib1g-dev ffmpeg \
        mesa-utils x11-apps less nano \
    && rm -rf /var/lib/apt/lists/*

# ---- Gazebo Harmonic + ros_gz (Humble 向けの非公式バイナリ, packages.osrfoundation.org) ----
RUN curl -fsSL https://packages.osrfoundation.org/gazebo.gpg -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
        > /etc/apt/sources.list.d/gazebo-stable.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        gz-harmonic \
        ros-humble-ros-gzharmonic \
        ros-humble-rviz2 \
        ros-humble-rqt-image-view \
        ros-humble-teleop-twist-keyboard \
        ros-humble-tf2-ros \
    && rm -rf /var/lib/apt/lists/*

# ---- PyTorch ----------------------------------------------------------------
COPY docker/constraints.txt /tmp/constraints.txt
RUN python3 -m pip install -c /tmp/constraints.txt torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0

# ---- OmniVLA (コミット固定) ---------------------------------------------------
# 公式 pyproject.toml は "av" の後ろのカンマ抜けで TOML として不正 (pip install -e . が失敗する) ため修正してから入れる
# Ubuntu 22.04 の pip 22 は pyproject のみのプロジェクトを editable で入れられないので通常インストール
# (使うのは prismatic パッケージだけ。/opt/OmniVLA のサンプル画像などはそのまま残る)
# 通常インストールでは __init__.py の無い rlds/oxe/utils が抜けて import が壊れるので追加する
RUN git init -q ${OMNIVLA_ROOT} && cd ${OMNIVLA_ROOT} \
    && git remote add origin ${OMNIVLA_REPO} \
    && git fetch -q --depth 1 origin ${OMNIVLA_COMMIT} && git checkout -q FETCH_HEAD \
    && sed -i -E 's/^([[:space:]]*)"av"[[:space:]]*$/\1"av",/' pyproject.toml \
    && python3 -c "from pip._vendor import tomli; tomli.loads(open('pyproject.toml').read())" \
    && touch prismatic/vla/datasets/rlds/oxe/utils/__init__.py \
    && python3 -m pip install -c /tmp/constraints.txt ${OMNIVLA_ROOT} \
    && python3 -m pip install -c /tmp/constraints.txt scipy pyyaml pytest

# ---- 本リポジトリ ------------------------------------------------------------
# 実行時は docker-compose で /workspace にマウントする (イメージ単体でも動くようにコピーもしておく)
COPY . /workspace
WORKDIR /workspace
RUN source /opt/ros/humble/setup.bash \
    && mkdir -p /opt/omnivla_ws \
    && colcon --log-base /opt/omnivla_ws/log build --symlink-install \
        --base-paths /workspace/ros2_ws/src \
        --build-base /opt/omnivla_ws/build --install-base /opt/omnivla_ws/install

# 動作確認 (import できるか)
RUN source /opt/ros/humble/setup.bash && cd /workspace \
    && PYTHONPATH=/workspace python3 -c "import torch, transformers, peft, prismatic, omnivla_nav.policy; \
print('torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__)" \
    && python3 tests/test_core.py

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
