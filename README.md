# omnivla_gazebo

[OmniVLA](https://github.com/NHirose/OmniVLA)（ナビゲーション用 VLA）を Gazebo 上で動かすための Docker 環境です。
ゴールまでの経路を 1m ごとに撮影した**サブゴール画像**を用意し、OmniVLA が「現在のカメラ画像」と「今のサブゴール画像」から進路を出します。
サブゴールに着いたら次の画像へ切り替え、最終ゴールまで自律移動します。

📘 **図解付きの手順ガイド: https://mamoru1126.github.io/omnivla_gazebo/**

## 必要なもの

- Linux + NVIDIA GPU（VRAM 24GB 目安。推論 約16GB / 学習 約20GB）
- Docker + Docker Compose v2 + NVIDIA Container Toolkit
- ディスク 50GB 程度（イメージ + チェックポイント + 収集データ）

## 手順

### 1. セットアップ

```bash
git clone https://github.com/mamoru1126/omnivla_gazebo && cd omnivla_gazebo
cp .env.example .env
docker compose build
docker compose run --rm shell bash scripts/download_checkpoints.sh   # 公式の重み (約16GB)
```

### 2. 学習データを集める

ランダムな経路を自動走行して、画像と位置を記録します。

```bash
HEADLESS=true WORLD=office_0 docker compose up sim            # 端末1
WORLD=office_0 EPISODES=150 docker compose run --rm explore   # 端末2 → data/raw/office_0/
```

### 3. 学習する

```bash
docker compose stop sim
docker compose run --rm train    # → runs/<run>/checkpoints/step_005000/
```

### 4. サブゴール画像を作る

現在位置からゴール座標まで、1m ごとに画像を撮影します。

```bash
docker compose up sim                                          # 端末1
docker compose run --rm shell ros2 launch omnivla_gazebo topomap.launch.py \
    world:=office_0 out_dir:=/data/goals/demo goal_x:=7.5 goal_y:=4.0
```

### 5. 走らせる

```bash
# スタート位置に戻す
docker compose run --rm shell ros2 run omnivla_gazebo teleport --world office_0 --goal_dir /data/goals/demo
# 走行
FINETUNED_DIR=/runs/<run>/checkpoints/step_005000 GOAL_PATH=/data/goals/demo docker compose run --rm nav
```

走行中の様子は RViz で確認できます。手順 4 の `sim` を `RVIZ=true` を付けて起動してください。

```bash
RVIZ=true docker compose up sim
```

| 表示 | 中身 |
|---|---|
| Debug | 現在画像 + 予測軌跡、サブゴール画像、俯瞰図 (`/omnivla/debug_image`) |
| OmniVLA prediction | 予測軌跡（オレンジの線, `/omnivla/path`） |
| GroundTruthOdom | ロボットの実際の位置と軌跡 (`/odom`) |

RViz を使わずに画像だけ見る場合は `docker compose run --rm shell ros2 run rqt_image_view rqt_image_view /omnivla/debug_image`。

### 6. 成功率を測る（任意）

```bash
FINETUNED_DIR=/runs/<run>/checkpoints/step_005000 docker compose run --rm nav    # 端末2
docker compose run --rm shell ros2 launch omnivla_gazebo eval.launch.py \
    world:=office_0 num_tasks:=20 mode:=route label:=finetuned                    # 端末3
```

## 走行ログ

走行ごとに `log/nav/<日時>/` に保存されます。

```bash
python3 tools/plot_nav_log.py log/nav/latest    # 軌跡の図とレポートを作成
```

共有するときは [omnivla_gazebo_logs](https://github.com/mamoru1126/omnivla_gazebo_logs)（private）に push します。

## 主な設定

`ros2_ws/src/omnivla_gazebo/config/navigator.yaml`

| パラメータ | 既定 | 内容 |
|---|---|---|
| `subgoal_radius` | 0.3 | サブゴールに着いたとみなす距離 [m] |
| `reach_angle_deg` | 25 | サブゴールに着いたとみなす向きの差 [°] |
| `goal_radius` | 0.4 | 最終ゴールの到達距離 [m] |
| `controller` | `trajectory` | 予測軌跡を速度指令に変える方式 |

## フォルダ構成

| フォルダ | 中身 |
|---|---|
| `omnivla_nav/` | 推論・制御・到達判定などのコア（ROS 非依存） |
| `ros2_ws/` | ROS 2 ノード、launch、ワールド |
| `training/` | 学習スクリプト |
| `tools/` | ワールド生成、ログ解析 |
| `docs/` | 公式スクリプトから変えた点 |

## 補足

- 到達判定とサブゴール画像の撮影には、シミュレータの正確な位置を使っています。実機では自己位置推定に置き換えが必要です。
- 言語での指示は、学習していないので精度は期待できません。
