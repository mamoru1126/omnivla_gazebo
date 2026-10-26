# omnivla_gazebo

[OmniVLA](https://github.com/NHirose/OmniVLA)（ナビゲーション用 VLA）を Gazebo 上で動かすための Docker 環境です。
ゴールまでの経路を 1m ごとに撮影した**サブゴール画像**を用意し、OmniVLA が「現在のカメラ画像」と「今のサブゴール画像」から進路を出します。
サブゴールに着いたら次の画像へ切り替え、最終ゴールまで自律移動します。

📘 **仕組みの解説 (図解): https://mamoru1126.github.io/omnivla_gazebo/**

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

各手順は「シミュレータ起動 → 作業 → シミュレータ停止」で完結します。
シミュレータは端末1で起動し、端末2で作業します。止めるときは端末1で `Ctrl+C`（または `docker compose stop sim`）。

### 2. 学習データを集める

ランダムな経路を自動走行して、画像と位置を記録します。
走行中にわざと経路から外し、そこから戻る場面も自動で記録します（`perturb: true`）。これが無いと曲がり角で少しずれただけで衝突します。

```bash
# 端末1: シミュレータ起動 (GUI なしで速く)
HEADLESS=true WORLD=office_0 docker compose up sim

# 端末2: データ収集 → data/raw/office_0/
WORLD=office_0 EPISODES=150 docker compose run --rm explore

# 端末1: 終わったらシミュレータ停止 (Ctrl+C)
```

### 3. 学習する

シミュレータは止めた状態で実行します（GPU を空けるため）。
曲がる場面と、経路から戻る場面を多めに学習します（`turn_sample_ratio`, `recovery_sample_ratio`）。

```bash
docker compose run --rm train    # → runs/<run>/checkpoints/step_005000/
```

### 4. サブゴール画像を作る

スタート位置からゴール座標まで、1m ごとに画像を撮影して `data/goals/demo/` に保存します。

```bash
# 端末1: シミュレータ起動
docker compose up sim

# 端末2: 撮影
docker compose run --rm shell ros2 launch omnivla_gazebo topomap.launch.py \
    world:=office_0 out_dir:=/data/goals/demo goal_x:=7.5 goal_y:=4.0

# 端末1: 終わったらシミュレータ停止 (Ctrl+C)
```

一度作れば、同じゴールへは何度でも走らせられます。

### 5. 走らせる

```bash
# 端末1: シミュレータ起動 (RViz で様子を見るなら RVIZ=true を付ける)
RVIZ=true docker compose up sim

# 端末2: ロボットをスタート位置に戻してから走行
docker compose run --rm shell ros2 run omnivla_gazebo teleport --world office_0 --goal_dir /data/goals/demo
FINETUNED_DIR=/runs/<run>/checkpoints/step_005000 GOAL_PATH=/data/goals/demo docker compose run --rm nav

# 端末1: 終わったらシミュレータ停止 (Ctrl+C)
```

もう一度走らせるときは、端末2の 2 行（teleport → nav）だけを繰り返します。

RViz の表示:

| 表示 | 中身 |
|---|---|
| Debug | 現在画像 + 予測軌跡、サブゴール画像、俯瞰図 (`/omnivla/debug_image`) |
| OmniVLA prediction | 予測軌跡（オレンジの線, `/omnivla/path`） |
| GroundTruthOdom | ロボットの実際の位置と軌跡 (`/odom`) |

RViz を使わずに画像だけ見る場合は `docker compose run --rm shell ros2 run rqt_image_view rqt_image_view /omnivla/debug_image`。

### 6. 成功率を測る（任意）

ランダムな経路でサブゴール撮影 → 走行 → 判定を自動で繰り返します。

```bash
# 端末1: シミュレータ起動
docker compose up sim

# 端末2: navigator をゴールなしで起動
FINETUNED_DIR=/runs/<run>/checkpoints/step_005000 docker compose run --rm nav

# 端末3: 評価 (20 本) → runs/eval/<label>_<world>_<日時>/summary.json
docker compose run --rm shell ros2 launch omnivla_gazebo eval.launch.py \
    world:=office_0 num_tasks:=20 mode:=route label:=finetuned

# 終わったら端末2・端末1を Ctrl+C
```

### 7. 言葉と要所の指示で走らせる（実験）

1m ごとのサブゴール画像（topomap）でコースを覚えさせる代わりに、人に道を教えるときのように「要所でだけ」指示を出して走らせます。
要所と要所の間は VLA に任せます（道なりに進む・人や障害物を避ける）。

#### 7-1. 言葉だけで走らせる（ゴールなし）

```bash
# 端末1: シミュレータ起動
docker compose up sim

# 端末2: 言葉の指示だけで走らせる (止めるまで走り続ける)
MODALITY=language INSTRUCTION="move toward blue sofa" docker compose run --rm nav

# 端末3: 走行中に指示を変える
docker compose run --rm shell ros2 topic pub --once /omnivla/instruction std_msgs/msg/String "{data: 'turn left'}"

# 止める
docker compose run --rm shell ros2 topic pub --once /omnivla/enable std_msgs/msg/Bool "{data: false}"
```

#### 7-2. 言葉がどれくらい効くか測る

開けた場所から「turn left」「go straight」などを、物の手前から「move toward red sofa」などを、1 つずつ 10 秒与えて、指示どおりに動いたかを数えます。

```bash
# 端末1: シミュレータ起動
docker compose up sim

# 端末2: navigator をゴールなしで起動 (FINETUNED_DIR なし = 公式の重み)
docker compose run --rm nav

# 端末3: 評価 → runs/eval/<label>_<world>_<日時>/summary.json
docker compose run --rm shell ros2 launch omnivla_gazebo eval.launch.py \
    world:=office_0 mode:=language num_tasks:=5 object_probes:=10 label:=original
```

| 指示の種類 | 成功の条件 |
|---|---|
| left / right | その向きに 45° 以上曲がった |
| straight | 向きの変化が 20° 以内で、0.6m 以上前に進んだ |
| object | 指示した物の 0.8m 以内まで近づいた |

`summary.json` の `by_kind`（種類ごと）と `by_instruction`（言い方ごと）に成功率が出ます。言い方は `instructions:="left:turn left;left:go to the left;..."` で変えられます。
Gazebo のデータでの追加学習は言葉を使っていないので、まず公式の重みで測り、`FINETUNED_DIR` を付けた場合と比べてください。

#### 7-3. ルート帳（要所ごとの指示）を作って走らせる

経路を計画して曲がり角などの「要所」だけを拾い、`route.yaml`（ルート帳）を作ります。
要所に着いたら、次の指示に自動で切り替わります。

```bash
# 端末1: シミュレータ起動
docker compose up sim

# 端末2: ルート帳を作る → data/routes/demo/ (route.yaml, 要所の画像 key_*.jpg, 経路と要所の図 route_map.png)
docker compose run --rm shell ros2 launch omnivla_gazebo topomap.launch.py mode:=routebook \
    style:=language switch:=image world:=office_0 out_dir:=/data/routes/demo goal_x:=7.5 goal_y:=4.0

# 端末2: スタート位置に戻してから走らせる
docker compose run --rm shell ros2 run omnivla_gazebo teleport --world office_0 --goal_dir /data/routes/demo
GOAL_PATH=/data/routes/demo docker compose run --rm nav
```

`route.yaml` は手で書き換えられます（指示の文言、次へ進む条件など。書き方は `omnivla_nav/routebook.py` の先頭）。例:

```yaml
steps:
  - instruction: turn right           # 言葉で指示
    until: {turn_deg: -75, image: key_0.jpg}
  - instruction: go straight
    until: {image: key_1.jpg, after_m: 6.2, distance: 16.6}   # 要所の景色が見えたら次へ
  - goal_image: key_2.jpg             # 景色 (画像) に向かう
```

| style（要所と要所の間の指示） | 中身 |
|---|---|
| `language` | 言葉（go straight / turn left / turn right） |
| `image` | 次の要所の景色（topomap を要所だけに間引いたもの） |
| `pose` | 次の要所の相対位置（オドメトリで測る） |
| `language_pose` | 言葉 + 相対位置 |

| switch（次の要所に着いたことの判定） | 中身 |
|---|---|
| `image` | 次の要所の景色が見えたら（予定の道のりの 6 割を進んでから判定） |
| `odom` | オドメトリだけ（予定の道のり・曲がった角度） |
| `pose` | Gazebo の真値の位置（切り替えが正しい場合の上限を見るため。実機では自己位置が必要） |

#### 7-4. 指示の出し方を比べる

`mode:=routebook` で、ランダムな経路ごとにルート帳を作って走らせ、成功率を測ります。
`summary.json` には成功率のほか、ルート帳を最後まで進めた割合（`route_done_rate`）と、平均で何ステップ目まで進んだか（`mean_route_progress`）が出ます。

```bash
# 端末2: navigator をゴールなしで起動しておく
docker compose run --rm nav

# 端末3: 言葉 + 景色で切り替え
docker compose run --rm shell ros2 launch omnivla_gazebo eval.launch.py \
    world:=office_0 mode:=routebook style:=language switch:=image num_tasks:=20 label:=lang_image

# 端末3: 切り替えを真値にした場合 (言葉の指示そのものがどこまで効くか)
docker compose run --rm shell ros2 launch omnivla_gazebo eval.launch.py \
    world:=office_0 mode:=routebook style:=language switch:=pose num_tasks:=20 label:=lang_oracle
```

同じ `seed` なら同じ経路になるので、`style` と `switch` だけを変えて比べられます。従来の topomap（`mode:=route`）とも比べてください。

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
| `route_image_threshold` | 0.9 | ルート帳: 要所の景色が見えたとみなす類似度 |
| `route_step_timeout` | 0 | ルート帳: 1 ステップの制限時間 [s]（0 = なし。超えたら失敗で停止） |

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
  ルート帳の `switch:=image` / `switch:=odom` は、走行中に正確な位置を使いません（要所の画像を撮るときだけ使います）。
- Gazebo のデータでの追加学習には言語の指示が入っていません（言葉の効き具合は 7-2 で測れます）。
