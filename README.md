# omnivla_gazebo

[OmniVLA](https://github.com/NHirose/OmniVLA) (ICRA 2026, 全方位モーダルなナビゲーション用 VLA) を
**Gazebo (Harmonic) 上で動かし、評価し、Gazebo のデータでファインチューニングする**ための Docker 環境です。

```
 ┌──────── Gazebo Harmonic ────────┐   /camera/image_raw    ┌──────── OmniVLA navigator (3Hz) ───────────┐
 │ 手続き生成ワールド (office/park)  │ ─────────────────────▶ │ 現在画像 + サブゴール画像 (+相対ゴール姿勢)  │
 │ 差動二輪ロボット + 前方カメラ     │   /odom (真値)          │   → OmniVLA 7B → 8 点の waypoint            │
 │                                  │ ◀───────────────────── │   → (v, w)  → サブゴール到達で次の画像へ     │
 └──────────────────────────────────┘   /cmd_vel             └─────────────────────────────────────────────┘
          │ 自動走行 (A* + pure pursuit) で記録 (GNM 形式)                         ▲
          ▼                                                                      │ LoRA アダプタ
   /data/raw/<world>/<軌跡>/{0.jpg,…, traj_data.pkl}  ──▶  training/finetune_omnivla.py
```

- **推論**: 現在のカメラ画像と「目的地の画像」から軌跡を生成して移動、を 3Hz で繰り返す。
  目的地が遠い場合はルート上のサブゴール画像列 (topomap) を順にたどって最終ゴールに到達する。
- **データ収集**: 占有格子地図上で経路計画したエキスパート走行を自動で繰り返し、画像と真値姿勢を記録。
- **ファインチューニング**: 公式と同じ LoRA 設定で、Gazebo の見え方に適応させる (24GB GPU で可)。
- **評価**: ランダムなスタート/ゴールで成功率・SPL を測り、ゼロショットと学習後を比較。
- 公式スクリプトの「入力画像が固定」問題などの確認結果は [docs/omnivla_input_review.md](docs/omnivla_input_review.md)。

## 動作環境

| 項目 | 要件 |
|---|---|
| OS | Linux (Ubuntu 22.04/24.04 で想定)。GUI を出すなら X11 |
| GPU | NVIDIA, VRAM 24GB 推奨 (RTX 3090/4090 等)。推論 ≈ 16GB、学習 (LoRA + gradient checkpointing, batch 2) ≈ 20GB 前後 |
| ソフト | Docker + Docker Compose v2 + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)、NVIDIA ドライバ 530 以上 (CUDA 12.1) |
| ディスク | イメージ ≈ 15GB + チェックポイント ≈ 16GB + 収集データ |
| ネットワーク | ビルド時: GitHub, PyPI, packages.ros.org, packages.osrfoundation.org / 初回のみ Hugging Face |

コンテナの中身: Ubuntu 22.04 / Python 3.10 (OmniVLA 公式の想定) / ROS 2 Humble / Gazebo Harmonic (`ros-humble-ros-gzharmonic`) /
PyTorch 2.2.0 / OmniVLA (コミット固定, `pyproject.toml` の不具合を修正してインストール)。

## クイックスタート

```bash
git clone https://github.com/mamoru1126/omnivla_gazebo && cd omnivla_gazebo
cp .env.example .env
docker compose build                                   # 初回は 20〜40 分程度

# 1) 公式チェックポイント (omnivla-original, 約 16GB) を ./checkpoints に取得
docker compose run --rm shell bash scripts/download_checkpoints.sh

# 2) GPU とモデルの動作確認 (公式サンプル画像で image/language/pose の推論 → ./runs/smoke_test/*.jpg)
docker compose run --rm shell python3 scripts/smoke_test_policy.py

# 3) Gazebo 起動 (GUI)。別の端末で以降を実行
xhost +local:root
docker compose up sim

# 4) ゴール画像列 (topomap) を作る: スポーン位置から (7.5, 4.0) まで経路計画し、1m ごとにテレポートして撮影
docker compose run --rm shell ros2 launch omnivla_gazebo topomap.launch.py world:=office_0 out_dir:=/data/goals/demo goal_x:=7.5 goal_y:=4.0

# 5) OmniVLA で自律移動 (サブゴール画像を順にたどる)
GOAL_PATH=/data/goals/demo docker compose run --rm nav
```

途中経過は `rqt_image_view /omnivla/debug_image` (現在画像に予測軌跡を投影, ゴール画像, 俯瞰図) か
`RVIZ=true docker compose up sim` の RViz で確認できます。`/omnivla/status` に JSON で状態が出ます。

> `goal_x`, `goal_y` は地図上の空き位置を指定してください。地図は `ros2_ws/src/omnivla_gazebo/maps/<world>.png`
> (黒=障害物, 赤=スポーン位置, 1px=0.05m の 2 倍表示) や `<world>.yaml` で確認できます。
> 座標を指定するランチ引数は `8.0` のように小数で書いてください。

### もう一度走らせる (再実行の手順)

手順 4 の topomap 作成は、撮影後にロボットを**スタート地点へ・経路の進行方向を向けて**戻します。
ただし次の場合はロボットがその姿勢にいないので、走らせる前にスタート姿勢へ戻してください。

- 1 回走らせた後 (ロボットはゴール付近にいる)
- Gazebo (`sim`) を再起動した後 (ワールドのスポーン姿勢に戻る。`office_0` はゴールと逆の −x 向き)

ゴールが真後ろにあると、OmniVLA はゴール画像と現在画像の対応が取れないので、後ろ向きのまま前進してしまいます。

```bash
# Gazebo は起動したまま (docker compose up sim)

# 1) ロボットを topomap 作成時のスタート姿勢に戻す (poses.yaml の start を読む)
docker compose run --rm shell ros2 run omnivla_gazebo teleport --world office_0 --goal_dir /data/goals/demo

# 2) 走らせる
GOAL_PATH=/data/goals/demo docker compose run --rm nav
```

- navigator は起動するとすぐ走り出します (`autostart: true`)。必ず 1) の後に起動してください。
  走行中の navigator を止めずに再スタートしたい場合は、1) の後に
  `ros2 topic pub --once /omnivla/goal_dir std_msgs/msg/String "{data: /data/goals/demo}"` と
  `ros2 topic pub --once /omnivla/enable std_msgs/msg/Bool "{data: true}"` を送ります。
- 任意の姿勢から始めたい場合: `ros2 run omnivla_gazebo teleport --world office_0 --x -1.98 --y 0.0 --yaw 0.0` (yaw はラジアン)。
- topomap を作り直す場合: スタート地点は「実行時のロボットの位置」になります。先に上の 1) でスタートへ戻すか
  `start_x:=… start_y:=…` を指定してください。同じ `out_dir` に上書きするなら `overwrite:=true`
  (無いと「ディレクトリが空ではない」で止まります)。
- 2026-10-05 より前に作った topomap には start が記録されていないので、作り直すか `--x --y --yaw` で戻してください。

### 走行ログ (自動で保存)

navigator は走行ごとに `log/nav/<日時>/` (リポジトリ直下) へログを保存します。何もしなくても毎回残ります。

| ファイル | 中身 |
|---|---|
| `steps.csv` | 1 推論 (≈0.33 秒) ごと: 真値の位置姿勢, サブゴール番号と相対位置, 予測 8 点, 指令 (v, w), レイテンシ |
| `events.log` | サブゴール切替・到達・停止などの出来事 |
| `summary.json` | 到達したか, 最終距離, 走行距離, 終了理由 |
| `meta.json` | 使ったモデル (finetuned_dir), modality, 制御則, 全パラメータ, ゴール列の姿勢 |
| `debug/*.jpg`, `raw/*.jpg` | 毎ステップのデバッグ画像とカメラ画像 / `goals/*.jpg` 使ったゴール画像 |

解析 (GPU 不要): `python3 tools/plot_nav_log.py log/nav/latest` で
`overview.png` (地図上の走行軌跡と予測軌跡), `timeline.png`, `report.txt`
(予測がサブゴールと逆を向いたステップ = モデル側の問題 / 指令が予測と逆のステップ・予測した旋回を実行できていない区間
= 制御側の問題) を作ります。

うまくいかなかった走行は、そのディレクトリだけ push してください (1 走行 10〜30MB 程度):
```bash
git add log/nav/<日時> && git commit -m "nav log" && git push
```
画像が不要なら `log_debug_images` / `log_raw_images` を false に、ログ自体を止めるなら `log_dir` を空にします (`config/navigator.yaml`)。

## 使い方の詳細

### ナビゲーション (`navigate.launch.py`)

| 引数 | 既定 | 説明 |
|---|---|---|
| `goal_path` | `""` | ゴール画像 1 枚、または topomap ディレクトリ (`0.jpg, 1.jpg, …, poses.yaml`) |
| `modality` | `image` | `image` (画像のみ, id 6) / `image_pose` (5) / `pose` (4) / `language` (7) / `language_pose` (8) |
| `instruction` | `""` | 言語指示。公式の学習データに合わせ `move toward blue trash bin` のような形 |
| `finetuned_dir` | `""` | ファインチューニング結果 (`/runs/<run>/checkpoints/step_XXXXXX`) |
| `vla_path` | `/checkpoints/omnivla-original` | ベースモデル (または `merge_lora.py` の出力) |
| `controller` | `trajectory` | `trajectory`: 予測軌跡の旋回・速度をそのまま実行 / `upstream`: 公式 run_omnivla.py の式 / `pure_pursuit` |
| `reach_check` | `auto` | サブゴール到達判定。`auto`=ノードに姿勢があれば真値距離、無ければ画像類似度 (DINOv2 特徴) |

その他 (到達半径, 速度上限, 推論周期など) は `ros2_ws/src/omnivla_gazebo/config/navigator.yaml`。
公式の制御則・到達判定から変えている点 (いずれも `navigator.yaml` で無効化できる):

| 項目 | 内容 | パラメータ |
|---|---|---|
| 制御則 | 公式の式 (`waypoints[4]` だけから v, w を計算) は角速度が 0.3rad/s 止まりで、鋭く曲がる予測ほど大回りする (学習データのお手本は最大 0.8rad/s で曲がる)。`trajectory` は予測 wp0..wp4 (各 (k+1)/3 秒後) の道のりと向きの変化を時間で最小二乗フィットして、予測した旋回の速さ・曲率・減速をそのまま実行する | `controller`, `track_max_v`, `track_max_w` |
| 予測速度の尊重 | 予測軌跡が短い (モデルが減速を予測) ときは速度を落とす。公式は 0.1m 先でも上限速度で進み、壁の手前でも減速しない | `respect_predicted_speed` |
| サブゴール到達の向き条件 | 途中のサブゴールは半径内に入るだけでなく、ロボットの向きとその画像を撮った向きの差が `reach_angle_deg` (25°) 以内で到達。曲がり角で向きがずれたまま次へ切り替わって角を内側に詰めるのを防ぐ (最終ゴールには使わない, 0 で無効) | `reach_angle_deg` |
| サブゴールの通過判定 | 半径 (`subgoal_radius`) に入らなくても、`pass_radius` 以内で真横より後ろに来たら通過扱い。サブゴールの周りを回り続けるのを防ぐ (最終ゴールには使わない) | `pass_radius`, `pass_angle_deg` |
| 動けない時の停止 | 前進指令中に `stuck_timeout` 秒動かなければ (障害物に押し付け) 停止して走行を終了 (`summary.json` の reason=stuck) | `stuck_timeout` |

実行中に `/omnivla/goal_dir` (String), `/omnivla/goal_image` (+ `/omnivla/goal_pose`), `/omnivla/enable` (Bool) で
ゴール変更・開始/停止ができます。

ループの中身 (`omnivla_gazebo/navigator_node.py`):
1. 最新のカメラ画像と真値オドメトリを取得
2. topomap の現在ノードに到達していれば次のノードへ (最終ノード到達で停止)
3. OmniVLA(現在画像, サブゴール画像[, 相対姿勢]) → 正規化 waypoint (8, 4) → ×0.1m でメートルに
4. 予測軌跡から (v, w) を計算して `/cmd_vel` へ (`trajectory`: wp0..wp4 を再現する一定の (v, w))

推論部分は ROS 非依存の `omnivla_nav/policy.py` にあり、実機や別のシミュレータからも同じように使えます:
```python
from omnivla_nav.policy import OmniVLAPolicy, PolicyConfig
policy = OmniVLAPolicy(PolicyConfig(vla_path="/checkpoints/omnivla-original"))
out = policy.predict(current_pil, goal_image=goal_pil, modality="image")
out.waypoints   # (8, 4) [x前[m], y左[m], cos(yaw), sin(yaw)]
```

### ゴール画像の作り方 (`topomap.launch.py`)

- `mode:=route` (Gazebo 専用): 経路を自動計画し `spacing` m ごとにテレポートして撮影 → スタートに戻す
- `mode:=distance`: `ros2 run teleop_twist_keyboard teleop_twist_keyboard` で操縦しながら一定距離ごとに保存
- `mode:=manual`: Enter を押したときに保存
- 画像 1 枚だけをゴールにする場合は、その画像ファイルを `goal_path` に指定 (同名 `.yaml` に `{x, y, yaw}` があれば到達判定に使用)

### データ収集

**自動 (推奨)**: エピソードごとにランダムな位置へテレポート → ランダムなゴールまで経路計画 (毎回少し揺らぐ) →
pure pursuit で 0.3m/s 走行しながら 3Hz で記録 → ゴールで停止 (停止フレームも記録)。
```bash
WORLD=office_0 docker compose up sim                         # 端末 1 (HEADLESS=true で GUI なし・高速)
WORLD=office_0 EPISODES=100 docker compose run --rm explore  # 端末 2 → /data/raw/office_0/
```
**手動**: `ros2 launch omnivla_gazebo collect_teleop.launch.py out_dir:=/data/raw/teleop` + teleop_twist_keyboard。

**外乱つき収集 (DART, 既定で有効)**: 走行中 3〜8 秒ごとに 0.8〜2.5 秒だけお手本の指令をランダムな旋回に置き換えて
ロボットを経路から外し (最大 0.7m 程度, 障害物の近くとゴール手前では行わない)、その後お手本が経路へ戻る様子を記録します。
外乱中のフレームは `traj_data.pkl` の `perturbed` に記録され、正解ラベルには使いません (外乱直後の「立て直し」だけを学習)。
お手本は常に経路の上を走るので、これが無いとモデルは「経路から外れた状態」を見たことがなく、
少しずれただけで予測が崩れて衝突します。無効にするなら `explore.launch.py perturb:=false`。

保存形式は GNM (visualnav-transformer) と同じ (`<軌跡>/0.jpg…`, `traj_data.pkl` = `{"position": (N,2), "yaw": (N,), "perturbed": (N,)}`)。
確認 (コンテナ内): `python3 training/inspect_dataset.py /data/raw --num_viz 16` (1 フレームあたりの移動量が 0.1m 前後か、
正解軌跡 (緑) が画像上の進行方向と一致するかを見る)。

目安: ワールド (seed) を変えながら 1〜3 時間分 (数万フレーム)。1 エピソード ≈ 30〜60 秒。
見た目の多様性が汎化に効くので、`python3 tools/generate_worlds.py --worlds office park --seeds 1 2 3` で
レイアウト・テクスチャ配置の違うワールドを追加して集めるのがおすすめです (生成後 `WORLD=office_1` などで起動)。

### ファインチューニング

```bash
docker compose stop sim                       # GPU メモリを空ける
docker compose run --rm train                 # = python3 training/finetune_omnivla.py --config training/configs/finetune_gazebo.yaml
# 最初に配管と VRAM だけ確認したいとき
docker compose run --rm shell python3 training/finetune_omnivla.py --config training/configs/finetune_gazebo.yaml --dry_run true

# 途中のチェックポイントから学習を再開する (例: step 1000 から 2000 step 追加 ≒ 2.5 時間)
docker compose run --rm shell python3 training/finetune_omnivla.py  --config training/configs/finetune_gazebo.yaml --resume_from /runs/<run>/checkpoints/step_001000 --max_steps 2000

# 学習したモデルで走らせる (走行ログは log/nav/<日時>/ に自動保存)
docker compose up sim                                                                   # 端末 1
docker compose run --rm shell ros2 run omnivla_gazebo teleport --world office_0 --goal_dir /data/goals/demo
FINETUNED_DIR=/runs/<run>/checkpoints/step_003000 GOAL_PATH=/data/goals/demo docker compose run --rm nav

# 走行ログの解析 (地図上の軌跡・予測と、曲がるべき所で曲がっているかのレポート)
docker compose run --rm shell python3 tools/plot_nav_log.py log/nav/latest
```

- `<run>` は `runs/` 以下の学習ごとのディレクトリ (例: `omnivla_gazebo_20261005_080359`)。
  再開した学習は新しい `<run>` に保存され、step 番号は続きから数えます (step 1000 + 2000 → `step_003000`)。
- 学習中は 500 step ごとに `[val step …] ADE=… turn: ADE=… heading_err=…` が出ます。
  `turn:` の値が下がっていれば曲がり角の学習が進んでいます。途中のチェックポイントで試すときは学習を Ctrl+C で止めます
  (`saved …/step_XXXXXX` が出てから。走行と学習は同時に GPU に載りません)。

- 学習サンプル: 軌跡上の時刻 t の画像を現在画像、同じ軌跡の未来フレームをゴール画像/ゴール姿勢とし (hindsight relabeling)、
  真値オドメトリから未来 8 点の waypoint を正解とする。modality は `image`:`image_pose`:`pose` = 2:1:1 (設定可)。
- 公式と同じ: 全 Linear 層への LoRA (r=32), action head / pose projector も学習, 損失 = MSE + 0.1×平滑化項, lr 1e-4。
- 出力: `/runs/<run>/checkpoints/step_XXXXXX/` (LoRA アダプタ + ヘッド, 数百 MB), `metrics.csv` (JSON lines),
  `viz/step_XXXXXX/*.jpg` (検証サンプルの予測 (青) と正解 (緑))。`val_at_start: true` で学習前 = ゼロショットの誤差も記録。
- 曲がるサンプルの重み付け: 自動収集の走行はほとんど直進なので、そのままだとモデルは「とりあえず直進」を覚えます。
  既定では「この先 1m 以内に 45° 以上曲がる」サンプル (自然には約 1 割) を学習の 5 割で引きます
  (`turn_sample_ratio`, `turn_threshold_deg`, `turn_horizon`。0 で一様)。
  検証も半分を曲がるサンプルにして、`turn: ADE / FDE / heading_err` を別に表示します。曲がり角で失敗するならここを見ます。
- 立て直しサンプルの重み付け: 外乱つきで集めたデータがあれば、外乱直後 (経路から外れた状態から戻る) のサンプルを
  学習の 3 割以上で引きます (`recovery_sample_ratio`)。検証も 3 割をそのサンプルにして `recovery: ADE …` を表示します。
- 途中から再開: `--resume_from` で指定した step から `--max_steps` だけ追加で学習 (コマンドは上)。
  学習率は設定 (`learning_rate`, `lr_decay_step`) に従います (step 番号は続きから数える)。
- VRAM が足りない場合: `batch_size: 1` + `grad_accumulation_steps` を増やす、`lora_target: llm` (視覚エンコーダに LoRA を入れない)。
- 重要: `metric_waypoint_spacing` (既定 0.1m) は推論側と一致させる (navigator は `finetune_meta.json` から自動で読む)。

学習後の評価と利用 (`python3 …` / `ros2 …` はコンテナ内で実行: `docker compose run --rm shell bash`):
```bash
# 記録データ上の予測誤差 (ADE/FDE) をゼロショットと比較
python3 training/eval_offline.py --split /runs/<run>/split.json                                   # ベース
python3 training/eval_offline.py --split /runs/<run>/split.json --finetuned_dir /runs/<run>/checkpoints/step_005000
# Gazebo で走らせる
FINETUNED_DIR=/runs/<run>/checkpoints/step_005000 GOAL_PATH=/data/goals/demo docker compose run --rm nav
# 公式形式のマージ済みモデルを作る (公式 run_omnivla.py でも使える)
python3 training/merge_lora.py --finetuned_dir /runs/<run>/checkpoints/step_005000 --out_dir /checkpoints/omnivla-gazebo
```

### 曲がり角で経路から外れる・ぶつかる場合 (追加データ + 追加学習)

`log/nav/20261006_005858` (全 5000 step 学習後, 机の角で stuck) の解析で分かった原因は 2 つです。

1. **制御**: 公式の式 (`upstream`) は角速度 0.3rad/s 止まりで、鋭く曲がる予測ほど大回りします。
   机の角では予測の約半分の速さでしか曲がれず、経路から 0.15 → 0.6m 外れました。→ `controller: trajectory` (既定) に変更。
2. **学習データ**: お手本は常に経路の上を走るので、モデルは「経路から外れた状態」を学習していません。
   ずれが 0.1m 以内の間は予測はお手本とほぼ一致していましたが (2.7 秒先の向き +40° vs お手本 +41°)、
   0.6m 外れると予測が崩れて逆向き (−27〜−57°) になりました。→ 外乱つき収集 (DART) のデータを足して追加学習。

運動学シミュレーション (同じ demo 経路, 予測に学習後相当の誤差 3cm/6° を入れた 10 試行) での到達数:

| 予測モデルの性質 | `upstream` | `trajectory` |
|---|---|---|
| 経路から外れても戻れない (今のモデル相当) | 2/10 | 5/10 |
| 外れた状態から 2 割だけ戻れる (立て直しを学習したモデル相当) | 7/10 (机の角で衝突) | **10/10** |

制御だけでは足りず、立て直しの学習が必要です。手順 (`git pull` だけで反映, イメージの再ビルドは不要):

```bash
# 準備) 制御則の変数は NAV_CONTROLLER (既定 trajectory)。古い .env / シェルの CONTROLLER は使われない
grep -n CONTROLLER .env; env | grep CONTROLLER     # NAV_CONTROLLER=upstream があれば消す
#       走行ログの meta.json / report.txt の controller が trajectory になっているかでも確認できる

# 0) まず今のモデルのまま trajectory 制御で試す
docker compose up sim                                                                  # 端末 1
docker compose run --rm shell ros2 run omnivla_gazebo teleport --world office_0 --goal_dir /data/goals/demo
FINETUNED_DIR=/runs/<run>/checkpoints/step_005000 GOAL_PATH=/data/goals/demo docker compose run --rm nav

# 1) 外乱つきでデータを追加収集 (既存データと同じ /data/raw/office_0 に増える, 1 エピソード ≈ 40〜70 秒)
docker compose stop sim && HEADLESS=true docker compose up sim                         # 端末 1 (GUI なしで速く)
WORLD=office_0 EPISODES=150 SEED=1 docker compose run --rm explore                      # 端末 2
docker compose run --rm shell python3 training/inspect_dataset.py /data/raw --num_viz 0  # perturbed_frames > 0 を確認

# 2) 追加学習: step 5000 から 3000 step (7000 まで lr 1e-4, その後 1e-5)。古いデータと新しいデータを混ぜて学習する
docker compose stop sim
docker compose run --rm shell python3 training/finetune_omnivla.py --config training/configs/finetune_gazebo.yaml --resume_from /runs/<run>/checkpoints/step_005000 --max_steps 3000 --lr_decay_step 7000
#    ログの "recovery samples … -> sampled at >= 30%" と、500 step ごとの [val] recovery: ADE が下がるのを確認

# 3) 走らせる (新しい <run2> の step_008000)
FINETUNED_DIR=/runs/<run2>/checkpoints/step_008000 GOAL_PATH=/data/goals/demo docker compose run --rm nav

# 4) 1 本だけでなく成功率で確認: navigator をゴール無しで起動し、ランダムな経路 20 本 (サブゴール列) を評価
FINETUNED_DIR=/runs/<run2>/checkpoints/step_008000 docker compose run --rm nav         # 端末 2
docker compose run --rm shell ros2 launch omnivla_gazebo eval.launch.py \
    world:=office_0 num_tasks:=20 mode:=route min_dist:=4.0 max_dist:=12.0 label:=dart  # 端末 3
```

時間に余裕があれば 2) の代わりに、全データでベースモデルから学習し直す (`docker compose run --rm train`) 方がより確実です。
失敗した走行は今までどおり `log/nav/<日時>` を push してください。

### シミュレーション評価 (`eval.launch.py`)

navigator をゴール無しで起動しておき、評価ノードがゴール撮影 → スタートへテレポート → ゴール送信 → 監視を繰り返します。
```bash
docker compose up sim                                                              # 端末 1
docker compose run --rm nav                                                        # 端末 2 (FINETUNED_DIR で切替)
docker compose run --rm shell ros2 launch omnivla_gazebo eval.launch.py \
    world:=office_0 num_tasks:=20 mode:=single label:=zeroshot                     # 端末 3
```
`mode:=single` は 2〜5m 先のゴール画像 1 枚、`mode:=route` は経路沿いのサブゴール列。
結果は `/runs/eval/<label>_<world>_<時刻>/{results.csv, summary.json}` (成功率, SPL, 衝突回数, 最終距離)。
`seed` が同じならタスクも同じなので、ゼロショットと学習後を同条件で比較できます。

## ディレクトリ構成

```
docker/                Dockerfile, entrypoint, pip constraints
docker-compose.yml     sim / nav / explore / train / shell サービス
omnivla_nav/           ROS 非依存のコア (推論ラッパ, 制御, 地図・経路計画, データ形式, 可視化)
  policy.py            OmniVLAPolicy: 画像/姿勢/言語を引数で受け取る推論
  omnivla_model.py     モデル読み込み (公式 / LoRA), プロンプト, forward (推論と学習で共通)
  controller.py        予測軌跡 → (v, w) (trajectory / 公式の式 / pure pursuit)
  sim_map.py           SDF → 占有格子, Dijkstra 経路計画
ros2_ws/src/omnivla_gazebo/
  omnivla_gazebo/      ROS 2 ノード (navigator, auto_explorer, data_collector, topomap_recorder, eval_runner, teleport)
  launch/ config/ worlds/ models/ maps/ rviz/
training/              finetune_omnivla.py, gazebo_dataset.py, eval_offline.py, merge_lora.py, inspect_dataset.py
tools/generate_worlds.py   手続き的なワールド・テクスチャ生成
scripts/               download_checkpoints.sh, smoke_test_policy.py
tests/                 単体テスト (python3 tests/test_core.py)
docs/                  公式スクリプトの確認結果
```

## Gazebo 環境について

- ロボット (`models/omnivla_robot`): 差動二輪, 前方カメラ 640x480 / 水平画角 1.75rad (≒100°) / 高さ 0.35m / 15Hz。
  OmniVLA の学習データ (GNM, FrodoBots など) に多い広角・低視点に寄せています。`/odom` は OdometryPublisher による
  **ワールド座標の真値**で、ラベル作成・到達判定・評価に使います (実機では位置推定に置き換える部分)。
- ワールド: `tools/generate_worlds.py` が木目床・レンガ・ポスター・本棚・草地などの手続き的テクスチャと
  レイアウトを生成 (`office_0`: 部屋と廊下の屋内, `park_0`: 樹木・ベンチ・車のある屋外)。
  SDF の box/cylinder/sphere から占有格子を作るので、自作ワールドも同じ要素で作れば収集・評価に使えます。

## トラブルシューティング

- **Gazebo の画面が出ない**: ホストで `xhost +local:root`。`DISPLAY` が `.env`/環境変数で正しいか。
  ハイブリッド GPU のノート PC ではホスト側で NVIDIA を使う設定が必要な場合があります。GUI が不要なら `HEADLESS=true`。
- **カメラ画像が来ない (`waiting for camera image`)**: `docker compose logs sim` で ogre2 のエラーを確認。
  `NVIDIA_DRIVER_CAPABILITIES=all` (compose で設定済み) と NVIDIA Container Toolkit が必要です。
- **テレポートが失敗する**: `gz service -l | grep set_pose` で確認。コンテナ間で `GZ_PARTITION` を揃えています
  (compose 外から触る場合は `export GZ_PARTITION=omnivla`)。
- **ROS のトピックが見えない**: 全サービスが `network_mode: host`, 同じ `ROS_DOMAIN_ID`。ホストの ROS 2 と混ざる場合は ID を変える。
- **CUDA out of memory**: Gazebo と navigator を同じ GPU で動かすと 17〜18GB。学習時は sim/nav を止める。
- **WSL2**: GPU (CUDA) は使えますが Gazebo の GUI/描画は環境依存です。`HEADLESS=true` を試してください。

## 検証状況・既知の制約

- `tests/test_core.py` (座標変換が公式の GPS→ゴール計算と一致すること, 行動ラベル, 制御則 (trajectory が予測軌跡を再現すること,
  正しい予測での閉ループ走行), SDF→地図, 経路計画, エキスパート走行と外乱つき走行の運動学シミュレーション, データ入出力) はパスしています。Docker ビルド時にも実行されます。
- この環境の作成時点では GPU/Docker デーモンが無い環境で作業したため、**Docker ビルド、Gazebo 実行、7B モデルでの推論・学習は未実行**です。
  初回は `scripts/smoke_test_policy.py` と `finetune_omnivla.py --dry_run true` で確認してください。
- 衛星画像 modality (0-3) は扱っていません。言語 modality は推論のみ (Gazebo データに指示文が無いため学習では使わない)。
- OmniVLA-edge (軽量版) は対象外です。
