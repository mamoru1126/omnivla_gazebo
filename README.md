# omnivla_gazebo

[OmniVLA](https://github.com/NHirose/OmniVLA) (ICRA 2026, ナビゲーション用 VLA) を **Gazebo (Harmonic) 上で動かし、
Gazebo のデータでファインチューニングして、サブゴール画像をたどって自律移動させる**ための Docker 環境です。

```
 ┌──────── Gazebo Harmonic ────────┐   /camera/image_raw    ┌──────── OmniVLA navigator (3Hz) ───────────┐
 │ 手続き生成ワールド (office/park)  │ ─────────────────────▶ │ 現在画像 + サブゴール画像                    │
 │ 差動二輪ロボット + 前方カメラ     │   /odom (真値)          │   → OmniVLA 7B → 8 点の waypoint (約 2.7 秒) │
 │                                  │ ◀───────────────────── │   → (v, w)  → サブゴール到達で次の画像へ     │
 └──────────────────────────────────┘   /cmd_vel             └─────────────────────────────────────────────┘
          │ 自動走行 (経路計画 + 外乱) で記録                                        ▲
          ▼                                                                      │ LoRA アダプタ
   /data/raw/<world>/<軌跡>/{0.jpg,…, traj_data.pkl}  ──▶  training/finetune_omnivla.py
```

- **走行**: ゴールまでの経路を 1m ごとに撮影した画像 (topomap) を用意し、OmniVLA が「現在画像 + 今のサブゴール画像」から
  進路を出す → サブゴールに着いたら次の画像へ、を最終ゴールまで繰り返す。
- **学習**: Gazebo 内で自動走行させてデータを集め、公式と同じ LoRA 設定で Gazebo の見え方に適応させる。
- 公式スクリプトから変えた点 (入力画像の固定など) は [docs/omnivla_input_review.md](docs/omnivla_input_review.md)。

## 動作環境

| 項目 | 要件 |
|---|---|
| OS | Linux (Ubuntu 22.04/24.04)。GUI を出すなら X11 |
| GPU | NVIDIA, VRAM 24GB (RTX 3090/4090 等)。推論 ≈ 16GB、学習 ≈ 20GB |
| ソフト | Docker + Docker Compose v2 + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)、NVIDIA ドライバ 530 以上 |
| ディスク | イメージ ≈ 15GB + チェックポイント ≈ 16GB + 収集データ |

コンテナ: Ubuntu 22.04 / Python 3.10 / ROS 2 Humble / Gazebo Harmonic / PyTorch 2.2.0 / OmniVLA (コミット固定)。

## 手順

`docker compose run --rm shell ...` はコンテナ内で実行するコマンドです。Gazebo (`sim`) と学習は同じ GPU に同時に載らないので、
学習の前に `sim` を止めます。

### 1. セットアップ

```bash
git clone https://github.com/mamoru1126/omnivla_gazebo && cd omnivla_gazebo
cp .env.example .env
docker compose build                                                  # 初回 20〜40 分
docker compose run --rm shell bash scripts/download_checkpoints.sh    # 公式チェックポイント (約 16GB) → ./checkpoints
docker compose run --rm shell python3 scripts/smoke_test_policy.py    # GPU とモデルの動作確認 (任意)
```

### 2. 学習データを集める

ランダムなスタート/ゴール間を経路計画して自動走行し、画像と姿勢を 3Hz で記録します。
走行中にときどきわざと経路から外し、そこから戻る様子も記録します (外乱つき収集, 既定で有効)。
これが無いとモデルは「経路から外れた状態」を知らず、曲がり角で少しずれただけで衝突します。

```bash
HEADLESS=true WORLD=office_0 docker compose up sim                    # 端末 1 (GUI なしで速く)
WORLD=office_0 EPISODES=150 docker compose run --rm explore           # 端末 2 → /data/raw/office_0/ (1 エピソード ≈ 1 分)
docker compose run --rm shell python3 training/inspect_dataset.py /data/raw --num_viz 16   # 確認 (任意)
```

見た目の多様性が汎化に効くので、`python3 tools/generate_worlds.py --worlds office park --seeds 1 2 3` で
ワールドを増やして集めるのもおすすめです (`WORLD=office_1` などで起動)。

### 3. ファインチューニング

```bash
docker compose stop sim
docker compose run --rm train          # = finetune_omnivla.py --config training/configs/finetune_gazebo.yaml (5000 step)
```

- 結果は `runs/<run>/checkpoints/step_XXXXXX/` (`<run>` 例: `omnivla_gazebo_20261007_045312`)。
- 500 step ごとに `[val step …] ADE=… turn: …` が出ます (`turn` = 曲がる場面, `recovery` = 経路から戻る場面の誤差)。
- 曲がる場面と経路から戻る場面は少ないので、多めに引いて学習します (`turn_sample_ratio`, `recovery_sample_ratio`)。
- 中断した学習の再開: `--resume_from runs/<run>/checkpoints/step_XXXXXX --max_steps <追加 step 数>`。

### 4. ゴール画像列 (topomap) を作る

スタート (ロボットの現在位置) からゴール座標まで経路計画し、1m ごとにテレポートして撮影 → スタートに戻します。

```bash
docker compose up sim                                                 # 端末 1
docker compose run --rm shell ros2 launch omnivla_gazebo topomap.launch.py \
    world:=office_0 out_dir:=/data/goals/demo goal_x:=7.5 goal_y:=4.0
```

ゴール座標は地図 `ros2_ws/src/omnivla_gazebo/maps/<world>.png` (黒=障害物) の空き位置を小数で指定します。
作り直すときは `overwrite:=true`。

### 5. 走らせる

```bash
# ロボットを topomap のスタート姿勢に戻す (走行後や sim 再起動後は必須)
docker compose run --rm shell ros2 run omnivla_gazebo teleport --world office_0 --goal_dir /data/goals/demo
# 走行 (起動するとすぐ走り出す)
FINETUNED_DIR=/runs/<run>/checkpoints/step_005000 GOAL_PATH=/data/goals/demo docker compose run --rm nav
```

途中経過は `rqt_image_view /omnivla/debug_image` (現在画像 + 予測軌跡, サブゴール画像, 俯瞰図) で見られます。

### 6. 評価 (成功率)

navigator をゴール無しで起動し、評価ノードがランダムな経路でゴール撮影 → スタートへ移動 → 走行 → 判定を繰り返します。

```bash
FINETUNED_DIR=/runs/<run>/checkpoints/step_005000 docker compose run --rm nav     # 端末 2
docker compose run --rm shell ros2 launch omnivla_gazebo eval.launch.py \
    world:=office_0 num_tasks:=20 mode:=route min_dist:=4.0 max_dist:=12.0 label:=finetuned   # 端末 3
```

結果は `runs/eval/<label>_<world>_<時刻>/summary.json` (成功率, SPL, 衝突回数)。`seed` が同じならタスクも同じです。
`FINETUNED_DIR` を空にするとゼロショット (公式モデルそのまま) と比較できます。

## 走行ログ

navigator は走行ごとに `log/nav/<日時>/` へログを保存します (`log/` は git 管理外)。

| ファイル | 中身 |
|---|---|
| `steps.csv` | 推論ごとの位置姿勢, サブゴール, 予測 8 点, 指令 (v, w) |
| `events.log` / `summary.json` | サブゴールの切り替え, 到達したか, 終了理由 |
| `meta.json` | モデル, 全パラメータ, ゴール列 |
| `debug/`, `raw/`, `goals/` | 毎ステップのデバッグ画像, カメラ画像, ゴール画像 |

解析 (GPU 不要): `python3 tools/plot_nav_log.py log/nav/latest` → `overview.png` (地図上の軌跡と予測), `timeline.png`,
`report.txt` (予測がサブゴールと逆 = モデル側の問題 / 予測した旋回を実行できていない = 制御側の問題)。

解析を頼むときは、ログ用リポジトリ (`mamoru1126/omnivla_gazebo_logs`, private) に push します。

```bash
cp -r log/nav/<日時> ../omnivla_gazebo_logs/nav/
cd ../omnivla_gazebo_logs && git add nav/<日時> && git commit -m "nav log <日時>" && git push
```

## 主な設定

走行の設定は `ros2_ws/src/omnivla_gazebo/config/navigator.yaml`。

| パラメータ | 既定 | 内容 |
|---|---|---|
| `modality` | `image` | ゴールの与え方。`image` / `image_pose` / `pose` / `language` (言語は学習していない) |
| `controller` | `trajectory` | 予測軌跡 → (v, w)。`trajectory` は予測した旋回・速度をそのまま実行 (公式の式は `upstream`) |
| `subgoal_radius` | 0.3 | サブゴールに着いたとみなす距離 [m] |
| `reach_angle_deg` | 25 | 途中のサブゴールは向きの差もこれ以内で到達 (曲がり角で内側に詰めるのを防ぐ) |
| `pass_radius` | 1.0 | この距離以内でサブゴールが後ろに回ったら通過扱い (サブゴールの周りを回り続けない) |
| `goal_radius` | 0.4 | 最終ゴールの到達距離 [m] |
| `stuck_timeout` | 4.0 | 前進指令中にこの秒数動かなければ停止 |

`docker compose run --rm nav` の環境変数: `GOAL_PATH`, `FINETUNED_DIR`, `MODALITY`, `NAV_CONTROLLER`, `WORLD`。

推論部分は ROS 非依存なので、実機や別のシミュレータからも使えます:
```python
from omnivla_nav.policy import OmniVLAPolicy, PolicyConfig
policy = OmniVLAPolicy(PolicyConfig(vla_path="/checkpoints/omnivla-original", finetuned_dir="..."))
out = policy.predict(current_pil, goal_image=goal_pil, modality="image")
out.waypoints   # (8, 4) [x前[m], y左[m], cos(yaw), sin(yaw)], k 点目は (k+1)/3 秒後
```

## ディレクトリ構成

```
docker/ docker-compose.yml   Docker 環境 (sim / nav / explore / train / shell)
omnivla_nav/                 ROS 非依存のコア: 推論, 制御, 到達判定, 地図・経路計画, データ形式
ros2_ws/src/omnivla_gazebo/  ROS 2 ノード (navigator, auto_explorer, topomap_recorder, eval_runner, teleport),
                             launch / config / worlds / models / maps
training/                    ファインチューニング, データセット, オフライン評価, LoRA のマージ
tools/                       ワールド生成, 走行ログ解析
scripts/ tests/ docs/        チェックポイント取得・動作確認, 単体テスト, 公式スクリプトの確認結果
```

## Gazebo 環境

- ロボット: 差動二輪, 前方カメラ 640x480 / 水平画角約 100° / 高さ 0.35m。`/odom` はワールド座標の真値で、
  学習ラベル・到達判定・評価に使います (実機では自己位置推定に置き換える部分)。
- ワールド: `office_0` (部屋と廊下の屋内), `park_0` (屋外)。`tools/generate_worlds.py` で増やせます。

## トラブルシューティング

- **Gazebo の画面が出ない**: ホストで `xhost +local:root`。GUI が不要なら `HEADLESS=true`。
- **カメラ画像が来ない**: `docker compose logs sim` で描画エラーを確認 (NVIDIA Container Toolkit が必要)。
- **テレポートできない / トピックが見えない**: コンテナ間で `GZ_PARTITION`, `ROS_DOMAIN_ID` を揃えています (`.env`)。
- **CUDA out of memory**: 学習時は `docker compose stop sim`。
- **ゴールと逆を向いて走り出す**: スタート姿勢に戻していない。手順 5 の teleport を先に実行。

## 制約

- 到達判定と topomap の撮影にシミュレータの真値位置を使っています。実機では自己位置推定か画像類似度 (`reach_check: image`) に置き換えが必要です。
- 言語 modality は推論のみ (Gazebo データに指示文が無いため学習していない)。衛星画像 modality と OmniVLA-edge は対象外。
