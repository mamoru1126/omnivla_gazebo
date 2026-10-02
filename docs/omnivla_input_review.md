# OmniVLA 公式スクリプトの入力・前提の確認結果

対象: [NHirose/OmniVLA](https://github.com/NHirose/OmniVLA) コミット `5182600cb4a9ee07684e17cdd2a6cbafc56b8a68`
(Docker イメージはこのコミットに固定。行番号はこのコミットのもの)

結論: **公式の推論スクリプトは入力画像 (と GPS) がファイルパス/定数で固定されたデモ**で、そのままでは
ロボットやシミュレータの映像を入れられない。さらにモデル側にも「224x224・2 枚」という固定の前提がある。
本リポジトリでは `omnivla_nav/policy.py` (推論) と `training/` (学習) でこれらを解消している。

## 1. `inference/run_omnivla.py` (OmniVLA 本体 7B)

| 場所 | 内容 | 影響 | 本リポジトリでの対応 |
|---|---|---|---|
| L156-157 | 現在画像が `./inference/current_img.jpg` に固定 (毎 tick 同じファイルを読む) | カメラ画像を入れられない | `OmniVLAPolicy.predict(current, ...)` に PIL/ndarray を渡す。ROS ノードは `/camera/image_raw` を渡す |
| L572 | ゴール画像が `./inference/goal_img.jpg` に固定 (起動時に 1 回だけ読む) | ゴールを切り替えられない | `goal_image` 引数。navigator は topomap (サブゴール画像列) を順に切り替える |
| L132-134 | 現在位置 (GPS 緯度経度・コンパス) が定数 | pose 系 modality が常に同じ入力 | Gazebo の真値オドメトリから相対ゴール (x前, y左, yaw) を計算して `goal_pose` に渡す |
| L119 | `run()` が 1 回推論して `break` | ループしない | navigator が 3Hz で推論→制御を繰り返す |
| L560-565, L160, L172-175, L420 | modality フラグ (`pose_goal` 等) とモデル (`vla`, `action_head` …) が `__main__` のグローバル変数 | import して使えない | 全てクラス/引数化 |
| L203 | `clip_angle` が未定義 (waypoint が原点のとき NameError) | まれにクラッシュ | `omnivla_nav/controller.py` で実装 |
| L12, L156, L484 | `sys.path.insert(0, '..')` と相対パス (`./omnivla-original`) | リポジトリ直下から実行する必要あり | 絶対パス/パラメータ化 |

その他の暗黙の前提 (公式のまま踏襲している):
- `metric_waypoint_spacing = 0.1` (L129): 出力 waypoint 1 単位 = 0.1m。ゴール姿勢も 0.1m 単位で正規化、30m でクリップ (L128)
- 制御: `waypoints[4]` を `DT=1/3` で追従、`v<=0.3m/s, w<=0.3rad/s` に曲率を保って制限 (L193-235)

## 2. `inference/run_omnivla_edge.py` (軽量版 OmniVLA-edge)

| 場所 | 内容 |
|---|---|
| L154-155 | 現在画像が固定ファイル |
| L443 | ゴール画像が固定ファイル、さらに 96x96 に縮小 |
| L145-151 | GPS から計算した `goal_pose` を **テスト用の固定値で上書き** (コメントに「GPS を使うならこのブロックを消せ」とある) |
| L161 | 観測履歴 (context 6 枚) に **現在画像を 6 回複製** して入れている (本来は過去 5 フレーム + 現在) |
| L172-173 | 衛星画像はダミーの黒画像 |
| L430-431 | 入力解像度 96x96 (履歴・ゴール) と 224x224 (CLIP 用) |

本リポジトリは 7B 版を対象にしている (edge 版を使う場合も上記の固定を外す必要がある)。

## 3. モデル側の固定の前提 (変更不可 = 再学習が必要)

- **入力解像度 224x224 / resize-naive**: チェックポイントの `preprocessor_config.json` が `image_resize_strategy: resize-naive`。
  任意サイズの画像を入れられるが、**アスペクト比を無視して 224x224 に潰される** (640x480 なら横方向に圧縮)。
  学習時も同じ処理なので、推論と学習で同じ前処理を通すことが重要 (本リポジトリは両方とも processor の `apply_transform` を使用)。
- **画像は必ず 2 枚 (現在 + ゴール)**: `num_images_in_input=2`。
  `modeling_prismatic.py` L981-984 の注意マスクが `256` パッチ x 2 画像 + pose 1 トークンを**ハードコード**しており、
  解像度やパッチ数、画像枚数を変えると壊れる。ゴール画像を使わない modality でもダミー画像を入れて注意マスクで遮断する。
- **modality id** (0-8): `_build_multimodal_attention_MMN` が id ごとにゴール画像・pose・衛星画像トークンをマスク。
  言語は「マスク」ではなくプロンプトを `No language instruction` に置き換えることで無効化している。

## 4. 学習スクリプト `vla-scripts/train_omnivla.py` / `prismatic/vla/datasets/dummy_dataset.py`

| 場所 | 内容 | 本リポジトリでの対応 |
|---|---|---|
| dummy_dataset.py L110-111 | 全サンプルが同じ 2 枚の jpg (current_img / goal_img) | `training/gazebo_dataset.py` で記録データから作る |
| dummy_dataset.py L81 | 行動ラベルが乱数 | 真値オドメトリから未来 8 点の waypoint を計算 |
| dummy_dataset.py L153 | ランダムクロップの座標が 224x224 前提で決め打ち (`224-hoffset`)。それ以外のサイズでは左上だけを切り出してしまう | クロップ量を画像サイズに比例させてから 224 に縮小 |
| dummy_dataset.py (import) | ダミーでも `vint_train` (MBRA リポジトリ) が必要 | 不要 |
| train_omnivla.py L10 | `TRAIN_MODE = False` が既定で、**このままでは勾配が流れない** (デバッグモード) | 常に学習モード |
| train_omnivla.py L104, L756-772 | MBRA (別リポジトリ + `MBRA/mbra.pth`) を必須でロード (行動の再ラベル付け用) | Gazebo は真値があるので不要 |
| train_omnivla.py L144 | 保存のたびに LoRA をマージして 15GB のモデルを書き出す | アダプタ + ヘッドのみ保存。必要なら `training/merge_lora.py` |
| train_omnivla.py L807 | チェックポイントディレクトリ内のファイルを書き換える (`update_auto_map` 等) | 書き換えない |
| train_omnivla.py (全体) | torchrun 前提 (1 GPU でも `dist.barrier()` を呼ぶ) | 1 GPU は普通に `python3` で実行可、torchrun も可 |

## 5. インストール (`pyproject.toml`)

- L66: `"av"` の後ろにカンマが無く **TOML として不正** → `pip install -e .` が失敗する。Dockerfile で修正してからインストールしている。
- `flash-attn` は README では学習用に推奨されているが、コード上は import されておらず無くても動く (SDPA が使われる)。
