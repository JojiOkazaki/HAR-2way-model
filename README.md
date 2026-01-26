# HAR-2way-model
2経路(画像ベース行動認識モデルおよび骨格ベース行動認識モデル)を結合し、UCF101および自作データセットにより文埋め込みを学習する単一人物を前提とした行動認識モデル。

## モデルアーキテクチャ
### 次元
- `B`: バッチ数
- `T`: フレーム数
- `C_img`: 画像チャネル(`(R, G, B)`)
- `C_skel`: キーポイントチャネル(`(x, y, conf)`)
- `H`: 人物bboxの画像高さ
- `W`: 人物bboxの画像幅
- `J`: 人物キーポイント数
- `d_img`: 画像ベース行動認識モデルの出力次元
- `d_skel`: 骨格ベース行動認識モデルの出力次元
- `d`: 文埋め込み(および融合モデル)の出力次元

### 入力、出力データ
- `x_img`: 単一人物の画像入力(形状: `(B, T, C_img, H, W)`)
- `x_skel`: 単一人物の骨格入力(形状: `(B, T, J, C_skel)`)
- `f_img`: 画像ベース行動認識モデルの出力(形状: `(B, d_img)`)
- `f_skel`: 骨格ベース行動認識モデルの出力(形状: `(B, d_skel)`)
- `y`: 融合モデルの出力(形状: `(B, d)`)
- `y'`: 文埋め込み(形状: `(B, d)`)
- `text_gt`: 教師文章

### 画像ベース行動認識モデル
- `f_img = TransformerEncoder(CNN(x_img))`

### 骨格ベース行動認識モデル
- `f_skel = ST-GCN(x_skel)`

### 結合モデル
- `y = MLP(concat(f_img, f_skel))`

### 教師埋め込み
- `y' = SentenceBERT(text_gt)`

# 各コンポーネントについて
## dataset_builder
学習用データセットを生成するためのモジュール。
UCF101の動画（avi）およびYOLOv8x-Poseにより生成された骨格情報（json）から、PyTorchで利用可能なDataset（`.pt`）を作成する。
生成されるデータは、画像ベースのbranchと骨格ベースのbranchに分かれて保存される。
詳細な仕様、前提となるデータ構成、実行方法については`dataset_builder/README.md` を参照。

## training
2経路モデルを学習するためのモジュール。
dataset_builderにより作成されたデータセットから、train/valに分割し学習を行う。
学習されたモデルは、モデル構造、ログ、グラフが保存される。
詳細な仕様、前提となるデータ構成、実行方法については`training/README.md` を参照。

# ディレクトリ構造
```
.
├─ dataset_builder/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ data_generation.py
│  ├─ environment.yml
│  ├─ main.py
│  ├─ script_writer.py
│  └─ video_frame_utils.py
├─ training/
│  ├─ configs/
│  │  ├─ infer.yaml
│  │  └─ train.yaml
│  ├─ modules/
│  │  ├─ dataset/
│  │  │  ├─ __init__.py
│  │  │  └─ dataset.py
│  │  ├─ models/
│  │  │  ├─ blocks/
│  │  │  │  ├─ __init__.py
│  │  │  │  ├─ base.py
│  │  │  │  ├─ conv.py
│  │  │  │  ├─ fc.py
│  │  │  │  └─ graph.py
│  │  │  ├─ __init__.py
│  │  │  ├─ cnn.py
│  │  │  ├─ mlp.py
│  │  │  ├─ stgcn.py
│  │  │  └─ transformer_encoder.py
│  │  ├─ networks/
│  │  │  ├─ __init__.py
│  │  │  ├─ full_model.py
│  │  │  ├─ image_branch.py
│  │  │  └─ skeleton_branch.py
│  │  ├─ trainer/
│  │  │  ├─ __init__.py
│  │  │  └─ trainer.py
│  │  ├─ utils/
│  │  │  ├─ __init__.py
│  │  │  ├─ early_stopper.py
│  │  │  ├─ errors.py
│  │  │  ├─ logger.py
│  │  │  ├─ lr_scheduler.py
│  │  │  ├─ runtime.py
│  │  │  ├─ seed.py
│  │  │  └─ skeleton.py
│  │  └─ __init__.py
│  ├─ check_env.py
│  ├─ environment.yml
│  ├─ infer.py
│  └─ main.py
├─ .gitignore
├─ config_base.py
├─ config_local.py
└─ README.md
```