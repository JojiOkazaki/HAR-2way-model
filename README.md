# HAR-2way-model
画像ベース行動認識モデルおよび骨格ベース行動認識モデルを結合した2経路によるUCF101の行動認識モデル。

## 各コンポーネントについて
### dataset_builder
学習用データセットを生成するためのモジュール。
UCF101の動画（avi）およびYOLOv8x-Poseにより生成された骨格情報（json）から、PyTorchで利用可能なDataset（`.pt`）を作成する。
生成されるデータは、画像ベースのbranchと骨格ベースのbranchに分かれて保存される。
詳細な仕様、前提となるデータ構成、実行方法については`dataset_builder/README.md` を参照。

### training
2経路モデルを学習するためのモジュール。
dataset_builderにより作成されたデータセットから、train/valに分割し学習を行う。
学習されたモデルは、モデル構造、ログ、グラフが保存される。
詳細な仕様、前提となるデータ構成、実行方法については`training/README.md` を参照。

# ディレクトリ構造
```
./HAR-2way-model
│  .gitignore
│  config_base.py
│  config_local.py
│  README.md
│
├─dataset_builder
│      config.py
│      data_generation.py
│      environment.yml
│      main.py
│      README.md
│      script_writer.py
│      video_frame_utils.py
│      __init__.py
│
└─training
    │  check_env.py
    │  environment.yml
    │  main.py
    │  README.md
    │
    ├─configs
    │      train.yaml
    │
    └─modules
        │  __init__.py
        │
        ├─dataset
        │      dataset.py
        │      __init__.py
        │
        ├─models
        │      block.py
        │      cnn.py
        │      gcn.py
        │      mlp.py
        │      transformer_encoder.py
        │      __init__.py
        │
        ├─networks
        │      full_model.py
        │      image_branch.py
        │      skeleton_branch.py
        │      __init__.py
        │
        ├─trainer
        │      trainer.py
        │      __init__.py
        │
        └─utils
                logger.py
                lr_scheduler.py
                runtime.py
                seed.py
                skeleton.py
                __init__.py
```