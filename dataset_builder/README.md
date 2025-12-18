
# dataset_builder
## 概要
UCF101 の avi ファイルおよび YOLOv8x-Pose により生成された json ファイルから、
学習用の PyTorch Dataset（`.pt`）を作成する。

- image_branch には  
  `{'images': (T, N, C_img, H, W), 'frames': (T,), 'label': int}`  
- skeleton_branch には  
  `{'skeletons': (T, N, K, C_kp), 'scores': (T, N, K), 'frames': (T,), 'label': int}`  

を出力する。

画像は画素値を 255 で割ることで 0–1 に正規化され、
骨格データは (x, y) 座標をそれぞれ (W, H) で割ることで 0–1 に正規化される。
スコアは元データの値（0–1）をそのまま使用する。

## 前提となるデータ構成
以下のディレクトリ構成を前提とする。
```
RAW_DATA_ROOT/
└─ UCF-101/
  └─ <class_name>/*.avi

DATASET_ROOT/
└─ preprocessed/keypoints_yolov8xpose/
  └─ <class_name>/*.json
```

クラス名のディレクトリ構成は avi / json で一致している必要がある。

## 補足仕様
- 各フレームで扱う人物数は N 人に固定される  
- 人物数が N 未満の場合、不足分はゼロ埋めのダミーデータで補完される  
- 検出できなかった人物の bbox / keypoints / score はすべて 0 となる  

## 出力
```
DATASET_ROOT/
└─ torch/
  ├─ image_branch/
  │ └─ <class_name>/.pt
  └─ skeleton_branch/
    └─ <class_name>/.pt
```

## conda 環境の作成
```bash
conda env create -f dataset_builder/environment.yml -n har-2way-model-dataset-builder -y
```

## 実行
実行は、リポジトリのトップディレクトリで行う。

```bash
python -m dataset_builder.main
```