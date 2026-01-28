# dataset_builder
# 概要
本モジュールは、モデル学習に必要な画像/骨格データセットを作成するためのパイプラインである。
生の動画ファイル(`.avi`/`.mp4`)、姿勢推定モデルによる推定結果ファイル(`.json`)、アノテーションファイル(`.json`)から、PyTorchでの学習に直接利用可能な`.pt`形式へデータを変換する。
主な機能を以下に示す。

- 人物領域のクロップ: 動画フレームから特定の人物(`person_id`)のBounding Boxに基づいて画像をクロップし、指定サイズにリサイズする。
- 骨格座標の正規化: COCO形式に従う17点のキーポイントの骨格座標に対し、腰を中心とした平行移動と正規化を行う。
- 文埋め込み: SentenceBERT(`stsb-xlm-r-multilingual`)を使用して、行動ラベルに対応する文章をベクトル化し、教師データとする。
- データセット分割: 生成された全データから、train/val/testデータのファイルリストを自動生成する。

# 環境構築
本モジュールはAnaconda環境での動作を想定している。`dataset_builder/environment.yml`を用いて依存ライブラリを一括インストールする。

```bash
conda env create -f dataset_builder/environment.yml
conda activate har-2way-model-dataset-builder
```

実行はリポジトリルート(`HAR-2way-model/`)で以下の通りである。

```bash
python -m dataset_builder.main
```

実行時に参照するデータセットルートは、`config_base.py` / `config_local.py` に定義する次のいずれかで決まる(優先順)。

- `DATASETS_ROOT`
- `DATASET_ROOT`
- `RAW_DATA_ROOT`

例(`HAR-2way-model/datasets/` をルートとして使う場合): 

```python
# config_local.py(例)
DATASETS_ROOT = r"D:\datas\HAR-2way-model\datasets"
```

# データディレクトリ構造
データセットは`DATASETS_ROOT`配下に複数置ける構造を前提とする。
各データセットは `raw/`(入力)と `processed/`(生成物)を持つ。
また、UCF101の公式splitファイル(trainlist/testlist)は`raw/`の外に別管理する。

```
HAR-2way-model/
└─ datasets/                       # DATASETS_ROOT(複数データセットの親)
   ├─ katorilab/                   # データセット名
   │  ├─ raw/                      # 入力データ
   │  │  ├─ videos/                # 動画ファイル (.mp4 / .avi)
   │  │  ├─ keypoints/             # 骨格推定JSON (YOLOv8 Pose等)
   │  │  └─ annotations/           # アノテーションJSON
   │  └─ processed/                # 自動生成されるディレクトリ
   │     ├─ pt/                    # 生成された .pt ファイル
   │     └─ splits/                # 分割リスト置き場
   │        ├─ all_list.txt         # 全人物行の一覧(共通インデックス)
   │        └─ default/             # サブスプリット名(katorilabは通常 default)
   │           ├─ train_list.txt
   │           ├─ val_list.txt
   │           └─ test_list.txt
   │
   └─ ucf101/
      ├─ raw/
      │  ├─ videos/
      │  ├─ keypoints/
      │  └─ annotations/
      ├─ processed/
      │  ├─ pt/
      │  └─ splits/
      │     ├─ all_list.txt         # 全人物行の一覧(共通インデックス)
      │     ├─ split01/             # サブスプリット(split01/02/03 等)
      │     │  ├─ train_list.txt
      │     │  ├─ val_list.txt
      │     │  └─ test_list.txt
      │     ├─ split02/
      │     │  ├─ train_list.txt
      │     │  ├─ val_list.txt
      │     │  └─ test_list.txt
      │     └─ split03/
      │        ├─ train_list.txt
      │        ├─ val_list.txt
      │        └─ test_list.txt
      └─ original_splits/           # UCF101公式の分割ファイル(rawの外)
         ├─ split01/
         │  ├─ trainlist01.txt
         │  └─ testlist01.txt
         ├─ split02/
         │  ├─ trainlist02.txt
         │  └─ testlist02.txt
         └─ split03/
            ├─ trainlist03.txt
            └─ testlist03.txt
```

備考: 

- `raw/` 配下の各ファイル(動画・JSON)はファイル名(stem)が一致している必要がある。
- `processed/pt/` は `.pt` の生成物であり、複数のサブスプリットで共通に使う。
- `processed/splits/` は `all_list.txt`(共通)と、サブスプリットごとの `train/val/test_list.txt` を保持する。

# 設定
設定は用途により2か所に分かれる。

## 1. データ生成パラメータ(`dataset_builder/config.py`)
必要に応じて以下を変更する。

- `SENTENCE_TRANSFORMER_MODEL_NAME`: 文埋め込みに使用するSentenceBERTモデル名
- `NUM_WORKERS`: 並列処理を行うワーカープロセス数
- `T`: 生成する時系列データのフレーム長
- `H, W`: 人物領域画像の出力サイズ
- `J`: 骨格キーポイント数
- `MIN_FRAMES`: 入力動画またはJSONのフレーム数がこの値未満の場合、生成から除外
- `MAX_P`: 1動画あたりに処理する最大人物数

## 2. データセットルート／分割比率(`config_base.py` / `config_local.py`)
データセットの配置場所と、分割リスト生成の比率(train/val/test)などは `config_base.py`(必要なら `config_local.py` で上書き)で定義する。

- データセットルート(優先順): `DATASETS_ROOT` → `DATASET_ROOT` → `RAW_DATA_ROOT`
- 分割比率: `TRAIN_RATIO`, `VAL_RATIO`, `TEST_RATIO`
- 乱数seed: `SPLIT_SEED`

# 出力データ形式
生成される`.pt`ファイルは、以下のキーを持つ辞書オブジェクトである。
テンソルは全てCPU上に配置される。

| キー | 形状 (Shape) | データ型 | 説明 |
| --- | --- | --- | --- |
| `images` | `"(P, T, 3, H, W)"` | `uint8` | クロップされた人物画像(0-255) |
| `skeletons` | `"(P, T, J, 2)"` | `float32` | 正規化済み骨格座標 |
| `scores` | `"(P, T, J)"` | `float32` | 骨格キーポイントの信頼度 |
| `frames` | `"(P, T)"` | `int64` | 元動画におけるフレーム番号 |
| `labels` | `"(P, D)"` | `float32` | SentenceBERTによる文埋め込みベクトル |
| `person_ids` | `List[int]` | - | アノテーションに基づくPerson ID |
| `label_ids` | `List[int]` | - | アノテーションに基づくLabel ID |
| `label_sentences` | `List[str]` | - | アノテーションの元の文章 |
| `video_stem` | `str` | - | 元動画のファイル名(拡張子なし) |

- `P`: 人物数
- `T`: フレーム数 (`config.T`)
- `J`: キーポイント数 (`config.J`)
- `D`: 文埋め込み次元

# 入力JSONファイルのデータ構造
## 骨格推定JSON(`keypoints/*.json`)
YOLOv8 Pose等の出力を想定しており、フレームごとの人物リストを含む。

```
{
  "video_id": "sample_video.mp4",
  "num_frames": 100,
  "frames": [
    {
      "frame_index": 0,
      "timestamp_ms": 0,
      "persons": [
        {
          "person_id": 1,
          "bbox_xyxy": [
            100.0, 50.0, 200.0, 300.0
          ],
          "keypoints_coco": [
            [320.0, 240.0, 0.95],
            [325.0, 238.0, 0.88],
            ...
            // 合計17点の [x, y, conf] リスト
          ]
        }
      ]
    },
    ...
  ]
}
```

## アノテーションJSON(`annotations/*.json`)
動画内の人物IDに対して、正解ラベルと文章を紐付ける。

```
{
  "annotations": [
    {
      "person_id": 1,            // 必須: keypoints側のIDと対応
      "label": 0,                // 必須: クラスID (int)
      "sentence": "行動を表す文章。", // 必須: 文埋め込みの元テキスト
      
      // 以下はオプション (存在しない場合はデフォルト値が使用される)
      "is_error": false          // trueの場合、この人物は無視される (Default: false)
      // "annotator_id": 101     // 記録用 (ツールでは使用されない)
    }
  ]
}
```
# スプリットファイルの形式と運用
## 1. `processed/splits/all_list.txt`
全データから作る共通インデックス。各行は人物単位である。

- 形式: `<stem>.pt <person_id> <label_id>`

例: 

```
v_ApplyEyeMakeup_g01_c01.pt 1 0
v_ApplyEyeMakeup_g01_c01.pt 2 0
```

## 2. `processed/splits/<サブスプリット>/{train,val,test}_list.txt`
学習時に参照する動画(pt)単位のリストである。

- 形式(2列): `<stem>.pt 0`

例: 

```
v_ApplyEyeMakeup_g01_c01.pt 0
v_ApplyEyeMakeup_g01_c02.pt 0
```

※ 2列目は現在 `0` 固定で出力している(学習側のリスト仕様に合わせるため)。

## 3. UCF101公式 split(`original_splits/`)
UCF101公式の train/test リストは形式が異なるため、生成物(`processed/splits/`)とは分離して保管する。

- `trainlistXX.txt`: `<ClassName>/<video>.avi <label>`(ラベルは 1-based)
- `testlistXX.txt`: `<ClassName>/<video>.avi`(ラベル列なし)

これらを元に、実際に存在する `.pt` に対応付けた `processed/splits/split01/` 等の `train/val/test_list.txt` を作成して運用する。
