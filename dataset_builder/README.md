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
本モジュールはAnaconda環境での動作を想定している。
`dataset_builder/environment.yml`を用いて、全依存ライブラリを一括でインストールできる。

```
conda env create -f dataset_builder/environment.yml
conda activate har-2way-model-dataset-builder
```

また、実行時は以下のように実行する。

```
python -m dataset_builder.main
```


# データディレクトリ構造
生の動画ファイル、骨格推定結果ファイル、アノテーションファイルが含まれる複数のデータセットは、以下の構造であることを前提とする。
各ファイル(動画、JSON)のファイル名(stem)は一致している必要がある。

```
.
├─ katorilab/               # データセット名
│  ├─ processed/            # 自動生成されるディレクトリ
│  │  ├─ pt/                # 生成された .pt ファイル
│  │  │  ├─ 20250802_100426.pt
│  │  │  └─ ...
│  │  └─ splits/            # 生成されたリストファイル
│  │     └─ all_list.txt
│  └─ raw/                  # 入力データ
│     ├─ annotations/       # アノテーションJSON
│     │  ├─ 20250802_100426.json
│     │  └─ ...
│     ├─ keypoints/         # 骨格推定JSON (YOLOv8 Pose等)
│     │  ├─ 20250802_094407.json
│     │  └─ ...
│     └─ videos/            # 動画ファイル (.mp4 / .avi)
│        ├─ 20250802_094407.mp4
│        └─ ...
└─ ucf101/
   └─ ...
```

# 設定
データセット生成のパラメータは`dataset_builder/config.py`にて定義されている。
必要に応じて以下の値を変更する。

- `SENTENCE_TRANSFORMER_MODEL_NAME`: 文埋め込みに使用するSentenceBERTモデル名(デフォルト: `stsb-xlm-r-multilingual`)。
- `NUM_WORKERS`: 並列処理を行うワーカープロセス数。
- `T`: 生成する時系列データのフレーム長(デフォルト: `32`)。
- `H, W`: 人物領域画像の出力サイズ(デフォルト: `112x112`)。
- `J`: 骨格キーポイント数(デフォルト: 17点のCOCO形式)。
- `MIN_FRAMES`: 入力動画またはJSONのフレーム数がこの値未満の場合、データ生成から除外する。
- `MAX_P`: 1動画あたりに処理する最大人物数。これを超える人物は無視される。

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






