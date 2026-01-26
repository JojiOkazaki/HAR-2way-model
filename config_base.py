# config_base.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# 論理パスによるデフォルト設定
RAW_DATA_ROOT = PROJECT_ROOT / "raw_datas"

# ルート配下にデータセット名ごとのディレクトリを置く（例: ucf101, katorilab）
DATASETS_ROOT = PROJECT_ROOT / "datasets"

# 互換用: 単一データセット運用のコードが DATASET_ROOT を参照しても動くようにする
DEFAULT_DATASET_NAME = "ucf101"
DATASET_ROOT = DATASETS_ROOT / DEFAULT_DATASET_NAME

ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
