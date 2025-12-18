from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# 論理パスによるデフォルト設定
RAW_DATA_ROOT = PROJECT_ROOT / "raw_datas"
DATASET_ROOT = PROJECT_ROOT / "datasets"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
