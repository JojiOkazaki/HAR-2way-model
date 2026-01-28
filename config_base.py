
# config_base.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# 論理パスによるデフォルト設定
RAW_DATA_ROOT = PROJECT_ROOT / "raw_datas"

# ルート配下にデータセット名ごとのディレクトリを置く（例: ucf101, katorilab）
DATASETS_ROOT = PROJECT_ROOT / "datasets"

# 互換用: 単一データセット運用のコードが DATASET_ROOT を参照しても動くようにする
DATASET_NAME = "ucf101"
DATASET_ROOT = DATASETS_ROOT / DATASET_NAME

ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"

# 比率は合計が 1.0 になるようにしてください
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

# split 生成の乱数シード（再現性用）
SPLIT_SEED = 42

# groupバランスを無効にしたい場合
ENABLE_GROUP_BALANCE = True

# groupバランスの強さ（大きいほど group 偏りを強く抑える）
GROUP_BALANCE_ALPHA = 5.0
