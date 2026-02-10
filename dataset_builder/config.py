# dataset_builder/config.py

# SentenceBERT（SentenceTransformer）で文章埋め込みを作る際のモデル名
SENTENCE_TRANSFORMER_MODEL_NAME = "stsb-xlm-r-multilingual"

# 並列処理のワーカー数
NUM_WORKERS = 4

# 時系列長（各人物のフレーム列はこの長さで作る）
T = 32

# 画像（人物切り出し）の出力サイズ
H = 112
W = 112

# キーポイント数（COCO形式）
J = 17

# サンプリング前の足切り（min(video_frames, json_frames) がこれ未満なら除外）
MIN_FRAMES = 32

# 1動画あたり候補人物数（先に多めに拾って、後段で「主人物優先」で絞り込む）
MAX_P_CANDIDATE = 10

# 1動画あたり最終的に残す人物数（.pt に保存する上限）
MAX_P = 3

# 有効人物判定の最小フレーム数（scores.max(dim=-1)>0 のフレーム数がこれ以上）
MIN_VALID_T = 16
