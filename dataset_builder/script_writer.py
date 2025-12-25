import os
import torch
import orjson
from sentence_transformers import SentenceTransformer

import dataset_builder.config as config
from dataset_builder.data_generation import generate_video_sample

_ST_MODEL = None
_LABEL_EMB_CACHE = {}

def _get_st_model():
    global _ST_MODEL
    if _ST_MODEL is None:
        _ST_MODEL = SentenceTransformer(config.SENTENCE_TRANSFORMER_MODEL_NAME)
    return _ST_MODEL

def _get_label_embedding(label_id: int, sentence: str) -> torch.Tensor:
    # 各workerプロセス内で label_id ごとに一度だけ encode してキャッシュ
    if label_id not in _LABEL_EMB_CACHE:
        model = _get_st_model()
        emb = model.encode(sentence, convert_to_tensor=True)  # shape: (D,)
        _LABEL_EMB_CACHE[label_id] = emb.detach().cpu().to(torch.float32)
    return _LABEL_EMB_CACHE[label_id]

def write_video_sample(video_dir, json_dir, out_img_dir, out_skel_dir, label, label_id, label_sentence, video_filename, P, T, J, W, H):
    # 出力先ディレクトリの取得
    out_img_dir = out_img_dir / label
    out_skel_dir = out_skel_dir / label
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_skel_dir, exist_ok=True)

    # 元データパスの取得
    video_path = video_dir / label / video_filename
    json_path = json_dir / label / video_filename.replace(".avi", ".json")

    # 出力先パスの取得
    out_img_path = out_img_dir / video_filename.replace(".avi", ".pt")
    out_skel_path = out_skel_dir / video_filename.replace(".avi", ".pt")

    # 元データパスがない場合は処理しない
    if not (os.path.exists(video_path) and os.path.exists(json_path)):
        return

    # jsonデータの取得
    with open(json_path, "rb") as f:
        json_data = orjson.loads(f.read())

    # 処理
    frames, img_data, skel_data, score_data = generate_video_sample(video_path, json_data, P, T, J, W, H)
    label_emb = _get_label_embedding(label_id, label_sentence) # shape: (D, )
    labels = label_emb.unsqueeze(0).repeat(P, 1).contiguous() # shape: (P, D)

    # ptファイルを作成
    torch.save(
        {"images": img_data, "frames": frames, "labels": labels,
        "label_id": label_id, "label_sentence": label_sentence},
        out_img_path,
    )
    torch.save(
        {"skeletons": skel_data, "scores": score_data, "frames": frames, "labels": labels,
        "label_id": label_id, "label_sentence": label_sentence},
        out_skel_path,
    )
