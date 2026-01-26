# dataset_builder/script_writer.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import orjson
import torch
from sentence_transformers import SentenceTransformer

import dataset_builder.config as config
from dataset_builder.data_generation import generate_video_sample


_ST_MODEL: Optional[SentenceTransformer] = None
_SENT_EMB_CACHE: Dict[str, torch.Tensor] = {}


def _get_st_model() -> SentenceTransformer:
    global _ST_MODEL
    if _ST_MODEL is None:
        _ST_MODEL = SentenceTransformer(config.SENTENCE_TRANSFORMER_MODEL_NAME)
    return _ST_MODEL


def _get_sentence_embedding(sentence: str) -> torch.Tensor:
    """
    workerプロセス内キャッシュ（sentence文字列キー）
    returns: (D,) float32 CPU
    """
    if sentence not in _SENT_EMB_CACHE:
        model = _get_st_model()
        emb = model.encode(sentence, convert_to_tensor=True)  # (D,)
        _SENT_EMB_CACHE[sentence] = emb.detach().cpu().to(torch.float32).contiguous()
    return _SENT_EMB_CACHE[sentence]


def _read_json(path: Path) -> Any:
    with open(path, "rb") as f:
        return orjson.loads(f.read())


def _resolve_video_path(videos_dir: Path, stem: str) -> Optional[Path]:
    avi = videos_dir / f"{stem}.avi"
    mp4 = videos_dir / f"{stem}.mp4"

    has_avi = avi.is_file()
    has_mp4 = mp4.is_file()

    if has_avi and has_mp4:
        raise ValueError(f"Both .avi and .mp4 exist for the same stem (not allowed): {stem}")
    if has_avi:
        return avi
    if has_mp4:
        return mp4
    return None


def _get_video_frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    return n


def _extract_json_frame_count(json_data: dict) -> int:
    if not isinstance(json_data, dict):
        return 0
    frames = json_data.get("frames", [])
    if not isinstance(frames, list):
        return 0
    num_frames = int(json_data.get("num_frames", len(frames)))
    return min(num_frames, len(frames))


def _load_annotation_records(annotation_path: Path) -> List[dict]:
    ann_json = _read_json(annotation_path)
    if not isinstance(ann_json, dict):
        raise ValueError(f"annotations JSON root must be an object: {annotation_path}")

    ann_list = ann_json.get("annotations", [])
    if not isinstance(ann_list, list):
        raise ValueError(f"Invalid annotations format (annotations must be a list): {annotation_path}")

    records: List[dict] = []
    for rec in ann_list:
        if not isinstance(rec, dict):
            continue
        if bool(rec.get("is_error", False)):
            continue
        records.append(rec)
    return records


def _consume_sentence_for_pair(
    remaining_records: List[dict],
    person_id: int,
    label_id: int,
    annotation_path: Path,
) -> str:
    pid = int(person_id)
    lid = int(label_id)

    for idx, rec in enumerate(remaining_records):
        if not isinstance(rec, dict):
            continue
        try:
            r_pid = int(rec.get("person_id", -999999))
            r_lid = int(rec.get("label", -999999))
        except Exception:
            continue

        if r_pid != pid or r_lid != lid:
            continue

        sent = rec.get("sentence", None)
        if not isinstance(sent, str):
            raise ValueError(f"annotations record missing valid sentence: {annotation_path}")

        del remaining_records[idx]
        return sent

    raise ValueError(
        f"Cannot find matching annotation record for (person_id={pid}, label={lid}) in {annotation_path}"
    )


def write_video_sample(
    dataset_dir: Path,
    stem: str,
    person_label_pairs: List[Tuple[int, int]],
    T: int,
    J: int,
    W: int,
    H: int,
    MIN_FRAMES: int,
    MAX_P: int,
) -> bool:
    """
    1動画につき1ファイル（単一pt）を作成する。

    出力先（フラット）:
      <dataset_dir>/processed/pt/<stem>.pt

    pt（images は uint8）:
      images:    (P,T,3,H,W)  uint8  (0..255)
      skeletons: (P,T,J,2)    float32
      scores:    (P,T,J)      float32
      frames:    (P,T)        int64
      labels:    (P,D)        float32
      person_ids: list[int]
      label_ids:  list[int]
      label_sentences: list[str]
    """
    dataset_dir = Path(dataset_dir)

    raw_dir = dataset_dir / "raw"
    videos_dir = raw_dir / "videos"
    keypoints_dir = raw_dir / "keypoints"
    annotations_dir = raw_dir / "annotations"

    out_pt_dir = dataset_dir / "processed" / "pt"
    out_pt_dir.mkdir(parents=True, exist_ok=True)
    out_pt_path = out_pt_dir / f"{stem}.pt"

    video_path = _resolve_video_path(videos_dir, stem)
    if video_path is None:
        return False

    keypoints_path = keypoints_dir / f"{stem}.json"
    annotations_path = annotations_dir / f"{stem}.json"
    if not (keypoints_path.is_file() and annotations_path.is_file()):
        return False

    json_data = _read_json(keypoints_path)
    if not isinstance(json_data, dict):
        return False

    # MIN_FRAMES 判定（サンプリング前）
    video_frames = _get_video_frame_count(video_path)
    json_frames = _extract_json_frame_count(json_data)
    if min(video_frames, json_frames) < int(MIN_FRAMES):
        return False

    if not person_label_pairs:
        return False

    # MAX_P 適用（all_list 出現順）
    person_label_pairs = person_label_pairs[: int(MAX_P)]

    person_ids: List[int] = [int(pid) for pid, _ in person_label_pairs]
    label_ids: List[int] = [int(lid) for _, lid in person_label_pairs]

    # annotations から sentence を確定（重複があっても消費して一意に割り当て）
    remaining = _load_annotation_records(annotations_path)
    label_sentences: List[str] = []
    for pid, lid in person_label_pairs:
        s = _consume_sentence_for_pair(remaining, pid, lid, annotations_path)
        label_sentences.append(s)

    # 文埋め込み（人物ごと）
    emb_list = [_get_sentence_embedding(s) for s in label_sentences]
    labels = torch.stack(emb_list, dim=0).contiguous()  # (P,D) float32 CPU

    # データ生成
    frames, images, skeletons, scores = generate_video_sample(
        video_path=video_path,
        json_data=json_data,
        person_ids=person_ids,
        T=int(T),
        J=int(J),
        W=int(W),
        H=int(H),
    )

    person_has_keypoint = scores.max(dim=-1).values > 0  # (P,T)
    valid_len = person_has_keypoint.sum(dim=-1)          # (P,)

    keep = (valid_len >= config.MIN_VALID_T)

    # 全部落ちたらこの動画は捨てる（ptを作らない）
    if int(keep.sum().item()) == 0:
        return False
    
    keep_cpu = keep.detach().cpu()

    frames = frames[keep]
    images = images[keep]
    skeletons = skeletons[keep]
    scores = scores[keep]
    labels = labels[keep]

    # listも同じkeepで落とす
    keep_list = keep_cpu.tolist()
    person_ids = [pid for pid, k in zip(person_ids, keep_list) if k]
    label_ids = [lid for lid, k in zip(label_ids, keep_list) if k]
    label_sentences = [s for s, k in zip(label_sentences, keep_list) if k]

    # dtype/CPU/contiguous を保証
    frames = frames.detach().cpu().to(torch.long).contiguous()              # (P,T)
    images = images.detach().cpu().to(torch.uint8).contiguous()             # (P,T,3,H,W)
    skeletons = skeletons.detach().cpu().to(torch.float32).contiguous()     # (P,T,J,2)
    scores = scores.detach().cpu().to(torch.float32).contiguous()           # (P,T,J)

    torch.save(
        {
            "images": images,
            "skeletons": skeletons,
            "scores": scores,
            "frames": frames,
            "labels": labels,
            "person_ids": person_ids,
            "label_ids": label_ids,
            "label_sentences": label_sentences,
            "video_stem": str(stem),
        },
        out_pt_path,
    )

    return True
