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
    MAX_P_CANDIDATE: int,
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

    # 候補人数上限（出現順）: ここでは「候補」を多めに保持し、後段で“主人物優先”で絞る
    max_p_candidate = int(MAX_P_CANDIDATE)
    if max_p_candidate <= 0:
        return False
    person_label_pairs = person_label_pairs[:max_p_candidate]

    # 最終的に保存する P（学習に使う P 上限）
    max_p_out = int(getattr(config, "MAX_P", 3))
    if max_p_out <= 0:
        return False

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

    # データ生成（候補人物ぶん生成）
    frames, images, skeletons, scores = generate_video_sample(
        video_path=video_path,
        json_data=json_data,
        person_ids=person_ids,
        T=int(T),
        J=int(J),
        W=int(W),
        H=int(H),
    )

    # 有効フレーム数（人物ごと）
    person_has_keypoint = scores.max(dim=-1).values > 0  # (P,T)
    valid_len = person_has_keypoint.sum(dim=-1)          # (P,)

    keep = (valid_len >= int(getattr(config, "MIN_VALID_T", 16)))

    # 全部落ちたらこの動画は捨てる（ptを作らない）
    if int(keep.sum().item()) == 0:
        return False

    keep_idx = keep.nonzero(as_tuple=False).squeeze(1)  # (K,)

    # “主人物”優先で上位Pを選ぶ:
    #  1) valid_len が大きいほど優先
    #  2) 同程度なら平均 score が高いほど優先
    scores_kept = scores.index_select(0, keep_idx)  # (K,T,J)
    mean_score = scores_kept.mean(dim=(1, 2))       # (K,)
    valid_kept = valid_len.index_select(0, keep_idx).to(mean_score.dtype)  # (K,)

    composite = valid_kept * 1000.0 + mean_score  # valid_len を主にする
    k_out = min(int(max_p_out), int(keep_idx.numel()))
    if k_out <= 0:
        return False

    top_in_kept = torch.topk(composite, k=k_out, largest=True, sorted=True).indices  # (k_out,)
    select_idx = keep_idx.index_select(0, top_in_kept)  # (k_out,)

    # Tensor を選別（同じインデックスで揃える）
    frames = frames.index_select(0, select_idx)
    images = images.index_select(0, select_idx)
    skeletons = skeletons.index_select(0, select_idx)
    scores = scores.index_select(0, select_idx)
    labels = labels.index_select(0, select_idx)

    # list も同じインデックスで揃える
    sel = [int(i) for i in select_idx.tolist()]
    person_ids = [person_ids[i] for i in sel]
    label_ids = [label_ids[i] for i in sel]
    label_sentences = [label_sentences[i] for i in sel]

    # dtype/CPU/contiguous を保証
    frames = frames.detach().cpu().to(torch.long).contiguous()              # (P,T)
    images = images.detach().cpu().to(torch.uint8).contiguous()             # (P,T,3,H,W)
    skeletons = skeletons.detach().cpu().to(torch.float32).contiguous()     # (P,T,J,2)
    scores = scores.detach().cpu().to(torch.float32).contiguous()           # (P,T,J)
    labels = labels.detach().cpu().to(torch.float32).contiguous()           # (P,D)

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
