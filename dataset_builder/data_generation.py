# dataset_builder/data_generation.py

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from dataset_builder.video_frame_utils import sample_frame_ids
from dataset_builder.video_frame_utils import read_frames_by_id
from dataset_builder.video_frame_utils import find_closest_past_frame


def _extract_json_frame_count(json_data: dict) -> int:
    num_frames = int(json_data.get("num_frames", 0))
    frames = json_data.get("frames", [])
    if not isinstance(frames, list):
        return 0
    return min(num_frames, len(frames))


def _is_valid_person_record(person: dict, J: int) -> bool:
    if not isinstance(person, dict):
        return False
    if person.get("person_id", None) is None:
        return False
    bbox = person.get("bbox_xyxy", None)
    kps = person.get("keypoints_coco", None)
    if bbox is None or kps is None:
        return False
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False
    if not isinstance(kps, list) or len(kps) < int(J):
        return False
    return True


def _crop_person_image_u8(img_rgb: np.ndarray, bbox_xyxy: List[float], out_size: Tuple[int, int]) -> np.ndarray:
    """
    img_rgb: (H_img, W_img, 3) uint8 RGB
    bbox_xyxy: [x1,y1,x2,y2] (float/int)
    out_size: (out_w, out_h)

    returns: (3, out_h, out_w) uint8
    """
    h, w = img_rgb.shape[:2]
    out_w, out_h = int(out_size[0]), int(out_size[1])
    x1, y1, x2, y2 = bbox_xyxy

    x1 = int(max(0, min(w - 1, x1)))
    x2 = int(max(0, min(w, x2)))
    y1 = int(max(0, min(h - 1, y1)))
    y2 = int(max(0, min(h, y2)))

    if x2 <= x1 or y2 <= y1:
        return np.zeros((3, out_h, out_w), dtype=np.uint8)

    crop = img_rgb[y1:y2, x1:x2]
    crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)  # (out_h,out_w,3)
    crop = np.transpose(crop, (2, 0, 1)).astype(np.uint8, copy=False)        # (3,out_h,out_w)
    return crop


def normalize_skeleton_person_based(
    kps: np.ndarray,
    scores: np.ndarray,
    *,
    score_thr: float = 0.05,
    min_scale: float = 1e-3,
    max_abs_norm: float = 5.0,
    use_fallback_hip_scale: bool = True,
    return_ok: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, bool]]:
    """
    COCO17 前提の人物基準正規化:
      - hip_center で平行移動
      - shoulder 幅（両肩が一定以上の信頼度なら）でスケール
      - shoulder が使えない場合は hip 間距離でフォールバック（use_fallback_hip_scale=True）
      - 正規化後の座標が極端（|x| が max_abs_norm を超える等）なら無効扱い

    kps:    (J,2) float
    scores: (J,)  float

    returns:
      - return_ok=False: (J,2) float32
      - return_ok=True : ((J,2) float32, ok: bool)
        ok=False の場合はゼロ配列を返す
    """
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6

    def _zeros_like_input() -> np.ndarray:
        try:
            return np.zeros_like(kps, dtype=np.float32)
        except Exception:
            j = int(scores.shape[0]) if hasattr(scores, "shape") else 0
            return np.zeros((j, 2), dtype=np.float32)

    kps = np.asarray(kps, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)

    ok = True

    if kps.ndim != 2 or kps.shape[1] != 2 or scores.ndim != 1:
        ok = False
    elif kps.shape[0] != scores.shape[0]:
        ok = False
    elif kps.shape[0] <= max(LEFT_HIP, RIGHT_HIP, LEFT_SHOULDER, RIGHT_SHOULDER):
        ok = False
    elif not np.isfinite(kps).all() or not np.isfinite(scores).all():
        ok = False

    if not ok:
        out = _zeros_like_input()
        return (out, False) if return_ok else out

    # hip 必須（平行移動の基準）
    if scores[LEFT_HIP] <= 0 or scores[RIGHT_HIP] <= 0:
        out = _zeros_like_input()
        return (out, False) if return_ok else out

    hip_center = 0.5 * (kps[LEFT_HIP] + kps[RIGHT_HIP])
    kps_c = kps - hip_center  # centered

    # scale candidates
    shoulder_ok = (scores[LEFT_SHOULDER] >= float(score_thr)) and (scores[RIGHT_SHOULDER] >= float(score_thr))
    scale: Optional[float] = None

    if shoulder_ok:
        scale = float(np.linalg.norm(kps_c[LEFT_SHOULDER] - kps_c[RIGHT_SHOULDER]))

    if (scale is None) and bool(use_fallback_hip_scale):
        scale = float(np.linalg.norm(kps[LEFT_HIP] - kps[RIGHT_HIP]))

    if (scale is None) or (not np.isfinite(scale)) or (scale < float(min_scale)):
        out = _zeros_like_input()
        return (out, False) if return_ok else out

    kps_n = kps_c / float(scale)

    if (not np.isfinite(kps_n).all()) or (float(np.abs(kps_n).max()) > float(max_abs_norm)):
        out = _zeros_like_input()
        return (out, False) if return_ok else out

    out = kps_n.astype(np.float32, copy=False)
    return (out, True) if return_ok else out


def generate_video_sample(
    video_path,
    json_data: dict,
    person_ids: List[int],
    T: int,
    J: int,
    W: int,
    H: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    1動画から、指定 person_ids 分のデータを生成する。

    returns:
      frames:    (P,T)        int64
      images:    (P,T,3,H,W)  uint8  (0..255)
      skeletons: (P,T,J,2)    float32
      scores:    (P,T,J)      float32
    """
    if not isinstance(json_data, dict):
        raise ValueError("json_data must be a dict.")
    if not isinstance(person_ids, list) or len(person_ids) == 0:
        raise ValueError("person_ids must be a non-empty list[int].")

    cap = cv2.VideoCapture(str(video_path))
    try:
        video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()

    json_frames = _extract_json_frame_count(json_data)
    common_frames = min(video_frames, json_frames)
    if common_frames <= 0:
        raise RuntimeError("No usable frames (min(video_frames, json_frames) <= 0).")

    frames_list = json_data.get("frames", [])
    if not isinstance(frames_list, list):
        raise ValueError("json_data['frames'] must be a list.")

    P = len(person_ids)
    pid_list = [int(pid) for pid in person_ids]
    pid_set = set(pid_list)

    # personごとの出現区間（valid record がある最初と最後）+ 実在フレーム列
    first_seen: Dict[int, Optional[int]] = {pid: None for pid in pid_set}
    last_seen: Dict[int, Optional[int]] = {pid: None for pid in pid_set}
    seen_frames: Dict[int, List[int]] = {pid: [] for pid in pid_set}

    for f in range(common_frames):
        frame_obj = frames_list[f]
        if not isinstance(frame_obj, dict):
            continue
        persons = frame_obj.get("persons", [])
        if not isinstance(persons, list):
            continue

        for p in persons:
            if not isinstance(p, dict):
                continue
            pid = p.get("person_id", None)
            if pid is None:
                continue
            try:
                pid_i = int(pid)
            except Exception:
                continue
            if pid_i not in pid_set:
                continue
            if not _is_valid_person_record(p, int(J)):
                continue

            if first_seen[pid_i] is None:
                first_seen[pid_i] = f
            last_seen[pid_i] = f
            seen_frames[pid_i].append(f)

    # personごとのフレーム列 (P,T)
    per_person_frames: List[List[int]] = []
    for pid in pid_list:
        seen = seen_frames.get(int(pid), [])
        if len(seen) > 0:
            # 実際に person_id が存在したフレーム列からサンプルする
            rel = sample_frame_ids([len(seen)], int(T))
            fids = [int(seen[int(r)]) for r in rel]
            per_person_frames.append(fids)
            continue

        # 1度も出てこないperson_id（基本は後段で落ちる）
        fids = sample_frame_ids([common_frames], int(T))
        fids = [int(max(0, min(common_frames - 1, x))) for x in fids]
        per_person_frames.append(fids)

    # 動画から必要フレームをまとめて読む（setで高速化）
    unique_frames = sorted({fid for fids in per_person_frames for fid in fids})
    imgs = read_frames_by_id(str(video_path), set(unique_frames))  # {frame_id: RGB uint8}

    # フレームごとの pid->person を作る（必要フレームだけ）
    frame_pid_map: Dict[int, Dict[int, dict]] = {}
    for fid in unique_frames:
        if fid >= common_frames:
            continue
        frame_obj = frames_list[fid]
        if not isinstance(frame_obj, dict):
            continue
        persons = frame_obj.get("persons", [])
        if not isinstance(persons, list):
            continue

        m: Dict[int, dict] = {}
        for p in persons:
            if not isinstance(p, dict):
                continue
            pid = p.get("person_id", None)
            if pid is None:
                continue
            try:
                pid_i = int(pid)
            except Exception:
                continue
            if pid_i not in pid_set:
                continue
            if not _is_valid_person_record(p, int(J)):
                continue
            if pid_i not in m:
                m[pid_i] = p
        frame_pid_map[fid] = m

    # 出力（ゼロ初期化）
    frames_out = np.zeros((P, int(T)), dtype=np.int64)
    images_out = np.zeros((P, int(T), 3, int(H), int(W)), dtype=np.uint8)
    skel_out = np.zeros((P, int(T), int(J), 2), dtype=np.float32)
    score_out = np.zeros((P, int(T), int(J)), dtype=np.float32)

    for p_idx, pid in enumerate(pid_list):
        fids = per_person_frames[p_idx]

        for t_idx, fid in enumerate(fids):
            frames_out[p_idx, t_idx] = int(fid)

            img = imgs.get(int(fid), None)
            if img is None:
                try:
                    alt = find_closest_past_frame(unique_frames, int(fid))
                    img = imgs.get(int(alt), None)
                except Exception:
                    img = None
            if img is None:
                continue

            person_rec = frame_pid_map.get(int(fid), {}).get(pid, None)
            if person_rec is None:
                continue

            bbox = person_rec.get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0])
            kps = person_rec.get("keypoints_coco", None)
            if (not isinstance(bbox, list)) or len(bbox) != 4 or (not isinstance(kps, list)) or len(kps) < int(J):
                continue

            images_out[p_idx, t_idx] = _crop_person_image_u8(img, bbox, (int(W), int(H)))

            kps_slice = kps[: int(J)]
            try:
                sk = np.array([row[:2] for row in kps_slice], dtype=np.float32)  # (J,2)
                sc = np.array([row[2] for row in kps_slice], dtype=np.float32)   # (J,)
            except Exception:
                continue

            sk_n, ok = normalize_skeleton_person_based(sk, sc, return_ok=True)
            skel_out[p_idx, t_idx] = sk_n

            if ok:
                score_out[p_idx, t_idx] = sc
            else:
                score_out[p_idx, t_idx] = 0.0

    frames_tensor = torch.from_numpy(frames_out).to(dtype=torch.long).contiguous()
    images_tensor = torch.from_numpy(images_out).to(dtype=torch.uint8).contiguous()
    skel_tensor = torch.from_numpy(skel_out).to(dtype=torch.float32).contiguous()
    score_tensor = torch.from_numpy(score_out).to(dtype=torch.float32).contiguous()

    return frames_tensor, images_tensor, skel_tensor, score_tensor
