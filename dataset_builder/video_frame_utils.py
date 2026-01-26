# dataset_builder/video_frame_utils.py

from __future__ import annotations

from bisect import bisect_right
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Union

import cv2
import numpy as np


PathLike = Union[str, Path]


def sample_frame_ids(segment_lens: List[int], T: int) -> List[int]:
    """
    フレームインデックスのサンプリング（0始まりの整数）。

    使い方（本プロジェクト想定）:
      - segment_lens=[N] のとき、0..N-1 から T 個を等間隔にサンプル
      - N < T のときは重複を許容（端の値が繰り返される）

    戻り値:
      - 長さ T の List[int]
      - 各値は 0..(sum(segment_lens)-1) の範囲

    注意:
      本プロジェクトでは segment_lens はほぼ [N] で使われる想定ですが、
      複数セグメントが来ても動くように「連結した1本の区間」として扱います。
    """
    if not isinstance(T, int) or T <= 0:
        raise ValueError(f"T must be a positive int, got {T!r}")
    if not isinstance(segment_lens, list) or len(segment_lens) == 0:
        raise ValueError("segment_lens must be a non-empty list[int]")

    lens = [int(x) for x in segment_lens]
    total = sum(max(0, x) for x in lens)
    if total <= 0:
        return [0] * T

    if total == 1:
        return [0] * T

    # 等間隔サンプル（端点含む）
    xs = np.linspace(0, total - 1, num=T, dtype=np.float64)
    ids = np.rint(xs).astype(np.int64)
    ids = np.clip(ids, 0, total - 1).astype(np.int64)
    return [int(x) for x in ids.tolist()]


def find_closest_past_frame(sorted_frame_ids: Sequence[int], target_frame_id: int) -> int:
    """
    sorted_frame_ids（昇順）から target_frame_id 以下で最大の要素を返す。
    すべて target より大きい場合は最小要素を返す。

    sorted_frame_ids が空なら例外。
    """
    if not sorted_frame_ids:
        raise ValueError("sorted_frame_ids must be non-empty")

    xs = sorted_frame_ids
    t = int(target_frame_id)

    # rightmost index where xs[i] <= t
    i = bisect_right(xs, t) - 1
    if i < 0:
        return int(xs[0])
    return int(xs[i])


def _ensure_int_ids(frame_ids: Iterable[int]) -> List[int]:
    ids: List[int] = []
    for x in frame_ids:
        try:
            xi = int(x)
        except Exception:
            continue
        if xi < 0:
            continue
        ids.append(xi)
    # unique + sort
    return sorted(set(ids))


def read_frames_by_id(video_path: PathLike, frame_ids) -> Dict[int, np.ndarray]:
    vp = str(video_path)
    ids = _ensure_int_ids(frame_ids)
    if len(ids) == 0:
        return {}

    ids.sort()

    cap = cv2.VideoCapture(vp)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {vp}")

    # gap がこれより大きい場合は grab で追わずに seek する
    SEEK_GAP = 240

    out: Dict[int, np.ndarray] = {}
    try:
        pos = None  # 次に read されるフレーム番号（概念上）

        for fid in ids:
            fid = int(fid)

            if pos is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
                pos = fid

            gap = fid - pos

            if gap < 0:
                # 戻る場合は seek
                cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
                pos = fid
            elif gap > 0:
                if gap <= SEEK_GAP:
                    # 近いなら grab でスキップ
                    for _ in range(gap):
                        ok = cap.grab()
                        if not ok:
                            break
                    pos = fid
                else:
                    # 遠いなら seek
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
                    pos = fid

            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                # 失敗したら次へ（動画によっては末尾近辺で失敗することがある）
                pos = fid + 1
                continue

            pos = fid + 1

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if frame_rgb.dtype != np.uint8:
                frame_rgb = frame_rgb.astype(np.uint8, copy=False)
            out[fid] = frame_rgb

    finally:
        cap.release()

    return out


