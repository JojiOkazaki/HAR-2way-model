# dataset_builder/main.py

from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from multiprocessing import Pool, freeze_support
from pathlib import Path
from typing import Dict, List, Tuple

import orjson
from tqdm import tqdm

import dataset_builder.config as config
from dataset_builder.script_writer import write_video_sample

from config_base import *  # noqa: F401,F403
try:
    from config_local import *  # noqa: F401,F403
except ImportError:
    pass


VIDEO_EXTS = {".avi", ".mp4"}


@dataclass(frozen=True)
class AllListRow:
    video_pt: str     # "<stem>.pt"
    person_id: int
    label_id: int


def _to_path(p) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _resolve_datasets_root() -> Path:
    # 優先順: DATASETS_ROOT -> DATASET_ROOT -> RAW_DATA_ROOT
    root = globals().get("DATASETS_ROOT", None)
    if root is None:
        root = globals().get("DATASET_ROOT", None)
    if root is None:
        root = globals().get("RAW_DATA_ROOT", None)
    if root is None:
        raise RuntimeError("DATASETS_ROOT / DATASET_ROOT / RAW_DATA_ROOT のいずれも定義されていません。")
    return _to_path(root)


def _is_dataset_dir(dataset_dir: Path) -> bool:
    raw_dir = dataset_dir / "raw"
    return (
        dataset_dir.is_dir()
        and (raw_dir / "videos").is_dir()
        and (raw_dir / "keypoints").is_dir()
        and (raw_dir / "annotations").is_dir()
    )


def find_dataset_dirs(datasets_root: Path) -> List[Path]:
    """
    datasets_root が
      - データセット親ディレクトリ（katorilab/, ucf101/ が並ぶ）でも
      - 単一データセットディレクトリ（raw/ を持つ）でも
    動くようにする。
    """
    datasets_root = _to_path(datasets_root)
    if not datasets_root.exists():
        raise FileNotFoundError(f"datasets_root not found: {datasets_root}")

    if _is_dataset_dir(datasets_root):
        return [datasets_root]

    if not datasets_root.is_dir():
        raise NotADirectoryError(f"datasets_root is not a directory: {datasets_root}")

    ds_dirs: List[Path] = []
    for child in sorted(datasets_root.iterdir(), key=lambda p: p.name):
        if _is_dataset_dir(child):
            ds_dirs.append(child)
    return ds_dirs


def ensure_processed_dirs(dataset_dir: Path) -> None:
    (dataset_dir / "processed" / "pt").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "processed" / "splits").mkdir(parents=True, exist_ok=True)


def _collect_video_files(videos_dir: Path) -> Dict[str, Path]:
    """
    raw/videos から stem -> video_path を作る。
    同一stemで .avi と .mp4 が共存していたらエラー。
    """
    by_stem: Dict[str, List[Path]] = defaultdict(list)

    for p in videos_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in VIDEO_EXTS:
            continue
        by_stem[p.stem].append(p)

    dup = {stem: paths for stem, paths in by_stem.items() if len(paths) >= 2}
    if dup:
        msg = ["raw/videos に同一stemの動画が複数あります（.avi と .mp4 の共存を想定していません）。"]
        for stem, paths in sorted(dup.items(), key=lambda x: x[0]):
            msg.append(f"- {stem}: " + ", ".join(str(pp) for pp in sorted(paths)))
        raise ValueError("\n".join(msg))

    return {stem: paths[0] for stem, paths in by_stem.items()}


def _read_json(path: Path) -> dict:
    with open(path, "rb") as f:
        obj = orjson.loads(f.read())
    if not isinstance(obj, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return obj


def build_all_list(dataset_dir: Path) -> Path:
    """
    processed/splits/all_list.txt を自動生成する。

    all_list.txt の各行:
      "<stem>.pt <person_id> <label_id>"

    対象 stem は videos/keypoints/annotations が揃っているもののみ。
    katorilab 等の is_error=true は除外。
    1動画あたり MAX_P を超える人物は除外（annotations配列順で先頭から MAX_P まで）。
    """
    dataset_dir = _to_path(dataset_dir)
    ensure_processed_dirs(dataset_dir)

    raw_dir = dataset_dir / "raw"
    videos_dir = raw_dir / "videos"
    keypoints_dir = raw_dir / "keypoints"
    annotations_dir = raw_dir / "annotations"

    splits_dir = dataset_dir / "processed" / "splits"
    all_list_path = splits_dir / "all_list.txt"

    video_map = _collect_video_files(videos_dir)
    keypoint_stems = {p.stem for p in keypoints_dir.glob("*.json") if p.is_file()}
    annotation_stems = {p.stem for p in annotations_dir.glob("*.json") if p.is_file()}

    common_stems = sorted((set(video_map.keys()) & keypoint_stems & annotation_stems))

    max_p = int(getattr(config, "MAX_P", 30))

    lines: List[str] = []
    for stem in common_stems:
        ann_path = annotations_dir / f"{stem}.json"
        ann_json = _read_json(ann_path)

        ann_list = ann_json.get("annotations", [])
        if not isinstance(ann_list, list):
            raise ValueError(f"Invalid annotations format (annotations must be a list): {ann_path}")

        kept = 0
        for rec in ann_list:
            if kept >= max_p:
                break
            if not isinstance(rec, dict):
                continue
            if bool(rec.get("is_error", False)):
                continue
            if "person_id" not in rec or "label" not in rec:
                raise ValueError(f"annotations record must have person_id and label: {ann_path}")

            person_id = int(rec["person_id"])
            label_id = int(rec["label"])

            lines.append(f"{stem}.pt {person_id} {label_id}")
            kept += 1

    all_list_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return all_list_path


def read_all_list(all_list_path: Path) -> List[AllListRow]:
    all_list_path = _to_path(all_list_path)
    if not all_list_path.is_file():
        raise FileNotFoundError(f"all_list.txt not found: {all_list_path}")

    rows: List[AllListRow] = []
    for line_no, raw_line in enumerate(all_list_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 3:
            raise ValueError(f"{all_list_path}:{line_no}: expected 3 columns, got {len(parts)}: {raw_line!r}")
        video_pt, pid_s, lid_s = parts
        if not video_pt.lower().endswith(".pt"):
            raise ValueError(f"{all_list_path}:{line_no}: first column must end with .pt: {raw_line!r}")

        rows.append(AllListRow(
            video_pt=video_pt,
            person_id=int(pid_s),
            label_id=int(lid_s),
        ))
    return rows


def build_jobs_from_all_list(dataset_dir: Path, all_list_path: Path) -> List[Tuple]:
    """
    all_list を動画単位にまとめて 1動画=1job にする。
    person 次元の順序は all_list の出現順。
    """
    rows = read_all_list(all_list_path)

    grouped: "OrderedDict[str, List[Tuple[int, int]]]" = OrderedDict()
    for row in rows:
        stem = Path(row.video_pt).stem
        grouped.setdefault(stem, []).append((row.person_id, row.label_id))

    min_frames = int(getattr(config, "MIN_FRAMES", 30))
    max_p = int(getattr(config, "MAX_P", 30))

    jobs: List[Tuple] = []
    for stem, person_label_pairs in grouped.items():
        # 念のためここでも MAX_P を適用（all_list側で適用済みでも安全）
        person_label_pairs = person_label_pairs[:max_p]

        jobs.append((
            _to_path(dataset_dir),
            stem,
            person_label_pairs,        # [(person_id, label_id), ...]
            int(config.T),
            int(config.J),
            int(config.W),
            int(config.H),
            min_frames,
            max_p,
        ))
    return jobs


def _worker(args):
    return write_video_sample(*args)


def process_all(datasets_root: Path, num_workers: int) -> None:
    datasets_root = _to_path(datasets_root)
    dataset_dirs = find_dataset_dirs(datasets_root)
    if not dataset_dirs:
        raise RuntimeError(f"有効なデータセットディレクトリが見つかりません: {datasets_root}")

    # all_list 生成
    all_list_paths: Dict[Path, Path] = {}
    for ds_dir in dataset_dirs:
        ensure_processed_dirs(ds_dir)
        all_list_paths[ds_dir] = build_all_list(ds_dir)

    # jobs 作成
    jobs: List[Tuple] = []
    for ds_dir, all_list_path in all_list_paths.items():
        jobs.extend(build_jobs_from_all_list(ds_dir, all_list_path))

    print(f"Datasets: {len(dataset_dirs)}")
    print(f"Total videos (jobs): {len(jobs)}")
    print(
        f"Using {num_workers} workers, "
        f"T={config.T}, J={config.J}, W={config.W}, H={config.H}, "
        f"MIN_FRAMES={getattr(config, 'MIN_FRAMES', 30)}, MAX_P={getattr(config, 'MAX_P', 30)}"
    )

    if not jobs:
        print("No jobs to process.")
        return

    with Pool(processes=num_workers) as pool:
        for _ in tqdm(pool.imap_unordered(_worker, jobs), total=len(jobs)):
            pass

    print("Done.")


if __name__ == "__main__":
    freeze_support()
    process_all(
        _resolve_datasets_root(),
        int(getattr(config, "NUM_WORKERS", 4)),
    )
