# dataset_builder/main.py

from __future__ import annotations

import math
import random
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
    video_pt: str  # "<stem>.pt"
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

        rows.append(
            AllListRow(
                video_pt=video_pt,
                person_id=int(pid_s),
                label_id=int(lid_s),
            )
        )
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

        jobs.append(
            (
                _to_path(dataset_dir),
                stem,
                person_label_pairs,  # [(person_id, label_id), ...]
                int(config.T),
                int(config.J),
                int(config.W),
                int(config.H),
                min_frames,
                max_p,
            )
        )
    return jobs


def _worker(args):
    return write_video_sample(*args)


def _normalize_ratios(train_r: float, val_r: float, test_r: float) -> Dict[str, float]:
    for name, r in [("TRAIN_RATIO", train_r), ("VAL_RATIO", val_r), ("TEST_RATIO", test_r)]:
        if not isinstance(r, (int, float)) or math.isnan(float(r)) or math.isinf(float(r)):
            raise ValueError(f"{name} must be a finite number. got: {r!r}")
        if float(r) < 0:
            raise ValueError(f"{name} must be >= 0. got: {r!r}")

    s = float(train_r) + float(val_r) + float(test_r)
    if s <= 0:
        raise ValueError(f"TRAIN_RATIO+VAL_RATIO+TEST_RATIO must be > 0. got sum={s}")

    return {
        "train": float(train_r) / s,
        "val": float(val_r) / s,
        "test": float(test_r) / s,
    }


def _allocate_counts(total: int, ratios: Dict[str, float]) -> Dict[str, int]:
    """
    total を ratios に従って整数配分する（largest remainder method）。
    """
    if total < 0:
        raise ValueError(f"total must be >=0. got: {total}")

    keys = ["train", "val", "test"]
    raw = {k: total * float(ratios[k]) for k in keys}
    floors = {k: int(math.floor(raw[k])) for k in keys}
    rem = {k: raw[k] - floors[k] for k in keys}

    used = sum(floors.values())
    left = total - used
    order = sorted(keys, key=lambda k: (-rem[k], k))  # tie: stable by name

    out = dict(floors)
    i = 0
    while left > 0:
        k = order[i % len(order)]
        out[k] += 1
        left -= 1
        i += 1
    return out


def _collect_existing_pt_stems(dataset_dir: Path) -> set[str]:
    pt_dir = _to_path(dataset_dir) / "processed" / "pt"
    if not pt_dir.is_dir():
        return set()
    stems = set()
    for p in pt_dir.glob("*.pt"):
        if p.is_file():
            stems.add(p.stem)
    return stems


def build_split_lists(dataset_dir: Path, all_list_path: Path) -> None:
    """
    all_list.txt（人物行）をもとに、動画（pt）単位で split を作る。
    出力（2列）:
      processed/splits/train_list.txt
      processed/splits/val_list.txt
      processed/splits/test_list.txt

    各行: "<stem>.pt 0"
    """
    dataset_dir = _to_path(dataset_dir)
    ensure_processed_dirs(dataset_dir)

    train_r = globals().get("TRAIN_RATIO", None)
    val_r = globals().get("VAL_RATIO", None)
    test_r = globals().get("TEST_RATIO", None)
    if train_r is None or val_r is None or test_r is None:
        raise RuntimeError("TRAIN_RATIO / VAL_RATIO / TEST_RATIO が定義されていません（config_base.py を確認してください）。")

    ratios = _normalize_ratios(float(train_r), float(val_r), float(test_r))
    seed = int(globals().get("SPLIT_SEED", 42))

    existing_stems = _collect_existing_pt_stems(dataset_dir)
    if not existing_stems:
        print(f"[split] skip: no pt files found: {dataset_dir / 'processed' / 'pt'}")
        return

    # stem -> label_id -> count（人物数）
    video_label_counts: Dict[str, Dict[int, int]] = {}
    label_totals: Dict[int, int] = defaultdict(int)

    rows = read_all_list(all_list_path)
    for row in rows:
        stem = Path(row.video_pt).stem
        if stem not in existing_stems:
            continue
        d = video_label_counts.setdefault(stem, defaultdict(int))  # type: ignore[assignment]
        d[int(row.label_id)] += 1  # persons per label in this video

    # defaultdict を通常dictへ（以降の型の扱いを簡単にする）
    video_label_counts = {stem: dict(cnts) for stem, cnts in video_label_counts.items()}

    if not video_label_counts:
        print(f"[split] skip: no videos after filtering by existing pt: {dataset_dir}")
        return

    for cnts in video_label_counts.values():
        for lid, c in cnts.items():
            label_totals[int(lid)] += int(c)

    labels = sorted(label_totals.keys())
    if not labels:
        print(f"[split] skip: no labels found in all_list after filtering: {all_list_path}")
        return

    # 重み（レアラベルを少し強める）
    weights: Dict[int, float] = {lid: 1.0 / max(1, int(label_totals[lid])) for lid in labels}

    # splitごとの目標ラベル数（人物カウント）
    target_label: Dict[str, Dict[int, float]] = {
        s: {lid: float(ratios[s]) * float(label_totals[lid]) for lid in labels}
        for s in ["train", "val", "test"]
    }

    # splitごとの目標動画数
    stems_all = sorted(video_label_counts.keys())
    N = len(stems_all)
    target_n = _allocate_counts(N, ratios)

    # 動画の並び順：レアラベルを多く含むものを先に割り当てる
    rng = random.Random(seed)
    items = list(video_label_counts.items())
    rng.shuffle(items)

    def rarity_score(counts: Dict[int, int]) -> float:
        s = 0.0
        for lid, c in counts.items():
            s += float(c) * weights.get(int(lid), 0.0)
        return s

    def person_total(counts: Dict[int, int]) -> int:
        return int(sum(int(x) for x in counts.values()))

    items.sort(
        key=lambda kv: (
            -rarity_score(kv[1]),
            -person_total(kv[1]),
            kv[0],
        )
    )

    cur_n: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
    cur_label: Dict[str, Dict[int, int]] = {
        "train": defaultdict(int),
        "val": defaultdict(int),
        "test": defaultdict(int),
    }  # type: ignore[assignment]

    assigned: Dict[str, List[str]] = {"train": [], "val": [], "test": []}

    def delta_cost(split: str, counts_v: Dict[int, int]) -> float:
        """
        split に counts_v を足したときの L1誤差（重み付き）の増分を返す。
        """
        d = 0.0
        cur = cur_label[split]
        tgt = target_label[split]
        for lid, c in counts_v.items():
            lid = int(lid)
            c = int(c)
            cur_l = int(cur.get(lid, 0))
            tgt_l = float(tgt[lid])
            w = float(weights.get(lid, 1.0))
            d += w * (abs((cur_l + c) - tgt_l) - abs(cur_l - tgt_l))
        return d

    for stem, counts_v in items:
        candidates = [s for s in ["train", "val", "test"] if cur_n[s] < int(target_n[s])]
        if not candidates:
            candidates = ["train", "val", "test"]

        scored: List[Tuple[float, int, str]] = []
        for s in candidates:
            dc = delta_cost(s, counts_v)
            remaining = int(target_n[s]) - int(cur_n[s])
            scored.append((dc, -remaining, s))  # remaining大を優先（-でソート簡略化）

        scored.sort(key=lambda x: (x[0], x[1], x[2]))
        best_split = scored[0][2]

        assigned[best_split].append(stem)
        cur_n[best_split] += 1
        cur = cur_label[best_split]
        for lid, c in counts_v.items():
            cur[int(lid)] += int(c)

    splits_dir = dataset_dir / "processed" / "splits"

    def write_list(path: Path, stems: List[str]) -> None:
        stems = sorted(stems)
        lines = [f"{stem}.pt 0" for stem in stems]
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    write_list(splits_dir / "train_list.txt", assigned["train"])
    write_list(splits_dir / "val_list.txt", assigned["val"])
    write_list(splits_dir / "test_list.txt", assigned["test"])

    print(
        f"[split] {dataset_dir.name}: "
        f"train={len(assigned['train'])}, val={len(assigned['val'])}, test={len(assigned['test'])} "
        f"(target train={target_n['train']}, val={target_n['val']}, test={target_n['test']})"
    )


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

    # split list 生成（pt作成後）
    for ds_dir, all_list_path in all_list_paths.items():
        build_split_lists(ds_dir, all_list_path)

    print("Done.")


if __name__ == "__main__":
    freeze_support()
    process_all(
        _resolve_datasets_root(),
        int(getattr(config, "NUM_WORKERS", 4)),
    )
