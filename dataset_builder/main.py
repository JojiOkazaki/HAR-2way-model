# dataset_builder/main.py

from __future__ import annotations

import math
import random
import re
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from multiprocessing import Pool, freeze_support
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Optional, Tuple

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

_UCF_GROUP_RE = re.compile(r"_g(\d+)_c(\d+)", re.IGNORECASE)

_ORIG_TRAINLIST_RE = re.compile(r"^trainlist(\d+)\.txt$", re.IGNORECASE)
_ORIG_TESTLIST_RE = re.compile(r"^testlist(\d+)\.txt$", re.IGNORECASE)


def _extract_ucf_group_id(stem: str) -> int | None:
    """stem から UCF101 の group id (gXX) を抽出する。"""
    m = _UCF_GROUP_RE.search(stem)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


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


def ensure_processed_dirs(dataset_dir: Path, subsplit_name: str | None = None) -> None:
    """
    processed/pt と processed/splits を作成する。
    subsplit_name が指定されていれば processed/splits/<subsplit_name>/ も作成する。
    """
    dataset_dir = _to_path(dataset_dir)

    (dataset_dir / "processed" / "pt").mkdir(parents=True, exist_ok=True)

    splits_root = dataset_dir / "processed" / "splits"
    splits_root.mkdir(parents=True, exist_ok=True)

    if subsplit_name is not None:
        (splits_root / subsplit_name).mkdir(parents=True, exist_ok=True)


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


@dataclass(frozen=True)
class AllListRow:
    video_pt: str  # "<stem>.pt"
    person_id: int
    label_id: int


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

    splits_root = dataset_dir / "processed" / "splits"
    all_list_path = splits_root / "all_list.txt"

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


def _normalize_train_val_ratios(train_r: float, val_r: float) -> Dict[str, float]:
    for name, r in [("TRAIN_RATIO", train_r), ("VAL_RATIO", val_r)]:
        if not isinstance(r, (int, float)) or math.isnan(float(r)) or math.isinf(float(r)):
            raise ValueError(f"{name} must be a finite number. got: {r!r}")
        if float(r) < 0:
            raise ValueError(f"{name} must be >= 0. got: {r!r}")

    s = float(train_r) + float(val_r)
    if s <= 0:
        raise ValueError(f"TRAIN_RATIO+VAL_RATIO must be > 0. got sum={s}")

    return {
        "train": float(train_r) / s,
        "val": float(val_r) / s,
    }


def _allocate_counts_train_val(total: int, ratios_tv: Dict[str, float]) -> Dict[str, int]:
    """total を train/val の2分割比率に従って整数配分する（largest remainder method）。"""
    if total < 0:
        raise ValueError(f"total must be >=0. got: {total}")
    if "train" not in ratios_tv or "val" not in ratios_tv:
        raise ValueError(f"ratios_tv must have train and val. got: {ratios_tv}")

    raw_train = total * float(ratios_tv["train"])
    raw_val = total * float(ratios_tv["val"])
    floor_train = int(math.floor(raw_train))
    floor_val = int(math.floor(raw_val))
    rem_train = raw_train - floor_train
    rem_val = raw_val - floor_val

    used = floor_train + floor_val
    left = total - used

    out = {"train": floor_train, "val": floor_val}

    order = ["train", "val"]
    rem = {"train": rem_train, "val": rem_val}
    order.sort(key=lambda k: (-rem[k], k))

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


def _find_original_splits_root(dataset_dir: Path) -> Optional[Path]:
    """
    dataset_dir 内に original_splits がある場合、そのルートを返す。

    想定場所（優先順）:
      - <dataset_dir>/original_splits/
      - <dataset_dir>/raw/original_splits/
    """
    dataset_dir = _to_path(dataset_dir)
    candidates = [
        dataset_dir / "original_splits",
        dataset_dir / "raw" / "original_splits",
    ]
    for p in candidates:
        if p.is_dir():
            return p
    return None


def _discover_original_split_pairs(dataset_dir: Path) -> Dict[int, Tuple[Path, Path]]:
    """
    original_splits 配下から trainlistXX.txt / testlistXX.txt のペアを探索して返す。

    戻り値: {split_id: (trainlist_path, testlist_path)}
    """
    root = _find_original_splits_root(dataset_dir)
    if root is None:
        return {}

    train_map: Dict[int, List[Path]] = defaultdict(list)
    test_map: Dict[int, List[Path]] = defaultdict(list)

    for p in root.rglob("*.txt"):
        if not p.is_file():
            continue

        m = _ORIG_TRAINLIST_RE.match(p.name)
        if m:
            try:
                split_id = int(m.group(1))
            except Exception:
                continue
            train_map[split_id].append(p)
            continue

        m = _ORIG_TESTLIST_RE.match(p.name)
        if m:
            try:
                split_id = int(m.group(1))
            except Exception:
                continue
            test_map[split_id].append(p)
            continue

    pairs: Dict[int, Tuple[Path, Path]] = {}
    for split_id in sorted(set(train_map.keys()) & set(test_map.keys())):
        train_path = sorted(train_map[split_id])[0]
        test_path = sorted(test_map[split_id])[0]
        pairs[int(split_id)] = (train_path, test_path)

    return pairs


def _unique_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _token_to_stem(token: str) -> Optional[str]:
    """
    original split の 1列目（"ClassName/v_XXX.avi" 等）を stem（拡張子なしファイル名）に変換する。
    """
    if not isinstance(token, str):
        return None
    s = token.strip()
    if not s:
        return None

    # クォートが付いている場合に備える
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
        if not s:
            return None

    # Windowsの "\" を POSIX風にそろえる
    s = s.replace("\\", "/")

    stem = PurePosixPath(s).stem
    if not stem:
        return None
    return stem


def _read_original_split_stems(trainlist_path: Path, testlist_path: Path) -> Tuple[List[str], List[str]]:
    """
    trainlist / testlist を読み、video stem のリストを返す。

    - trainlist は通常 "<path> <label_id>" の2列
    - testlist は通常 "<path>" の1列

    どちらも 1列目だけを使う（ラベル列は無視）。
    """
    trainlist_path = _to_path(trainlist_path)
    testlist_path = _to_path(testlist_path)

    train_stems_raw: List[str] = []
    for raw_line in trainlist_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if not parts:
            continue
        stem = _token_to_stem(parts[0])
        if stem is not None:
            train_stems_raw.append(stem)

    test_stems_raw: List[str] = []
    for raw_line in testlist_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if not parts:
            continue
        stem = _token_to_stem(parts[0])
        if stem is not None:
            test_stems_raw.append(stem)

    return _unique_preserve_order(train_stems_raw), _unique_preserve_order(test_stems_raw)


def _split_train_val_balanced(
    train_pool: List[str],
    video_label_counts: Dict[str, Dict[int, int]],
    train_r: float,
    val_r: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    """
    train_pool を train/val に分割する（動画単位）。

    all_list.txt 由来の label（人物カウント）分布が train/val で近くなるように、
    乱数シードをずらした複数候補を生成し、最良候補を採用する。

    方針:
      1) train_pool 全体での label 合計（人物カウント）を計算
      2) train/val の目標 label 合計を比率から計算（train_r と val_r を正規化）
      3) 候補生成（複数回）:
           - 動画の順序をランダムにしつつ
           - 「割り当て後の目標との差（重み付きL1）」が最小になる方へ貪欲に入れる
      4) 完成した split をスコア化（全クラスの差の大きさ + サイズ誤差）
      5) スコア最小の候補を返す
    """
    if not train_pool:
        return [], []

    ratios_tv = _normalize_train_val_ratios(float(train_r), float(val_r))
    target_n_tv = _allocate_counts_train_val(len(train_pool), ratios_tv)

    # train_pool 全体の label 合計（人物カウント）
    label_totals_pool: Dict[int, int] = defaultdict(int)
    for stem in train_pool:
        for lid, c in video_label_counts.get(stem, {}).items():
            label_totals_pool[int(lid)] += int(c)

    labels = sorted(label_totals_pool.keys())

    # ラベルが取れないなら単純分割
    rng0 = random.Random(seed)
    if not labels:
        pool = list(train_pool)
        rng0.shuffle(pool)
        train_n = int(target_n_tv["train"])
        return pool[:train_n], pool[train_n:]

    # 目標 label 合計（人物カウント）
    target_label_tv: Dict[str, Dict[int, float]] = {
        s: {lid: float(ratios_tv[s]) * float(label_totals_pool[lid]) for lid in labels}
        for s in ["train", "val"]
    }

    # レアクラスを強める重み（人物数ベース）
    weights: Dict[int, float] = {lid: 1.0 / max(1, int(label_totals_pool[lid])) for lid in labels}

    def score_split(train_stems: List[str], val_stems: List[str]) -> float:
        # label合計
        cur_train: Dict[int, int] = defaultdict(int)
        cur_val: Dict[int, int] = defaultdict(int)

        for stem in train_stems:
            for lid, c in video_label_counts[stem].items():
                cur_train[int(lid)] += int(c)
        for stem in val_stems:
            for lid, c in video_label_counts[stem].items():
                cur_val[int(lid)] += int(c)

        # 目標との差（重み付きL1）
        dist = 0.0
        for lid in labels:
            w = float(weights[lid])
            dist += w * abs(float(cur_train.get(lid, 0)) - float(target_label_tv["train"][lid]))
            dist += w * abs(float(cur_val.get(lid, 0)) - float(target_label_tv["val"][lid]))

        # 動画数の目標との差（強めに罰）
        size_penalty = 0.0
        size_penalty += 10.0 * abs(len(train_stems) - int(target_n_tv["train"]))
        size_penalty += 10.0 * abs(len(val_stems) - int(target_n_tv["val"]))

        return dist + size_penalty

    def build_once(rng: random.Random) -> Tuple[List[str], List[str]]:
        # 動画ごとの希少度（レアクラスを多く含む動画を先に処理）
        items = [(stem, video_label_counts[stem]) for stem in train_pool]
        rng.shuffle(items)

        def rarity_score(counts: Dict[int, int]) -> float:
            s = 0.0
            for lid, c in counts.items():
                s += float(c) * float(weights.get(int(lid), 0.0))
            return s

        def person_total(counts: Dict[int, int]) -> int:
            return int(sum(int(x) for x in counts.values()))

        items.sort(key=lambda kv: (-rarity_score(kv[1]), -person_total(kv[1])))

        cur_n = {"train": 0, "val": 0}
        cur_label = {"train": defaultdict(int), "val": defaultdict(int)}
        out = {"train": [], "val": []}

        def delta_cost(split: str, counts_v: Dict[int, int]) -> float:
            # その split に入れた時の「目標との差」の増分（重み付きL1）
            d = 0.0
            cur = cur_label[split]
            tgt = target_label_tv[split]
            for lid, c in counts_v.items():
                lid = int(lid)
                c = int(c)
                cur_l = int(cur.get(lid, 0))
                tgt_l = float(tgt[lid])
                w = float(weights.get(lid, 1.0))
                d += w * (abs((cur_l + c) - tgt_l) - abs(cur_l - tgt_l))
            return d

        for stem, counts_v in items:
            # まず動画数の上限を守る
            candidates = [s for s in ["train", "val"] if cur_n[s] < int(target_n_tv[s])]
            if not candidates:
                candidates = ["train", "val"]

            # コストが小さい方へ入れる（同点は乱数で崩す）
            scored = []
            for s in candidates:
                dc = delta_cost(s, counts_v)
                remaining = int(target_n_tv[s]) - int(cur_n[s])
                scored.append((dc, -remaining, rng.random(), s))
            scored.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
            chosen = scored[0][3]

            out[chosen].append(stem)
            cur_n[chosen] += 1
            for lid, c in counts_v.items():
                cur_label[chosen][int(lid)] += int(c)

        return out["train"], out["val"]

    # ここが改善点：複数回試して最良を採用
    # 試行回数は環境変数/設定で変えられるようにしてもよいが、まず固定値で。
    tries = int(globals().get("SPLIT_TRIES", 64))

    best_train: List[str] = []
    best_val: List[str] = []
    best_score: float | None = None

    # seed から派生させて再現性を保つ
    base = random.Random(seed)
    seeds = [base.randint(0, 2**31 - 1) for _ in range(max(1, tries))]

    for s in seeds:
        rng = random.Random(int(s))
        tr, va = build_once(rng)
        sc = score_split(tr, va)
        if best_score is None or sc < best_score:
            best_score = sc
            best_train, best_val = tr, va

    # 念のためサイズが崩れていたら補正（基本はここに来ない想定）
    # ただし、何らかの理由で候補生成時に目標数を守れなかった場合の保険。
    if len(best_train) + len(best_val) != len(train_pool):
        pool = list(train_pool)
        rng0.shuffle(pool)
        train_n = int(target_n_tv["train"])
        best_train, best_val = pool[:train_n], pool[train_n:]

    return best_train, best_val


def build_split_lists(
    dataset_dir: Path,
    all_list_path: Path,
    subsplit_name: str,
    original_trainlist_path: Path | None = None,
    original_testlist_path: Path | None = None,
) -> None:
    """
    all_list.txt（人物行）をもとに、動画（pt）単位で split を作る。

    出力（2列）:
      processed/splits/<subsplit_name>/train_list.txt
      processed/splits/<subsplit_name>/val_list.txt
      processed/splits/<subsplit_name>/test_list.txt

    各行: "<stem>.pt 0"

    original_trainlist_path / original_testlist_path が指定されている場合:
      - testlist を test に固定する
      - trainlist を train/val に分割する（TRAIN_RATIO と VAL_RATIO を正規化して使用）
      - 出力は常に "<stem>.pt 0"

    original_* が指定されていない場合:
      - UCF101 形式（stem が *_gXX_cYY を含む）かつ dataset_dir.name が "ucf101" の場合:
          test は g01..g07 を固定し、残りを train/val に分割する（TRAIN_RATIO と VAL_RATIO を正規化）
      - それ以外は train/val/test を比率で自動割当する（従来どおり）
    """
    dataset_dir = _to_path(dataset_dir)
    ensure_processed_dirs(dataset_dir, subsplit_name=subsplit_name)

    train_r = globals().get("TRAIN_RATIO", None)
    val_r = globals().get("VAL_RATIO", None)
    test_r = globals().get("TEST_RATIO", None)
    if train_r is None or val_r is None or test_r is None:
        raise RuntimeError(
            "TRAIN_RATIO / VAL_RATIO / TEST_RATIO が定義されていません（config_base.py を確認してください）。"
        )

    ratios = _normalize_ratios(float(train_r), float(val_r), float(test_r))
    seed = int(globals().get("SPLIT_SEED", 42))

    existing_stems = _collect_existing_pt_stems(dataset_dir)
    if not existing_stems:
        print(f"[split] skip: no pt files found: {dataset_dir / 'processed' / 'pt'}")
        return

    # stem -> label_id -> count（人物数）
    video_label_counts: Dict[str, Dict[int, int]] = {}
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

    stems_all = sorted(video_label_counts.keys())

    splits_dir = dataset_dir / "processed" / "splits" / subsplit_name

    def write_list(path: Path, stems: List[str]) -> None:
        stems = list(stems)

        # train/val だけ並び順をランダム化（再現性あり）
        if path.name == "train_list.txt":
            rng = random.Random(seed + 0)
            rng.shuffle(stems)
        elif path.name == "val_list.txt":
            rng = random.Random(seed + 1)
            rng.shuffle(stems)
        else:
            # test は従来どおり名前順にしておく（必要ならここも shuffle に変更可）
            stems.sort()

        lines = [f"{stem}.pt 0" for stem in stems]
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    # ===== original_splits の trainlist/testlist を基に split を作る =====
    if original_trainlist_path is not None or original_testlist_path is not None:
        if original_trainlist_path is None or original_testlist_path is None:
            raise ValueError("original_trainlist_path と original_testlist_path は両方指定してください。")

        orig_train_stems, orig_test_stems = _read_original_split_stems(original_trainlist_path, original_testlist_path)

        stems_set = set(stems_all)
        test_set = {stem for stem in orig_test_stems if stem in stems_set}
        train_pool = [stem for stem in orig_train_stems if stem in stems_set and stem not in test_set]

        if not train_pool and not test_set:
            raise ValueError(
                "original_splits が指定されていますが、processed/pt のstemと一致しません。"
                f"\n  dataset_dir: {dataset_dir}"
                f"\n  trainlist: {original_trainlist_path}"
                f"\n  testlist: {original_testlist_path}"
            )

        train_stems, val_stems = _split_train_val_balanced(
            train_pool=train_pool,
            video_label_counts=video_label_counts,
            train_r=float(train_r),
            val_r=float(val_r),
            seed=seed,
        )
        test_stems = list(test_set)

        write_list(splits_dir / "train_list.txt", train_stems)
        write_list(splits_dir / "val_list.txt", val_stems)
        write_list(splits_dir / "test_list.txt", test_stems)

        print(
            f"[split] {dataset_dir.name}/{subsplit_name}: "
            f"train={len(train_stems)}, val={len(val_stems)}, test={len(test_stems)} "
            f"(from original_splits)"
        )
        return

    # ===== UCF101: test を g01..g07 で固定し、残りを train/val に分割 =====
    use_ucf_fixed_test = bool(globals().get("USE_UCF101_FIXED_TEST_SPLIT", True))
    ds_name = dataset_dir.name.lower()

    def _get_ucf101_test_groups() -> set[int]:
        cfg = globals().get("UCF101_TEST_GROUPS", None)
        if cfg is None:
            return set(range(1, 8))  # split1: g01..g07
        if isinstance(cfg, (set, list, tuple)):
            return {int(x) for x in cfg}
        if isinstance(cfg, str):
            s = cfg.strip()
            if not s:
                return set(range(1, 8))
            # "1-7" 形式
            if "-" in s and "," not in s:
                a, b = s.split("-", 1)
                return set(range(int(a), int(b) + 1))
            # "1,2,3" / "1 2 3" 形式
            parts = re.split(r"[\s,]+", s)
            return {int(p) for p in parts if p}
        raise TypeError(f"UCF101_TEST_GROUPS must be set/list/tuple/str. got: {type(cfg)}")

    # 形式判定（安全のため、名前判定に加えて group 抽出も確認する）
    stem_to_group: Dict[str, int | None] = {stem: _extract_ucf_group_id(stem) for stem in stems_all}
    detected_groups = [g for g in stem_to_group.values() if g is not None]
    looks_like_ucf = False
    if ds_name in {"ucf101", "ucf-101"}:
        looks_like_ucf = True
    else:
        if len(detected_groups) >= int(0.9 * len(stems_all)) and detected_groups:
            gmin = min(int(g) for g in detected_groups)
            gmax = max(int(g) for g in detected_groups)
            if 1 <= gmin and gmax <= 25:
                looks_like_ucf = True

    if use_ucf_fixed_test and looks_like_ucf:
        test_groups = _get_ucf101_test_groups()
        test_set = {stem for stem, g in stem_to_group.items() if g is not None and int(g) in test_groups}
        train_pool = [stem for stem in stems_all if stem not in test_set]

        if test_set and train_pool:
            train_stems, val_stems = _split_train_val_balanced(
                train_pool=train_pool,
                video_label_counts=video_label_counts,
                train_r=float(train_r),
                val_r=float(val_r),
                seed=seed,
            )
            test_stems = list(test_set)

            write_list(splits_dir / "train_list.txt", train_stems)
            write_list(splits_dir / "val_list.txt", val_stems)
            write_list(splits_dir / "test_list.txt", test_stems)

            print(
                f"[split] {dataset_dir.name}/{subsplit_name}: "
                f"train={len(train_stems)}, val={len(val_stems)}, test={len(test_stems)} "
                f"(fixed test groups={sorted(test_groups)})"
            )
            return

    # ===== フォールバック: train/val/test を同時に割り当てる =====

    # label 合計（人物カウント）
    label_totals: Dict[int, int] = defaultdict(int)
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

        scored: List[Tuple[float, int, float, str]] = []
        for s in candidates:
            dc = delta_cost(s, counts_v)
            remaining = int(target_n[s]) - int(cur_n[s])
            scored.append((dc, -remaining, rng.random(), s))  # tie を乱数で崩す

        scored.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
        best_split = scored[0][3]

        assigned[best_split].append(stem)
        cur_n[best_split] += 1
        cur = cur_label[best_split]
        for lid, c in counts_v.items():
            cur[int(lid)] += int(c)

    write_list(splits_dir / "train_list.txt", assigned["train"])
    write_list(splits_dir / "val_list.txt", assigned["val"])
    write_list(splits_dir / "test_list.txt", assigned["test"])

    print(
        f"[split] {dataset_dir.name}/{subsplit_name}: "
        f"train={len(assigned['train'])}, val={len(assigned['val'])}, test={len(assigned['test'])} "
        f"(target train={target_n['train']}, val={target_n['val']}, test={target_n['test']})"
    )


def _default_subsplit_name(dataset_dir: Path) -> str:
    name = _to_path(dataset_dir).name.lower()
    if name in {"ucf101", "ucf-101"}:
        return "split01"
    return "default"


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
        orig_pairs = _discover_original_split_pairs(ds_dir)
        if orig_pairs:
            for split_id, (trainlist_path, testlist_path) in orig_pairs.items():
                subsplit_name = f"split{int(split_id):02d}"
                build_split_lists(
                    ds_dir,
                    all_list_path,
                    subsplit_name=subsplit_name,
                    original_trainlist_path=trainlist_path,
                    original_testlist_path=testlist_path,
                )
        else:
            build_split_lists(
                ds_dir,
                all_list_path,
                subsplit_name=_default_subsplit_name(ds_dir),
            )

    print("Done.")


if __name__ == "__main__":
    freeze_support()
    process_all(
        _resolve_datasets_root(),
        int(getattr(config, "NUM_WORKERS", 4)),
    )
