# training/search.py
"""
Video-to-Text Retrieval (Semantic Search) for HAR-2way-model

What this script does
---------------------
1) Run inference for (all) preprocessed video samples (.pt) and build a video embedding index.
2) Encode a text query with SentenceTransformer and retrieve Top-N most similar videos.

Assumptions
-----------
- Dataset is already converted into `.pt` files by dataset_builder.
- Each `.pt` contains (at least) keys: images, skeletons, scores, labels, and optionally label_ids.
- The trained model maps video -> text-embedding space (same dimension as SentenceTransformer embeddings).

Typical usage
-------------
# Build index (once) then interactive search
python training/search.py --config training/configs/infer.yaml --target full --build

# Or: one-shot query (will auto-build index if missing)
python training/search.py --config training/configs/infer.yaml --target full --query "walking in the room" --topk 5

Notes
-----
- This script intentionally does NOT modify any __init__.py.
- If `sentence-transformers` is not installed, install it via:
    pip install sentence-transformers
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.modules.models import CNN, STGCN, MLP, TransformerEncoder
from training.modules.networks import ImageBranch, SkeletonBranch, FullModel
from training.modules.utils import build_coco17_adj

# __init__.py を変更しないため明示import
from training.modules.dataset.dataset import SyncedDataset, pad_person_collate

from config_base import *  # PROJECT_ROOT, DATASETS_ROOT, DATASET_ROOT, ARTIFACT_ROOT, ...
try:
    from config_local import *  # noqa: F401,F403
except ImportError:
    pass


# -----------------------------
# helpers
# -----------------------------
def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _torch_load_state_dict(path: Path, map_location: torch.device) -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # 古いPyTorch互換
        return torch.load(path, map_location=map_location)


def resolve_dataset_root(train_cfg: dict, infer_cfg: dict) -> Path:
    """
    優先順位: train_cfg.dataset.name > infer_cfg.dataset.name > config_(base|local) の DATASET_ROOT
    """
    name = None
    ds_train = train_cfg.get("dataset", None)
    ds_infer = infer_cfg.get("dataset", None)

    if isinstance(ds_train, dict):
        name = ds_train.get("name", None)
    if name is None and isinstance(ds_infer, dict):
        name = ds_infer.get("name", None)

    if name:
        return (DATASETS_ROOT / name).resolve()

    return Path(DATASET_ROOT).resolve()


def apply_train_data_defaults(train_cfg: dict) -> dict:
    """
    train_cfg.data に processed_dir / split がある場合、
    img_pt_dir / skel_pt_dir / train_file_list / val_file_list を自動生成する（未指定のみ）。
    """
    data = (train_cfg.get("data", {}) or {})
    processed = data.get("processed_dir", None)
    split = data.get("split", "default")

    if processed:
        data.setdefault("img_pt_dir", f"{processed}/pt")
        data.setdefault("skel_pt_dir", f"{processed}/pt")
        data.setdefault("train_file_list", f"{processed}/splits/{split}/train_list.txt")
        data.setdefault("val_file_list", f"{processed}/splits/{split}/val_list.txt")

    train_cfg["data"] = data
    return train_cfg


def create_model(params: dict, device: torch.device) -> torch.nn.Module:
    """
    FullModel(image_branch + skeleton_branch + fusion MLP)
    """
    p = copy.deepcopy(params)
    p["skel"]["stgcn"]["adj"] = build_coco17_adj(device=device)

    model = FullModel(
        ImageBranch(
            CNN(**p["img"]["cnn"]),
            TransformerEncoder(**p["img"]["transformer"]),
        ),
        SkeletonBranch(
            STGCN(**p["skel"]["stgcn"])
        ),
        MLP(**p["mlp"]),
    )
    return model


# -----------------------------
# dataset wrapper (path aware)
# -----------------------------
class SyncedDatasetWithPath(SyncedDataset):
    """
    SyncedDataset の返り値に relpath を付与するラッパー。
    """

    def __getitem__(self, idx):
        frames, keypoints, scores, labels, label_ids = super().__getitem__(idx)
        # Use path relative to img_torch_path so it is stable across machines.
        rel = str(self.samples[idx].relative_to(self.img_torch_path))
        return frames, keypoints, scores, labels, label_ids, rel


def pad_person_collate_with_path(batch):
    """
    pad_person_collate + relpath list を返す。
    """
    relpaths = [b[-1] for b in batch]
    base = [b[:-1] for b in batch]  # (frames, keypoints, scores, labels, label_ids)
    out = pad_person_collate(base)
    if len(out) != 5:
        raise RuntimeError(f"unexpected pad_person_collate output length: {len(out)}")
    frames, keypoints, scores, labels, label_ids = out
    return frames, keypoints, scores, labels, label_ids, relpaths


# -----------------------------
# index format
# -----------------------------
@dataclass
class VideoEmbeddingIndex:
    """
    A simple in-memory index for cosine similarity search.
    embeddings must be L2-normalized (N,D) on CPU.
    """

    embeddings: torch.Tensor
    relpaths: List[str]
    meta: Dict

    def __post_init__(self):
        if not isinstance(self.embeddings, torch.Tensor):
            raise TypeError("embeddings must be a torch.Tensor")
        if self.embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2D (N,D), got {tuple(self.embeddings.shape)}")
        if self.embeddings.device.type != "cpu":
            raise ValueError("embeddings must be on CPU")
        if len(self.relpaths) != int(self.embeddings.size(0)):
            raise ValueError("relpaths length mismatch")


def save_index(path: Path, index: VideoEmbeddingIndex) -> None:
    ensure_dir(path.parent)
    payload = {
        "embeddings": index.embeddings,  # CPU tensor
        "relpaths": index.relpaths,
        "meta": index.meta,
    }
    torch.save(payload, path)


def load_index(path: Path) -> VideoEmbeddingIndex:
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict):
        raise ValueError("index file is not a dict")
    embeddings = data.get("embeddings", None)
    relpaths = data.get("relpaths", None)
    meta = data.get("meta", {}) or {}
    if embeddings is None or relpaths is None:
        raise ValueError("index file missing embeddings/relpaths")
    # safety: enforce float32 + contiguous
    embeddings = embeddings.detach().float().contiguous()
    # ensure normalized (best-effort; do not modify too much)
    embeddings = F.normalize(embeddings, dim=-1, eps=1e-6)
    return VideoEmbeddingIndex(embeddings=embeddings, relpaths=list(relpaths), meta=dict(meta))


# -----------------------------
# embedding builders
# -----------------------------
def _pick_device(runtime_device: Optional[str], prefer_cuda: bool = True) -> torch.device:
    if runtime_device is None:
        if prefer_cuda and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    runtime_device = str(runtime_device)
    if runtime_device.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(runtime_device)


@torch.inference_mode()
def build_video_index(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    target: str,
    min_valid_t: int = 16,
    use_amp: bool = True,
    max_videos: Optional[int] = None,
) -> Tuple[VideoEmbeddingIndex, Dict[str, int]]:
    """
    Build a video-level embedding per dataset item.
    If a sample has multiple valid persons, we average their person embeddings.

    Returns:
      index, stats
    """
    if target not in ("full", "img", "skel"):
        raise ValueError(f"target must be one of full/img/skel, got {target}")

    model.eval()
    use_autocast = (device.type == "cuda") and bool(use_amp)

    all_embs: List[torch.Tensor] = []
    all_paths: List[str] = []

    stats = {
        "videos_seen": 0,
        "videos_indexed": 0,
        "videos_skipped_no_valid_person": 0,
        "persons_total": 0,
        "persons_valid": 0,
    }

    for batch in tqdm(loader, desc="Build video embeddings"):
        frames, keypoints, scores, _labels, _label_ids, relpaths = batch
        B, P = scores.shape[:2]
        stats["videos_seen"] += int(B)
        stats["persons_total"] += int(B * P)

        # valid persons on CPU
        person_has_keypoint = scores.max(dim=-1).values > 0  # (B,P,T)
        valid_len = person_has_keypoint.sum(dim=-1)          # (B,P)
        person_valid = valid_len >= int(min_valid_t)         # (B,P)

        valid_flat = person_valid.reshape(-1)                # (B*P,)
        idx = valid_flat.nonzero(as_tuple=False).squeeze(1)  # (N,)

        if idx.numel() == 0:
            stats["videos_skipped_no_valid_person"] += int(B)
            if max_videos is not None and stats["videos_seen"] >= int(max_videos):
                break
            continue

        # map each person to its video id (0..B-1)
        video_ids = torch.arange(B).repeat_interleave(P)     # (B*P,)
        video_ids = video_ids[idx]                           # (N,)

        # pick valid persons
        frames_f = frames.reshape(B * P, *frames.shape[2:])[idx]
        keypoints_f = keypoints.reshape(B * P, *keypoints.shape[2:])[idx]
        scores_f = scores.reshape(B * P, *scores.shape[2:])[idx]

        stats["persons_valid"] += int(idx.numel())

        frames_f = frames_f.to(device, non_blocking=True)
        keypoints_f = keypoints_f.to(device, non_blocking=True)
        scores_f = scores_f.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_autocast):
            full_out, img_out, skel_out = model(frames_f, keypoints_f, scores_f)
            outs = {"full": full_out, "img": img_out, "skel": skel_out}
            emb = outs[target].float()

        emb = F.normalize(emb, dim=-1, eps=1e-6)  # (N,D)

        # aggregate persons -> video
        D = int(emb.size(1))
        sum_emb = torch.zeros((B, D), device=device)
        cnt = torch.zeros((B, 1), device=device)

        sum_emb.index_add_(0, video_ids.to(device), emb)
        cnt.index_add_(0, video_ids.to(device), torch.ones((emb.size(0), 1), device=device))

        valid_video_mask = (cnt.squeeze(1) > 0)

        # move per-video embeddings to CPU
        for i in range(B):
            if not bool(valid_video_mask[i].item()):
                stats["videos_skipped_no_valid_person"] += 1
                continue
            v = (sum_emb[i] / cnt[i]).detach()
            v = F.normalize(v, dim=-1, eps=1e-6)
            all_embs.append(v.cpu())
            all_paths.append(relpaths[i])
            stats["videos_indexed"] += 1

        if max_videos is not None and stats["videos_seen"] >= int(max_videos):
            break

    if not all_embs:
        raise RuntimeError("No video embeddings were created (all samples might have no valid persons).")

    embs = torch.stack(all_embs, dim=0).contiguous().float()  # (N,D) on CPU
    embs = F.normalize(embs, dim=-1, eps=1e-6)

    index = VideoEmbeddingIndex(
        embeddings=embs,
        relpaths=all_paths,
        meta={
            "target": target,
            "min_valid_t": int(min_valid_t),
            "embed_dim": int(embs.size(1)),
            "videos_seen": int(stats["videos_seen"]),
            "videos_indexed": int(stats["videos_indexed"]),
            "videos_skipped_no_valid_person": int(stats["videos_skipped_no_valid_person"]),
        },
    )
    return index, stats


# -----------------------------
# text encoder (SentenceTransformer)
# -----------------------------
def load_sentence_transformer(model_name_or_path: str, device: torch.device):
    """
    Lazy import so training env works even without sentence-transformers (until search time).
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required for text queries.\n"
            "Install it with: pip install sentence-transformers"
        ) from e

    # Try the given name/path first. If it is a short Sentence-Transformers ID
    # (e.g., "stsb-xlm-r-multilingual"), also try "sentence-transformers/<id>" as fallback.
    try:
        st = SentenceTransformer(model_name_or_path)
    except Exception:
        alt = None
        if ("/" not in str(model_name_or_path)) and (not Path(str(model_name_or_path)).exists()):
            alt = f"sentence-transformers/{model_name_or_path}"
        if alt is None:
            raise
        st = SentenceTransformer(alt)

    # SentenceTransformer expects device string
    st = st.to(str(device))
    return st


@torch.inference_mode()
def encode_text(st_model, text: str) -> torch.Tensor:
    """
    Returns: (D,) CPU float32 L2-normalized
    """
    # normalize_embeddings=True gives L2-normalized vectors
    emb = st_model.encode([text], convert_to_tensor=True, normalize_embeddings=True)
    if emb.ndim == 2:
        emb = emb[0]
    emb = emb.detach().float().cpu()
    emb = F.normalize(emb, dim=-1, eps=1e-6)
    return emb


def search_topk(index: VideoEmbeddingIndex, query_emb: torch.Tensor, topk: int) -> List[Tuple[int, float, str]]:
    """
    Returns list of (rank_idx, score, relpath)
    """
    if query_emb.ndim != 1:
        raise ValueError("query_emb must be 1D (D,)")
    if int(query_emb.numel()) != int(index.embeddings.size(1)):
        raise ValueError(
            f"embedding dim mismatch: query={int(query_emb.numel())}, index={int(index.embeddings.size(1))}"
        )

    # cosine similarity because both are normalized
    sims = (index.embeddings @ query_emb).float()  # (N,)
    k = min(int(topk), int(sims.numel()))
    vals, idxs = sims.topk(k)
    out = []
    for r, (i, v) in enumerate(zip(idxs.tolist(), vals.tolist()), 1):
        out.append((r, float(v), index.relpaths[int(i)]))
    return out


def format_output_path(relpath: str, output_ext: Optional[str] = None, basename_only: bool = False) -> str:
    """
    relpath: indexに保存されている相対パス（基本は *.pt）
    output_ext: 例 '.avi' や 'mp4' を指定すると拡張子を置換して表示する
    basename_only: True ならファイル名のみ表示する
    """
    p = Path(relpath)
    if output_ext:
        ext = str(output_ext)
        if not ext.startswith('.'):
            ext = '.' + ext
        try:
            p = p.with_suffix(ext)
        except Exception:
            # 変なsuffixでも落ちないように保険
            pass
    return p.name if basename_only else str(p)


# -----------------------------
# main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Video semantic search (video->text embedding retrieval)")
    parser.add_argument("--config", default="training/configs/infer.yaml", help="Path to infer.yaml")
    parser.add_argument("--artifact_path", default=None, help="Override infer.yaml artifact_path (e.g. runs/20260201215901)")
    parser.add_argument("--target", default="full", choices=["full", "img", "skel"], help="Which head to index/search")
    parser.add_argument("--index_path", default=None, help="Where to save/load the embedding index (.pt)")
    parser.add_argument("--build", action="store_true", help="Force rebuild index")
    parser.add_argument("--topk", type=int, default=5, help="Top-N to show")
    parser.add_argument("--query", default=None, help="One-shot query. If omitted, runs interactive mode.")
    parser.add_argument("--sentence_transformer", default=None, help="SentenceTransformer model name/path (default: from infer.yaml)")
    parser.add_argument("--device", default=None, help="Override device (e.g. cuda, cuda:0, cpu)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for embedding build (dataset items per batch)")
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader num_workers override")
    parser.add_argument("--min_valid_t", type=int, default=None, help="Min valid frames (person) threshold override")
    parser.add_argument("--no_amp", action="store_true", help="Disable autocast AMP on CUDA")
    parser.add_argument("--max_videos", type=int, default=None, help="(debug) limit number of videos to index")

    # optional: restrict dataset by a file_list (two-column format: relpath class_id)
    parser.add_argument("--file_list", default=None, help="Optional file_list path (absolute or relative to dataset_root)")
    parser.add_argument("--output_ext", default=None, help="(optional) replace suffix for display (e.g. .avi)")
    parser.add_argument("--basename", action="store_true", help="Print only base filename (no directories)")

    args = parser.parse_args()

    infer_cfg_path = Path(args.config)
    infer_cfg = load_yaml(infer_cfg_path)

    # run_dir
    artifact_path = args.artifact_path or infer_cfg.get("artifact_path", None)
    if not artifact_path:
        raise KeyError("artifact_path is missing (provide --artifact_path or set it in infer.yaml).")

    run_dir = (ARTIFACT_ROOT / artifact_path).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    timestamp = run_dir.name

    # train config from the run directory
    train_cfg_path = run_dir / f"{timestamp}_config.yaml"
    if not train_cfg_path.exists():
        raise FileNotFoundError(f"train config not found: {train_cfg_path}")

    train_cfg = load_yaml(train_cfg_path)
    train_cfg = apply_train_data_defaults(train_cfg)

    # checkpoint
    remembers_best_name = train_cfg.get("logging", {}).get("best_model_name", "best_model.pt")
    ckpt_path = run_dir / f"{timestamp}_{remembers_best_name}"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    # dataset root
    dataset_root = resolve_dataset_root(train_cfg, infer_cfg)

    # data dirs
    img_pt_dir = (dataset_root / train_cfg["data"]["img_pt_dir"]).resolve()
    skel_pt_dir = (dataset_root / train_cfg["data"]["skel_pt_dir"]).resolve()

    if not img_pt_dir.exists():
        raise FileNotFoundError(f"img_pt_dir not found: {img_pt_dir}")
    if not skel_pt_dir.exists():
        raise FileNotFoundError(f"skel_pt_dir not found: {skel_pt_dir}")

    # runtime
    runtime_device = args.device
    if runtime_device is None and isinstance(train_cfg.get("runtime", None), dict):
        runtime_device = train_cfg["runtime"].get("device", None)

    device = _pick_device(runtime_device, prefer_cuda=True)

    # build model
    model_params = train_cfg["model"]["architecture"]
    model = create_model(model_params, device=device).to(device)
    state_dict = _torch_load_state_dict(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # loader settings
    infer_block = (infer_cfg.get("infer", {}) or {})
    default_bs = int(infer_block.get("batch_size", 8))
    batch_size = int(args.batch_size or default_bs)
    num_workers = int(args.num_workers if args.num_workers is not None else int(train_cfg["data"].get("num_workers", 0)))

    min_valid_t = int(args.min_valid_t if args.min_valid_t is not None else int(infer_block.get("min_valid_t", 16)))

    # dataset (all .pt by default)
    file_list_path = None
    if args.file_list:
        p = Path(args.file_list)
        if not p.is_absolute():
            p = (dataset_root / p).resolve()
        file_list_path = p
        if not file_list_path.exists():
            raise FileNotFoundError(f"file_list not found: {file_list_path}")

    ds = SyncedDatasetWithPath(
        img_torch_path=img_pt_dir,
        skel_torch_path=skel_pt_dir,
        file_list=file_list_path,
        img_augment=None,
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        drop_last=False,
        collate_fn=pad_person_collate_with_path,
    )

    # index paths
    if args.index_path:
        index_path = Path(args.index_path)
        if not index_path.is_absolute():
            index_path = (Path.cwd() / index_path).resolve()
    else:
        # default: save under the run directory
        index_path = (run_dir / "search" / f"video_index_{args.target}.pt").resolve()

    # build/load index
    if args.build or (not index_path.exists()):
        print(f"[Index] building embeddings: target={args.target} -> {index_path}")
        index, stats = build_video_index(
            model=model,
            loader=loader,
            device=device,
            target=args.target,
            min_valid_t=min_valid_t,
            use_amp=(not args.no_amp),
            max_videos=args.max_videos,
        )
        # add meta
        index.meta.update(
            {
                "run_dir": str(run_dir),
                "artifact_path": str(artifact_path),
                "checkpoint": str(ckpt_path),
                "dataset_root": str(dataset_root),
                "img_pt_dir": str(img_pt_dir),
                "skel_pt_dir": str(skel_pt_dir),
                "file_list": str(file_list_path) if file_list_path is not None else None,
            }
        )
        save_index(index_path, index)
        print(f"[Index] done. videos_indexed={stats['videos_indexed']}, skipped={stats['videos_skipped_no_valid_person']}")
    else:
        print(f"[Index] loading: {index_path}")
        index = load_index(index_path)
        print(f"[Index] loaded. videos={index.embeddings.size(0)}, dim={index.embeddings.size(1)}, target={index.meta.get('target')}")

    # sentence transformer name/path
    st_name = args.sentence_transformer
    if st_name is None:
        st_name = infer_block.get("sentence_transformer", None)
    if st_name is None:
        raise KeyError(
            "sentence_transformer is missing.\n"
            "Provide --sentence_transformer or set infer.sentence_transformer in infer.yaml."
        )

    # load text encoder (use CPU unless user explicitly uses CUDA device)
    # (Text encoding is typically light, but keeping device consistent is convenient.)
    st_device = device if device.type == "cuda" else torch.device("cpu")
    st_model = load_sentence_transformer(st_name, device=st_device)

    # query loop
    if args.query is not None:
        q = str(args.query).strip()
        if not q:
            raise ValueError("--query is empty")
        q_emb = encode_text(st_model, q)
        results = search_topk(index, q_emb, topk=args.topk)
        print(f"\nQuery: {q}")
        for rank, score, relpath in results:
            print(f"{rank:2d}. {score:+.4f}\t{format_output_path(relpath, args.output_ext, args.basename)}")
        return

    # interactive mode
    print("\nEnter a query sentence (empty line to exit).")
    while True:
        try:
            q = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if not q:
            break

        q_emb = encode_text(st_model, q)
        results = search_topk(index, q_emb, topk=args.topk)

        print("-" * 60)
        for rank, score, relpath in results:
            print(f"{rank:2d}. {score:+.4f}\t{format_output_path(relpath, args.output_ext, args.basename)}")


if __name__ == "__main__":
    main()
