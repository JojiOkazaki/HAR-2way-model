# infer.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import argparse
import copy
import csv

import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from training.modules.models import CNN, STGCN, MLP, TransformerEncoder
from training.modules.networks import ImageBranch, SkeletonBranch, FullModel
from training.modules.utils import build_coco17_adj

# __init__.py を変更しないため明示import
from training.modules.dataset.dataset import SyncedDataset, pad_person_collate

from config_base import *  # PROJECT_ROOT, DATASETS_ROOT, DATASET_ROOT, ARTIFACT_ROOT, DATASET_NAME 等

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


def resolve_eval_file_list_rel(train_cfg: dict, infer_cfg: dict) -> str:
    """
    優先順位:
      1) infer.infer.test_file_list
      2) infer.data.processed_dir + infer.data.split
      3) train_cfg.data.processed_dir + train_cfg.data.split
    """
    infer_block = (infer_cfg.get("infer", {}) or {})
    if infer_block.get("test_file_list", None):
        return str(infer_block["test_file_list"])

    infer_data = (infer_cfg.get("data", {}) or {})
    train_data = (train_cfg.get("data", {}) or {})

    processed = infer_data.get("processed_dir", None) or train_data.get("processed_dir", None)
    split = infer_data.get("split", None) or train_data.get("split", None) or "default"

    if not processed:
        raise KeyError(
            "infer.infer.test_file_list is missing and processed_dir is not provided (infer.data or train_cfg.data)."
        )

    return f"{processed}/splits/{split}/test_list.txt"


def resolve_proto_file_list_rel(train_cfg: dict, infer_cfg: dict, default_eval_rel: str) -> str:
    """
    クラスプロトタイプを作るための file_list。
    優先順位:
      1) infer.infer.prototype_file_list
      2) train_cfg.data.train_file_list（存在すれば）
      3) eval file_list
    """
    infer_block = (infer_cfg.get("infer", {}) or {})
    if infer_block.get("prototype_file_list", None):
        return str(infer_block["prototype_file_list"])

    train_data = (train_cfg.get("data", {}) or {})
    if train_data.get("train_file_list", None):
        return str(train_data["train_file_list"])

    return str(default_eval_rel)


def create_model(params: dict, device: torch.device) -> torch.nn.Module:
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
# metrics / confusion
# -----------------------------
@dataclass
class MetricState:
    recall_hits: Dict[int, int]            # k -> hits
    total: int
    confusion_by_k: Dict[int, np.ndarray]  # k -> (C,C) float64
    per_label_hits: Dict[int, Dict[int, int]]   # label_idx -> (k -> hits)
    per_label_total: Dict[int, int]            # label_idx -> total


def save_confusion_png(
    out_path: Path,
    confusion: np.ndarray,
    normalize: str = "true",  # "none" / "true" / "pred" / "all"
) -> None:
    cm = confusion.astype(np.float32)

    if normalize == "true":
        denom = cm.sum(axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        cm = cm / denom
    elif normalize == "pred":
        denom = cm.sum(axis=0, keepdims=True)
        denom[denom == 0] = 1.0
        cm = cm / denom
    elif normalize == "all":
        denom = cm.sum()
        denom = denom if denom != 0 else 1.0
        cm = cm / denom
    elif normalize == "none":
        pass
    else:
        raise ValueError(f"unknown normalize={normalize}")

    plt.figure(figsize=(12, 10))
    plt.imshow(cm, aspect="auto")
    plt.title(f"Confusion Matrix (normalize={normalize})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _map_unknown_label_ids(label_ids: torch.Tensor, unknown_ids: List[int]) -> torch.Tensor:
    if not unknown_ids:
        return label_ids
    out = label_ids.clone()
    for u in unknown_ids:
        out[out == int(u)] = -1
    return out


def build_class_prototypes_from_pt_labels(
    loader: DataLoader,
    min_valid_t: int,
    unknown_label_ids: List[int],
) -> Tuple[torch.Tensor, List[int]]:
    """
    `.pt` 内の (label_ids, labels) から label_id ごとの代表埋め込み（平均）を作る。
    returns:
      protos: (C, D) float32, L2-normalized (CPU)
      label_id_list: length C, index->label_id
    """
    sums: Dict[int, torch.Tensor] = {}
    counts: Dict[int, int] = {}

    with torch.inference_mode():
        for batch in loader:
            if len(batch) == 4:
                frames, keypoints, scores, labels = batch
                label_ids = None
            elif len(batch) == 5:
                frames, keypoints, scores, labels, label_ids = batch
            else:
                raise ValueError(f"unexpected batch tuple length: {len(batch)}")

            if label_ids is None:
                raise RuntimeError("label_ids is not provided by dataset/collate.")

            label_ids = _map_unknown_label_ids(label_ids.long(), unknown_label_ids)

            B, P = scores.shape[:2]
            D = labels.size(-1)

            person_has_keypoint = scores.max(dim=-1).values > 0  # (B,P,T)
            valid_len = person_has_keypoint.sum(dim=-1)          # (B,P)
            person_valid = valid_len >= int(min_valid_t)         # (B,P)

            valid_flat = person_valid.reshape(-1)  # (B*P,)
            idx = valid_flat.nonzero(as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                continue

            labels_f = labels.reshape(B * P, D)[idx].float()      # (N,D) CPU
            label_ids_f = label_ids.reshape(B * P)[idx].long()    # (N,) CPU

            known_mask = label_ids_f >= 0
            if not bool(known_mask.any()):
                continue

            labels_f = labels_f[known_mask]
            label_ids_f = label_ids_f[known_mask]

            for lid, emb in zip(label_ids_f.tolist(), labels_f):
                lid = int(lid)
                if lid not in sums:
                    sums[lid] = emb.double().clone()
                    counts[lid] = 1
                else:
                    sums[lid] += emb.double()
                    counts[lid] += 1

    if not sums:
        raise RuntimeError("class prototypes を作れませんでした（label_ids >= 0 のデータが無い/全て無効人物の可能性）。")

    label_id_list = sorted(sums.keys())
    protos = torch.stack([(sums[lid] / float(counts[lid])).float() for lid in label_id_list], dim=0)  # (C,D)
    protos = F.normalize(protos, dim=-1, eps=1e-6)
    return protos, label_id_list


def update_metrics_classification(
    state: MetricState,
    sim_logits: torch.Tensor,     # (N,C) on device
    true_idx: torch.Tensor,       # (N,) int64 on device (0..C-1)
    ks: List[int],
) -> None:
    """
    recall@k + confusion@k を更新
    confusion@k は 1サンプルを top-k へ 1/k ずつ配る
    """
    t_np = true_idx.detach().cpu().numpy().astype(np.int64)

    for k in ks:
        k_eff = min(int(k), sim_logits.size(1))
        topk = sim_logits.topk(k_eff, dim=1).indices  # (N,k_eff)

        hit = (topk == true_idx.unsqueeze(1)).any(dim=1)
        state.recall_hits[k] += int(hit.sum().item())

        p = topk.detach().cpu().numpy().astype(np.int64)
        w = 1.0 / float(k_eff)
        cm = state.confusion_by_k[k]

        for ti, row in zip(t_np, p):
            for pj in row:
                cm[ti, pj] += w

        # per-label recall@k
        hit_np = hit.detach().cpu().numpy().astype(np.bool_)
        for ti, hi in zip(t_np, hit_np):
            state.per_label_total[int(ti)] = state.per_label_total.get(int(ti), 0) + 1
            if int(ti) not in state.per_label_hits:
                state.per_label_hits[int(ti)] = {kk: 0 for kk in ks}
            if bool(hi):
                state.per_label_hits[int(ti)][k] += 1

    state.total += int(true_idx.numel())


def write_per_label_recall_csv(
    out_path: Path,
    state: MetricState,
    ks: List[int],
    label_id_list: List[int],
) -> None:
    """
    label_idx (0..C-1) ごとに recall@k を出す（label_id も併記）
    """
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label_idx", "label_id", "total"] + [f"R@{k}" for k in ks])

        for label_idx, label_id in enumerate(label_id_list):
            total = state.per_label_total.get(label_idx, 0)
            row = [label_idx, int(label_id), int(total)]
            for k in ks:
                hits = state.per_label_hits.get(label_idx, {}).get(k, 0)
                row.append(float(hits) / float(total) if total > 0 else float("nan"))
            w.writerow(row)


# -----------------------------
# main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/configs/infer.yaml")
    args = parser.parse_args()

    infer_cfg_path = Path(args.config)
    infer_cfg = load_yaml(infer_cfg_path)

    artifact_path = infer_cfg["artifact_path"]
    run_dir = (ARTIFACT_ROOT / artifact_path).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    timestamp = run_dir.name
    train_cfg_path = run_dir / f"{timestamp}_config.yaml"
    train_cfg = load_yaml(train_cfg_path)
    train_cfg = apply_train_data_defaults(train_cfg)

    # checkpoint
    best_model_name = f'{timestamp}_{train_cfg["logging"]["best_model_name"]}'
    ckpt_path = run_dir / best_model_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    # dataset root
    dataset_root = resolve_dataset_root(train_cfg, infer_cfg)

    infer_block = (infer_cfg.get("infer", {}) or {})
    recall_ks = [int(k) for k in infer_block.get("recall_ks", list(range(1, 11)))]
    targets = infer_block.get("targets", ["full", "img", "skel"])
    out_dir = run_dir / infer_block.get("out_dir", "infer")
    ensure_dir(out_dir)

    # valid person threshold
    min_valid_t = int(infer_block.get("min_valid_t", 16))

    # unknown label ids (e.g., 999)
    unknown_label_ids = infer_block.get("unknown_label_ids", [999])
    if unknown_label_ids is None:
        unknown_label_ids = []
    unknown_label_ids = [int(x) for x in unknown_label_ids]

    # device
    runtime_device = None
    if isinstance(train_cfg.get("runtime", None), dict):
        runtime_device = train_cfg["runtime"].get("device", None)
    if runtime_device is None:
        runtime_device = "cuda" if torch.cuda.is_available() else "cpu"

    if str(runtime_device).startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(runtime_device)

    # dataset paths
    img_pt_dir = (dataset_root / train_cfg["data"]["img_pt_dir"]).resolve()
    skel_pt_dir = (dataset_root / train_cfg["data"]["skel_pt_dir"]).resolve()

    eval_file_list_rel = resolve_eval_file_list_rel(train_cfg, infer_cfg)
    eval_file_list = (dataset_root / eval_file_list_rel).resolve()
    if not eval_file_list.exists():
        raise FileNotFoundError(f"eval file_list not found: {eval_file_list}")

    proto_file_list_rel = resolve_proto_file_list_rel(train_cfg, infer_cfg, eval_file_list_rel)
    proto_file_list = (dataset_root / proto_file_list_rel).resolve()
    if not proto_file_list.exists():
        raise FileNotFoundError(f"prototype file_list not found: {proto_file_list}")

    num_workers = int(train_cfg["data"].get("num_workers", 0))
    batch_size = int(infer_block.get("batch_size", train_cfg.get("training", {}).get("batch_size", 8)))

    def make_loader(file_list_path: Path) -> DataLoader:
        ds = SyncedDataset(
            img_torch_path=img_pt_dir,
            skel_torch_path=skel_pt_dir,
            file_list=file_list_path,
            img_augment=None,
        )
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            drop_last=False,
            collate_fn=pad_person_collate,
        )

    # loaders
    proto_loader = make_loader(proto_file_list)
    eval_loader = make_loader(eval_file_list)

    # 1) build class prototypes
    class_protos_cpu, label_id_list = build_class_prototypes_from_pt_labels(
        proto_loader,
        min_valid_t=min_valid_t,
        unknown_label_ids=unknown_label_ids,
    )
    C, D = class_protos_cpu.shape
    lid_to_cidx = {int(lid): i for i, lid in enumerate(label_id_list)}

    # save label map
    label_map_path = out_dir / "label_map.yaml"
    with label_map_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "class_count": int(C),
                "index_to_label_id": [int(x) for x in label_id_list],
                "prototype_file_list": str(proto_file_list_rel),
                "unknown_label_ids": [int(x) for x in unknown_label_ids],
            },
            f,
            allow_unicode=True,
            sort_keys=False,
        )

    # move prototypes to device
    class_protos = class_protos_cpu.to(device)

    # 2) build model
    model_params = train_cfg["model"]["architecture"]
    model = create_model(model_params, device=device).to(device)
    state_dict = _torch_load_state_dict(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # 3) metric states
    metrics: Dict[str, MetricState] = {}
    for t in targets:
        metrics[t] = MetricState(
            recall_hits={k: 0 for k in recall_ks},
            total=0,
            confusion_by_k={k: np.zeros((C, C), dtype=np.float64) for k in recall_ks},
            per_label_hits={},
            per_label_total={},
        )

    # 4) eval loop
    with torch.inference_mode():
        for batch in eval_loader:
            if len(batch) == 4:
                frames, keypoints, scores, labels = batch
                label_ids = None
            elif len(batch) == 5:
                frames, keypoints, scores, labels, label_ids = batch
            else:
                raise ValueError(f"unexpected batch tuple length: {len(batch)}")

            if label_ids is None:
                raise RuntimeError("label_ids is not provided. pt に label_ids が無い可能性があります。")

            label_ids = _map_unknown_label_ids(label_ids.long(), unknown_label_ids)

            B, P = scores.shape[:2]

            # valid persons (mask on CPU)
            person_has_keypoint = scores.max(dim=-1).values > 0  # (B,P,T)
            valid_len = person_has_keypoint.sum(dim=-1)          # (B,P)
            person_valid = valid_len >= min_valid_t              # (B,P)

            valid_flat = person_valid.reshape(-1)
            idx = valid_flat.nonzero(as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                continue

            # select valid persons
            frames_f = frames.reshape(B * P, *frames.shape[2:])[idx]
            keypoints_f = keypoints.reshape(B * P, *keypoints.shape[2:])[idx]
            scores_f = scores.reshape(B * P, *scores.shape[2:])[idx]
            label_ids_f = label_ids.reshape(B * P)[idx].long()

            # keep known label_ids that exist in prototypes
            known_mask = label_ids_f >= 0
            if not bool(known_mask.any()):
                continue

            frames_f = frames_f[known_mask]
            keypoints_f = keypoints_f[known_mask]
            scores_f = scores_f[known_mask]
            label_ids_f = label_ids_f[known_mask]

            lids_list = [int(x) for x in label_ids_f.tolist()]
            keep = [i for i, lid in enumerate(lids_list) if lid in lid_to_cidx]
            if len(keep) == 0:
                continue

            keep_t = torch.as_tensor(keep, dtype=torch.long)
            frames_f = frames_f.index_select(0, keep_t)
            keypoints_f = keypoints_f.index_select(0, keep_t)
            scores_f = scores_f.index_select(0, keep_t)
            lids_list = [lids_list[i] for i in keep]

            true_idx = torch.as_tensor([lid_to_cidx[lid] for lid in lids_list], device=device, dtype=torch.long)

            # move to device
            frames_f = frames_f.to(device, non_blocking=True)
            keypoints_f = keypoints_f.to(device, non_blocking=True)
            scores_f = scores_f.to(device, non_blocking=True)

            # forward
            full_out, img_out, skel_out = model(frames_f, keypoints_f, scores_f)
            outs: Dict[str, torch.Tensor] = {"full": full_out, "img": img_out, "skel": skel_out}

            # dimension check
            for t in targets:
                if t not in outs:
                    raise ValueError(f"unknown target {t}. available: {list(outs.keys())}")
                if outs[t].size(-1) != D:
                    raise RuntimeError(f"output dim mismatch: target={t}, out_dim={outs[t].size(-1)}, proto_dim={D}")

            for t in targets:
                pred = F.normalize(outs[t].float(), dim=-1, eps=1e-6)   # (N,D)
                sim = pred @ class_protos.float().T                     # (N,C)
                update_metrics_classification(metrics[t], sim, true_idx, recall_ks)

    # 5) save outputs
    summary_lines: List[str] = []
    for t in targets:
        st = metrics[t]
        if st.total == 0:
            summary_lines.append(f"{t}: total=0 (no valid samples)")
            continue

        rec = {k: st.recall_hits[k] / st.total for k in recall_ks}
        summary_lines.append(
            f"{t}: total={st.total}, " + ", ".join([f"R@{k}={rec[k]:.4f}" for k in recall_ks])
        )

        # confusion pngs
        for k in recall_ks:
            k_dir = out_dir / f"top{k}"
            ensure_dir(k_dir)
            png_path = k_dir / f"confusion_{t}.png"
            save_confusion_png(png_path, st.confusion_by_k[k], normalize="true")

        # per-label recall
        per_label_csv = out_dir / f"per_label_recall_{t}.csv"
        write_per_label_recall_csv(per_label_csv, st, recall_ks, label_id_list)

        # metrics yaml
        out_metrics = {
            "target": t,
            "total": int(st.total),
            "recall": {f"R@{k}": float(rec[k]) for k in recall_ks},
            "min_valid_t": int(min_valid_t),
            "eval_file_list": str(eval_file_list_rel),
            "prototype_file_list": str(proto_file_list_rel),
            "class_count": int(C),
            "label_map": "label_map.yaml",
            "unknown_label_ids": [int(x) for x in unknown_label_ids],
            "per_label_recall_csv": per_label_csv.name,
        }
        with (out_dir / f"metrics_{t}.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(out_metrics, f, allow_unicode=True, sort_keys=False)

    summary_text = "\n".join(summary_lines)
    print(summary_text)
    (out_dir / "summary.txt").write_text(summary_text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
