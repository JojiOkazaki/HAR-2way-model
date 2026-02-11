# training/main.py
import os
import copy
import time
from datetime import datetime
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional, Iterator

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import GradScaler

from training.modules.models import CNN, STGCN, MLP, TransformerEncoder
from training.modules.networks import ImageBranch, SkeletonBranch, FullModel
from training.modules.trainer import Trainer
from training.modules.utils import Logger, lr_lambda
from training.modules.utils import setup_runtime, load_yaml, format_hhmmss
from training.modules.utils import build_coco17_adj
from training.modules.utils import EarlyStopper

# GPU版augment（training/modules/utils/augment.py）
from training.modules.utils.augment import ImageOnlyAugment, SkeletonAugment

# __init__.py を変更しないため明示import
from training.modules.dataset.dataset import SyncedDataset, pad_person_collate

from config_base import *

try:
    from config_local import *  # noqa: F401,F403
except ImportError:
    pass


# -----------------------------
# config helpers
# -----------------------------
def resolve_dataset_root(cfg: dict) -> Path:
    ds = (cfg.get("dataset", {}) or {})
    name = ds.get("name", None)
    if name:
        return (DATASETS_ROOT / name).resolve()
    return Path(DATASET_ROOT).resolve()


def apply_data_path_defaults(cfg: dict) -> dict:
    """
    data.processed_dir / data.split がある場合に、関連パスを補完する。
    既に明示指定がある場合は上書きしない。
    """
    data = (cfg.get("data", {}) or {})
    processed = data.get("processed_dir", None)
    split = data.get("split", "default")

    if processed:
        data.setdefault("img_pt_dir", f"{processed}/pt")
        data.setdefault("skel_pt_dir", f"{processed}/pt")
        data.setdefault("train_file_list", f"{processed}/splits/{split}/train_list.txt")
        data.setdefault("val_file_list", f"{processed}/splits/{split}/val_list.txt")

    cfg["data"] = data
    return cfg


# -----------------------------
# finetune helpers
# -----------------------------
def _torch_load_any(path: Path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _load_pretrained_if_needed(model: torch.nn.Module, cfg: dict, device: torch.device) -> None:
    ft = (cfg.get("finetune", {}) or {})
    ckpt = ft.get("init_checkpoint", None)
    if not ckpt:
        return

    ckpt_path = Path(ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = (ARTIFACT_ROOT / ckpt_path).resolve()

    if not ckpt_path.exists():
        raise FileNotFoundError(f"init_checkpoint not found: {ckpt_path}")

    strict = bool(ft.get("strict", True))
    state = _torch_load_any(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=strict)


def _freeze_by_prefix(model: torch.nn.Module, cfg: dict) -> None:
    ft = (cfg.get("finetune", {}) or {})
    prefixes = ft.get("freeze_prefixes", []) or []
    if not prefixes:
        return

    for name, p in model.named_parameters():
        if any(name.startswith(pref) for pref in prefixes):
            p.requires_grad = False


# -----------------------------
# model / loader
# -----------------------------
def create_model(params: dict) -> torch.nn.Module:
    # imgの設定ブロックを取得
    img_params = params["img"]

    # backbone設定を取得 (デフォルトは resnet18 とする)
    backbone_type = img_params.get("backbone", "resnet18")

    return FullModel(
        ImageBranch(
            transformer=TransformerEncoder(**img_params["transformer"]),
            backbone_type=backbone_type,      # 追加: バックボーンの種類
            cnn_params=img_params.get("cnn")  # 追加: custom用パラメータ
        ),
        SkeletonBranch(
            STGCN(**params["skel"]["stgcn"]),
        ),
        MLP(**params["mlp"]),
    )


def build_weighted_sampler_from_file_list(file_list: Path, alpha: float = 1.0) -> WeightedRandomSampler:
    """
    file_list: 各行 "relpath class_id"（2列）
    alpha: 1.0 で完全逆頻度、0.5 で緩める
    """
    ids: List[int] = []
    with file_list.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"{file_list}:{line_no}: expected 2 columns, got {len(parts)}")
            _, id_str = parts
            ids.append(int(id_str))

    if not ids:
        raise ValueError(f"empty file_list: {file_list}")

    cnt = Counter(ids)
    weights = [(1.0 / cnt[i]) ** float(alpha) for i in ids]

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )


def create_loader(
    image_dir: Path,
    skeleton_dir: Path,
    file_list: Optional[Path],
    batch_size: int,
    num_workers: int,
    is_train: bool = True,
    sampler=None,
) -> DataLoader:
    """
    注意:
      - Dataset側のaugmentは使わない（= CPU augmentを無効化する）
      - augmentは main.py 側で device 上（GPU）で適用する
    """
    persistent = bool(num_workers and int(num_workers) > 0)
    return DataLoader(
        SyncedDataset(
            image_dir,
            skeleton_dir,
            file_list=file_list,
            img_augment=None,      # CPU augment 無効
            skel_augment=None,     # CPU augment 無効
        ),
        batch_size=batch_size,
        shuffle=(is_train and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        collate_fn=pad_person_collate,
        drop_last=False,
    )


class DeviceAugmentLoader:
    """
    DataLoaderの出力バッチを main process で device に転送し、device上でaugmentを適用する。

    Trainer側は従来通り .to(device) するが、既に同一device上なら実質no-opになる。
    """

    def __init__(
        self,
        loader: DataLoader,
        device: torch.device,
        img_augment=None,
        skel_augment=None,
        enabled: bool = True,
    ):
        self.loader = loader
        self.device = device
        self.img_augment = img_augment
        self.skel_augment = skel_augment
        self.enabled = bool(enabled)

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self) -> Iterator:
        for batch in self.loader:
            if len(batch) == 4:
                frames, keypoints, scores, label = batch
                label_ids = None
            elif len(batch) == 5:
                frames, keypoints, scores, label, label_ids = batch
            else:
                raise ValueError(f"unexpected batch tuple length: {len(batch)}")

            # move to device (GPU)
            frames = frames.to(self.device, non_blocking=True)
            keypoints = keypoints.to(self.device, non_blocking=True)
            scores = scores.to(self.device, non_blocking=True)
            label = label.to(self.device, non_blocking=True)
            if label_ids is not None:
                label_ids = label_ids.to(self.device, non_blocking=True)

            # apply augmentation on device (train only)
            if self.enabled:
                if self.img_augment is not None:
                    # frames: (B, P, T, C, H, W)
                    frames = self.img_augment(frames)

                if self.skel_augment is not None:
                    # keypoints: (B, P, T, J, 2) -> (B*P, T, J, 2) でaugment
                    if keypoints.ndim != 5 or int(keypoints.size(-1)) != 2:
                        raise ValueError(f"unexpected keypoints shape: {tuple(keypoints.shape)}")
                    B, P, T, J, C = keypoints.shape
                    keypoints_f = keypoints.reshape(B * P, T, J, C)
                    keypoints_f = self.skel_augment(keypoints_f)
                    keypoints = keypoints_f.reshape(B, P, T, J, C)

            if label_ids is None:
                yield frames, keypoints, scores, label
            else:
                yield frames, keypoints, scores, label, label_ids


# -----------------------------
# (A) class prototype builder (from .pt labels/label_ids)
# -----------------------------
def _map_unknown_label_ids(label_ids: torch.Tensor, unknown_label_ids: List[int]) -> torch.Tensor:
    if not unknown_label_ids:
        return label_ids
    out = label_ids.clone()
    for u in unknown_label_ids:
        out[out == int(u)] = -1
    return out


@torch.inference_mode()
def build_class_prototypes_from_pt_labels(
    loader: DataLoader,
    min_valid_t: int,
    unknown_label_ids: List[int],
) -> Tuple[torch.Tensor, List[int], Dict[int, int]]:
    """
    `.pt` の labels / label_ids から
    label_id ごとの平均埋め込み（class prototype）を作る。

    returns:
      protos: (C, D) float32, L2-normalized (CPU)
      label_id_list: index -> label_id
      lid_to_cidx: label_id -> index
    """
    sums: Dict[int, torch.Tensor] = {}
    counts: Dict[int, int] = {}

    for batch in loader:
        if len(batch) == 4:
            frames, keypoints, scores, labels = batch
            label_ids = None
        elif len(batch) == 5:
            frames, keypoints, scores, labels, label_ids = batch
        else:
            raise ValueError(f"unexpected batch tuple length: {len(batch)}")

        if label_ids is None:
            raise RuntimeError("label_ids is not provided by dataset/collate. pt に label_ids が無い可能性があります。")

        label_ids = _map_unknown_label_ids(label_ids.long(), unknown_label_ids)

        B, P = scores.shape[:2]
        D = labels.size(-1)

        # 有効人物判定（学習と同じ思想）
        person_has_keypoint = scores.max(dim=-1).values > 0  # (B,P,T)
        valid_len = person_has_keypoint.sum(dim=-1)          # (B,P)
        person_valid = valid_len >= int(min_valid_t)         # (B,P)

        valid_flat = person_valid.reshape(-1)  # (B*P,)
        idx = valid_flat.nonzero(as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            continue

        labels_f = labels.reshape(B * P, D)[idx].float()      # (N,D)
        label_ids_f = label_ids.reshape(B * P)[idx].long()    # (N,)

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
    protos = F.normalize(protos, dim=-1, eps=1e-6)  # cosine用
    lid_to_cidx = {int(lid): i for i, lid in enumerate(label_id_list)}
    return protos.cpu(), label_id_list, lid_to_cidx


# -----------------------------
# train
# -----------------------------
def train(config: dict) -> None:
    config = apply_data_path_defaults(config)
    dataset_root = resolve_dataset_root(config)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    # logging
    _log_dir = config["logging"]["log_dir"]
    log_dir = (ARTIFACT_ROOT / _log_dir / timestamp).resolve()
    best_model_name = f'{timestamp}_{config["logging"]["best_model_name"]}'
    best_model_path = log_dir / best_model_name
    csv_name = f'{timestamp}_{config["logging"]["csv_name"]}'
    graph_dir = log_dir / config["logging"]["graph_dir"]
    graph_size = config["logging"]["graph_size"]

    # data paths
    img_pt_dir = (dataset_root / config["data"]["img_pt_dir"]).resolve()
    skel_pt_dir = (dataset_root / config["data"]["skel_pt_dir"]).resolve()

    train_file_list_rel = (config.get("data", {}) or {}).get("train_file_list", None)
    val_file_list_rel = (config.get("data", {}) or {}).get("val_file_list", None)
    train_file_list = (dataset_root / train_file_list_rel).resolve() if train_file_list_rel else None
    val_file_list = (dataset_root / val_file_list_rel).resolve() if val_file_list_rel else None

    num_workers = int(config["data"]["num_workers"])

    # model
    model_params = copy.deepcopy(config["model"]["architecture"])
    loss_weights = config["model"]["loss_weights"]

    # training
    batch_size = int(config["training"]["batch_size"])
    epochs = int(config["training"]["epochs"])
    accum_steps = int(config["training"]["accum_steps"])
    max_norm = float(config["training"]["max_norm"])
    patience = int(config["training"]["patience"])
    min_delta = float(config["training"]["min_delta"])
    recall_k = int(config["training"]["recall_k"])
    temperature = float(config["training"]["temperature"])

    # (A) mode
    loss_mode = str((config.get("training", {}) or {}).get("loss_mode", "inbatch_infonce"))
    min_valid_t = int((config.get("training", {}) or {}).get("min_valid_t", 16))
    unknown_label_ids = (config.get("training", {}) or {}).get("unknown_label_ids", [999])
    unknown_label_ids = [] if unknown_label_ids is None else [int(x) for x in unknown_label_ids]

    # optimizer
    lr = float(config["optimizer"]["lr"])
    weight_decay = float(config["optimizer"]["weight_decay"])
    warmup_epochs = int(config["optimizer"]["warmup_epochs"])
    min_lr_ratio = float(config["optimizer"]["min_lr_ratio"])

    # runtime
    device = torch.device(config["runtime"]["device"])

    # preprocess (augment objects are created here, but applied on device via DeviceAugmentLoader)
    img_aug = None
    img_aug_cfg = (config.get("preprocess", {}) or {}).get("img_aug", None)
    if isinstance(img_aug_cfg, dict):
        img_aug = ImageOnlyAugment(**img_aug_cfg)

    skel_aug = None
    skel_aug_cfg = (config.get("preprocess", {}) or {}).get("skel_aug", None)
    if isinstance(skel_aug_cfg, dict):
        skel_aug = SkeletonAugment(**skel_aug_cfg)

    # logger
    head_keys = ["full", "img", "skel"]
    csv_headers = (
        ["epoch"] +
        [f"t_loss_{k}" for k in head_keys] + ["t_loss_main"] +
        [f"v_loss_{k}" for k in head_keys] + ["v_loss_main"] +
        [f"t_acc_{k}" for k in head_keys] +
        [f"v_acc_{k}" for k in head_keys]
    )
    logger = Logger(log_dir)
    logger.create_csv(csv_headers, csv_name)
    logger.create_config(config, filename=f'{timestamp}_config.yaml')
    os.makedirs(graph_dir, exist_ok=True)

    # loaders
    print("Create Data Loader...")
    start_time = time.perf_counter()

    sampler = None
    sampler_cfg = (config.get("data", {}) or {}).get("weighted_sampler", None)
    if isinstance(sampler_cfg, dict) and bool(sampler_cfg.get("enabled", False)):
        if train_file_list is None:
            raise ValueError("weighted_sampler enabled but train_file_list is None")
        alpha = float(sampler_cfg.get("alpha", 1.0))
        sampler = build_weighted_sampler_from_file_list(train_file_list, alpha=alpha)

    train_loader_cpu = create_loader(
        img_pt_dir, skel_pt_dir, train_file_list,
        batch_size, num_workers,
        is_train=True,
        sampler=sampler,
    )
    val_loader = create_loader(
        img_pt_dir, skel_pt_dir, val_file_list,
        batch_size, num_workers,
        is_train=False,
        sampler=None,
    )

    # train loader: move + augment on device (GPU)
    train_loader = DeviceAugmentLoader(
        train_loader_cpu,
        device=device,
        img_augment=img_aug,
        skel_augment=skel_aug,
        enabled=True,
    )

    print(f"elapsed time: {format_hhmmss(time.perf_counter() - start_time)}")

    # (A) class prototypes (CPU) for proto_ce mode
    class_protos_cpu = None
    lid_to_cidx = None
    label_id_list = None
    if loss_mode == "proto_ce":
        if train_file_list is None:
            raise ValueError("loss_mode=proto_ce requires train_file_list")

        print("Build class prototypes from train labels...")
        start_time = time.perf_counter()

        # prototype 用ローダ（augmentationなし、samplerなし、shuffleなし）
        proto_bs = int((config.get("training", {}) or {}).get("prototype_batch_size", max(batch_size, 8)))
        proto_loader = create_loader(
            img_pt_dir, skel_pt_dir, train_file_list,
            batch_size=proto_bs,
            num_workers=num_workers,
            is_train=False,
            sampler=None,
        )

        class_protos_cpu, label_id_list, lid_to_cidx = build_class_prototypes_from_pt_labels(
            proto_loader,
            min_valid_t=min_valid_t,
            unknown_label_ids=unknown_label_ids,
        )

        # 保存（推論やデバッグ用）
        ensure_path = log_dir
        ensure_path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"class_protos": class_protos_cpu, "label_id_list": label_id_list},
            log_dir / f"{timestamp}_class_prototypes.pt"
        )
        with (log_dir / f"{timestamp}_label_map.yaml").open("w", encoding="utf-8") as f:
            import yaml
            yaml.safe_dump(
                {
                    "class_count": int(class_protos_cpu.size(0)),
                    "index_to_label_id": [int(x) for x in label_id_list],
                    "unknown_label_ids": [int(x) for x in unknown_label_ids],
                    "min_valid_t": int(min_valid_t),
                    "loss_mode": "proto_ce",
                },
                f,
                allow_unicode=True,
                sort_keys=False,
            )

        print(f"elapsed time: {format_hhmmss(time.perf_counter() - start_time)}")

    # model
    print("Create Model...")
    start_time = time.perf_counter()

    # adj を投入
    model_params["skel"]["stgcn"]["adj"] = build_coco17_adj(device)
    model = create_model(model_params).to(device)

    # finetune
    _load_pretrained_if_needed(model, config, device)
    _freeze_by_prefix(model, config)

    # optimizer params（freezeを除外）
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr
    )
    scaler = GradScaler() if device.type == "cuda" else None
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda e: lr_lambda(e, warmup_epochs, epochs, min_lr_ratio)
    )

    # Trainer
    trainer_kwargs = dict(
        model=model,
        head_keys=head_keys,
        device=device,
        accum_steps=accum_steps,
        loss_weights=loss_weights,
        recall_k=recall_k,
    )

    # proto_ce 用パラメータ
    if loss_mode == "proto_ce":
        trainer_kwargs.update(
            dict(
                loss_mode="proto_ce",
                class_protos=class_protos_cpu,   # (C,D) CPU tensor
                lid_to_cidx=lid_to_cidx,         # label_id -> class_index
                min_valid_t=min_valid_t,
                unknown_label_ids=unknown_label_ids,
            )
        )
    else:
        trainer_kwargs.update(
            dict(
                loss_mode="inbatch_infonce",
                min_valid_t=min_valid_t,
                unknown_label_ids=unknown_label_ids,
            )
        )

    trainer = Trainer(**trainer_kwargs)

    early_stopper = EarlyStopper(patience=patience, min_delta=min_delta, path=best_model_path)
    print(f"elapsed time: {format_hhmmss(time.perf_counter() - start_time)}")

    # training loop
    print("Training...")
    RESET = "\033[0m"
    GREEN = "\033[32m"
    RED = "\033[31m"

    def fmt_colored(val, prev, higher_is_better):
        s = f"{val:.3f}"
        if prev is None:
            return s
        if val == prev:
            return s
        good = (val > prev) if higher_is_better else (val < prev)
        return f"{GREEN if good else RED}{s}{RESET}"

    prev_metrics = None

    for epoch in range(epochs):
        start_time = time.perf_counter()

        train_losses, train_accs = trainer.run_one_epoch(
            train_loader, optimizer, scaler,
            max_norm=max_norm,
            temperature=temperature,
            is_train=True
        )
        val_losses, val_accs = trainer.run_one_epoch(
            val_loader,
            max_norm=max_norm,
            temperature=temperature,
            is_train=False
        )

        row = [epoch + 1]
        row.extend(train_losses[k] for k in head_keys)
        row.append(train_losses["main"])
        row.extend(val_losses[k] for k in head_keys)
        row.append(val_losses["main"])
        row.extend(train_accs[k] for k in head_keys)
        row.extend(val_accs[k] for k in head_keys)
        logger.update_csv(row)

        cur_metrics = {
            "t_acc_full": train_accs["full"],
            "v_acc_full": val_accs["full"],
            "t_loss_main": train_losses["main"],
            "v_loss_main": val_losses["main"],
            "t_loss_full": train_losses["full"],
            "v_loss_full": val_losses["full"],
            "t_acc_img": train_accs["img"],
            "v_acc_img": val_accs["img"],
            "t_acc_skel": train_accs["skel"],
            "v_acc_skel": val_accs["skel"],
        }
        prev = prev_metrics or {}

        print(
            f"[Epoch {epoch+1}]"
            f"acc:{fmt_colored(cur_metrics['t_acc_full'], prev.get('t_acc_full'), True)}/"
            f"{fmt_colored(cur_metrics['v_acc_full'], prev.get('v_acc_full'), True)} | "
            f"main:{fmt_colored(cur_metrics['t_loss_main'], prev.get('t_loss_main'), False)}/"
            f"{fmt_colored(cur_metrics['v_loss_main'], prev.get('v_loss_main'), False)} | "
            f"full:{fmt_colored(cur_metrics['t_loss_full'], prev.get('t_loss_full'), False)}/"
            f"{fmt_colored(cur_metrics['v_loss_full'], prev.get('v_loss_full'), False)} | "
            f"img:{fmt_colored(cur_metrics['t_acc_img'], prev.get('t_acc_img'), True)}/"
            f"{fmt_colored(cur_metrics['v_acc_img'], prev.get('v_acc_img'), True)} | "
            f"skel:{fmt_colored(cur_metrics['t_acc_skel'], prev.get('t_acc_skel'), True)}/"
            f"{fmt_colored(cur_metrics['v_acc_skel'], prev.get('v_acc_skel'), True)}"
        )

        prev_metrics = cur_metrics

        early_stopper(val_losses["main"], model)
        if early_stopper.should_stop:
            break

        scheduler.step()

        items = {"main": ["loss"], **{k: ["loss", "acc"] for k in head_keys}}
        for k, names in items.items():
            for name in names:
                logger.create_graph(
                    x_axis_header="epoch",
                    y_axis_headers=[f"t_{name}_{k}", f"v_{name}_{k}"],
                    title=f"{name} ({k})",
                    figsize=graph_size,
                    filename=graph_dir / f"{name}_{k}.png"
                )

        print(f"elapsed time: {format_hhmmss(time.perf_counter() - start_time)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/configs/train.yaml")
    args = parser.parse_args()

    config = load_yaml(args.config)
    setup_runtime(config["training"]["seed"])
    train(config)
