import os
import torch
import copy
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from datetime import datetime

from training.modules.models import CNN
from training.modules.models import STGCN
from training.modules.models import MLP
from training.modules.models import TransformerEncoder
from training.modules.networks import ImageBranch
from training.modules.networks import SkeletonBranch
from training.modules.networks import FullModel
from training.modules.trainer import Trainer
from training.modules.utils import Logger
from training.modules.utils import lr_lambda
from training.modules.utils import setup_runtime, load_yaml
from training.modules.utils import build_coco17_adj
from training.modules.utils import EarlyStopper
from training.modules.dataset import SyncedDataset, ImageOnlyAugment

from config_base import *

try:
    from config_local import *
except ImportError:
    pass

'''
def build_branch(params, branch_class, branch_key):
    BRANCH_MAP = {"cnn": cnn, "gcn": gcn}
    return branch_class(
        BRANCH_MAP[branch_key](**params[branch_key]),
        TransformerEncoder(**params["transformer"]),
    )
'''

def create_model(params):
    return FullModel(
        ImageBranch(
            CNN(**params["img"]["cnn"]),
            TransformerEncoder(**params["img"]["transformer"]),
        ),
        SkeletonBranch(
            STGCN(**params["skel"]["stgcn"])
        ),
        MLP(**params["mlp"]),
    )

def create_loader(image_dir, skeleton_dir, file_list, batch_size, num_workers, is_train=True, img_aug=None):
    return DataLoader(
        SyncedDataset(
            image_dir,
            skeleton_dir,
            file_list=file_list,
            img_augment=img_aug if is_train else None,
        ), 
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

def train(config):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    # 変数の取り出し
    # logging
    _log_dir = config["logging"]["log_dir"]
    log_dir = ARTIFACT_ROOT / _log_dir / timestamp
    best_model_name = f'{timestamp}_{config["logging"]["best_model_name"]}'
    best_model_path = log_dir / best_model_name
    csv_name = f'{timestamp}_{config["logging"]["csv_name"]}'
    graph_dir = log_dir / config["logging"]["graph_dir"]
    graph_size = config["logging"]["graph_size"]
    # data
    img_pt_dir = DATASET_ROOT / config["data"]["img_pt_dir"]
    skel_pt_dir = DATASET_ROOT / config["data"]["skel_pt_dir"]
    train_file_list = DATASET_ROOT / config["data"]["train_file_list"]
    val_file_list = DATASET_ROOT / config["data"]["val_file_list"]
    num_workers = config["data"]["num_workers"]
    # model
    model_params = copy.deepcopy(config["model"]["architecture"])
    loss_weights = config["model"]["loss_weights"]
    # training
    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    accum_steps = config["training"]["accum_steps"]
    max_norm = config["training"]["max_norm"]
    patience = config["training"]["patience"]
    min_delta = config["training"]["min_delta"]
    # optimizer
    lr = config["optimizer"]["lr"]
    weight_decay = config["optimizer"]["weight_decay"]
    warmup_epochs = config["optimizer"]["warmup_epochs"]
    min_lr_ratio = config["optimizer"]["min_lr_ratio"]
    # runtime
    device = torch.device(config["runtime"]["device"])
    # preprocess
    img_aug = None
    img_aug_cfg = (config.get("preprocess", {}) or {}).get("img_aug", None)

    if isinstance(img_aug_cfg, dict):
        img_aug = ImageOnlyAugment(**img_aug_cfg)

    # ロガーの設定
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

    # データローダーの定義
    print("Create Data Loader...")
    train_loader = create_loader(
        img_pt_dir, skel_pt_dir, train_file_list, 
        batch_size, num_workers, 
        is_train=True, img_aug=img_aug
    )
    val_loader = create_loader(
        img_pt_dir, skel_pt_dir, val_file_list, 
        batch_size, num_workers, 
        is_train=False
    )

    # モデルなどの定義
    print("Create Model...")
    model_params["skel"]["stgcn"]["adj"] = build_coco17_adj(device)
    model = create_model(model_params).to(device)

    decay, no_decay = [], []
    for name, param in model.named_parameters():
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

    trainer = Trainer(model, head_keys, device, accum_steps=accum_steps, loss_weights=loss_weights)
    early_stopper = EarlyStopper(patience=patience, min_delta=min_delta, path=best_model_path)

    # 学習
    print("Training...")
    for epoch in range(epochs):
        train_losses, train_accs = trainer.run_one_epoch(train_loader, optimizer, scaler, max_norm=max_norm, is_train=True)
        val_losses, val_accs = trainer.run_one_epoch(val_loader, max_norm=max_norm, is_train=False)

        # ログの更新
        row = [epoch + 1]
        row.extend(train_losses[k] for k in head_keys)
        row.append(train_losses["main"])
        row.extend(val_losses[k] for k in head_keys)
        row.append(val_losses["main"])
        row.extend(train_accs[k] for k in head_keys)
        row.extend(val_accs[k] for k in head_keys)
        logger.update_csv(row)
        
        print(
            f"[Epoch {epoch+1}]"
            f"acc:{train_accs['full']:.3f}/{val_accs['full']:.3f} | "
            f"main:{train_losses['main']:.3f}/{val_losses['main']:.3f} | "
            f"full:{train_losses['full']:.3f}/{val_losses['full']:.3f} | "
            f"img:{train_losses['img']:.3f}/{val_losses['img']:.3f} | "
            f"skel:{train_losses['skel']:.3f}/{val_losses['skel']:.3f}"
        )

        # 早期終了
        early_stopper(val_losses["main"], model)
        if early_stopper.should_stop:
            break

        scheduler.step()
    
        # グラフの描画
        items = {
            "main": ["loss"],
            **{k: ["loss", "acc"] for k in head_keys}
        }

        for k, names in items.items():
            for name in names:
                logger.create_graph(
                    x_axis_header="epoch",
                    y_axis_headers=[f"t_{name}_{k}", f"v_{name}_{k}"],
                    title=f"{name} ({k})",
                    figsize=graph_size,
                    filename=graph_dir / f"{name}_{k}.png"
                )



if __name__ == "__main__":
    config = load_yaml("training/configs/train.yaml")
    setup_runtime(config["training"]["seed"])
    train(config)
