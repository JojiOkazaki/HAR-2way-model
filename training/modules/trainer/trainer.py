# training/modules/trainer/trainer.py
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.amp import autocast
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from tqdm import tqdm


class Trainer:
    """
    loss_mode:
      - "inbatch_infonce": 既存の in-batch 類似度行列に対する（multi-positive）InfoNCE
      - "proto_ce":        クラスプロトタイプ（全クラス候補）に対する cross entropy

    main.py 側から以下が渡される想定:
      - loss_mode
      - class_protos (C,D) CPU tensor（proto_ce のみ）
      - lid_to_cidx  dict[int,int] label_id -> class_index（proto_ce のみ）
      - min_valid_t
      - unknown_label_ids（例: [999]）

    augmentation:
      - img_augment: 画像入力(frames)に対するaugment callable（GPU上で適用する想定）
      - skel_augment: 骨格入力(keypoints)に対するaugment callable（GPU上で適用する想定）
        skel_augment は (P,T,J,2) を受ける実装を想定する（バッチ次元BはTrainer側でループ適用）
    """

    def __init__(
        self,
        model,
        head_keys: List[str],
        device: torch.device,
        accum_steps: int = 1,
        loss_weights=None,
        recall_k: int = 1,
        loss_mode: str = "inbatch_infonce",
        class_protos: Optional[torch.Tensor] = None,
        lid_to_cidx: Optional[dict] = None,
        min_valid_t: int = 16,
        unknown_label_ids: Optional[List[int]] = None,
        img_augment: Optional[Callable] = None,
        skel_augment: Optional[Callable] = None,
    ):
        self.model = model
        self.head_keys = list(head_keys)
        self.accum_steps = int(accum_steps)
        self.recall_k = int(recall_k)

        self.loss_mode = str(loss_mode)
        self.min_valid_t = int(min_valid_t)
        self.unknown_label_ids = [int(x) for x in (unknown_label_ids or [])]

        self.set_device(device)
        self.set_loss_weights(loss_weights)
        self.set_augmenters(img_augment=img_augment, skel_augment=skel_augment)

        # proto_ce 用
        self.class_protos = None
        self._lid_table = None
        self._lid_max = -1
        if self.loss_mode == "proto_ce":
            if class_protos is None or lid_to_cidx is None:
                raise ValueError("loss_mode=proto_ce requires class_protos and lid_to_cidx")
            self.set_class_protos(class_protos, lid_to_cidx)

    # -----------------------------
    # setters
    # -----------------------------
    def _check_loss_weights(self, loss_weights):
        if isinstance(loss_weights, dict):
            invalid = set(loss_weights) - set(self.head_keys)
            if invalid:
                raise KeyError(f"invalid loss weight keys: {invalid}")

        elif isinstance(loss_weights, (list, tuple)):
            if len(loss_weights) != len(self.head_keys):
                raise ValueError("loss_weights list length must match head_keys length")

        else:
            raise TypeError("loss_weights must be dict, list, tuple, or None")

    def set_model(self, model):
        self.model = model

    def set_head_keys(self, head_keys):
        self.head_keys = list(head_keys)

    def set_device(self, device: torch.device):
        self.device = device
        self.use_autocast = device.type == "cuda"

    def set_accum_steps(self, accum_steps: int):
        self.accum_steps = int(accum_steps)

    def set_loss_weights(self, loss_weights):
        if loss_weights is None:
            self.loss_weights = {k: 1.0 for k in self.head_keys}
            return

        if isinstance(loss_weights, (list, tuple)):
            self._check_loss_weights(loss_weights)
            self.loss_weights = {k: float(w) for k, w in zip(self.head_keys, loss_weights)}
            return

        self._check_loss_weights(loss_weights)
        self.loss_weights = {k: float(v) for k, v in loss_weights.items()}

    def set_augmenters(self, img_augment: Optional[Callable] = None, skel_augment: Optional[Callable] = None):
        self.img_augment = img_augment
        self.skel_augment = skel_augment

    def set_class_protos(self, class_protos_cpu: torch.Tensor, lid_to_cidx: dict):
        """
        class_protos_cpu: (C,D) CPU tensor
        lid_to_cidx: dict[label_id -> class_index]
        """
        if not isinstance(class_protos_cpu, torch.Tensor):
            raise TypeError("class_protos must be a torch.Tensor")

        # normalize + move to device
        protos = class_protos_cpu.detach().float()
        protos = F.normalize(protos, dim=-1, eps=1e-6).to(self.device)
        self.class_protos = protos  # (C,D)

        # vectorized mapping table for label_id -> class_index
        keys = [int(k) for k in lid_to_cidx.keys()]
        self._lid_max = max(keys) if keys else -1
        if self._lid_max >= 0:
            table = torch.full((self._lid_max + 1,), -1, dtype=torch.long)
            for lid, cidx in lid_to_cidx.items():
                lid = int(lid)
                cidx = int(cidx)
                if 0 <= lid <= self._lid_max:
                    table[lid] = cidx
            self._lid_table = table.to(self.device)
        else:
            self._lid_table = None

    # -----------------------------
    # augmentation (on device)
    # -----------------------------
    def _apply_augment_on_device(
        self,
        frames: torch.Tensor,     # (B,P,T,C,H,W) float
        keypoints: torch.Tensor,  # (B,P,T,J,2) float
        is_train: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dataset側(__getitem__)でaugmentしない構成を前提に、
        Trainer側で device 上（GPU）に転送した後で augment を適用する。

        注意:
          - img_augment / skel_augment は「1サンプル（1動画）」単位のcallを想定するため、
            バッチ次元Bでループして適用する（CPU版datasetの挙動に合わせる）。
          - padded persons も含まれるが、scores で valid 判定して落とすため問題ない。
        """
        if (not is_train) or (self.img_augment is None and self.skel_augment is None):
            return frames, keypoints

        with torch.no_grad():
            B = int(frames.size(0))
            if self.img_augment is not None:
                for i in range(B):
                    frames[i] = self.img_augment(frames[i])

            if self.skel_augment is not None:
                for i in range(B):
                    keypoints[i] = self.skel_augment(keypoints[i])

        return frames, keypoints

    # -----------------------------
    # forward
    # -----------------------------
    def forward(self, frames, keypoints, scores) -> Dict[str, torch.Tensor]:
        outputs_dict = {}
        with autocast("cuda", enabled=self.use_autocast):
            outputs = self.model(frames, keypoints, scores)
            for key, output in zip(self.head_keys, outputs):
                outputs_dict[key] = output
        return outputs_dict

    # -----------------------------
    # in-batch loss (existing)
    # -----------------------------
    @staticmethod
    def _make_ids_unique_for_unknown(label_ids: torch.Tensor) -> torch.Tensor:
        """label_ids < 0 を各サンプル固有IDに置換（unknown同士が正例にならないようにする）"""
        ids = label_ids.clone()
        unknown = ids < 0
        if unknown.any():
            B = ids.numel()
            if (~unknown).any():
                base = int(ids[~unknown].max().item()) + 1
            else:
                base = 0
            uniq = torch.arange(B, device=ids.device, dtype=torch.long) + base
            ids[unknown] = uniq[unknown]
        return ids

    @staticmethod
    def _multi_positive_inbatch_loss(logits: torch.Tensor, pos_mask: torch.Tensor) -> torch.Tensor:
        """
        Multi-positive InfoNCE（行方向）:
        L_i = -log sum_{j in P(i)} exp(logits_ij) + log sum_{a} exp(logits_ia)
        logits: (B,B)
        pos_mask: (B,B) bool（True が正例。対角は True 推奨）
        """
        log_denom = torch.logsumexp(logits, dim=1)  # (B,)
        logits_pos = logits.masked_fill(~pos_mask, float("-inf"))
        log_num = torch.logsumexp(logits_pos, dim=1)  # (B,)
        return -(log_num - log_denom).mean()

    def compute_contrastive_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        label: torch.Tensor,
        temperature: float = 0.07,
        label_ids: Optional[torch.Tensor] = None,
    ):
        losses_dict = {}
        main_loss = 0.0

        label_norm = F.normalize(label.float(), dim=-1, eps=1e-6).detach()
        B = label_norm.size(0)

        for key, value in outputs.items():
            pred = value.float()
            pred_norm = F.normalize(pred, dim=-1, eps=1e-6)

            if B < 2:
                cos = F.cosine_similarity(pred_norm, label_norm, dim=-1)
                loss = (1.0 - cos).mean()
            else:
                logits = (pred_norm @ label_norm.T) / float(temperature)

                if label_ids is None:
                    targets = torch.arange(B, device=logits.device)
                    loss = 0.5 * (
                        F.cross_entropy(logits, targets) +
                        F.cross_entropy(logits.T, targets)
                    )
                else:
                    ids = torch.as_tensor(label_ids, device=logits.device, dtype=torch.long)
                    ids = self._make_ids_unique_for_unknown(ids)
                    pos_mask = (ids.unsqueeze(0) == ids.unsqueeze(1))  # (B,B)

                    loss_row = self._multi_positive_inbatch_loss(logits, pos_mask)
                    loss_col = self._multi_positive_inbatch_loss(logits.T, pos_mask.T)
                    loss = 0.5 * (loss_row + loss_col)

            loss = loss * self.loss_weights[key]
            losses_dict[key] = loss
            main_loss = main_loss + loss

        return main_loss, losses_dict

    def recall_at_k_inbatch(
        self,
        pred: torch.Tensor,   # (B,D)
        label: torch.Tensor,  # (B,D)
        label_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pred = F.normalize(pred.float(), dim=-1, eps=1e-6)
        label = F.normalize(label.float(), dim=-1, eps=1e-6)

        sim = pred @ label.T  # (B,B)
        B = sim.size(0)
        if B == 0:
            return torch.tensor(float("nan"), device=sim.device)
        if B == 1:
            return torch.tensor(1.0, device=sim.device)

        k_eff = min(self.recall_k, B)
        topk = sim.topk(k_eff, dim=1).indices  # (B,k)

        if label_ids is None:
            gt = torch.arange(B, device=sim.device).unsqueeze(1)  # (B,1)
            hit = (topk == gt).any(dim=1).float()
            return hit.mean()

        ids = torch.as_tensor(label_ids, device=sim.device, dtype=torch.long)
        ids = self._make_ids_unique_for_unknown(ids)
        gt_mask = (ids.unsqueeze(0) == ids.unsqueeze(1))  # (B,B)
        hit = gt_mask.gather(1, topk).any(dim=1).float()
        return hit.mean()

    # -----------------------------
    # proto_ce loss / acc
    # -----------------------------
    def _map_unknown_label_ids(self, label_ids: torch.Tensor) -> torch.Tensor:
        if not self.unknown_label_ids:
            return label_ids
        out = label_ids.clone()
        for u in self.unknown_label_ids:
            out[out == int(u)] = -1
        return out

    def _label_ids_to_cidx(self, label_ids: torch.Tensor) -> torch.Tensor:
        """
        label_ids: (N,) on device
        returns:   (N,) class_index, unknown -> -1
        """
        if self._lid_table is None or self._lid_max < 0:
            return torch.full_like(label_ids, -1)

        out = torch.full_like(label_ids, -1)
        m = (label_ids >= 0) & (label_ids <= self._lid_max)
        if m.any():
            out[m] = self._lid_table[label_ids[m]]
        return out

    def compute_proto_ce_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        target_cidx: torch.Tensor,   # (N,) on device, 0..C-1
        temperature: float,
    ):
        losses_dict = {}
        main_loss = 0.0

        for key, value in outputs.items():
            pred = F.normalize(value.float(), dim=-1, eps=1e-6)
            logits = (pred @ self.class_protos.float().T) / float(temperature)  # (N,C)
            loss = F.cross_entropy(logits, target_cidx)
            loss = loss * self.loss_weights[key]
            losses_dict[key] = loss
            main_loss = main_loss + loss

        return main_loss, losses_dict

    def recall_at_k_proto(
        self,
        pred: torch.Tensor,          # (N,D)
        target_cidx: torch.Tensor,   # (N,)
        temperature: float,
    ) -> torch.Tensor:
        pred = F.normalize(pred.float(), dim=-1, eps=1e-6)
        logits = (pred @ self.class_protos.float().T) / float(temperature)  # (N,C)
        k_eff = min(int(self.recall_k), int(logits.size(1)))
        topk = logits.topk(k_eff, dim=1).indices  # (N,k)
        hit = (topk == target_cidx.unsqueeze(1)).any(dim=1).float()
        return hit.mean()

    # -----------------------------
    # optimizer step
    # -----------------------------
    def optimizer_step(self, optimizer, scaler, max_norm: float = 1.0):
        if scaler is not None:
            scaler.unscale_(optimizer)
            clip_grad_norm_(self.model.parameters(), max_norm=max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            clip_grad_norm_(self.model.parameters(), max_norm=max_norm)
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

    # -----------------------------
    # epoch loop
    # -----------------------------
    def run_one_epoch(
        self,
        loader,
        optimizer=None,
        scaler=None,
        max_norm: float = 1.0,
        temperature: float = 0.07,
        is_train: bool = True,
    ):
        total = 0
        accum_counter = 0

        avg_accs = {k: 0.0 for k in self.head_keys}
        avg_losses = {k: 0.0 for k in self.head_keys}
        avg_losses["main"] = 0.0

        if is_train:
            self.model.train()
            optimizer.zero_grad(set_to_none=True)
        else:
            self.model.eval()

        for batch in tqdm(loader):
            if len(batch) == 4:
                frames, keypoints, scores, label = batch
                label_ids = None
            elif len(batch) == 5:
                frames, keypoints, scores, label, label_ids = batch
            else:
                raise ValueError(f"unexpected batch tuple length: {len(batch)}")

            # move to device
            frames = frames.to(self.device, non_blocking=True)
            keypoints = keypoints.to(self.device, non_blocking=True)
            scores = scores.to(self.device, non_blocking=True)
            label = label.to(self.device, non_blocking=True)
            if label_ids is not None:
                label_ids = label_ids.to(self.device, non_blocking=True)

            # GPU augmentation (train only)
            frames, keypoints = self._apply_augment_on_device(frames, keypoints, is_train=is_train)

            # valid person selection (B,P,T,J)
            person_has_keypoint = scores.max(dim=-1).values > 0  # (B,P,T)
            valid_len = person_has_keypoint.sum(dim=-1)          # (B,P)
            person_valid = valid_len >= int(self.min_valid_t)    # (B,P)

            valid_flat = person_valid.reshape(-1)  # (B*P,)
            valid_count = int(valid_flat.sum().item())
            if valid_count == 0:
                continue

            B, P = scores.shape[:2]

            frames = frames.reshape(B * P, *frames.shape[2:])
            keypoints = keypoints.reshape(B * P, *keypoints.shape[2:])
            scores = scores.reshape(B * P, *scores.shape[2:])
            label = label.reshape(B * P, *label.shape[2:])
            if label_ids is not None:
                label_ids = label_ids.reshape(B * P)

            idx = valid_flat.nonzero(as_tuple=False).squeeze(1)
            frames = frames[idx]
            keypoints = keypoints[idx]
            scores = scores[idx]
            label = label[idx]
            if label_ids is not None:
                label_ids = label_ids[idx]

            # unknown label mapping (e.g. 999 -> -1)
            if label_ids is not None:
                label_ids = self._map_unknown_label_ids(label_ids.long())

            # -----------------------------
            # forward + loss
            # -----------------------------
            outputs = self.forward(frames, keypoints, scores)

            if self.loss_mode == "proto_ce":
                if label_ids is None:
                    raise RuntimeError("loss_mode=proto_ce requires label_ids")

                target = self._label_ids_to_cidx(label_ids)  # (N,)
                keep = target >= 0
                if not bool(keep.any()):
                    continue

                # filter all tensors to known classes only
                target = target[keep]
                frames = frames[keep]
                keypoints = keypoints[keep]
                scores = scores[keep]
                label = label[keep]  # (not used in loss, but kept for shape consistency)

                outputs = self.forward(frames, keypoints, scores)
                main_loss, losses = self.compute_proto_ce_losses(outputs, target, temperature=float(temperature))

            else:
                # in-batch InfoNCE（従来）
                main_loss, losses = self.compute_contrastive_losses(
                    outputs, label, temperature=float(temperature), label_ids=label_ids
                )
                target = None  # for type

            # -----------------------------
            # accumulate metrics
            # -----------------------------
            batch_size_eff = int(frames.size(0))
            total += batch_size_eff

            avg_losses["main"] += float(main_loss.item()) * batch_size_eff
            for key, v in losses.items():
                avg_losses[key] += float(v.item()) * batch_size_eff

            for key, v in outputs.items():
                pred_flat = v.reshape(-1, v.size(-1))
                if self.loss_mode == "proto_ce":
                    r = self.recall_at_k_proto(pred_flat, target, temperature=float(temperature))
                else:
                    label_flat = label.reshape(-1, label.size(-1))
                    r = self.recall_at_k_inbatch(pred_flat, label_flat, label_ids=label_ids)
                avg_accs[key] += float(r.item()) * batch_size_eff

            # -----------------------------
            # backward
            # -----------------------------
            if is_train:
                accum_counter += 1
                loss_scaled = main_loss / float(self.accum_steps)

                if scaler is not None:
                    scaler.scale(loss_scaled).backward()
                else:
                    loss_scaled.backward()

                if accum_counter == int(self.accum_steps):
                    self.optimizer_step(optimizer, scaler, max_norm=float(max_norm))
                    accum_counter = 0

        if is_train and accum_counter > 0:
            self.optimizer_step(optimizer, scaler, max_norm=float(max_norm))

        if total == 0:
            for k in avg_losses:
                avg_losses[k] = float("nan")
            for k in avg_accs:
                avg_accs[k] = float("nan")
            return avg_losses, avg_accs

        for k in avg_losses:
            avg_losses[k] /= float(total)
        for k in avg_accs:
            avg_accs[k] /= float(total)

        return avg_losses, avg_accs
