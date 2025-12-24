from torch.amp import autocast
from tqdm import tqdm
import torch
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F

class Trainer():
    def __init__(self, model, head_keys, device, accum_steps=1, loss_weights=None):
        self.model = model
        self.head_keys = head_keys
        self.accum_steps = accum_steps
        self.set_device(device)
        self.set_loss_weights(loss_weights)

    def _check_loss_weights(self, loss_weights):
        if isinstance(loss_weights, dict):
            invalid = set(loss_weights) - set(self.head_keys)
            if invalid:
                raise KeyError(f"invalid loss weight keys: {invalid}")

        elif isinstance(loss_weights, (list, tuple)):
            if len(loss_weights) != len(self.head_keys):
                raise ValueError(
                    "loss_weights list length must match head_keys length"
                )

        else:
            raise TypeError("loss_weights must be dict, list, tuple, or None")

    def set_model(self, model):
        self.model = model
    
    def set_head_keys(self, head_keys):
        self.head_keys = head_keys

    def set_device(self, device):
        self.device = device
        self.use_autocast = device.type == "cuda"
    
    def set_accum_steps(self, accum_steps):
        self.accum_steps = accum_steps
    
    def set_loss_weights(self, loss_weights):
        if loss_weights is None:
            self.loss_weights = {k: 1.0 for k in self.head_keys}

        elif isinstance(loss_weights, (list, tuple)):
            self._check_loss_weights(loss_weights)
            self.loss_weights = {
                k: float(w) for k, w in zip(self.head_keys, loss_weights)
            }

        else:
            self._check_loss_weights(loss_weights)
            self.loss_weights = loss_weights

    def forward(self, frames, keypoints, scores):
        outputs_dict = {}

        with autocast("cuda", enabled=self.use_autocast):
            outputs = self.model(frames, keypoints, scores)
            for key, output in zip(self.head_keys, outputs):
                outputs_dict[key] = output
        
        return outputs_dict

    def compute_aux_losses(self, outputs, label, valid_flat=None):
        losses_dict = {}
        main_loss = 0

        label_flat = label.reshape(-1)  # shape: (B*P, )
        if valid_flat is not None:
            label_flat = label_flat[valid_flat] # 有効人物だけ処理する

        with autocast("cuda", enabled=self.use_autocast):
            for key, value in outputs.items():
                logits_flat = value.reshape(-1, value.size(-1))  # shape: (B*P, num_classes)
                if valid_flat is not None:
                    logits_flat = logits_flat[valid_flat] # 有効人物だけ処理する

                loss = F.cross_entropy(logits_flat, label_flat)
                loss *= self.loss_weights[key]

                losses_dict[key] = loss
                main_loss += loss

        return main_loss, losses_dict

    def optimizer_step(self, optimizer, scaler, max_norm=1.0):
        if scaler is not None:
            # 勾配の前処理
            scaler.unscale_(optimizer)
            clip_grad_norm_(self.model.parameters(), max_norm=max_norm)

            # 重みの更新
            scaler.step(optimizer)
            scaler.update()
        
        else:
            clip_grad_norm_(self.model.parameters(), max_norm=max_norm)
            optimizer.step()

        # 勾配のクリア
        optimizer.zero_grad(set_to_none=True)

    def run_one_epoch(self, loader, optimizer=None, scaler=None, max_norm=1.0, is_train=True):
        total = 0
        accum_counter = 0
        avg_accs = {k: 0 for k in self.head_keys}
        avg_losses = {k: 0 for k in self.head_keys}
        avg_losses["main"] = 0

        if is_train:
            self.model.train()
            optimizer.zero_grad(set_to_none=True)
        else:
            self.model.eval()
        
        for frames, keypoints, scores, label in tqdm(loader):
            # データをデバイスに移動
            frames = frames.to(self.device, non_blocking=True)
            keypoints = keypoints.to(self.device, non_blocking=True)
            scores = scores.to(self.device, non_blocking=True)
            label = label.to(self.device, non_blocking=True)

            person_has_keypoint = scores.max(dim=-1).values > 0 # shape: (B, P, T)
            person_valid = person_has_keypoint.any(dim=-1) # shape: (B,P)
            valid_flat = person_valid.reshape(-1) # shape: (B*P, )
            valid_count = int(valid_flat.sum().item())
            if valid_count == 0:
                continue  # このバッチは欠損人物しかいない
            
            # モデルの実行、ロスの計算
            outputs = self.forward(frames, keypoints, scores)
            main_loss, losses = self.compute_aux_losses(outputs, label, valid_flat=valid_flat)

            # 出力用の平均ロス、平均精度の算出
            batch_size = valid_count
            total += batch_size

            avg_losses["main"] += main_loss.item() * batch_size

            for key, value in losses.items():
                avg_losses[key] += value.item() * batch_size

            for key, value in outputs.items():
                logits_flat = value.reshape(-1, value.size(-1)) # shape: (B*P, num_classes)
                pred_flat = logits_flat.argmax(dim=1) # shape: (B*P,　)
                label_flat = label.reshape(-1) # shape: (B*P,　)
                pred_flat = pred_flat[valid_flat]
                label_flat = label_flat[valid_flat]
                avg_accs[key] += pred_flat.eq(label_flat).sum().item()

            # 勾配の計算
            if is_train:
                # 勾配をためる
                accum_counter += 1
                main_loss /= self.accum_steps

                if scaler is not None:
                    scaler.scale(main_loss).backward()
                else:
                    main_loss.backward()

                # accum_stepsごとに勾配蓄積の重み更新を行う
                if accum_counter == self.accum_steps:
                    self.optimizer_step(optimizer, scaler, max_norm=max_norm)
                    accum_counter = 0

        # train時に端数のデータがあった時
        if is_train and accum_counter > 0:
            self.optimizer_step(optimizer, scaler, max_norm=max_norm)

        # 平均ロス、平均精度の計算
        if total == 0:
            for k in avg_losses:
                avg_losses[k] = float("nan")
            for k in avg_accs:
                avg_accs[k] = float("nan")
            return avg_losses, avg_accs

        for key in avg_losses:
            avg_losses[key] /= total
        for key in avg_accs:
            avg_accs[key] /= total

        return avg_losses, avg_accs
