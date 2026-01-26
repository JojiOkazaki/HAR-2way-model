from torch.amp import autocast
from tqdm import tqdm
import torch
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F

class Trainer():
    def __init__(self, model, head_keys, device, accum_steps=1, loss_weights=None, recall_k=1):
        self.model = model
        self.head_keys = head_keys
        self.accum_steps = accum_steps
        self.set_device(device)
        self.set_loss_weights(loss_weights)
        self.recall_k = recall_k

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

        label_flat = label  # shape: (B, D)
        if valid_flat is not None:
            label_flat = label_flat[valid_flat] # 有効人物だけ処理する, shape: (B', D)

        with autocast("cuda", enabled=self.use_autocast):
            for key, value in outputs.items():
                pred_flat = value.reshape(-1, value.size(-1))  # shape: (B', D)
                if valid_flat is not None:
                    pred_flat = pred_flat[valid_flat] # 有効人物だけ処理する

                cos = F.cosine_similarity(pred_flat, label_flat, dim=-1) # shape: (B', )
                loss = (1.0 - cos).mean()

                loss *= self.loss_weights[key]
                losses_dict[key] = loss
                main_loss += loss

        return main_loss, losses_dict

    def compute_contrastive_losses(self, outputs, label, temperature=0.07):
        losses_dict = {}
        main_loss = 0.0

        # label: (B, D)
        label_norm = F.normalize(label, dim=-1).detach()
        B = label_norm.size(0)

        with autocast("cuda", enabled=self.use_autocast):
            for key, value in outputs.items():
                pred = value  # (B, D) の想定
                pred_norm = F.normalize(pred, dim=-1)

                if B < 2:
                    cos = F.cosine_similarity(pred_norm, label_norm, dim=-1)
                    loss = (1.0 - cos).mean()
                else:
                    logits = (pred_norm.float() @ label_norm.float().T) / float(temperature)
                    targets = torch.arange(B, device=logits.device)
                    loss = 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))

                loss = loss * self.loss_weights[key]
                losses_dict[key] = loss
                main_loss = main_loss + loss

        return main_loss, losses_dict

    def recall_at_k(self, pred, label):
        """
        pred:  (B, D)
        label: (B, D)
        return: スカラー（0〜1）
        """
        pred = F.normalize(pred, dim=-1)
        label = F.normalize(label, dim=-1)

        # (B,B) 類似度行列。i行の正解は i列。
        sim = pred.float() @ label.float().T
        B = sim.size(0)
        if B == 0:
            return torch.tensor(float("nan"), device=sim.device)
        if B == 1:
            return torch.tensor(1.0, device=sim.device)  # 正例しかないので常に当たる

        k_eff = min(self.recall_k, B)
        topk = sim.topk(k_eff, dim=1).indices  # (B, k)
        gt = torch.arange(B, device=sim.device).unsqueeze(1)  # (B,1)
        hit = (topk == gt).any(dim=1).float()  # (B,)
        return hit.mean()  # 0〜1

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

    def run_one_epoch(self, loader, optimizer=None, scaler=None, max_norm=1.0, temperature=0.07, is_train=True):
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

            person_has_keypoint = scores.max(dim=-1).values > 0  # (B,P,T)
            valid_len = person_has_keypoint.sum(dim=-1)          # (B,P)

            MIN_VALID_T = 16  # 例: 32中16フレーム以上（ここは調整用）
            person_valid = valid_len >= MIN_VALID_T              # (B,P)

            valid_flat = person_valid.reshape(-1) # shape: (B*P, )
            valid_count = int(valid_flat.sum().item())
            if valid_count == 0:
                continue  # このバッチは欠損人物しかいない
            
            B, P = scores.shape[:2]
            frames = frames.reshape(B * P, *frames.shape[2:])
            keypoints = keypoints.reshape(B * P, *keypoints.shape[2:])
            scores = scores.reshape(B * P, *scores.shape[2:])
            label = label.reshape(B * P, *label.shape[2:])

            idx = valid_flat.nonzero(as_tuple=False).squeeze(1)  # shape: (valid_count,)
            frames = frames[idx]
            keypoints = keypoints[idx]
            scores = scores[idx]
            label = label[idx]

            valid_flat = None

            # モデルの実行、ロスの計算
            outputs = self.forward(frames, keypoints, scores)
            main_loss, losses = self.compute_contrastive_losses(outputs, label, temperature=temperature)

            # 出力用の平均ロス、平均精度の算出
            batch_size = frames.size(0)
            total += batch_size

            avg_losses["main"] += main_loss.item() * batch_size

            for key, value in losses.items():
                avg_losses[key] += value.item() * batch_size

            for key, value in outputs.items():
                pred_flat = value.reshape(-1, value.size(-1))  # (B*P, D)
                label_flat = label.reshape(-1, label.size(-1)) # (B*P, D)

                r = self.recall_at_k(pred_flat, label_flat)
                avg_accs[key] += (r.item() * batch_size)  # 後で total で割る

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
