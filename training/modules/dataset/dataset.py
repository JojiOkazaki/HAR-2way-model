import os
from glob import glob

import torch
from torch.utils.data import Dataset

import torch
import torchvision.transforms.functional as TF

class ImageOnlyAugment:
    def __init__(
        self,
        p=0.8,
        jitter=0.2,
        hue=0.05,
        noise_std=0.03,
        p_erasing=0.25,
        erasing_scale=(0.02, 0.15),
    ):
        self.p = p
        self.jitter = jitter
        self.hue = hue
        self.noise_std = noise_std
        self.p_erasing = p_erasing
        self.erasing_scale = erasing_scale

    @staticmethod
    def _u(a, b, device):
        return (a + (b - a) * torch.rand((), device=device)).item()

    def __call__(self, frames):
        *lead, C, H, W = frames.shape
        device = frames.device
        x = frames.reshape(-1, C, H, W)

        # 1) Color jitter
        if torch.rand((), device=device).item() < self.p:
            j = self.jitter
            b = self._u(max(0.0, 1.0 - j), 1.0 + j, device)
            c = self._u(max(0.0, 1.0 - j), 1.0 + j, device)
            s = self._u(max(0.0, 1.0 - j), 1.0 + j, device)
            h = self._u(-self.hue, self.hue, device)

            # 適用順をランダム化
            ops = ["b", "c", "s", "h"]
            for i in torch.randperm(4, device=device).tolist():
                if ops[i] == "b":
                    x = TF.adjust_brightness(x, b)
                elif ops[i] == "c":
                    x = TF.adjust_contrast(x, c)
                elif ops[i] == "s":
                    x = TF.adjust_saturation(x, s)
                else:
                    x = TF.adjust_hue(x, h)

        # 2) Gaussian noise
        if self.noise_std > 0:
            x = (x + torch.randn_like(x) * self.noise_std).clamp(0.0, 1.0)

        # 3) Random erasing
        if torch.rand((), device=device).item() < self.p_erasing:
            area = H * W
            for i in range(x.shape[0]):
                erase_area = area * self._u(self.erasing_scale[0], self.erasing_scale[1], device)
                r = self._u(0.3, 3.3, device)
                eh = int(round((erase_area * r) ** 0.5))
                ew = int(round((erase_area / r) ** 0.5))
                if eh <= 0 or ew <= 0 or eh >= H or ew >= W:
                    continue
                y0 = int(torch.randint(0, H - eh + 1, (1,), device=device).item())
                x0 = int(torch.randint(0, W - ew + 1, (1,), device=device).item())
                x[i, :, y0:y0+eh, x0:x0+ew] = 0.0

        return x.reshape(*lead, C, H, W)

class SyncedDataset(Dataset):
    def __init__(self, img_torch_path, skel_torch_path, file_list=None, img_augment=None):
        self.img_torch_path = img_torch_path
        self.skel_torch_path = skel_torch_path
        self.img_augment = img_augment

        # サンプルを取得する
        if file_list is None:
            self.samples = sorted(self.img_torch_path.glob("*/*.pt"))
        else:
            self.samples = []
            with open(file_list, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    path, label = line.split()
                    img_path = self.img_torch_path / path
                    self.samples.append(img_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, id):
        # 画像ptファイルパスを取得
        img_path = self.samples[id]

        # 画像ptファイルパスから骨格ptファイルパスを取得
        labelname = img_path.parent.name
        filename = img_path.name
        skel_path = self.skel_torch_path / labelname / filename

        if not skel_path.exists():
            raise FileNotFoundError(skel_path)

        # 画像、骨格ptファイルを読み込み
        img_data = torch.load(img_path, weights_only=True)
        skel_data = torch.load(skel_path, weights_only=True)

        # 各データを取得
        frames = img_data["images"] # shape: (P, T, C_img, H, W)
        keypoints = skel_data["skeletons"] # shape: (P, T, J, C_kp)
        scores = skel_data["scores"] # shape: (P, T, J)
        labels = img_data["labels"] # shape: (P, D)

        frames = frames.float()
        keypoints = keypoints.float()
        scores = scores.float()
        labels = labels.float()

        # 画像データの前処理
        if self.img_augment is not None:
            frames = self.img_augment(frames)

        return frames, keypoints, scores, labels
