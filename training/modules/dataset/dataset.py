import os
from glob import glob

import torch
from torch.utils.data import Dataset

class SyncedDataset(Dataset):
    def __init__(self, img_torch_path, skel_torch_path, file_list=None):
        self.img_torch_path = img_torch_path
        self.skel_torch_path = skel_torch_path

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
        frames = img_data["images"] # shape: (T, N, C_img, H, W)
        keypoints = skel_data["skeletons"] # shape: (T, N, K, C_kp)
        scores = skel_data["scores"] # shape: (T, N, K)
        label = torch.tensor(img_data["label"], dtype=torch.long) # shape: 'int'

        return frames.float(), keypoints.float(), scores.float(), label
