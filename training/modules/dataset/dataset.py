import torch
from pathlib import Path
from torch.utils.data import Dataset
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


def _torch_load(path: Path):
    try:
        return torch.load(path, weights_only=True)
    except TypeError:
        # 古いPyTorch互換
        return torch.load(path)


def pad_person_collate(batch):
    """
    P（人物数）が可変なので、バッチ内の最大Pにゼロ埋めして揃える。

    batch:
      [(frames, keypoints, scores, labels), ...]
      または
      [(frames, keypoints, scores, labels, label_ids), ...]

    それぞれ:
      frames    : (P,T,C,H,W)
      keypoints : (P,T,J,2)
      scores    : (P,T,J)
      labels    : (P,D)
      label_ids : (P,)  # int（同一ラベル判定用。無い場合は -1）

    returns:
      frames    : (B,Pmax,T,C,H,W)
      keypoints : (B,Pmax,T,J,2)
      scores    : (B,Pmax,T,J)
      labels    : (B,Pmax,D)
      label_ids : (B,Pmax)  # 返せる場合のみ
    """
    if len(batch) == 0:
        raise ValueError("empty batch")

    if len(batch[0]) == 4:
        frames_list, keypoints_list, scores_list, labels_list = zip(*batch)
        label_ids_list = None
    elif len(batch[0]) == 5:
        frames_list, keypoints_list, scores_list, labels_list, label_ids_list = zip(*batch)
    else:
        raise ValueError(f"unexpected sample tuple length: {len(batch[0])}")

    B = len(frames_list)
    P_max = max(int(x.shape[0]) for x in frames_list) if B > 0 else 0

    # 基本shape（P以外は共通前提）
    _, T, C, H, W = frames_list[0].shape
    _, _, J, Ck = keypoints_list[0].shape
    _, D = labels_list[0].shape

    frames_out = torch.zeros((B, P_max, T, C, H, W), dtype=frames_list[0].dtype)
    keypoints_out = torch.zeros((B, P_max, T, J, Ck), dtype=keypoints_list[0].dtype)
    scores_out = torch.zeros((B, P_max, T, J), dtype=scores_list[0].dtype)
    labels_out = torch.zeros((B, P_max, D), dtype=labels_list[0].dtype)

    label_ids_out = None
    if label_ids_list is not None:
        label_ids_out = torch.full((B, P_max), -1, dtype=torch.long)

    for i in range(B):
        P = int(frames_list[i].shape[0])
        if P == 0:
            continue
        frames_out[i, :P] = frames_list[i]
        keypoints_out[i, :P] = keypoints_list[i]
        scores_out[i, :P] = scores_list[i]
        labels_out[i, :P] = labels_list[i]
        if label_ids_out is not None:
            label_ids_out[i, :P] = label_ids_list[i]

    if label_ids_out is None:
        return frames_out, keypoints_out, scores_out, labels_out
    return frames_out, keypoints_out, scores_out, labels_out, label_ids_out

class SyncedDataset(Dataset):
    def __init__(self, img_torch_path, skel_torch_path, file_list=None, img_augment=None):
        self.img_torch_path = Path(img_torch_path)
        self.skel_torch_path = Path(skel_torch_path) if skel_torch_path is not None else None
        self.img_augment = img_augment

        # 新形式: 単一pt（images/skeletons/scores/labels 同梱）
        self._single_pt = (
            self.skel_torch_path is None
            or self.skel_torch_path.resolve() == self.img_torch_path.resolve()
        )

        if file_list is None:
            # フラット/サブフォルダ両対応
            self.samples = sorted(self.img_torch_path.rglob("*.pt"))
        else:
            self.samples = []
            with open(file_list, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 2:
                        raise ValueError(f"{file_list}:{line_no}: expected 2 columns, got {len(parts)}: {line!r}")
                    relpath, _ = parts
                    self.samples.append(self.img_torch_path / relpath)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        if not img_path.exists():
            raise FileNotFoundError(img_path)

        if self._single_pt:
            data = _torch_load(img_path)
            frames = data["images"]         # (P,T,C,H,W) uint8
            keypoints = data["skeletons"]   # (P,T,J,2)
            scores = data["scores"]         # (P,T,J)
            labels = data["labels"]         # (P,D)
            label_ids_raw = data.get("label_ids", None)  # list[int] or None
        else:
            # 旧形式（image_branch/skeleton_branch が別pt）の互換
            labelname = img_path.parent.name
            filename = img_path.name
            skel_path = self.skel_torch_path / labelname / filename
            if not skel_path.exists():
                raise FileNotFoundError(skel_path)

            img_data = _torch_load(img_path)
            skel_data = _torch_load(skel_path)

            frames = img_data["images"]
            keypoints = skel_data["skeletons"]
            scores = skel_data["scores"]
            labels = img_data["labels"]
            label_ids_raw = img_data.get("label_ids", None)

        # images: uint8(0..255) -> float(0..1)
        if frames.dtype == torch.uint8:
            frames = frames.float().div_(255.0)
        else:
            frames = frames.float()
            if frames.numel() > 0 and float(frames.max().item()) > 1.5:
                frames = frames.div(255.0)

        keypoints = keypoints.float()
        scores = scores.float()
        labels = labels.float()

        P = int(frames.shape[0])
        if label_ids_raw is None:
            label_ids = torch.full((P,), -1, dtype=torch.long)
        else:
            label_ids = torch.as_tensor(label_ids_raw, dtype=torch.long)
            if int(label_ids.numel()) != P:
                raise ValueError(
                    f"label_ids length mismatch: got {int(label_ids.numel())} but P={P} ({img_path})"
                )

        if self.img_augment is not None:
            frames = self.img_augment(frames)

        return frames, keypoints, scores, labels, label_ids
