# training/modules/utils/augment.py
from __future__ import annotations

import torch
import math
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

        if self.noise_std > 0:
            x = (x + torch.randn_like(x) * self.noise_std).clamp(0.0, 1.0)

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
                x[i, :, y0:y0 + eh, x0:x0 + ew] = 0.0

        return x.reshape(*lead, C, H, W)


class SkeletonAugment:
    def __init__(self, p=0.5, theta=15, scale=(0.9, 1.1), shift=0.05):
        self.p = p
        self.theta = theta
        self.scale = scale
        self.shift = shift

    def __call__(self, keypoints):
        if torch.rand(1).item() > self.p:
            return keypoints

        device = keypoints.device
        dtype = keypoints.dtype

        P, T, J, C = keypoints.shape

        if self.theta > 0:
            angle = (torch.rand((), device=device, dtype=dtype) * 2 - 1) * float(self.theta)
            rad = angle * (torch.pi / torch.tensor(180.0, device=device, dtype=dtype))
            cos_r = torch.cos(rad)
            sin_r = torch.sin(rad)

            center_x, center_y = 0.5, 0.5
            center_x = torch.tensor(center_x, device=device, dtype=dtype)
            center_y = torch.tensor(center_y, device=device, dtype=dtype)

            x = keypoints[..., 0] - center_x
            y = keypoints[..., 1] - center_y

            new_x = x * cos_r - y * sin_r + center_x
            new_y = x * sin_r + y * cos_r + center_y

            keypoints = torch.stack([new_x, new_y], dim=-1)

        if self.scale:
            s = (
                torch.rand((), device=device, dtype=dtype)
                * (float(self.scale[1]) - float(self.scale[0]))
                + float(self.scale[0])
            )
            center = torch.tensor(0.5, device=device, dtype=dtype)
            keypoints = (keypoints - center) * s + center

        if self.shift > 0:
            dx = (torch.rand((), device=device, dtype=dtype) * 2 - 1) * float(self.shift)
            dy = (torch.rand((), device=device, dtype=dtype) * 2 - 1) * float(self.shift)
            keypoints = keypoints.clone()
            keypoints[..., 0] += dx
            keypoints[..., 1] += dy

        return keypoints.clamp(0.0, 1.0)
