import torch.nn as nn

from .base import BaseBlock, GroupNorm

class ConvBlock(BaseBlock):
    def __init__(self, in_dim, out_dim, dropout=0.0, kernel_size=3, padding=1, max_pool_2d=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, padding=padding),
            GroupNorm(out_dim),
            nn.ReLU(),
            self._make_dropout(dropout),
            nn.MaxPool2d(max_pool_2d),
        )

    def forward(self, x):
        # input shape: (B, C_in, H, W)
        # output shape: (B, C_out, H', W')
        return self.block(x)
