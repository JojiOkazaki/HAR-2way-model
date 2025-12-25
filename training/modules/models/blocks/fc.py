import torch.nn as nn

from .base import BaseBlock

class FCBlock(BaseBlock):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            self._make_dropout(dropout),
        )

    def forward(self, x):
        # input shape: (B, in_dim)
        # output shape: (B, out_dim)
        return self.block(x)
