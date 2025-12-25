import torch
import torch.nn as nn

from training.modules.models.blocks import STGCNBlock
from training.modules.models.blocks import normalize_dropouts

class STGCN(nn.Module):
    def __init__(self, channels, adj, temporal_kernel_size=9, dropout=0.0, dropouts=None):
        super().__init__()
        
        # 出力次元数を明示的に書く
        self.out_dim = int(channels[-1])

        dropouts = normalize_dropouts(len(channels) - 1, dropout, dropouts)

        blocks = []
        for i, (in_dim, out_dim) in enumerate(zip(channels[:-1], channels[1:])):
            blocks.append(
                STGCNBlock(
                    in_dim, out_dim,
                    adj=adj,
                    temporal_kernel_size=temporal_kernel_size,
                    stride=1,
                    dropout=dropouts[i],
                )
            )

        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        # input shape: (B, T, J, C)
        # output shape: (B, out_dim)
        B, T, J, C = x.shape
        x = x.permute(0, 3, 1, 2).contiguous() # shape: (B, C, T, J)
        x = self.model(x) # shape: (B, out_dim, T, J)
        x = x.mean(dim=3) # shape: (B, out_dim, T)
        x = x.mean(dim=2) # shape: (B, out_dim)
        return x
