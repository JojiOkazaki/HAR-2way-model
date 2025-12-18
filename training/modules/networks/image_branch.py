import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class ImageOutput:
    trans: torch.Tensor
    pool: torch.Tensor
    cnn: torch.Tensor

class ImageBranch(nn.Module):
    def __init__(self, cnn, mh_attn_pool, transformer):
        super().__init__()
        self.cnn = cnn
        self.mh_attn_pool = mh_attn_pool
        self.transformer = transformer

    def forward(self, x, mask):
        # 入力形状の把握
        B, T, N, C, H, W = x.shape

        # MultiheadAttentionPooling用マスクの作成
        pool_mask = mask.reshape(B*T, N) # shape: (B*T, N)
        #pool_mask = pool_mask.unsqueeze(-1) # shape: (B*T, N, 1)

        # Transformer用マスクの作成
        frame_mask = (mask.sum(dim=-1) == 0) # shape: (B, T), Trueの時無効フレーム

        # CNN
        h_cnn = x.reshape(-1, C, H, W) # shape: (B*T*N, C, H, W)
        h_cnn = self.cnn(h_cnn) # shape: (B*T*N, d_cnn)
        h_cnn = h_cnn.reshape(B*T, N, -1) # shape: (B*T, N, d_cnn)

        # N人 -> 1人分のデータに集約
        h_pool = self.mh_attn_pool(h_cnn, pool_mask) # shape: (B*T, d_cnn)
        h_pool = h_pool.reshape(B, T, -1) # shape: (B, T, d_cnn)

        # TransformerEncoder
        h_trans = self.transformer(h_pool, frame_mask) # shape: (B, d_transformer)

        return ImageOutput(trans=h_trans, pool=h_pool, cnn=h_cnn)
