import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class SkeletonOutput:
    trans: torch.Tensor
    pool: torch.Tensor
    gcn: torch.Tensor

class SkeletonBranch(nn.Module):
    def __init__(self, gcn, mh_attn_pool, transformer):
        super().__init__()
        self.gcn = gcn
        self.mh_attn_pool = mh_attn_pool
        self.transformer = transformer
    
    def forward(self, x, confs):
        if torch.isnan(x).any():
            raise RuntimeError("NaN detected in skeleton input")
            
        # 入力形状の把握
        B, T, N, K, C = x.shape

        # 人物valid判定(1関節でもconf>0なら人物ありとする)
        person_valid = (confs.max(dim=-1).values > 0)

        # MultiheadAttentionPooling用マスクの作成
        pool_mask = person_valid.reshape(B*T, N) # shape: (B*T, N)

        # Transformer用マスクの作成
        frame_mask = (person_valid.sum(dim=-1) == 0) # shape: (B, T), Trueの時無効フレーム

        # GCN
        confs_exp = confs.unsqueeze(-1) # (B, T, N, K, 1)
        h_gcn = torch.cat([x, confs_exp], dim=-1) # (B, T, N, K, C+1)
        h_gcn = h_gcn.reshape(B*T*N, K, C+1) # shape: (B*T*N, K, C+1)
        h_gcn = self.gcn(h_gcn) # shape: (B*T*N, d_gcn)

        h_gcn = h_gcn.reshape(B*T, N, -1) # shape: (B*T, N, d_gcn)

        # N人 -> 1人分のデータに集約
        h_pool = self.mh_attn_pool(h_gcn, pool_mask) # shape: (B*T, d_gcn)
        h_pool = h_pool.reshape(B, T, -1) # shape: (B, T, d_gcn)

        # TransformerEncoder
        h_trans = self.transformer(h_pool, frame_mask) # shape: (B, d_transformer)

        return SkeletonOutput(trans=h_trans, pool=h_pool, gcn=h_gcn)
