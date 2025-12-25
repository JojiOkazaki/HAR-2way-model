import torch
import torch.nn as nn

from training.modules.utils import error

class SkeletonBranch(nn.Module):
    def __init__(self, stgcn):
        super().__init__()
        self.stgcn = stgcn
    
    def forward(self, x, confs):
        # キーポイントデータにNaNが入っていた場合
        if torch.isnan(x).any():
            error("NaN detected in skeleton input")
        
        # 入力形状の把握
        B, T, J, C = x.shape
        
        # キーポイント(x, y)に信頼度スコアを追加して(x, y, conf)の形にする
        confs_exp = confs.unsqueeze(-1) # (B, T, J, 1)
        w = 0.3 + 0.7 * confs_exp
        x_masked = x * w
        h_stgcn = torch.cat([x_masked, confs_exp], dim=-1) # (B, T, J, C+1)

        # ST-GCN
        h_stgcn = h_stgcn.reshape(B, T, J, C+1) # shape: (B, T, J, C+1)
        h_stgcn = self.stgcn(h_stgcn) # shape: (B, d_stgcn)

        return h_stgcn # shape: (B, d_stgcn)

    def out_dim(self):
        return self.stgcn.out_dim
