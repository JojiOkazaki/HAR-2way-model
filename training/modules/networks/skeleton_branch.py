import torch
import torch.nn as nn

class SkeletonBranch(nn.Module):
    def __init__(self, gcn, transformer):
        super().__init__()
        self.gcn = gcn
        self.transformer = transformer
    
    def forward(self, x, confs):
        # キーポイントデータにNaNが入っていた場合
        if torch.isnan(x).any():
            raise RuntimeError("NaN detected in skeleton input")
        
        # 入力形状の把握
        B, P, T, J, C = x.shape
        
        # キーポイント(x, y)に信頼度スコアを追加して(x, y, conf)の形にする
        confs_exp = confs.unsqueeze(-1) # (B, P, T, J, 1)
        h_gcn = torch.cat([x, confs_exp], dim=-1) # (B, P, T, J, C+1)

        # GCN
        h_gcn = h_gcn.reshape(B*P*T, J, C+1) # shape: (B*P*T, J, C+1)
        h_gcn = self.gcn(h_gcn) # shape: (B*P*T, d_gcn)
        h_gcn = h_gcn.reshape(B*P, T, -1) # shape: (B*P, T, d_gcn)

        # TransformerEncoder
        person_has_keypoint = confs.max(dim=-1).values > 0 # shape: (B, P, T)
        frame_mask = ~person_has_keypoint # Trueの時マスクが有効になる
        frame_mask = frame_mask.reshape(B*P, T) # shape: (B*P, T)
        h_trans = self.transformer(h_gcn, frame_mask) # shape: (B*P, d_trans)
        h_trans = h_trans.reshape(B, P, -1) # shape: (B, P, d_trans)

        return h_trans
