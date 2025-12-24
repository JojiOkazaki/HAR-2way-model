import torch
import torch.nn as nn

class ImageBranch(nn.Module):
    def __init__(self, cnn, transformer):
        super().__init__()
        self.cnn = cnn
        self.transformer = transformer

    def forward(self, x, confs):
        # 入力形状の把握
        B, P, T, C, H, W = x.shape

        # CNN
        h_cnn = x.reshape(-1, C, H, W) # shape: (B*P*T, C, H, W)
        h_cnn = self.cnn(h_cnn) # shape: (B*P*T, d_cnn)
        h_cnn = h_cnn.reshape(B*P, T, -1) # shape: (B*P, T, d_cnn)

        # TransformerEncoder
        person_has_keypoint = confs.max(dim=-1).values > 0 # shape: (B, P, T)
        frame_mask = ~person_has_keypoint # Trueの時マスクが有効になる
        frame_mask = frame_mask.reshape(B*P, T) # shape: (B*P, T)
        h_trans = self.transformer(h_cnn, frame_mask) # shape: (B*P, d_trans)
        h_trans = h_trans.reshape(B, P, -1) # shape: (B, P, d_trans)

        return h_trans
