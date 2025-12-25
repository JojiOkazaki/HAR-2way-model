import torch.nn as nn

class ImageBranch(nn.Module):
    def __init__(self, cnn, transformer):
        super().__init__()
        self.cnn = cnn
        self.transformer = transformer

    def forward(self, x, confs):
        # 入力形状の把握
        B, T, C, H, W = x.shape

        # CNN
        h_cnn = x.reshape(-1, C, H, W) # shape: (B*T, C, H, W)
        h_cnn = self.cnn(h_cnn) # shape: (B*T, d_cnn)
        h_cnn = h_cnn.reshape(B, T, -1) # shape: (B*P, T, d_cnn)

        # TransformerEncoder
        person_has_keypoint = confs.max(dim=-1).values > 0 # shape: (B, T)
        frame_mask = ~person_has_keypoint # Trueの時マスクが有効になる
        frame_mask = frame_mask.reshape(B, T) # shape: (B, T)
        h_trans = self.transformer(h_cnn, frame_mask) # shape: (B, d_trans)

        return h_trans # shape: (B, d_trans)

    def out_dim(self):
        return self.transformer.cls_token.size(-1)
