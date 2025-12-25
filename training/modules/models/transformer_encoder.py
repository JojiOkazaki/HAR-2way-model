import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_ff, max_len, dropout=0.0):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(max_len + 1, d_model))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_ff,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

    def forward(self, z, mask=None):
        batch_size, token_len, embed_dim = z.shape

        # CLSトークン付与
        cls = self.cls_token.expand(batch_size, 1, embed_dim) # [batch_size, 1, embed_dim]
        x = torch.cat([cls, z], dim=1) # [batch_size, token_len+1, embed_dim]

        # 位置埋め込み
        pos = self.pos_embed[:token_len+1].unsqueeze(0)  # [1, token_len+1, embed_dim]
        x = x + pos

        if mask is not None:
            cls_padding = torch.zeros(batch_size, 1, dtype=torch.bool, device=mask.device)
            src_key_padding_mask = torch.cat([cls_padding, mask], dim=1)  # [batch_size, token_len+1]
        else:
            src_key_padding_mask = None

        h = self.encoder(x, src_key_padding_mask=src_key_padding_mask)  # [batch_size, token_len+1, embed_dim]

        return h[:, 0, :]   # CLS
