import torch
import torch.nn as nn

# ========================================================================
# GCN層の定義
# ========================================================================
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, adj):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

        # 正規化
        A = adj + torch.eye(adj.size(0), device=adj.device)
        D = A.sum(dim=1)
        D_inv_sqrt = torch.diag(torch.pow(D, -0.5))
        A_hat = D_inv_sqrt @ A @ D_inv_sqrt

        self.register_buffer("A_hat", A_hat)

    def forward(self, x):
        h = self.A_hat @ x # shape: (B*T*N, K, C)
        return self.linear(h)

# ========================================================================
# ブロックの定義
# ========================================================================
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential()
    
    def forward(self, x):
        return self.block(x)

# ========================================================================
# 全結合ブロック
# ========================================================================
class FCBlock(Block):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim), # 多次元入力に対応、CNNもバッチサイズが小さいためこれを使用
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )

# ========================================================================
# 畳み込みブロック
# ========================================================================
class ConvBlock(Block):
    def __init__(self, in_dim, out_dim, kernel_size=3, padding=1, dropout=0.0, max_pool_2d=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.MaxPool2d(max_pool_2d),
        )

# ========================================================================
# グラフ畳み込みブロック
# ========================================================================
class GraphBlock(Block):
    def __init__(self, in_dim, out_dim, adj, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            GCNLayer(in_dim, out_dim, adj),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )

# ========================================================================
# GlobalMeanPooling
# ========================================================================
class GlobalMeanPooling(nn.Module):
    def forward(self, x):
        return x.mean(dim=1) # (N, F) -> (F, )

# ========================================================================
# GlobalSumPooling
# ========================================================================
class GlobalSumPooling(nn.Module):
    def forward(self, x):
        return x.sum(dim=0)

# ========================================================================
# GlobalMaxPooling
# ========================================================================
class GlobalMaxPooling(nn.Module):
    def forward(self, x):
        return x.max(dim=0).values

# ========================================================================
# MultiheadAttentionPooling
# ========================================================================
class MultiheadAttentionPooling(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.0):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )
        self.query = nn.Parameter(torch.empty(1, 1, dim))
        nn.init.xavier_uniform_(self.query)

    def forward(self, z, mask=None):
        batch_size = z.size(0)

        q = self.query.expand(batch_size, -1, -1)

        if mask is not None:
            all_false = (~mask).all(dim=1)
            if all_false.any():
                mask = mask.clone()
                mask[all_false, 0] = True
            key_padding_mask = ~mask
        else:
            key_padding_mask = None
        
        pooled, _ = self.attention(q, z, z, key_padding_mask=key_padding_mask)

        return pooled.squeeze(1)