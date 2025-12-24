import torch
import torch.nn as nn

# レイヤの定義
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

# ブロックの定義
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential()
    
    def forward(self, x):
        return self.block(x)

class FCBlock(Block):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim), # 多次元入力に対応、CNNもバッチサイズが小さいためこれを使用
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )

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

class GraphBlock(Block):
    def __init__(self, in_dim, out_dim, adj, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            GCNLayer(in_dim, out_dim, adj),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )

# プーリングの定義
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

class SpatialGraphConvBlock(nn.Module):
    """
    ST-GCN の Spatial Graph Convolution（簡易版）
    入力:  x (N, C_in, T, V)
    出力:  y (N, C_out, T, V)
    """
    def __init__(self, in_channels: int, out_channels: int, adj: torch.Tensor, bias: bool = True):
        super().__init__()
        A_hat = self._normalize_adj(adj)
        self.register_buffer("A_hat", A_hat)  # (V, V)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (N, C, T, V) -> (N, C, T, V)
        x = torch.einsum("nctv,vw->nctw", x, self.A_hat)
        return self.conv(x)
    
    def _normalize_adj(adj: torch.Tensor) -> torch.Tensor:
        """
        adj: (V, V) 0/1 adjacency (no self-loop ok)
        returns: A_hat (V, V) normalized with self-loop
        """
        A = adj + torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)
        D = A.sum(dim=1)  # (V,)
        D_inv_sqrt = torch.pow(D.clamp(min=1e-12), -0.5)
        D_inv_sqrt = torch.diag(D_inv_sqrt)
        return D_inv_sqrt @ A @ D_inv_sqrt

class STGCNBlock(nn.Module):
    """
    Spatial GraphConv + Temporal Conv + Residual
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adj: torch.Tensor,
        temporal_kernel_size: int = 9,
        stride: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        if temporal_kernel_size % 2 == 0:
            raise ValueError("temporal_kernel_size must be odd (for same padding).")

        pad_t = (temporal_kernel_size - 1) // 2

        self.gcn = SpatialGraphConvBlock(in_channels, out_channels, adj)

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(temporal_kernel_size, 1),
                stride=(stride, 1),
                padding=(pad_t, 0),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )

        if (in_channels == out_channels) and (stride == 1):
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1), bias=False),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T, V)
        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        x = x + res
        return self.relu(x)
