import torch
import torch.nn as nn

from .base import BaseBlock
from training.modules.utils import error

def _normalize_adj(adj):
    A = adj + torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)
    D = A.sum(dim=1)
    D_inv_sqrt = torch.pow(D.clamp(min=1e-12), -0.5)
    D_inv_sqrt = torch.diag(D_inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt

class ChannelLayerNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous() # (N, C, T, V) -> (N, T, V, C)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous() # (N, T, V, C) -> (N, C, T, V)
        return x

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, adj):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

        # 正規化
        A_hat = _normalize_adj(adj)

        self.register_buffer("A_hat", A_hat) # shape: (N, N)

    def forward(self, x):
        # input shape: (B, N, in_dim)
        # output shape: (B, N, out_dim)
        h = self.A_hat @ x
        return self.linear(h)

class GraphBlock(BaseBlock):
    def __init__(self, in_dim, out_dim, adj, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            GCNLayer(in_dim, out_dim, adj),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            self._make_dropout(dropout),
        )
    
    def forward(self, x):
        return self.block(x)

class SpatialGraphConvBlock(BaseBlock):
    def __init__(self, in_dim, out_dim, adj, bias=True):
        super().__init__()
        
        # adj: (3, K, K) or (K, K)
        # 正規化は skeleton.py 側で完了している前提とします。
        if adj.dim() == 2:
            # 後方互換性（念のため）
            adj = adj.unsqueeze(0)
        
        # adj をバッファとして登録
        self.register_buffer("A", adj) # (Kp, K, K)
        
        self.num_subsets = adj.size(0) # Kp = 3

        # 重み共有ではなく、パーティションごとに独立した重みを持たせる
        # 実装テクニック: 出力チャネルを Kp倍 にしたConv2dを使い、あとで分割して和をとる
        self.conv = nn.Conv2d(in_dim, out_dim * self.num_subsets, kernel_size=1, bias=bias)

    def forward(self, x):
        # x: (B, in_dim, T, V)
        N, C, T, V = x.size()
        
        # 1. 独立した重みで変換
        # out: (B, out_dim * Kp, T, V)
        x = self.conv(x)
        
        # 2. 形状変更 (B, Kp, out_dim, T, V)
        # channel方向を (Kp, out_dim) に分解
        x = x.view(N, self.num_subsets, -1, T, V)
        
        # 3. グラフ畳み込み (各パーティションのAを適用して和をとる)
        # x:      (N, Kp, out, T, V)
        # self.A: (Kp, V, W)   (ここで V=W=ノード数)
        #
        # einsum計算:
        # n: batch
        # k: kernel size (partitions)
        # c: out_channels
        # t: time
        # v: target node
        # w: source node
        #
        # 出力は (N, c, t, v)
        
        x = torch.einsum('nkctw,kvw->nctv', x, self.A)
        
        return x.contiguous()

class STGCNBlock(BaseBlock):
    def __init__(self, in_dim, out_dim, adj, temporal_kernel_size=9, stride=1, dropout=0.0, norm='batch'):
        super().__init__()

        if temporal_kernel_size % 2 == 0:
            error("temporal_kernel_size must be odd (for same padding).")

        pad_t = (temporal_kernel_size - 1) // 2

        self.gcn = SpatialGraphConvBlock(in_dim, out_dim, adj)

        # 正規化層を取得するヘルパー関数
        def get_norm(channels):
            if norm == 'batch':
                return nn.BatchNorm2d(channels)
            elif norm == 'layer':
                return ChannelLayerNorm(channels)
            else:
                raise ValueError(f"Unknown norm type: {norm}")

        # 時間畳み込み層 (切り替え対応)
        self.tcn = nn.Sequential(
            get_norm(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim,
                kernel_size=(temporal_kernel_size, 1),
                stride=(stride, 1),
                padding=(pad_t, 0),
                bias=False,
            ),
            get_norm(out_dim),
            self._make_dropout(dropout),
        )

        # 残差接続
        if (in_dim == out_dim) and (stride == 1):
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=(stride, 1), bias=False),
                get_norm(out_dim),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        x = x + res
        return self.relu(x)
