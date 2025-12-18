import torch
import torch.nn as nn

from training.modules.models.block import GraphBlock, GlobalMeanPooling

class GCN(nn.Module):
    def __init__(self, channels, adj, dropout=0.0, dropouts=None):
        super().__init__()

        num_layers = len(channels) - 1

        # dropoutsの作成
        dropouts = [dropout] * num_layers if dropouts is None else dropouts
        assert len(dropouts) == num_layers, f"dropouts length must be {num_layers}."

        # GCN 層を構築
        gcn_list = []
        for i, (in_dim, out_dim) in enumerate(zip(channels[:-1], channels[1:])):
            d = dropouts[i]
            gcn_list.append(GraphBlock(in_dim, out_dim, adj, dropout=d))

        # 最後の層はプーリング層
        gcn_list.append(GlobalMeanPooling())

        # Sequential モデル化
        self.model = nn.Sequential(*gcn_list)

    def forward(self, x):
        return self.model(x)
