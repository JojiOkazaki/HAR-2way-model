import torch
import torch.nn as nn

from training.modules.models.block import FCBlock

class MLP(nn.Module):
    def __init__(self, layers, dropout=0.0, dropouts=None):
        super().__init__()

        num_layers = len(layers) - 1

        # dropoutsの作成
        dropouts = [dropout] * num_layers if dropouts is None else dropouts
        assert len(dropouts) == num_layers, f"dropouts length must be {num_layers}."

        # FC層を構築
        fc_list = []
        for i, (in_dim, out_dim) in enumerate(zip(layers[:-2], layers[1:-1])):
            d = dropouts[i]
            fc_list.append(FCBlock(in_dim, out_dim, dropout=d))

        # 最後の層はLinear
        fc_list.append(nn.Linear(layers[-2], layers[-1]))

        # Sequential モデル化
        self.model = nn.Sequential(*fc_list)

    def forward(self, x):
        return self.model(x)
