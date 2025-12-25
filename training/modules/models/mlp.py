import torch.nn as nn

from training.modules.models.blocks import FCBlock
from training.modules.models.blocks import normalize_dropouts

class MLP(nn.Module):
    def __init__(self, layers, dropout=0.0, dropouts=None):
        super().__init__()

        dropouts = normalize_dropouts(len(layers) - 2, dropout, dropouts)

        fc_list = []
        for i, (in_dim, out_dim) in enumerate(zip(layers[:-2], layers[1:-1])):
            d = dropouts[i]
            fc_list.append(FCBlock(in_dim, out_dim, dropout=d))

        fc_list.append(nn.Linear(layers[-2], layers[-1]))

        self.model = nn.Sequential(*fc_list)

    def forward(self, x):
        # input shape: (B, in_dim)
        # output shape: (B, out_dim)
        return self.model(x)

    def out_dim(self):
        return self.model[-1].out_features
