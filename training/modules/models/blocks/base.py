import torch.nn as nn

from training.modules.utils import error

def normalize_dropouts(num_layers, dropout=0.0, dropouts=None):
    if dropouts is None:
        return [float(dropout)] * num_layers

    if len(dropouts) != num_layers:
        error(f"Dropouts length must be {num_layers}, but got {len(dropouts)}")

    return [float(d) for d in dropouts]

class BaseBlock(nn.Module):
    def _make_dropout(self, p: float):
        return nn.Dropout(p) if p > 0 else nn.Identity()

def GroupNorm(out_dim: int):
    for g in (32, 16, 8, 4, 2, 1):
        if out_dim % g == 0:
            return nn.GroupNorm(g, out_dim)
    return nn.GroupNorm(1, out_dim)  # 念のため
