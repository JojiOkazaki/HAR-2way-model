import torch
import torch.nn as nn

from training.modules.models.blocks import ConvBlock, FCBlock
from training.modules.models.blocks import normalize_dropouts

class CNN(nn.Module):
    def __init__(self, conv_channels, fc_layers, input_size, dropout=0.0, dropouts=None, kernel_size=3, padding=1, max_pool_2d=2,):
        super().__init__()
        
        # 層数の計算
        num_conv = len(conv_channels) - 1
        num_fc   = len(fc_layers) - 1
        total_layers = num_conv + num_fc

        dropouts = normalize_dropouts(total_layers, dropout, dropouts)

        # 畳み込み層
        conv_blocks = []
        for i, (in_ch, out_ch) in enumerate(zip(conv_channels[:-1], conv_channels[1:])):
            conv_blocks.append(ConvBlock(
                in_ch, out_ch,
                dropout=dropouts[i],
                kernel_size=kernel_size,
                padding=padding,
                max_pool_2d=max_pool_2d
            ))

        self.conv = nn.Sequential(*conv_blocks)
        
        # flatten形状の自動計算
        with torch.no_grad():
            dummy = torch.zeros(1, conv_channels[0], *input_size)
            h = self.conv(dummy)
            flatten_dim = h.flatten(1).size(1)

        # 全結合層
        fc_blocks = []
        prev_dim = flatten_dim
        for j, hidden_dim in enumerate(fc_layers[:-1]):
            d = dropouts[num_conv + j]
            fc_blocks.append(FCBlock(prev_dim, hidden_dim, dropout=d))
            prev_dim = hidden_dim
        
        fc_blocks.append(nn.Linear(prev_dim, fc_layers[-1]))

        self.fc = nn.Sequential(*fc_blocks)

        self.model = nn.Sequential(
            self.conv,
            nn.Flatten(),
            self.fc,
        )

    def forward(self, x):
        # input shape: (B, C, H, W)
        # output shape: (B, out_dim)
        return self.model(x)
