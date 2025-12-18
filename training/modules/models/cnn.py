import torch
import torch.nn as nn

from training.modules.models.block import FCBlock, ConvBlock

class CNN(nn.Module):
    def __init__(self, conv_channels, fc_layers, dropout=0.0, dropouts=None):
        super().__init__()

        # 層数の計算
        num_conv = len(conv_channels) - 1
        num_fc = len(fc_layers)
        total_layers = num_conv + num_fc

        # dropoutsの作成
        dropouts = [dropout] * total_layers if dropouts is None else dropouts
        assert len(dropouts) == total_layers, f"dropouts length must be {total_layers}."

        # 畳み込み層の作成
        conv_list = []
        for i, (in_ch, out_ch) in enumerate(zip(conv_channels[:-1], conv_channels[1:])):
            conv_list.append(ConvBlock(in_ch, out_ch, dropout=dropouts[i]))

        self.conv = nn.Sequential(*conv_list)

        # 画像次元の計算
        with torch.no_grad():
            dummy = torch.zeros(1, conv_channels[0], 112, 112) # [3, 64, 64]のダミー画像
            h = self.conv(dummy)
            flatten_dim = h.flatten(1).size(1)

        # 全結合層の作成
        fc_list = []
        prev_dim = flatten_dim
        for j, hidden_dim in enumerate(fc_layers[:-1]):
            d = dropouts[num_conv + j]
            fc_list.append(FCBlock(prev_dim, hidden_dim, dropout=d))
            prev_dim = hidden_dim
        
        # 最後の層はLinear -> LayerNorm
        fc_list.append(nn.Linear(prev_dim, fc_layers[-1]))
        fc_list.append(nn.LayerNorm(fc_layers[-1]))

        self.fc = nn.Sequential(*fc_list)

        # モデルの作成
        self.model = nn.Sequential(
            self.conv,
            nn.Flatten(),
            self.fc
        )

    def forward(self, x):
        return self.model(x)
