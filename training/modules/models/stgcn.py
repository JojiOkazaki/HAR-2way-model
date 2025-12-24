import torch
import torch.nn as nn

from training.modules.models.block import STGCNBlock

class STGCN(nn.Module):
    """
    入力:
      keypoints: (B, P, T, J, C) 例: C=2 (x,y) ※0~1正規化済み
      confs:     (B, P, T, J)    信頼度スコア

    出力:
      person_embed: (B, P, D)  ※時系列と関節を平均プーリングした人物表現
    """
    def __init__(
        self,
        channels,
        adj: torch.Tensor,
        temporal_kernel_size: int = 9,
        dropout: float = 0.0,
        dropouts=None,
        use_conf: bool = True,
        conf_as_gate: bool = False,
    ):
        """
        channels: 例 [C_in, 64, 128, 256]
          - C_in は keypoints のCに use_conf を加味して決める
        adj: (J, J) の隣接行列（COCO17なら17x17）
        use_conf:
          - True: 入力特徴に conf を 1ch 追加（x,y,conf）
        conf_as_gate:
          - True: 入力特徴を conf で乗算して欠損の影響を弱める（0ならゼロ化）
        """
        super().__init__()
        self.use_conf = use_conf
        self.conf_as_gate = conf_as_gate

        num_layers = len(channels) - 1
        dropouts = [dropout] * num_layers if dropouts is None else dropouts
        if len(dropouts) != num_layers:
            raise ValueError(f"dropouts length must be {num_layers}.")

        blocks = []
        for i, (c_in, c_out) in enumerate(zip(channels[:-1], channels[1:])):
            blocks.append(
                STGCNBlock(
                    in_channels=c_in,
                    out_channels=c_out,
                    adj=adj,
                    temporal_kernel_size=temporal_kernel_size,
                    stride=1,
                    dropout=float(dropouts[i]),
                )
            )

        self.model = nn.Sequential(*blocks)

        self.out_dim = int(channels[-1])


    def forward(self, keypoints: torch.Tensor, confs: torch.Tensor) -> torch.Tensor:
        if torch.isnan(keypoints).any():
            raise RuntimeError("NaN detected in keypoints input")

        # shapes
        B, P, T, J, C = keypoints.shape
        if confs.shape != (B, P, T, J):
            raise ValueError(f"confs shape must be (B,P,T,J) but got {tuple(confs.shape)}")

        x = keypoints
        if self.use_conf:
            x = torch.cat([x, confs.unsqueeze(-1)], dim=-1)  # (B,P,T,J,C+1)

        if self.conf_as_gate:
            gate = confs.unsqueeze(-1)  # (B,P,T,J,1)
            x = x * gate

        # (B,P,T,J,F) -> (B*P, F, T, J)
        x = x.reshape(B * P, T, J, x.size(-1)).permute(0, 3, 1, 2).contiguous()

        # ST-GCN
        x = self.model(x)  # (B*P, D, T, J)

        # Pool joints & time -> person embedding
        x = x.mean(dim=3)   # (B*P, D, T)
        x = x.mean(dim=2)   # (B*P, D)
        x = x.reshape(B, P, -1)  # (B, P, D)

        return x
