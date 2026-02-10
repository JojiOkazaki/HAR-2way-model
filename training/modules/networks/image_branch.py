import torch
import torch.nn as nn
import torchvision.models as models

# 自作CNNモジュールをインポート
from training.modules.models.cnn import CNN
from training.modules.utils import error

class ImageBranch(nn.Module):
    def __init__(self, transformer, backbone_type='resnet18', cnn_params=None):
        super().__init__()
        
        self.backbone_type = backbone_type

        # --- バックボーンの切り替えロジック ---
        if backbone_type == 'custom':
            if cnn_params is None:
                error("backbone='custom' requires 'cnn' params in config.")
            
            # 自作CNNの初期化
            self.backbone = CNN(**cnn_params)
            
            # CNNの出力次元を取得 (FC層の最後の次元)
            out_dim = cnn_params["fc_layers"][-1]

        elif backbone_type.startswith('resnet'):
            # ResNetの初期化
            if backbone_type == 'resnet18':
                self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
                out_dim = 512
            elif backbone_type == 'resnet50':
                self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
                out_dim = 2048
            else:
                # 必要に応じて他のResNetを追加
                self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
                out_dim = 512
            
            # ResNetの分類層(fc)を無効化し、特徴量(Pooling後)を出力とする
            self.backbone.fc = nn.Identity()
            
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")
        
        # ------------------------------------

        # Transformerの入力次元を取得
        if hasattr(transformer, 'd_model'):
            transformer_dim = transformer.d_model
        else:
            transformer_dim = transformer.cls_token.shape[-1]

        # 次元変換層 (Backbone出力 -> Transformer入力)
        if out_dim != transformer_dim:
            self.projector = nn.Linear(out_dim, transformer_dim)
        else:
            self.projector = nn.Identity()

        self.transformer = transformer
        self._out_dim = transformer_dim

    def forward(self, x, confs):
        # x shape: (B, T, C, H, W)
        B, T, C, H, W = x.shape

        # バッチと時間をマージしてバックボーンに入力
        x = x.reshape(-1, C, H, W) # (B*T, C, H, W)
        
        # バックボーンによる特徴抽出
        h_feat = self.backbone(x) # shape: (B*T, out_dim)
        
        # 次元変換
        h_feat = self.projector(h_feat) # shape: (B*T, transformer_dim)

        # Transformer用に形状を戻す
        h_feat = h_feat.reshape(B, T, -1) # (B, T, transformer_dim)

        # マスク作成 (信頼度が全て0のフレームはマスク)
        person_has_keypoint = confs.max(dim=-1).values > 0 # shape: (B, T)
        frame_mask = ~person_has_keypoint 
        
        # TransformerEncoder
        h_trans = self.transformer(h_feat, frame_mask)

        return h_trans

    def out_dim(self):
        return self._out_dim