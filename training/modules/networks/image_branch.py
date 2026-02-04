import torch
import torch.nn as nn
import torchvision.models as models

class ImageBranch(nn.Module):
    def __init__(self, transformer, model_name='resnet18'):
        super().__init__()
        
        # 1. 学習済みResNetのロード
        if model_name == 'resnet18':
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            out_dim = 512
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            out_dim = 2048
        else:
            # デフォルトなど必要に応じて追加
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            out_dim = 512
        
        # 2. 分類層(fc)を無効化
        self.backbone.fc = nn.Identity()
        
        # 3. Transformerの入力次元を取得 (修正箇所)
        # transformer.d_model が無い場合でも cls_token の形状から取得する
        if hasattr(transformer, 'd_model'):
            transformer_dim = transformer.d_model
        else:
            # cls_token shape: [1, 1, d_model] -> shape[-1] で取得
            transformer_dim = transformer.cls_token.shape[-1]

        # 4. ResNetの出力次元をTransformerの入力次元に合わせる線形層
        if out_dim != transformer_dim:
            self.projector = nn.Linear(out_dim, transformer_dim)
        else:
            self.projector = nn.Identity()

        self.transformer = transformer
        self._out_dim = transformer_dim

    def forward(self, x, confs):
        # x shape: (B, T, C, H, W)
        B, T, C, H, W = x.shape

        # ResNetは (N, C, H, W) を受け取るため、バッチと時間をマージ
        x = x.reshape(-1, C, H, W) # (B*T, C, H, W)
        
        # ResNetによる特徴抽出
        h_cnn = self.backbone(x) # shape: (B*T, 512) for ResNet18
        
        # 次元変換 (512 -> 128)
        h_cnn = self.projector(h_cnn) # shape: (B*T, 128)

        # Transformer用に形状を戻す
        h_cnn = h_cnn.reshape(B, T, -1) # (B, T, 128)

        # TransformerEncoder
        # confsからマスクを作成 (信頼度が全て0のフレームはマスクする)
        person_has_keypoint = confs.max(dim=-1).values > 0 # shape: (B, T)
        frame_mask = ~person_has_keypoint # Trueの時マスクが有効になる (padding mask)
        
        # TransformerEncoder.forward は (z, mask) を受け取る
        h_trans = self.transformer(h_cnn, frame_mask)

        return h_trans

    def out_dim(self):
        return self._out_dim