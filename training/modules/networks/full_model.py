import torch
import torch.nn as nn
from dataclasses import dataclass

from training.modules.networks.image_branch import ImageOutput
from training.modules.networks.skeleton_branch import SkeletonOutput

@dataclass
class FullModelOutput:
    logits: torch.Tensor
    img: ImageOutput
    skel: SkeletonOutput

class FullModel(nn.Module):
    def __init__(self, image_branch, skeleton_branch, mlp):
        super().__init__()
        self.image_branch = image_branch
        self.skeleton_branch = skeleton_branch
        self.mlp = mlp

        img_trans_dim = self.image_branch.transformer.cls_token.size(-1)
        skel_trans_dim = self.skeleton_branch.transformer.cls_token.size(-1)
        num_classes = self.mlp.model[-1].out_features

        self.image_head = nn.Linear(img_trans_dim, num_classes)
        self.skeleton_head = nn.Linear(skel_trans_dim, num_classes)
        self.fusion_norm = nn.LayerNorm(img_trans_dim + skel_trans_dim)

    def forward(self, images, skeletons, conf_scores):
        # ImageBranch用のマスク作成
        image_valid_mask = conf_scores.max(dim=-1).values > 0  # shape: (B, T, N)

        # ImageBranch
        img_out = self.image_branch(images, image_valid_mask)
        img_logits = self.image_head(img_out.trans)

        # SkeletonBranch
        skel_out = self.skeleton_branch(skeletons, conf_scores)
        skel_logits = self.skeleton_head(skel_out.trans)

        # FusionHead
        h_fused = torch.cat([img_out.trans, skel_out.trans], dim=1) # shape: (B, d_img+d_skel)
        h_fused = self.fusion_norm(h_fused)
        h_fused = self.mlp(h_fused) # shape: (B, d_mlp)

        return h_fused, img_logits, skel_logits # FullModelOutput(logits=h_fused, img=img_out, skel=skel_out)
