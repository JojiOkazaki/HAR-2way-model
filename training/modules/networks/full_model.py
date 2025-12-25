import torch
import torch.nn as nn

class FullModel(nn.Module):
    def __init__(self, image_branch, skeleton_branch, mlp):
        super().__init__()
        self.image_branch = image_branch
        self.skeleton_branch = skeleton_branch
        self.mlp = mlp

        self.img_out_dim = self.image_branch.out_dim()
        self.skel_out_dim = self.skeleton_branch.out_dim()
        num_classes = self.mlp.out_dim()

        self.image_head = nn.Linear(self.img_out_dim, num_classes)
        self.skeleton_head = nn.Linear(self.skel_out_dim, num_classes)
        self.fusion_norm = nn.LayerNorm(self.img_out_dim + self.skel_out_dim)

    def forward(self, images, skeletons, conf_scores):
        # ImageBranch
        img_out = self.image_branch(images, conf_scores) # shape: (B, d_img)
        img_logits = self.image_head(img_out)

        # SkeletonBranch
        skel_out = self.skeleton_branch(skeletons, conf_scores) # shape: (B, d_skel)
        skel_logits = self.skeleton_head(skel_out)

        # FusionHead
        h_fused = torch.cat([img_out, skel_out], dim=1) # shape: (B, d_img+d_skel)
        h_fused = self.fusion_norm(h_fused)
        h_fused = self.mlp(h_fused) # shape: (B, d_mlp)

        return h_fused, img_logits, skel_logits # それぞれshape: (B, d)
    