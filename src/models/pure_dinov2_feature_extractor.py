"""Feature extractor that finetunes the final few layers of a DINOv2 model."""

import torch
import torch.nn as nn
from einops import rearrange

OUT_DIMS = {"vits14": 384, "vitb14": 768, "vitl14": 1024, "vitg14": 1536}


class Pure_DINOv2_FeatureExtractor(nn.Module):
    def __init__(
        self,
        n_pretrained_blocks: int = 12,
        n_frozen_blocks: int = 9,
        vit_backbone: str = "vitb14",
        feature_dim: int = 768,
    ):
        super().__init__()

        assert vit_backbone in ["vits14", "vitb14", "vitl14", "vitg14"]
        assert feature_dim == OUT_DIMS[vit_backbone]

        self.output_subsample = 14
        self.n_pretrained_blocks = n_pretrained_blocks
        self.n_frozen_blocks = n_frozen_blocks
        self.feature_dim = feature_dim

        self.dinov2_model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_{}".format(vit_backbone)
        )
        assert n_pretrained_blocks == len(self.dinov2_model.blocks)
        assert n_frozen_blocks <= n_pretrained_blocks

        if self.n_frozen_blocks > 0:
            for param in self.dinov2_model.patch_embed.parameters():
                param.requires_grad = False
            self.dinov2_model.pos_embed.requires_grad = False
            self.dinov2_model.cls_token.requires_grad = False
        for blk in self.dinov2_model.blocks[: self.n_frozen_blocks]:
            for param in blk.parameters():
                param.requires_grad = False

    def forward(self, x):
        B, C, H, W = x.shape

        with torch.no_grad():
            x = self.dinov2_model.prepare_tokens_with_masks(x)
            for blk in self.dinov2_model.blocks[: self.n_frozen_blocks]:
                x = blk(x)
        for blk in self.dinov2_model.blocks[
            self.n_frozen_blocks : self.n_pretrained_blocks
        ]:
            x = blk(x)
        # x = self.dinov2_model.norm(x)

        # turn into extracted features; first token does not derive from an image patch
        x = rearrange(
            x[:, 1:],
            "b (h w) c -> b c h w",
            b=B,
            h=int(H // self.output_subsample),
            w=int(W // self.output_subsample),
            c=self.feature_dim,
        )

        return x
