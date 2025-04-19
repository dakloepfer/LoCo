"""Feature extractor that finetunes the final few layers of a DINO model."""

import torch
import torch.nn as nn
from einops import rearrange


class Pure_DINO_FeatureExtractor(nn.Module):
    def __init__(
        self,
        n_pretrained_blocks: int = 11,
        n_frozen_blocks: int = 8,
        vit_backbone: str = "vitb8",
        feature_dim: int = 768,
    ):
        super().__init__()

        assert n_pretrained_blocks == 11
        assert n_frozen_blocks <= n_pretrained_blocks

        if vit_backbone == "vits8":
            assert feature_dim == 384
        else:
            assert feature_dim == 768

        self.output_subsample = 8
        self.n_pretrained_blocks = n_pretrained_blocks
        self.n_frozen_blocks = n_frozen_blocks
        self.feature_dim = feature_dim

        self.dino_model = torch.hub.load(
            "facebookresearch/dino:main", "dino_{}".format(vit_backbone)
        )

        if self.n_frozen_blocks > 0:
            for param in self.dino_model.patch_embed.parameters():
                param.requires_grad = False
            self.dino_model.pos_embed.requires_grad = False
            self.dino_model.cls_token.requires_grad = False
        for blk in self.dino_model.blocks[: self.n_frozen_blocks]:
            for param in blk.parameters():
                param.requires_grad = False

    def forward(self, x):
        B, C, H, W = x.shape

        with torch.no_grad():
            x = self.dino_model.prepare_tokens(x)
            for blk in self.dino_model.blocks[: self.n_frozen_blocks]:
                x = blk(x)
        for blk in self.dino_model.blocks[
            self.n_frozen_blocks : self.n_pretrained_blocks
        ]:
            x = blk(x)

        # turn into extracted features; first token does not derive from an image patch
        x = rearrange(x[:, 1:], "b (h w) c -> b c h w", h=int(H // 8), w=int(W // 8))

        return x
