"""Feature extractor that finetunes the final few layers of a CroCo Encoder."""

import torch
import torch.nn as nn
from croco_code.models.croco import CroCoNet
from einops import rearrange


class CroCo_FeatureExtractor(nn.Module):
    def __init__(
        self,
        n_pretrained_blocks: int = 12,
        n_frozen_blocks: int = 9,
        vit_backbone: str = "base",
        feature_dim: int = 768,
        img_size: int = 224,
    ):
        super().__init__()

        assert n_pretrained_blocks == 12
        assert n_frozen_blocks <= n_pretrained_blocks
        assert feature_dim == 768

        if vit_backbone == "base":  # actually used to determine which checkpoint to use
            ckpt = torch.load("pretrained_weights/CroCo_V2_ViTBase_BaseDecoder.pth")
        elif vit_backbone == "small":
            ckpt = torch.load("pretrained_weights/CroCo_V2_ViTBase_SmallDecoder.pth")
        else:
            raise ValueError(
                f"Unknown ViT backbone (used to select the CroCo checkpoint): {vit_backbone}"
            )

        self.output_subsample = 16
        self.n_pretrained_blocks = n_pretrained_blocks
        self.n_frozen_blocks = n_frozen_blocks
        self.feature_dim = feature_dim
        self.img_size = img_size

        self.croco_model = CroCoNet(
            **ckpt.get("croco_kwargs", {}), img_size=self.img_size
        )
        self.croco_model.load_state_dict(ckpt["model"], strict=True)

        # remove a bunch of layers that we don't need
        self.decoder_embed = None
        self.dec_blocks = None
        self.dec_norm = None
        self.prediction_head = None

        if self.n_frozen_blocks > 0:
            for param in self.croco_model.patch_embed.parameters():
                param.requires_grad = False

        for blk in self.croco_model.enc_blocks[: self.n_frozen_blocks]:
            for param in blk.parameters():
                param.requires_grad = False

    def forward(self, x):
        B, C, H, W = x.shape
        if H != W:
            # pad to square on side that is too short
            if H < W:
                x = nn.functional.pad(x, (0, 0, 0, H - W), mode="constant", value=0)
            elif W < H:
                x = nn.functional.pad(x, (0, W - H, 0, 0), mode="constant", value=0)

        if H != self.img_size or W != self.img_size:
            x = nn.functional.interpolate(
                x, size=(self.img_size, self.img_size), mode="bilinear"
            )
        x = self.croco_model._encode_image(x)
        # turn into feature map
        h, w = x[0].shape[-2:]
        x = rearrange(
            x[0],
            "b (h w) c -> b c h w",
            b=B,
            h=int(self.img_size // self.output_subsample),
            w=int(self.img_size // self.output_subsample),
            c=self.feature_dim,
        )

        # crop away the padding in the feature map
        aspect_ratio = W / H
        target_w = int(round(aspect_ratio * self.img_size / self.output_subsample))
        if target_w > w:
            x = x[..., :target_w]
        elif target_w < w:
            target_h = int(
                round(self.img_size / (self.output_subsample * aspect_ratio))
            )
            x = x[..., :target_h, :]
        return x
