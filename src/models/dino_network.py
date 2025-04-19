import torch
import torch.nn as nn
from einops import rearrange

from src.utils.data import center_padding


class DINO_Network(nn.Module):
    def __init__(self, n_blocks=12, model="vitb8", use_value_facet=False):
        super().__init__()

        if model in ["vitb8", "vits8"]:
            self.output_subsample = 8
        elif model in ["vitb16", "vits16"]:
            self.output_subsample = 16

        self.n_blocks = n_blocks

        self.dino_model = torch.hub.load(
            "facebookresearch/dino:main", "dino_{}".format(model)
        ).eval()
        assert n_blocks <= len(self.dino_model.blocks)

        self.out_dim = 384 if model == "vits8" else 768

        self.dino_model.blocks = self.dino_model.blocks[:n_blocks]
        for param in self.dino_model.parameters():
            param.requires_grad = False

        self.use_value_facet = use_value_facet
        if use_value_facet:
            self.dino_model.blocks[-1].attn.attn_drop = nn.Identity()
            self.dino_model.blocks[-1].attn.proj = nn.Identity()
            self.dino_model.blocks[-1].attn.proj_drop = nn.Identity()
            self.dino_model.blocks[-1].drop_path = nn.Identity()
            self.dino_model.blocks[-1].norm2 = nn.Identity()
            self.dino_model.blocks[-1].mlp = nn.Identity()

    def forward(self, x):
        # pad to multiple of 14
        x = center_padding(x, self.output_subsample)

        B, C, H, W = x.shape

        with torch.no_grad():
            x = self.dino_model.prepare_tokens(x)

            # if I stop before the last block, the elements should have all the relevant information about the image patches
            # the attention masks that they use to segment images attend to these features
            for blk in self.dino_model.blocks[: self.n_blocks]:
                x = blk(x)

        x = rearrange(
            x[:, 1:],  # first token does not derive from an image patch
            "b (h w) c -> b c h w",
            h=int(H // self.output_subsample),
            w=int(W // self.output_subsample),
        )
        if self.use_value_facet:
            x = x[:, -self.out_dim :]

        return x
