import torch
import torch.nn as nn
from einops import rearrange, repeat

from src.utils.data import center_padding

OUT_DIMS = {"vits14": 384, "vitb14": 768, "vitl14": 1024, "vitg14": 1536}


class DINOv2_Network(nn.Module):
    def __init__(
        self,
        n_blocks=12,
        model="vitb14",
        use_value_facet=False,
        use_layer_norm=False,
        concat_class_token=False,
    ):
        super().__init__()

        assert model in ["vits14", "vitb14", "vitl14", "vitg14"]
        self.output_subsample = 14

        self.dinov2_model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_{}".format(model)
        ).eval()
        assert n_blocks <= len(self.dinov2_model.blocks)
        self.n_blocks = n_blocks

        self.out_dim = OUT_DIMS[model]

        self.dinov2_model.blocks = self.dinov2_model.blocks[:n_blocks]
        for param in self.dinov2_model.parameters():
            param.requires_grad = False

        self.use_value_facet = use_value_facet
        if use_value_facet:
            self.dinov2_model.blocks[-1].attn.attn_drop = nn.Identity()
            self.dinov2_model.blocks[-1].attn.proj = nn.Identity()
            self.dinov2_model.blocks[-1].attn.proj_drop = nn.Identity()
            self.dinov2_model.blocks[-1].ls1 = nn.Identity()
            self.dinov2_model.blocks[-1].drop_path1 = nn.Identity()
            self.dinov2_model.blocks[-1].norm2 = nn.Identity()
            self.dinov2_model.blocks[-1].mlp = nn.Identity()
            self.dinov2_model.blocks[-1].ls2 = nn.Identity()
            self.dinov2_model.blocks[-1].drop_path2 = nn.Identity()

        self.use_layer_norm = use_layer_norm
        self.concat_class_token = concat_class_token
        if self.concat_class_token:
            self.out_dim += self.out_dim

    def forward(self, x):

        # pad to multiple of 14
        x = center_padding(x, self.output_subsample)

        B, C, H, W = x.shape

        with torch.no_grad():
            x = self.dinov2_model.prepare_tokens_with_masks(x)

            # if I stop before the last block, the elements should have all the relevant information about the image patches
            # the attention masks that they use to segment images attend to these features
            for blk in self.dinov2_model.blocks:
                x = blk(x)

        if self.use_layer_norm:
            x = self.dinov2_model.norm(x)

        features = rearrange(
            x[:, 1:],
            "b (h w) c -> b c h w",
            h=int(H // self.output_subsample),
            w=int(W // self.output_subsample),
        )
        if self.use_value_facet:
            features = features[:, -self.out_dim :]
        if self.concat_class_token:
            cls_token = x[:, 0]
            cls_token = repeat(
                cls_token, "b c -> b c h w", h=features.shape[-2], w=features.shape[-1]
            )
            features = torch.cat((cls_token, features), dim=1)

        return features
