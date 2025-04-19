"""Feature extractor that finetunes a DINOv1 model using PEFT (https://huggingface.co/docs/peft/v0.11.0/en/index)"""

import hydra
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import DictConfig
from peft import get_peft_model


class DINO_PEFT(nn.Module):
    def __init__(
        self,
        vit_backbone: str = "vitb8",
        n_frozen_layers=-1,
        peft_config: DictConfig = None,
        **kwargs,
    ):
        super().__init__()

        if vit_backbone == "vits8":
            self.feature_dim = 384
        elif vit_backbone == "vitb8":
            self.feature_dim = 768

        self.output_subsample = 8

        dino_model = torch.hub.load(
            "facebookresearch/dino:main", "dino_{}".format(vit_backbone)
        )

        self.n_frozen_layers = n_frozen_layers
        if self.n_frozen_layers < 0:
            target_modules = ".*blocks.*(qkv|proj|fc1|fc2)"
            modules_to_save = ["patch_embed"]
        elif self.n_frozen_layers < 8:
            target_modules = (
                r".*blocks.(\b([{}-9]|1[0-2])\b).*(qkv|proj|fc1|fc2)".format(
                    self.n_frozen_layers + 1
                )
            )
            modules_to_save = None
        elif self.n_frozen_layers == 8:
            target_modules = r".*blocks.(\b(9|1[0-2])\b).*(qkv|proj|fc1|fc2)"
            modules_to_save = None
        elif self.n_frozen_layers < 12:
            target_modules = r".*blocks.(\b(1[{}-2])\b).*(qkv|proj|fc1|fc2)".format(
                min(self.n_frozen_layers - 10, 0)
            )
            modules_to_save = None
        else:
            target_modules = None
            modules_to_save = None

        peft_config_class = hydra.utils.instantiate(
            peft_config, target_modules=target_modules, modules_to_save=modules_to_save
        )
        self.model = get_peft_model(dino_model, peft_config_class)

        if self.n_frozen_layers < 0:
            # needed to internally use the correct patch size
            self.model.patch_embed.patch_size = (
                self.model.patch_embed.original_module.patch_size
            )

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.model.prepare_tokens(x)
        for blk in self.model.blocks:
            x = blk(x)

        # turn into extracted features; first token does not derive from an image patch
        x = rearrange(x[:, 1:], "b (h w) c -> b c h w", h=int(H // 8), w=int(W // 8))

        return x
