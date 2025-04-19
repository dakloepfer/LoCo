"""Feature extractor that uses frozen, pre-trained DINO model and computes residuals to the frozen DINO layers using a CNN, as used by the DinoTracker paper."""

import antialiased_cnns
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.utils.data import center_padding

OUT_DIMS = {
    "vits14": 384,
    "vitb14": 768,
    "vitl14": 1024,
    "vitg14": 1536,
    "vits8": 384,
    "vitb8": 768,
}


def align_cnn_vit_features(
    vit_features_bchw: torch.Tensor,
    cnn_features_bchw: torch.Tensor,
    vit_patch_size: int = 14,
    vit_stride: int = 7,
    cnn_stride: int = 8,
) -> torch.Tensor:
    """
    Assumptions:
    1. CNN layers are fully padded, thus the feature in the top left corner is centered at the [0, 0] pixel in the image.
    2. ViT patch embed layer has no padding, thus the feature in the top left corner is centered at [vit_patch / 2, vit_patch / 2].
    3. Feature and pixel positions are based on square pixels and refer to the center of the square
       (hence `align_corners=True` in grid_sample)
    :param vit_features_bchw: input ViT features (device and dtype will be set according th them)
    :param cnn_features_bchw: input CNN features to be aligned to ViT features
    :param vit_patch_size:
    :param vit_stride:
    :param cnn_stride:
    :return: CNN features sampled at ViT grid positions
    """
    with torch.no_grad():
        dtype = vit_features_bchw.dtype
        device = vit_features_bchw.device

        # number of features (ViT/CNN) we got
        v_sz = vit_features_bchw.shape[-2:]
        c_sz = cnn_features_bchw.shape[-2:]

        # pixel position of the bottom right feature
        c_br = [(s_ - 1) * cnn_stride for s_ in c_sz]

        # pixel locations of ViT features
        vit_x = (
            torch.arange(v_sz[1], dtype=dtype, device=device) * vit_stride
            + vit_patch_size / 2.0
        )
        vit_y = (
            torch.arange(v_sz[0], dtype=dtype, device=device) * vit_stride
            + vit_patch_size / 2.0
        )
        # map pixel locations to CNN feature locations in [-1, 1] scaled interval

        vit_grid_x, vit_grid_y = torch.meshgrid(
            -1.0 - (1.0 / c_br[1]) + (2.0 * vit_x / c_br[1]),
            -1 - (1.0 / c_br[0]) + (2.0 * vit_y / c_br[0]),
            indexing="xy",
        )
        grid = torch.stack([vit_grid_x, vit_grid_y], dim=-1)[None, ...].expand(
            vit_features_bchw.shape[0], -1, -1, -1
        )
    grid.requires_grad_(
        False
    )  # do not propagate gradients to the grid, only to the sampled features.
    aligned_cnn_features = F.grid_sample(
        cnn_features_bchw,
        grid=grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return aligned_cnn_features


class DeltaDINO_FeatureExtractor(nn.Module):

    def __init__(
        self,
        pretrained_model: str = "dino",
        vit_backbone: str = "vitb8",
        feature_dim: int = 768,
        cnn_config=None,
    ):
        super().__init__()

        # Create frozen & pre-trained DINO(v2) model
        assert feature_dim == OUT_DIMS[vit_backbone]
        self.model_type = pretrained_model
        if pretrained_model == "dinov2":
            self.pretrained_model = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_{}".format(vit_backbone)
            )
            # ['vits14', 'vitb14', 'vitl14', 'vitg14']
            self.vit_patch_size = 14
            self.vit_stride = 14

        elif pretrained_model == "dino":
            self.pretrained_model = torch.hub.load(
                "facebookresearch/dino:main", "dino_{}".format(vit_backbone)
            )
            # ['vits8', 'vitb8']
            self.vit_patch_size = 8
            self.vit_stride = 8

        else:
            raise NotImplementedError(
                f"Pretrained model {pretrained_model} not implemented"
            )

        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        self.output_subsample = self.vit_patch_size

        # create CNN layers
        channels = [3] + list(cnn_config.channels)
        assert channels[-1] == feature_dim
        dilations = list(cnn_config.dilations)
        self.downsample_layers = cnn_config.downsample_layers
        self.down_stride = cnn_config.down_stride
        kernel_size = cnn_config.kernel_size
        padding_mode = "reflect"

        self.layers_list = []
        for i in range(len(channels) - 1):
            is_last_layer = i == len(channels) - 2
            dilation = dilations[i]
            padding = (kernel_size + ((kernel_size - 1) * (dilation - 1))) // 2
            conv_layer = nn.Conv2d(
                channels[i],
                channels[i + 1],
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation,
                padding=padding,
                padding_mode=padding_mode,
            )
            # zero init
            if is_last_layer:
                conv_layer.weight.data = torch.zeros_like(conv_layer.weight.data).to(
                    conv_layer.weight.data.device
                )
                conv_layer.bias.data = torch.zeros_like(conv_layer.bias.data).to(
                    conv_layer.bias.data.device
                )

            self.layers_list.append(conv_layer)
            self.layers_list.append(nn.BatchNorm2d(channels[i + 1]))
            if is_last_layer:
                # initialize gamma of batch norm to inital_gamma
                self.layers_list[-1].weight.data.fill_(0.05)
            if not is_last_layer:
                self.layers_list.append(nn.ReLU())
            if self.downsample_layers[i]:
                self.layers_list.append(
                    antialiased_cnns.BlurPool(channels[i + 1], stride=self.down_stride)
                )

        self.layers = torch.nn.ModuleList(self.layers_list)

    def get_total_stride(self):
        # assumes that model does not contain upsampling layers
        n_down = sum(self.downsample_layers)
        return self.down_stride**n_down

    def forward(self, x):

        x = center_padding(x, self.output_subsample)

        B, C, H, W = x.shape
        # ViT pretrained features
        with torch.no_grad():
            if self.model_type == "dinov2":
                vit_output = self.pretrained_model.prepare_tokens_with_masks(x)
            elif self.model_type == "dino":
                vit_output = self.pretrained_model.prepare_tokens(x)

            for blk in self.pretrained_model.blocks:
                vit_output = blk(vit_output)
            vit_output = rearrange(
                vit_output[:, 1:],
                "b (h w) c -> b c h w",
                h=int(H // self.output_subsample),
                w=int(W // self.output_subsample),
            )

        # CNN features
        for layer in self.layers:
            x = layer(x)

        cnn_stride = self.get_total_stride()
        x = align_cnn_vit_features(
            vit_features_bchw=vit_output,
            cnn_features_bchw=x,
            cnn_stride=cnn_stride,
            vit_patch_size=self.vit_patch_size,
            vit_stride=self.vit_stride,
        )

        return vit_output + x
