import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from sklearn.decomposition import PCA

## Some visualisation functions
H, W = 256, 320


def torch_to_cv2_img(img):
    """Converts a PyTorch RGB image normalised to between 0 and 1 to a BGR image normalised to between 0 and 255."""
    img = rearrange(img, "c h w -> h w c").cpu().numpy() * 255
    if img.shape[-1] == 1:  # grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR).astype(np.uint8)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).astype(np.uint8)
    return img


def unnormalise(img):
    """undo the Imagenet normalisation"""
    img = (
        img
        * torch.tensor([0.229, 0.224, 0.225], device=img.device)[None, :, None, None]
    )
    img = (
        img
        + torch.tensor([0.485, 0.456, 0.406], device=img.device)[None, :, None, None]
    )

    return img


def colour_patches(
    imgs, patch_idxs, save_path=None, factor=8, colour=None, run_unnormalise=True
):
    """Colour patches in batches of images red and saves the finished image.

    Parameters
    ----------
    imgs (batch_size x 3 x H x W tensor):
        batch of images

    patch_idxs (batch_size or batch_size x 2 tensor):
        The indices of the patches in the images (one per image), with the index being the patch number in the flattened image.

    save_path (str):
        the file path to save the visualisation

    factor (int, optional):
        downsample-factor; the size of each patch in pixels, by default 8

    colour (tensor of shape batch_size x 3, optional):
        the colour to use for the respective patches in RGB, by default torch.tensor([1, 0, 0]) (Red)

    run_unnormalise (bool, optional):
        whether to un-do the Imagenet normalisation, by default True

    Returns
    -------
    imgs (batch_size x 3 x H x W tensor):
        the visualisation of the images with the patches coloured
    """
    if run_unnormalise:
        imgs = unnormalise(imgs)
    height, width = imgs.shape[-2:]

    if colour is None:
        colour = torch.tensor([1, 0, 0], device=imgs.device)
    if len(colour.shape) == 1:
        colour = repeat(colour, "c -> b c", b=imgs.shape[0])

    for i, patch_idx in enumerate(patch_idxs):
        if len(patch_idxs.shape) == 2:
            row = patch_idx[0]
            col = patch_idx[1]
        else:
            row = patch_idx // (width // factor)
            col = patch_idx % (width // factor)
        imgs[
            i, :, row * factor : (row + 1) * factor, col * factor : (col + 1) * factor
        ] = colour[i, :, None, None]

    # save img with OpenCV
    if save_path:
        save_img = rearrange(imgs, "b c h w -> c (b h) w")
        cv2.imwrite(save_path, torch_to_cv2_img(save_img))

    return imgs


def vis_image_with_patches(
    imgs1,
    imgs2,
    patch_idxs1,
    patch_idxs2,
    save_path,
    factor=8,
    colour=None,
    unnormalise=True,
):
    """Visualise patches in batches of image pairs by colouring them red and saves the finished image.

    Parameters
    ----------
    imgs1 (batch_size x 3 x H x W tensor):
        batch of the first image of the image pairs

    imgs2 (batch_size x 3 x H x W tensor):
        batch of the second image of the image pairs

    patch_idxs1 (batch_size tensor):
        The indices of the patches in the first image (one per image), with the index being the patch number in the flattened image.

    patch_idxs2 (batch_size tensor):
        The indices of the patches in the second image (one per image), with the index being the patch number in the flattened image.

    save_path (str):
        the file path to save the visualisation

    factor (int, optional):
        downsample-factor; the size of each patch in pixels, by default 8

    colour (tensor of shape batch_size x 3, optional):
        the colour to use for the respective patches in RGB, by default torch.tensor([1, 0, 0]) (Red)

    unnormalise (bool, optional):
        whether to un-do the Imagenet normalisation, by default True
    """
    imgs1 = colour_patches(imgs1, patch_idxs1, None, factor, colour, unnormalise)
    imgs2 = colour_patches(imgs2, patch_idxs2, None, factor, colour, unnormalise)

    img = torch.cat([imgs1, imgs2], dim=-1)
    img = rearrange(img, "b c h w -> c (b h) w")

    # save img with OpenCV
    cv2.imwrite(save_path, torch_to_cv2_img(img))


def vis_similarities(
    features1,
    features2,
    query_patches,
    imgs1=None,
    imgs2=None,
    save_path="vis_similarities.png",
    softmax=False,
    alpha=0.5,
    colormap="black-white",
    return_colormaps=False,
):
    """Visualise the similarities between the given query patches in images1 and the patches in images2, either just showing the heat map or overlaying the map on the second image.

    Parameters
    ----------
    features1 (batch_size x feature_dim x H/factor x W/factor tensor):
        the patch features of the first image

    features2 (batch_size x feature_dim x H/factor x W/factor tensor):
        the patch features of the second image

    query_patches (batch_size tensor):
        The indices of the patches in the first image (one per image), with the index being the patch number in the flattened image.

    imgs1 (batch_size x 3 x H x W tensor):
        The images of the first image of the image pairs

    imgs2 (batch_size x 3 x H x W tensor, optional):
        the images of the second image of the image pair; if not None then the similarity map is overlayed on the image.

    save_path (str, optional):
        The file path to save the resulting visualisation to, by default 'vis_similarities.png'

    softmax (bool, optional):
        whether to apply a softmax to the similarity map, by default False. Otherwise similarity map is normalised linearly.

    alpha (float, optional):
        The alpha value for the overlay, by default 0.5. A value of 1.0 results in just the similarity mask being shown.

    colormap (str, optional):
        The colormap to use for the similarity map, by default 'black-white'. If "red", "blue", or "green", the colourmap is a heatmap in that single colour. Colormap can also be 'jet' or 'viridis'.

    return_colormaps (bool, optional):
        Whether to return the (coloured) similarity maps, by default False

    Returns
    -------
    colormaps (batch_size x 3 x H/factor x W/factor tensor), optional:
        the coloured similarity maps, in RGB between 0 and 1.
    """

    feature_height, feature_width = features1.shape[-2:]
    batch_size = features1.shape[0]

    query_patches = query_patches.squeeze()
    assert features1.shape == features2.shape
    assert query_patches.shape[0] == batch_size

    # normalise features
    features1 = F.normalize(features1, dim=1)
    features2 = F.normalize(features2, dim=1)

    if len(query_patches.shape) == 2:
        query_features = features1[
            torch.arange(batch_size), :, query_patches[:, 0], query_patches[:, 1]
        ]
    else:
        query_features = rearrange(features1, "b d h w -> b (h w) d")[
            torch.arange(batch_size), query_patches
        ]

    similarity_maps = einsum(query_features, features2, "b d, b d h w -> b h w")
    if softmax:
        similarity_maps = rearrange(similarity_maps, "b h w -> b (h w)")
        similarity_maps = F.softmax(similarity_maps, dim=-1)
        similarity_maps = rearrange(
            similarity_maps, "b (h w) -> b h w", h=feature_height, w=feature_width
        )
    else:
        similarity_maps = (similarity_maps - similarity_maps.min()) / (
            similarity_maps.max() - similarity_maps.min()
        )

    similarity_maps = torch_to_cv2_img(rearrange(similarity_maps, "b h w -> 1 (b h) w"))
    # overlay the similarity map on the second image
    if colormap == "black-white":
        pass
    elif colormap.lower() == "red":
        similarity_maps = np.stack(
            [
                np.zeros_like(similarity_maps),
                np.zeros_like(similarity_maps),
                similarity_maps,
            ],
            axis=-1,
        )
    elif colormap.lower() == "blue":
        similarity_maps = np.stack(
            [
                similarity_maps,
                np.zeros_like(similarity_maps),
                np.zeros_like(similarity_maps),
            ],
            axis=-1,
        )
    elif colormap.lower() == "green":
        similarity_maps = np.stack(
            [
                np.zeros_like(similarity_maps),
                similarity_maps,
                np.zeros_like(similarity_maps),
            ],
            axis=-1,
        )
    elif colormap.lower() == "jet":
        similarity_maps = cv2.applyColorMap(similarity_maps, cv2.COLORMAP_JET)
    elif colormap.lower() == "viridis":
        similarity_maps = cv2.applyColorMap(similarity_maps, cv2.COLORMAP_VIRIDIS)
    else:
        raise ValueError(f"Unknown colormap {colormap}")

    if return_colormaps:
        return_value = cv2.cvtColor(similarity_maps, cv2.COLOR_BGR2RGB)
        return_value = rearrange(return_value, "(b h) w c -> b c h w", b=batch_size)
        return_value = torch.from_numpy(return_value).float() / 255

    if save_path is not None:
        img_height, img_width = imgs1.shape[-2:]
        assert imgs1.shape[0] == batch_size
        assert img_height % feature_height == 0
        assert img_width % feature_width == 0

        similarity_maps = rearrange(
            torch.from_numpy(similarity_maps), "(b h) w c -> b c h w", b=batch_size
        )
        similarity_maps = F.interpolate(
            similarity_maps, size=(img_height, img_width), mode="nearest-exact"
        )
        similarity_maps = rearrange(similarity_maps, "b c h w -> (b h) w c").numpy()

        imgs1 = colour_patches(imgs1, query_patches, None, img_height // feature_height)
        imgs1 = torch_to_cv2_img(rearrange(imgs1, "b c h w -> c (b h) w"))
        if imgs2 is not None:
            imgs2 = unnormalise(imgs2)
            imgs2 = torch_to_cv2_img(rearrange(imgs2, "b c h w -> c (b h) w"))

            imgs2 = (1 - alpha) * imgs2 + alpha * similarity_maps
        else:
            imgs2 = similarity_maps

        img = np.concatenate([imgs1, imgs2], axis=1)

        # save img with OpenCV
        cv2.imwrite(save_path, img)

    if return_colormaps:
        return return_value


def vis_principal_components(
    features, imgs=None, save_path=None, background_thresh=-1.0, return_pca_map=False
):
    """Visualise the first three principal components of the features provided, next to the source images. PCA is performed on the features of all the images provided, then the first three principal components are assigned to the three colour channels and visualised. Optionally, a threshold can be set to remove the background (all pixels with an absolute value of less than the threshold on the first principal component are set to 0).

    Parameters
    ----------
    features (batch_size x feature_dim x H//factor x W//factor tensor):
        the patch features of the images provided.

    imgs (batch_size x 3 x H x W tensor):
        the images that the features are extracted from.

    save_path (string):
        The file path to save the resulting visualisation to.

    background_thresh (float, optional):
        If >0, all patches whose absolute value of the first principal component is less than this threshold are set to 0, by default -1.0

    return_pca_map (bool, optional):
        whether to return the PCA map, by default False

    Returns
    -------
    pca_map (batch_size x 3 x H x W tensor), optional:
        the visualisation of the first three principal components of the features provided, in RGB ordering but not normalised.
    """

    feature_height, feature_width = features.shape[-2:]
    batch_size = features.shape[0]

    # normalise features
    features = F.normalize(features, dim=1)

    # perform PCA on all the feature vectors
    features = rearrange(features, "b d h w -> (b h w) d").cpu().detach().numpy()
    pca = PCA(n_components=3)
    pca.fit(features)

    # project the features onto the first three principal components
    features = pca.transform(features)

    # reshape the features back to the original shape
    features = rearrange(
        torch.from_numpy(features),
        "(b h w) d -> b d h w",
        b=batch_size,
        h=feature_height,
        w=feature_width,
    )
    if background_thresh > 0:
        features = features * (features[:, 0:1].abs() > background_thresh)
        return_value = features.clone()

    if save_path is not None:
        img_height, img_width = imgs.shape[-2:]
        assert img_height % feature_height == 0
        assert img_width % feature_width == 0
        assert features.shape[0] == imgs.shape[0]

        features = F.interpolate(
            features, size=(img_height, img_width), mode="nearest-exact"
        )
        imgs = unnormalise(imgs).cpu()
        vis = torch.cat([imgs, features], dim=-1)
        vis = rearrange(vis, "b c h w -> (b h) w c").cpu().numpy()

        # save visualisation with OpenCV
        cv2.imwrite(save_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR) * 255)

    if return_pca_map:
        return return_value
