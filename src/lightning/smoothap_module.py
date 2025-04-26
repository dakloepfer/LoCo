import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import faiss
import hydra
import numpy as np
import torch
import wandb
from einops import einsum, rearrange, repeat
from kornia.geometry.camera import PinholeCamera
from kornia.geometry.conversions import convert_points_from_homogeneous
from kornia.geometry.linalg import transform_points
from kornia.utils import create_meshgrid
from loguru import logger
from torch.nn import functional as F

import lightning.pytorch as pl
from src.models.concat_model import ConcatModel
from src.utils.metrics import mean_avg_precision, pixel_correspondence_matching_recall
from src.utils.misc import (
    add_statistics_to_logdict,
    batched_2d_index_select,
    conf_to_string,
    generate_all_combination_configs,
    so3_rotation_angle,
    squareform,
)
from src.utils.profiler import PassThroughProfiler

faiss_res = faiss.StandardGpuResources()


class PLModule(pl.LightningModule):
    def __init__(self, config, profiler=None):
        super().__init__()
        self.save_hyperparameters()

        self.config = config
        self.profiler = profiler or PassThroughProfiler()

        if not type(config.model.network) is list:
            self.model = hydra.utils.instantiate(
                config.model.network, _recursive_=False
            )
        else:
            self.model = ConcatModel(
                config.model.network, concat_dim=config.model.concat_dim
            )

        model_name = self.config.model.network._target_.replace(".", "_")
        if config.ckpt_path is None:
            self.model_id = model_name + conf_to_string(self.config.model)
        else:
            self.model_id = f"{model_name}-{config.ckpt_path.replace('/', '_')}"

        self.total_n_anchorpairs = self.config.model.total_n_anchorpairs

        self.feature_subsample = self.model.output_subsample

        self.loss = hydra.utils.instantiate(config.loss)

        if config.ckpt_path is not None:
            state_dict = torch.load(config.ckpt_path, map_location="cpu")
            if "model" not in state_dict and "model_state_dict" in state_dict:
                state_dict["model"] = state_dict["model_state_dict"]

            elif "state_dict" in state_dict:
                state_dict["model"] = state_dict["state_dict"]
                for key in list(state_dict["model"].keys()):
                    state_dict["model"][
                        key.replace("model.", "", 1).replace("module.", "", 1)
                    ] = state_dict["model"].pop(key)

            if "model" in state_dict:
                if type(self.model) is not ConcatModel:
                    self.model.load_state_dict(state_dict["model"])
                else:
                    self.model.models[0].load_state_dict(state_dict["model"])
            else:  # state_dict that only contains weights for the residual network
                self.model.layers.load_state_dict(state_dict)
            logger.info(f"Loaded pretrained weights from {config.ckpt_path}")

        self.devices = config.device_list
        self.manually_move_data = False
        if len(self.devices) > 2:
            self.model = self.model.to(self.devices[1])
            self.model = torch.nn.DataParallel(self.model, device_ids=self.devices[1:])
        elif len(self.devices) == 2:
            self.model = self.model.to(self.devices[1])
            self.manually_move_data = True
        elif len(self.devices) == 1:
            self.model = self.model.to(self.devices[0])
            self.manually_move_data = False

        self.pair_chunk_size = self.config.model.pair_chunk_size
        self.anchor_chunk_size = self.config.model.anchor_chunk_size

        # other loss stuff
        self.pos_radius = self.config.model.pos_radius
        self.neg_radius = self.config.model.neg_radius

        # plus sign in front of sqrt to choose the positive difference threshold
        if config.model.saturated_sigmoid_gradient_ratio < 0:  # don't saturate anything
            self.sim_difference_threshold = 3  # max similarity difference is 2
        else:
            self.sim_difference_threshold = -config.loss.sigmoid_temperature * np.log(
                2 / (1 + np.sqrt(1 - config.model.saturated_sigmoid_gradient_ratio)) - 1
            )

        self.max_nonanchor_pos_pairs = self.config.model.max_nonanchor_pos_pairs
        self.max_nonanchor_neg_pairs = self.config.model.max_nonanchor_neg_pairs
        self.subsample_patches = self.config.model.subsample_patches
        if self.subsample_patches >= 1:
            self.subsample_patches = -1.0  # don't subsample
        self.subsample_neg_pairs = self.config.model.subsample_neg_pairs
        if self.subsample_neg_pairs >= 1:
            self.subsample_neg_pairs = -1.0  # don't subsample
        self.frac_pos_pairs_from_same_img = (
            self.config.model.frac_pos_pairs_from_same_img
        )
        if self.frac_pos_pairs_from_same_img >= 1:
            self.frac_pos_pairs_from_same_img = -1.0  # don't subsample

        # stuff for validation / testing
        self.mean_ap_settings = generate_all_combination_configs(
            self.config.eval.mean_ap
        )
        self.pixel_corrs_settings = generate_all_combination_configs(
            self.config.eval.pixel_corrs
        )

    def configure_optimizers(self):
        return hydra.utils.instantiate(
            self.config.optimizer, params=self.model.parameters()
        )

    def on_fit_start(self) -> None:
        if len(self.devices) > 1:
            self.model.to(self.devices[1])

    def forward(self, imgs):
        """Extract features from the images (BxCxHxW)."""
        if self.manually_move_data:
            imgs = imgs.to(self.devices[1])
        features = self.model(imgs)
        # features = F.normalize(features, p=2, dim=1)
        return features.to(self.device)

    def training_step(self, batch, batch_idx):

        for key in batch.keys():
            if key == "img":
                continue
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].squeeze(0)

        imgs = batch["img"]
        anchor_pair_scene_idxs = batch["anchor_pair_scene_idxs"]
        img_scene_idxs = batch["scene_idx"]
        anchor_patch_idxs = batch["anchor_patch_idxs"]
        n_anchors, _ = anchor_patch_idxs.shape

        with self.profiler.profile("Training Step Feature Extraction"):
            if self.manually_move_data:
                imgs = imgs.to(self.devices[1])
            features = self.model(imgs).to(self.device)
            features = rearrange(features, "b c h w -> b (h w) c")
            features_norm = torch.linalg.norm(features.detach(), dim=-1)

            self.log_dict(
                {
                    "train/features_mean_norm": features_norm.mean(),
                    "train/features_std_norm": features_norm.std(),
                    "train/features_max_norm": features_norm.max(),
                    "train/features_min_norm": features_norm.min(),
                }
            )
            wandb.log(
                {
                    "raw_samples/train_raw_features": wandb.Histogram(
                        features_norm.flatten().cpu().numpy()
                    )
                }
            )
            features = F.normalize(features, p=2, dim=-1)
            feature_dim = features.shape[-1]
            n_patches_per_img = features.shape[1]

        with torch.no_grad():  # no grad needed here
            with self.profiler.profile("Computing potential positive/negative pairs"):
                (
                    pos_pair_patch_idxs,
                    neg_pair_patch_idxs,
                ) = self._compute_potential_pairs(
                    batch["img_patch_locations"],
                    img_scene_idxs,
                    anchor_patch_idxs,
                    anchor_pair_scene_idxs,
                )
            with self.profiler.profile("Adding more anchor pairs"):
                if n_anchors < self.total_n_anchorpairs:
                    n_anchors_to_add = self.total_n_anchorpairs - n_anchors
                    new_anchor_patch_idxs = torch.randperm(
                        len(pos_pair_patch_idxs), device=self.device
                    )[:n_anchors_to_add]
                    anchor_patch_idxs = torch.cat(
                        [anchor_patch_idxs, pos_pair_patch_idxs[new_anchor_patch_idxs]]
                    )
                    n_anchors = len(anchor_patch_idxs)
                    # remove new anchor pairs from pos_pairs
                    pos_pair_mask = torch.ones(
                        len(pos_pair_patch_idxs), dtype=torch.bool, device=self.device
                    )
                    pos_pair_mask[new_anchor_patch_idxs] = False
                    pos_pair_patch_idxs = pos_pair_patch_idxs[pos_pair_mask]

                    anchor_pair_patch_imgidxs = anchor_patch_idxs // n_patches_per_img
                    frac_anchors_from_same_img = (
                        (
                            anchor_pair_patch_imgidxs[:, 0]
                            == anchor_pair_patch_imgidxs[:, 1]
                        )
                        .float()
                        .mean()
                    )
                    self.log_dict(
                        {
                            "train/n_anchors_added": len(new_anchor_patch_idxs),
                            "train/n_pos_pairs_remaining": len(pos_pair_patch_idxs),
                            "train/n_neg_pairs_remaining": len(neg_pair_patch_idxs),
                            "train/frac_anchor_pairs_from_same_img": frac_anchors_from_same_img,
                        }
                    )

                pos_pair_patch_imgidxs = pos_pair_patch_idxs // n_patches_per_img
                frac_pos_pairs_from_same_img = (
                    (pos_pair_patch_imgidxs[:, 0] == pos_pair_patch_imgidxs[:, 1])
                    .float()
                    .mean()
                )

                self.log(
                    "train/frac_pos_pairs_from_same_img",
                    frac_pos_pairs_from_same_img,
                )

        # with gradient again
        with self.profiler.profile("Computing Anchor Pair Similarities"):
            # get features for anchor pairs
            anchor_features = self._get_patch_features(
                features, anchor_patch_idxs.flatten()
            )
            anchor_features = rearrange(
                anchor_features,
                "(n_anchors two) c -> n_anchors two c",
                n_anchors=n_anchors,
                two=2,
                c=feature_dim,
            )

            anchor_similarities = einsum(
                anchor_features[:, 0], anchor_features[:, 1], "n c, n c -> n"
            )

        with torch.no_grad():
            with self.profiler.profile(
                "Computing non-anchor pair similarities without grad"
            ):
                pos_pair_similarities = self._compute_pair_similarities(
                    pos_pair_patch_idxs, features, self.pair_chunk_size
                )
                neg_pair_similarities = self._compute_pair_similarities(
                    neg_pair_patch_idxs, features, self.pair_chunk_size
                )

            with self.profiler.profile("Selecting positive non-anchor pairs"):
                # sample non-anchor pairs
                (
                    nonanchor_pos_patch_idxs,
                    n_pos_within_sim_range,
                    n_pos_outside_sim_range_greater,
                ) = self._sample_nonanchor_pairs(
                    anchor_similarities,
                    pos_pair_similarities,
                    pos_pair_patch_idxs,
                    max_pairs=self.max_nonanchor_pos_pairs,
                    anchor_chunk_size=self.anchor_chunk_size,
                    mode="train",
                    pair_type="pos",
                )
                n_batch_positive_pairs = len(pos_pair_similarities)
                del pos_pair_similarities, pos_pair_patch_idxs

            with self.profiler.profile("Selecting negative non-anchor pairs"):
                (
                    nonanchor_neg_patch_idxs,
                    n_neg_within_sim_range,
                    n_neg_outside_sim_range_greater,
                ) = self._sample_nonanchor_pairs(
                    anchor_similarities,
                    neg_pair_similarities,
                    neg_pair_patch_idxs,
                    max_pairs=self.max_nonanchor_neg_pairs,
                    anchor_chunk_size=self.anchor_chunk_size,
                    mode="train",
                    pair_type="neg",
                )
                n_batch_negative_pairs = len(neg_pair_similarities)
                del neg_pair_similarities, neg_pair_patch_idxs

            with self.profiler.profile("Computing Correction Terms"):
                # masks for padded pairs
                batch["valid_nonanchor_pos_pair_mask"] = torch.all(
                    nonanchor_pos_patch_idxs != -1, dim=-1
                )
                batch["valid_nonanchor_neg_pair_mask"] = torch.all(
                    nonanchor_neg_patch_idxs != -1, dim=-1
                )

                # compute correction terms
                n_total_pos_pairs = batch["n_total_positive_pairs"]  # across all scenes
                n_total_neg_pairs = batch["n_total_negative_pairs"]  # across all scenes

                n_sampled_pos_pairs = batch["valid_nonanchor_pos_pair_mask"].sum(-1)
                n_sampled_neg_pairs = batch["valid_nonanchor_neg_pair_mask"].sum(-1)

                # divide through by n_total_pos_pairs for numerical stability
                batch["batch_correction_factor_positive"] = torch.clamp(
                    n_pos_within_sim_range - 1, min=0
                ) / torch.clamp(
                    (n_batch_positive_pairs - 1) * n_sampled_pos_pairs, min=1
                )  # clamp for numerical stability
                batch["batch_correction_factor_negative"] = (
                    (n_total_neg_pairs / (n_total_pos_pairs - 1))
                    * n_neg_within_sim_range
                    / torch.clamp(n_batch_negative_pairs * n_sampled_neg_pairs, min=1)
                )  # clamp for numerical stability
                batch["sampling_correction_factor_positive"] = (
                    n_pos_outside_sim_range_greater / max(n_batch_positive_pairs - 1, 1)
                )
                batch["sampling_correction_factor_negative"] = (
                    n_total_neg_pairs / (n_total_pos_pairs - 1)
                ) * (n_neg_outside_sim_range_greater / n_batch_negative_pairs)

                # this is just for logging, the effective number of pair comparisons used in the loss
                n_effective_pos_pairs = torch.sum(
                    (n_sampled_pos_pairs / n_pos_within_sim_range)
                    * n_batch_positive_pairs
                )
                n_effective_neg_pairs = torch.sum(
                    (n_sampled_neg_pairs / n_neg_within_sim_range)
                    * n_batch_negative_pairs
                )

                self.log_dict(
                    {
                        "train/n_effective_pos_comparisons": n_effective_pos_pairs,
                        "train/n_effective_neg_comparisons": n_effective_neg_pairs,
                        "train/n_effective_total_comparisons": n_effective_pos_pairs
                        + n_effective_neg_pairs,
                        "train/n_batch_positive_pairs": n_batch_positive_pairs,
                        "train/n_batch_negative_pairs": n_batch_negative_pairs,
                        "train/avg_n_sampled_pos_pairs": n_sampled_pos_pairs.float().mean(),
                        "train/avg_n_sampled_neg_pairs": n_sampled_neg_pairs.float().mean(),
                        "train/avg_n_pos_outside_sim_range_greater": n_pos_outside_sim_range_greater.float().mean(),
                        "train/avg_n_neg_outside_sim_range_greater": n_neg_outside_sim_range_greater.float().mean(),
                        "train/avg_n_pos_within_sim_range": n_pos_within_sim_range.float().mean(),
                        "train/avg_n_neg_within_sim_range": n_neg_within_sim_range.float().mean(),
                        "train/n_scenes_in_batch": float(
                            len(torch.unique(img_scene_idxs))
                        ),
                    }
                )

        # with grad again
        with self.profiler.profile("Re-calculating non-anchor similarities with grad"):
            # get features
            nonanchor_pos_features = self._get_patch_features(
                features, nonanchor_pos_patch_idxs.flatten()
            )
            nonanchor_pos_features = rearrange(
                nonanchor_pos_features,
                "(n_anchors n_nonanchor_pos_pairs two) c -> n_anchors n_nonanchor_pos_pairs two c",
                n_anchors=n_anchors,
                n_nonanchor_pos_pairs=nonanchor_pos_patch_idxs.shape[1],
                two=2,
                c=feature_dim,
            )
            nonanchor_neg_features = self._get_patch_features(
                features, nonanchor_neg_patch_idxs.flatten()
            )
            nonanchor_neg_features = rearrange(
                nonanchor_neg_features,
                "(n_anchors n_nonanchor_neg_pairs two) c -> n_anchors n_nonanchor_neg_pairs two c",
                n_anchors=n_anchors,
                n_nonanchor_neg_pairs=nonanchor_neg_patch_idxs.shape[1],
                two=2,
                c=feature_dim,
            )

            # compute similarities
            nonanchor_pos_similarities = einsum(
                nonanchor_pos_features[:, :, 0],
                nonanchor_pos_features[:, :, 1],
                "n m c, n m c -> n m",
            )
            nonanchor_neg_similarities = einsum(
                nonanchor_neg_features[:, :, 0],
                nonanchor_neg_features[:, :, 1],
                "n m c, n m c -> n m",
            )

        with self.profiler.profile("Computing Loss"):
            loss, log_data = self.loss(
                anchor_similarities,
                nonanchor_pos_similarities,
                nonanchor_neg_similarities,
                batch,
            )

        with self.profiler.profile("Adding logging statistics"):
            log_dict = {
                "loss": loss.detach().cpu(),
                "vec_smooth_ap": -loss.detach().cpu(),
                "n_anchor_pairs": float(n_anchors),
            }
            log_dict = add_statistics_to_logdict(
                log_dict, log_data, log_wandb_histogram=True
            )

            new_log_dict = {"train/loss": log_dict["loss"]}
            for key in log_dict.keys():
                new_log_dict[f"train/{key}"] = log_dict[key]

            self.log_dict(new_log_dict)

        return loss

    def _get_patch_features(
        self, features, patch_idxs, img_env_idxs=None, use_patch_env_idxs=False
    ):
        """Select the correct features given the patch indices (potentially across the entire environment) and the indices of the images in the batch w.r.t. the entire environment.

        Parameters
        ----------
        features (batch_size x n_patches_per_img x n_channels tensor):
            The extracted features.

        patch_idxs (n_patches tensor):
            a tensor of indices of patches for which the features are to be selected.
            if use_patch_env_idxs=True, the indices run over the entire environment, i.e. index 0 refers to the first patch in the first image of the environment, not necessarily the first patch in the first image in the batch.

        img_env_idxs (batch_size tensor):
            A tensor giving for each image in the batch the index of this image in the entire environment. Only needed if use_patch_env_idxs is True.

        use_patch_env_idxs (bool):
            If True, the indices in patch_idxs are assumed to be indices into the entire environment. If False, the indices in patch_idxs are assumed to be indices into the batch.

        Returns
        -------
        n_patches x n_channels tensor:
            The selected features for each patch.
        """
        if img_env_idxs is None:
            assert not use_patch_env_idxs
        else:
            assert use_patch_env_idxs

        batch_size, n_patches_per_img, _ = features.shape
        img_batch_idxs = torch.arange(batch_size, device=self.device)

        patch_within_img_idxs = patch_idxs % n_patches_per_img
        patch_img_env_idxs = patch_idxs // n_patches_per_img

        if use_patch_env_idxs:
            patch_img_batch_idxs = img_batch_idxs[
                torch.where(patch_img_env_idxs[:, None] == img_env_idxs[None])[1]
            ]
        else:
            patch_img_batch_idxs = patch_img_env_idxs

        patch_features = features[patch_img_batch_idxs, patch_within_img_idxs]

        return patch_features

    def _compute_potential_pairs(
        self,
        patch_locations,
        scene_idxs,
        anchor_patch_idxs=None,
        anchor_patch_scene_idxs=None,
    ):
        """Computes the positive and negative pairs that can be formed from patches in the batch.

        Parameters
        ----------
        patch_locations (batch_size x n_patches_per_img x 2 tensor):
            the 3D world-coordinates of the respective patches (2nd dimension is (h w)).

        scene_idxs (batch_size tensor):
            the indices of the scenes from which the patches come.

        anchor_patch_idxs (n_anchor_pairs x 2 tensor):
            the indices (across the batch) of the patches that form the anchor pairs.

        anchor_patch_scene_idxs (n_anchor_pairs tensor):
            the indices of the scenes from which the anchor patches come.

        Returns
        -------
        pos_pair_patch_idxs (n_pos_pairs x 2 tensor):
            Tensor containing the indices (across the batch) of the patches that can form positive pairs.

        neg_pair_patch_idxs (n_neg_pairs x 2 tensor):
            Tensor containing the indices (across the batch) of the patches that can form negative pairs.
        """
        assert torch.all(
            scene_idxs[1:] >= scene_idxs[:-1]
        ), "Image Scene Indices should be sorted!"

        n_patches_per_img = patch_locations.shape[1]

        unique_scene_idxs = torch.unique_consecutive(scene_idxs)

        pos_pair_patch_idxs = []
        neg_pair_patch_idxs = []

        for scene_idx in unique_scene_idxs:
            batch_mask = scene_idxs == scene_idx
            scene_patch_locations = patch_locations[batch_mask]

            batch_scene_start_idx = torch.where(batch_mask)[0][0]
            patch_scene_start_idx = batch_scene_start_idx * n_patches_per_img

            if anchor_patch_idxs is not None:
                scene_anchor_patch_idxs = anchor_patch_idxs[
                    (anchor_patch_scene_idxs == scene_idx)
                ]
                scene_anchor_patch_idxs -= patch_scene_start_idx
            else:
                scene_anchor_patch_idxs = None

            pos_pairs, neg_pairs = self._compute_potential_pairs_single_scene(
                scene_patch_locations, scene_anchor_patch_idxs
            )
            pos_pairs += patch_scene_start_idx
            neg_pairs += patch_scene_start_idx

            pos_pair_patch_idxs.append(pos_pairs)
            neg_pair_patch_idxs.append(neg_pairs)

        pos_pair_patch_idxs = torch.cat(pos_pair_patch_idxs)
        neg_pair_patch_idxs = torch.cat(neg_pair_patch_idxs)

        return pos_pair_patch_idxs, neg_pair_patch_idxs

    def _compute_potential_pairs_single_scene(
        self, patch_locations, anchor_patch_idxs=None
    ):
        """Computes the positive and negative pairs that can be formed from patches in the batch. This function assumes that all patches come from the same scene.

        Parameters
        ----------
        patch_locations (batch_size x n_patches_per_img x 2 tensor):
            the 3D world-coordinates of the respective patches (2nd dimension is (h w)).

        anchor_patch_idxs (n_anchor_pairs x 2 tensor):
            the indices (across the batch) of the patches that form the anchor pairs.

        Returns
        -------
        pos_pair_patch_idxs (n_pos_pairs x 2 tensor):
            Tensor containing the indices (across the batch) of the patches that can form positive pairs.

        neg_pair_patch_idxs (n_neg_pairs x 2 tensor):
            Tensor containing the indices (across the batch) of the patches that can form negative pairs.
        """

        if len(patch_locations.shape) == 3:
            n_patches_per_img = patch_locations.shape[1]
            patch_locations = rearrange(patch_locations, "b n t -> (b n) t")
            subsample_positive_sameimg_pairs = True
        else:
            subsample_positive_sameimg_pairs = False

        if self.subsample_patches > 0:
            n_kept_patches = int(self.subsample_patches * len(patch_locations))
            kept_patch_indices = torch.randperm(len(patch_locations))[:n_kept_patches]
            patch_locations = patch_locations[kept_patch_indices]
        else:
            kept_patch_indices = torch.arange(len(patch_locations), device=self.device)

        n_patches = patch_locations.shape[0]

        patch_distances = F.pdist(patch_locations).half()
        patch_distances = squareform(
            patch_distances, n_patches, low_mem=True, upper_only=True
        )

        pos_pair_mask = patch_distances <= self.pos_radius
        neg_pair_mask = (patch_distances > self.pos_radius) & (
            patch_distances <= self.neg_radius
        )
        del patch_distances

        if anchor_patch_idxs is not None:  # remove anchor pairs from positive pairs
            if self.subsample_patches <= 0.0:
                pos_pair_mask[anchor_patch_idxs[0], anchor_patch_idxs[1]] = False
                pos_pair_mask[anchor_patch_idxs[1], anchor_patch_idxs[0]] = False
            else:
                # convert anchor patch indices to kept patch indices
                kept_patch_indices = kept_patch_indices.to(anchor_patch_idxs.device)
                helper = kept_patch_indices[:, None, None] == anchor_patch_idxs[None]

                both_patches_kept = torch.all(torch.any(helper, dim=0), dim=-1)

                kept_anchors = torch.zeros(
                    (both_patches_kept.sum(), 2),
                    dtype=anchor_patch_idxs.dtype,
                    device=anchor_patch_idxs.device,
                )
                helper = helper[:, both_patches_kept]
                kept_anchors[:, 0] = torch.where(helper[:, :, 0])[0]
                kept_anchors[:, 1] = torch.where(helper[:, :, 1])[0]

                pos_pair_mask[kept_anchors[:, 0], kept_anchors[:, 1]] = False
                pos_pair_mask[kept_anchors[:, 1], kept_anchors[:, 0]] = False

        if self.frac_pos_pairs_from_same_img > 0.0 and subsample_positive_sameimg_pairs:
            # remove some pairs that are from the same image

            kept_patch_img_idxs = kept_patch_indices // n_patches_per_img
            same_img_mask = kept_patch_img_idxs[:, None] == kept_patch_img_idxs[None]

            n_pos_pairs_sameimg = torch.sum(pos_pair_mask & same_img_mask)
            n_pos_pairs_diffimg = torch.sum(pos_pair_mask & ~same_img_mask)

            subsample_factor = (
                self.frac_pos_pairs_from_same_img
                / (1 - self.frac_pos_pairs_from_same_img)
                * (n_pos_pairs_diffimg / n_pos_pairs_sameimg)
            )
            if (
                subsample_factor * n_pos_pairs_sameimg + n_pos_pairs_diffimg
            ) >= 100 and subsample_factor < 1.0:
                # make sure I am still left with an acceptable number of positive pairs

                same_img_removal_mask = (
                    torch.rand(same_img_mask.shape, device=self.device)
                    > subsample_factor
                ) & same_img_mask
                del same_img_mask

                pos_pair_mask = pos_pair_mask & (~same_img_removal_mask)
                del same_img_removal_mask

        elif self.frac_pos_pairs_from_same_img > 0.0:
            raise ValueError(
                "If subsampling positive pairs created from the same image, need to pass the patch locations for a batch of images into _compute_potential_pairs(), not a raw list of patch locations."
            )
        else:
            pass

        pos_pair_patch_idxs = torch.nonzero(pos_pair_mask, as_tuple=False)
        del pos_pair_mask
        neg_pair_patch_idxs = torch.nonzero(neg_pair_mask, as_tuple=False)
        del neg_pair_mask

        if self.subsample_neg_pairs > 0:
            n_kept_neg_pairs = int(self.subsample_neg_pairs * len(neg_pair_patch_idxs))
            kept_neg_pair_indices = torch.randperm(len(neg_pair_patch_idxs))[
                :n_kept_neg_pairs
            ]
            neg_pair_patch_idxs = neg_pair_patch_idxs[kept_neg_pair_indices]

        if self.subsample_patches > 0:
            # make subselected indices into actual patch indices
            pos_pair_patch_idxs = kept_patch_indices[pos_pair_patch_idxs]
            neg_pair_patch_idxs = kept_patch_indices[neg_pair_patch_idxs]

        return pos_pair_patch_idxs, neg_pair_patch_idxs

    def _compute_pair_similarities(self, pair_patch_idxs, features, pair_chunk_size):
        """Computes the similarities of the pairs of patches given their indices and the features of the patches by chunking the pairs to avoid memory issues.

        Parameters
        ----------
        pair_patch_idxs (n_pairs x 2 tensor):
            A tensor of patch indices (across the batch) of the patches making up the pairs.

        features (batch_size x n_patches_per_img x n_channels tensor):
            The feature maps for the batch of images.

        pair_chunk_size (int):
            The maximum number of pairs to process at once.

        Returns
        -------
        pair_similarities (n_pairs tensor):
            The cosine similarities of the pairs of patches.
        """
        pair_patch_idxs_chunks = pair_patch_idxs.split(pair_chunk_size, dim=0)

        pair_similarities = []

        for chunk in pair_patch_idxs_chunks:
            # for positive pairs
            pair_features = self._get_patch_features(features, chunk.flatten())
            pair_features = rearrange(
                pair_features,
                "(chunk_size two) c -> chunk_size two c",
                chunk_size=chunk.shape[0],
                two=2,
            )
            pair_similarities.append(
                einsum(
                    pair_features[:, 0],
                    pair_features[:, 1],
                    "n c, n c -> n",
                )
            )
        return torch.cat(pair_similarities)

    def _sample_nonanchor_pairs(
        self,
        anchor_similarities,
        pair_similarities,
        pair_patch_idxs,
        max_pairs=-1,
        anchor_chunk_size=512,
        mode="train",
        pair_type="pos",
    ):
        """For each anchor pair, sample at most max_pairs pairs from the provided pairs, so that the similarities of the sampled pairs are within a certain range of the similarity of the anchor pair.

        Parameters
        ----------
        anchor_similarities (n_anchor_pairs tensor):
            the similarities of the anchor pairs

        pair_similarities (n_nonanchor_pairs tensor):
            the similarities of the nonanchor pairs

        pair_patch_idxs (n_nonanchor_pairs x 2 tensor):
            the patch indices (across the batch) of the patches making up the nonanchor pairs

        max_pairs (int):
            maximum number of pairs to sample for each anchor pair. If <0, all pairs within the range are sampled.

        amchor_chunk_size (int):
            the maximum number of anchor pairs to process at once.

        mode (str):
            the mode in which the function is called. Needed for logging.

        pair_type (str):
            the type of pairs being sampled. Needed for logging.

        Returns
        -------
        nonachor_patch_idxs (n_anchor_pairs x min(max(n_nonanchor_pairs in range), max_pairs) x 2 tensor):
            the sampled nonanchor pairs for each anchor pair. Padded with -1 if not enough pairs within the range are available for some anchor pairs.

        n_within_sim_range (n_anchor_pairs tensor):
            the number of pairs within the range for each anchor pair. Needed for computing correction terms.

        n_outside_sim_range_greater (n_anchor_pairs tensor):
            the number of pairs outside the range with similarity greater than the anchor pair for each anchor pair. Needed for computing correction terms.
        """
        if max_pairs == 0:
            raise ValueError("Trying to sample no pairs, why are you calling this?")
        elif max_pairs < 0:
            max_pairs = len(pair_similarities)
        else:
            max_pairs = min(max_pairs, len(pair_similarities))
        n_anchors = len(anchor_similarities)

        n_within_sim_range = torch.zeros(
            n_anchors, device=self.device, dtype=torch.long
        )
        n_outside_sim_range_greater = torch.zeros(
            n_anchors, device=self.device, dtype=torch.long
        )
        sampled_pairs_idxs = -torch.ones(
            (n_anchors, max_pairs), dtype=torch.long, device=self.device
        )

        n_chunks = int(np.ceil(n_anchors / anchor_chunk_size))
        for i in range(n_chunks):
            start_idx = i * anchor_chunk_size
            end_idx = min((i + 1) * anchor_chunk_size, n_anchors)

            sim_differences = (
                pair_similarities[None, :]
                - anchor_similarities[start_idx:end_idx, None]
            )  # n_anchors (chunked) x n_potential_nonanchor_pairs
            pairs_within_sim_range = (
                torch.abs(sim_differences) <= self.sim_difference_threshold
            )
            n_within_sim_range[start_idx:end_idx] = torch.sum(
                pairs_within_sim_range, dim=-1
            )
            n_outside_sim_range_greater[start_idx:end_idx] = torch.sum(
                sim_differences > self.sim_difference_threshold, dim=-1
            )

            probs = pairs_within_sim_range.float()
            probs[n_within_sim_range[start_idx:end_idx] == 0] = 0.1
            sampled_pairs_idxs[start_idx:end_idx] = torch.multinomial(probs, max_pairs)

        sampled_pairs = torch.gather(
            pair_patch_idxs[None].expand(n_anchors, -1, -1),
            dim=1,
            index=sampled_pairs_idxs[..., None].expand(-1, -1, 2),
        )
        # set pairs outside sim range to -1
        for i in range(n_anchors):
            if n_within_sim_range[i] < max_pairs:
                sampled_pairs[i, n_within_sim_range[i] :] = -1

        # some logging
        frac_pairs_within_sim_range = n_within_sim_range / len(pair_similarities)
        frac_pairs_outside_sim_range_greater = n_outside_sim_range_greater / len(
            pair_similarities
        )
        self.log_dict(
            {
                f"{mode}/mean_frac_{pair_type}_within_simrange": frac_pairs_within_sim_range.mean(),
                f"{mode}/stdev_frac_{pair_type}_within_simrange": frac_pairs_within_sim_range.std(),
                f"{mode}/max_frac_{pair_type}_within_simrange": frac_pairs_within_sim_range.max(),
                f"{mode}/min_frac_{pair_type}_within_simrange": frac_pairs_within_sim_range.min(),
                f"{mode}/mean_frac_{pair_type}_outside_simrange_greater": frac_pairs_outside_sim_range_greater.mean(),
                f"{mode}/stdev_frac_{pair_type}_outside_simrange_greater": frac_pairs_outside_sim_range_greater.std(),
                f"{mode}/max_frac_{pair_type}_outside_simrange_greater": frac_pairs_outside_sim_range_greater.max(),
                f"{mode}/min_frac_{pair_type}_outside_simrange_greater": frac_pairs_outside_sim_range_greater.min(),
            }
        )
        wandb.log(
            {
                f"raw_samples/{mode}_frac_{pair_type}_within_simrange": wandb.Histogram(
                    frac_pairs_within_sim_range.flatten().cpu().numpy()
                ),
                f"raw_samples/{mode}_frac_{pair_type}_outside_simrange_greater": wandb.Histogram(
                    frac_pairs_outside_sim_range_greater.flatten().cpu().numpy()
                ),
            }
        )

        return sampled_pairs, n_within_sim_range, n_outside_sim_range_greater

    def on_validation_start(self):
        self.val_log_data = {}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, log_data = self._valtest_step(
            batch, batch_idx, mode="val", dataloader_idx=dataloader_idx
        )

        for key, value in log_data.items():
            if key not in self.val_log_data:
                self.val_log_data[key] = []
            if type(value) == torch.Tensor:
                value = value.cpu()
            self.val_log_data[key].append(value)

        # only aggregate loss from first task (first dataloader)
        # rest is done manually
        if dataloader_idx == 0:
            return loss
        else:
            return None

    def on_validation_epoch_end(self):
        for key, value in self.val_log_data.items():
            if len(value[0].shape) == 0:
                self.val_log_data[key] = torch.tensor(value)
            else:
                self.val_log_data[key] = torch.cat(value)

        log_dict = add_statistics_to_logdict({}, self.val_log_data)
        for key in list(log_dict.keys()):
            log_dict[f"val/{key}"] = log_dict.pop(key)

        for repr_error_key in [
            k for k in self.val_log_data.keys() if k.startswith("reprojection_errors")
        ]:
            setting_name = repr_error_key[20:]
            if f"gt_rotations_{setting_name}" in self.val_log_data:
                gt_rotations = self.val_log_data[f"gt_rotations_{setting_name}"]
            else:
                gt_rotations = None
            # TODO double-check this data collation and stuff with the keys etc. Also figure out how / where to compute the gt_rotations and how to pass them in
            matching_recalls = pixel_correspondence_matching_recall(
                self.val_log_data[repr_error_key],
                self.pixel_corrs_settings[0].pixel_thresholds,
                self.pixel_corrs_settings[0].angle_bins,
                gt_rotations,
            )

            for key, value in matching_recalls.items():
                log_dict[f"val/pixel_correspondence_recall_{setting_name}_{key}"] = (
                    value
                )
        self.log_dict(log_dict)

    def on_test_start(self):
        if len(self.devices) > 1:
            self.model.to(self.devices[1])
        self.test_log_data = {}

    def test_step(self, batch, batch_idx, dataloader_idx=0):

        loss, log_data = self._valtest_step(
            batch, batch_idx, mode="test", dataloader_idx=dataloader_idx
        )
        for key, value in log_data.items():
            if key not in self.test_log_data:
                self.test_log_data[key] = []
            if type(value) == torch.Tensor:
                value = value.cpu()
            self.test_log_data[key].append(value)

        # only aggregate loss from first task (first dataloader)
        # rest is done manually
        if dataloader_idx == 0:
            return loss
        else:
            return None

    def on_test_epoch_end(self):
        for key, value in self.test_log_data.items():
            if len(value[0].shape) == 0:
                self.test_log_data[key] = torch.tensor(value)
            else:
                self.test_log_data[key] = torch.cat(value)

        # TODO I think this does not work well with reporjection errors
        log_dict = add_statistics_to_logdict({}, self.test_log_data)
        for key in list(log_dict.keys()):
            log_dict[f"test/{key}"] = log_dict.pop(key)

        for repr_error_key in [
            k for k in self.test_log_data.keys() if k.startswith("reprojection_errors")
        ]:
            setting_name = repr_error_key[20:]
            if f"gt_rotations_{setting_name}" in self.test_log_data:
                gt_rotations = self.test_log_data[f"gt_rotations_{setting_name}"]
            else:
                gt_rotations = None

            matching_recalls = pixel_correspondence_matching_recall(
                self.test_log_data[repr_error_key],
                self.pixel_corrs_settings[0].pixel_thresholds,
                self.pixel_corrs_settings[0].angle_bins,
                gt_rotations,
            )

            for key, value in matching_recalls.items():
                log_dict[f"test/pixel_correspondence_recall_{setting_name}_{key}"] = (
                    value
                )

        self.log_dict(log_dict)

    def _valtest_step(self, batch, batch_idx, mode="val", dataloader_idx=0):
        """Compute validation and test metrics; skip computing the actual loss for speed."""

        # Compute mean avg precision
        if dataloader_idx == 0:
            loss = None
            log_data = {}
            for setting in self.mean_ap_settings:
                cur_loss, cur_log_data = self._compute_mean_avg_precision(
                    batch, batch_idx, setting, mode=mode
                )
                if loss is None:  # only use the first combination of configs
                    loss = cur_loss
                    self.log(f"{mode}/loss", loss)
                log_data.update(cur_log_data)

        # compute pixel correspondence matching recall
        elif dataloader_idx == 1:
            loss = None
            log_data = {}
            for setting in self.pixel_corrs_settings:
                cur_loss, cur_log_data = (
                    self._compute_pixel_correspondence_reprojection_errors(
                        batch, batch_idx, setting=setting, mode=mode
                    )
                )
                log_data.update(cur_log_data)

        else:
            raise NotImplementedError(
                f"Expected only two {mode} dataloaders, got dataloader_idx {dataloader_idx}. Is there a task missing?"
            )

        return loss, log_data

    def _compute_mean_avg_precision(self, batch, batch_idx, setting, mode="val"):
        """Compute the mean average precision from a batch of images of the same scene"""
        imgs = batch["img"]
        patch_locations = batch["img_patch_locations"]
        patch_locations = rearrange(patch_locations, "() b h w t -> (b h w) t")

        n_patches = patch_locations.shape[0]

        if "scene_idx" in batch:
            # only one scene at a time for validation
            assert torch.all(batch["scene_idx"] == batch["scene_idx"].flatten()[0])

        with self.profiler.profile("Feature Extraction"):
            if "features" not in batch:
                features = self(imgs)
                features = rearrange(features, "b c h w -> (b h w) c")
                batch["features"] = features
            else:
                features = batch["features"]

        n_patches = features.shape[0]

        with self.profiler.profile("Select query and target pairs"):
            if "query_patch_indices" not in batch:
                # randomly select query patches
                query_patch_indices = torch.randperm(n_patches, device=self.device)[
                    : setting.n_queries
                ]
            else:
                query_patch_indices = batch["query_patch_indices"]

            query_patch_features = features[query_patch_indices]
            query_patch_locations = patch_locations[query_patch_indices]

            # select sample patches for each query patch
            # only up to a certain distance from the query patch
            faiss_index = faiss.IndexFlatL2(3)
            faiss_index.add(patch_locations.cpu().numpy())
            lims, dists, idxs = faiss_index.range_search(
                query_patch_locations.cpu().numpy(), setting.max_radius**2
            )
            sample_patch_indices = torch.zeros(
                setting.n_queries,
                setting.n_samples,
                device=self.device,
                dtype=torch.long,
            )

            for i in range(setting.n_queries):

                n_possible_samples = int(lims[i + 1] - lims[i])
                if n_possible_samples <= setting.n_samples:
                    # fill up with just uniformly sampled patches; max_avgavgprec_radius should be chosen so that this does not happen
                    sample_patch_indices[i, :n_possible_samples] = torch.from_numpy(
                        idxs[lims[i] : lims[i + 1]]
                    ).to(device=self.device)
                    sample_patch_indices[i, n_possible_samples:] = torch.randint(
                        n_patches,
                        (setting.n_samples - n_possible_samples,),
                        device=self.device,
                    )
                else:
                    if setting.sample_weighted_by_distance:
                        # the probability distribution for a patch being at a certain distance is proportional to the square of the distance
                        # so to sample similar numbers of patches for each distance, I need to weight the sampling by the inverse square
                        # (strictly speaking that is only really the case up to a distance of 1.0m or so for Matterport3D, because for r > some distance the sphere of radius r starts to hit the boundaries of the environment (ceiling / floor, but also in the horizontal plane later on) and the number of patches at that distance starts to scale linear before dropping off. So strictly speaking, weighting the sampling by the square of the distance over-samples small-distance patches a bit, but that is where it is arguably more important to correctly estimate the mean average precision anyways.)

                        # calculate weights; distances are already squared
                        weights = 1 / np.clip(
                            dists[lims[i] : lims[i + 1]], a_min=1e-6, a_max=None
                        )
                        weights = weights / weights.sum()
                    else:
                        weights = None

                    current_indices = np.random.choice(
                        idxs[lims[i] : lims[i + 1]],
                        setting.n_samples,
                        replace=False,
                        p=weights,
                    )
                    sample_patch_indices[i] = torch.from_numpy(current_indices).to(
                        device=self.device
                    )

            sample_patch_features = features[
                sample_patch_indices
            ]  # n_queries x n_samples x feature_dim
            sample_patch_locations = patch_locations[
                sample_patch_indices
            ]  # n_queries x n_samples x 3

        with self.profiler.profile("Compute similarities and distances"):
            # compute similarities
            similarities = einsum(
                query_patch_features, sample_patch_features, "q c, q s c -> q s"
            )

            # compute distances
            distances = torch.linalg.norm(
                query_patch_locations[:, None] - sample_patch_locations, dim=-1
            )

        with self.profiler.profile("Compute Average Average Precision"):
            mAP = mean_avg_precision(distances, similarities)

        setting_name = ""
        for key, value in setting.items():
            if key in ["n_queries", "n_samples"]:
                continue
            key = key.lower()
            setting_name += f"{key}::{value}.."
        setting_name = setting_name[:-2]  # remove final dash

        return -mAP, {f"mean_avg_precision_{setting_name}": mAP}

    def _compute_pixel_correspondence_reprojection_errors(
        self,
        batch,
        batch_idx,
        setting,
        mode="val",
    ):
        """Compute the reprojection errors for the pixel correspondences from a batch of images of the same scene, like in [this](https://arxiv.org/pdf/2404.08636.pdf) paper. Following that paper, simply choose the nearest neighbour from pixels in A to B, and keep the top 1000 pixels based on Lowe's ratio test."""

        imgsA = batch["imgA"]
        imgsB = batch["imgB"]
        depthsA = batch["depthA"].float()
        depthsB = batch["depthB"].float()

        batch_size = depthsA.shape[0]

        intrinsicsA = torch.eye(4, device=self.device)[None].repeat(batch_size, 1, 1)
        intrinsicsB = torch.eye(4, device=self.device)[None].repeat(batch_size, 1, 1)
        intrinsicsA[:, :3, :3] = batch["intrinsicA"]
        intrinsicsB[:, :3, :3] = batch["intrinsicB"]

        poses_AtoB = batch["pose_AtoB"].float()

        depthsA = F.interpolate(
            depthsA[:, None], scale_factor=setting.scale_factor, mode="nearest-exact"
        )[:, 0]
        depthsB = F.interpolate(
            depthsB[:, None], scale_factor=setting.scale_factor, mode="nearest-exact"
        )[:, 0]
        intrinsicsA[:, :2] *= setting.scale_factor
        intrinsicsB[:, :2] *= setting.scale_factor

        height, width = depthsA.shape[-2:]

        with self.profiler.profile("Feature Extraction"):
            if "featuresA" not in batch:
                featuresA = self(imgsA)
                featuresB = self(imgsB)
                batch["featuresA"] = featuresA
                batch["featuresB"] = featuresB
            else:
                featuresA = batch["featuresA"]
                featuresB = batch["featuresB"]

        with self.profiler.profile("Compute Pixel Correspondences"):
            # need to use for-loop allow filtering (and therefore potentially different numbers of correspondences per image)
            # since I need to use a for-loop anyways for faiss, this is not too bad
            world_pointsA = torch.zeros(
                batch_size, setting.n_pixel_corrs, 3, device=self.device
            )
            world_pointsB = torch.zeros(
                batch_size, setting.n_pixel_corrs, 3, device=self.device
            )
            mask = torch.zeros(
                batch_size, setting.n_pixel_corrs, device=self.device, dtype=torch.bool
            )

            for i in range(batch_size):
                # print(
                #     f"Computing pixel correspondences for pair number {i+1} / {batch_size} in batch {batch_idx}",
                #     end="\r",
                # )
                cameraA = PinholeCamera(
                    intrinsicsA[i : i + 1],
                    torch.eye(4, device=self.device)[None],
                    torch.ones(1, device=self.device) * height,
                    torch.ones(1, device=self.device) * width,
                )
                cameraB = PinholeCamera(
                    intrinsicsB[i : i + 1],
                    poses_AtoB[i : i + 1],
                    torch.ones(1, device=self.device) * height,
                    torch.ones(1, device=self.device) * width,
                )

                cur_world_pointsA, cur_world_pointsB = (
                    self._compute_pixel_correspondences(
                        featuresA[i],
                        featuresB[i],
                        depthsA[i],
                        depthsB[i],
                        cameraA,
                        cameraB,
                        setting,
                    )
                )
                world_pointsA[i, : len(cur_world_pointsA)] = cur_world_pointsA
                world_pointsB[i, : len(cur_world_pointsB)] = cur_world_pointsB
                mask[i, : len(cur_world_pointsA)] = True

        with self.profiler.profile("Compute Reprojection Errors"):
            camerasB = PinholeCamera(
                intrinsicsB,
                poses_AtoB,
                torch.ones(batch_size, device=self.device) * height,
                torch.ones(batch_size, device=self.device) * width,
            )
            matches_AinB = camerasB.project(world_pointsA)
            matches_BinB = camerasB.project(world_pointsB)

            reprojection_errors = torch.linalg.norm(matches_AinB - matches_BinB, dim=-1)
            reprojection_errors = reprojection_errors[mask]

        setting_name = ""
        for key, value in setting.items():
            if key in ["pixel_thresholds", "angle_bins"]:
                continue
            key = key.lower()
            setting_name += f"{key}::{value}.."
        setting_name = setting_name[:-2]  # remove final dash

        gt_rotations = so3_rotation_angle(poses_AtoB[:, :3, :3])
        gt_rotations = repeat(gt_rotations, "b -> b n", n=setting.n_pixel_corrs)[mask]

        return None, {
            f"reprojection_errors_{setting_name}": reprojection_errors,
            f"gt_rotations_{setting_name}": gt_rotations,
        }

    def _compute_pixel_correspondences(
        self, featuresA, featuresB, depthsA, depthsB, cameraA, cameraB, setting
    ):
        """Compute setting.n_pixel_corrs pixel correspondences between two images A and B, given their features, depths, and camera matrices.

        Parameters
        ----------
        featuresA (feature_dim x feature_height x feature_width tensor):
            The features of the patches in image A.

        featuresB (feature_dim x feature_height x feature_width tensor):
            The features of the patches in image B.

        depthsA (height x width tensor):
            The depths of the pixels in image A.

        depthsB (height x width tensor):
            The depths of the pixels in image B.

        cameraA (kornia PinholeCamera):
            A PinholeCamera object for the images A.

        cameraB (kornia PinholeCamera):
            A PinholeCamera object for the images B.

        setting (yacs ConfigNode):
            A YACS config node containing the settings for the pixel correspondence computation.

        Returns
        -------
        world_pointsA (n_pixel_corrs x 3 tensor):
            The 3D world-coordinates of the pixels in image A that are used as matches.

        world_pointsB (n_pixel_corrs x 3 tensor):
            The 3D world-coordinates of the pixels in image B that are used as matches.
        """
        height, width = depthsA.shape
        feature_dim = featuresA.shape[0]

        mesh_grid = create_meshgrid(height, width, device=self.device)
        pixel_grid = (
            0.5
            * (mesh_grid + 1)
            * torch.tensor([width - 1, height - 1], device=self.device).view(1, 1, 1, 2)
            + 0.5
        )
        pixel_grid = rearrange(pixel_grid, "() h w two -> () (h w) two")

        depthsA = rearrange(depthsA, "h w -> () (h w) ()")
        depthsB = rearrange(depthsB, "h w -> () (h w) ()")

        worldcoordsA = cameraA.unproject(pixel_grid, depthsA)
        worldcoordsB = cameraB.unproject(pixel_grid, depthsB)

        # sample features on grid with interpolation
        featuresA = F.grid_sample(featuresA[None], mesh_grid, align_corners=False)
        featuresB = F.grid_sample(featuresB[None], mesh_grid, align_corners=False)

        featuresA = rearrange(featuresA, "() c h w -> () (h w) c")
        featuresB = rearrange(featuresB, "() c h w -> () (h w) c")

        valid_maskA = depthsA.squeeze(2) > 1e-6
        valid_maskB = depthsB.squeeze(2) > 1e-6

        # filtering means the network can never try to give a match for a pixel that does not have a match
        # which is kind of cheating; in the paper they don't do that
        if setting.filter_occlusions:
            cam_coords_AinB = transform_points(cameraB.extrinsics, worldcoordsA)
            Adepths_in_B = cam_coords_AinB[..., 2]
            pixel_coords_AinB = convert_points_from_homogeneous(
                transform_points(cameraB.intrinsics, cam_coords_AinB)
            )

            # occlusion check
            pixel_idxs_AinB = torch.round(pixel_coords_AinB).long()
            pixel_idxs_AinB = torch.clamp(
                pixel_idxs_AinB,
                torch.tensor([0, 0], device=self.device),
                torch.tensor([width - 1, height - 1], device=self.device),
            )
            B_depths_at_Apoints = batched_2d_index_select(
                rearrange(depthsB, "() (h w) () -> () () h w", h=height, w=width),
                torch.flip(pixel_idxs_AinB, dims=(-1,)),
            ).squeeze(2)

            # add some tolerance
            valid_maskA &= (
                (pixel_coords_AinB[..., 0] > 0 - 1e-3)
                & (pixel_coords_AinB[..., 0] < width + 1e-3)
                & (pixel_coords_AinB[..., 1] > 0 - 1e-3)
                & (pixel_coords_AinB[..., 1] < height + 1e-3)
                & (Adepths_in_B < B_depths_at_Apoints + 1e-2)
            )

        elif setting.filter_pixels_outside_image:
            # if I filter occlusions, I also filter pixels outside the image
            pixel_coords_AinB = cameraB.project(worldcoordsA)
            # add some tolerance
            valid_maskA = (
                (pixel_coords_AinB[..., 0] > 0 - 1e-3)
                & (pixel_coords_AinB[..., 0] < width + 1e-3)
                & (pixel_coords_AinB[..., 1] > 0 - 1e-3)
                & (pixel_coords_AinB[..., 1] < height + 1e-3)
            )

        featuresA = featuresA[valid_maskA]
        featuresB = featuresB[valid_maskB]
        worldcoordsA = worldcoordsA[valid_maskA]
        worldcoordsB = worldcoordsB[valid_maskB]
        pixel_grid = pixel_grid[valid_maskB]

        featuresA = featuresA.contiguous()
        featuresB = featuresB.contiguous()
        featuresA = F.normalize(featuresA, p=2, dim=-1)
        featuresB = F.normalize(featuresB, p=2, dim=-1)

        index = faiss.GpuIndexFlatL2(faiss_res, feature_dim)
        index.add(featuresB)
        # get top 2 matches in B; two are needed for some filtering algorithms
        top_k_dists, top_k_indices = index.search(featuresA, 2)
        # they use 1-cosine_similarity in the paper, which is 0.5 * squared euclidean distance
        top_k_dists *= 0.5

        if setting.filter in ["ratio_test", "lowe"]:
            # Lowe's ratio test
            top_k_dists = top_k_dists.clamp(min=1e-9)
            top0_dists = top_k_dists[:, 0]
            top1_dists = top_k_dists[:, 1]
            ratio = top0_dists / top1_dists
            weights = 1 - ratio

        elif setting.filter == "similarity":
            weights = -top_k_dists[:, 0]

        elif setting.filter in ["pixel_distance", "pixel_dists"]:
            top0_locs = pixel_grid[top_k_indices[:, 0]]
            top1_locs = pixel_grid[top_k_indices[:, 1]]
            pixel_dists = torch.linalg.norm(top0_locs - top1_locs, dim=-1)
            weights = -pixel_dists

        else:
            raise NotImplementedError(
                f"Unknown pixel correspondence filter {setting.filter}"
            )

        n_corrs = min(setting.n_pixel_corrs, len(weights))
        _, indices_A = torch.topk(weights, k=n_corrs)

        indices_B = top_k_indices[indices_A, 0]

        world_pointsA = worldcoordsA[indices_A]
        world_pointsB = worldcoordsB[indices_B]

        return world_pointsA, world_pointsB
