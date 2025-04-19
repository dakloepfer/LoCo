import os

import numpy as np
import torch
from einops import rearrange
from torch.nn import functional as F
from torch.utils import data

from src.utils.data import crop_to_aspect_ratio


class meanAP_BatchSampler(data.Sampler):
    """A batch sampler to make batches for the mean average precision calculation. Basically returns batches of images from the same scene that have some reasonable overlap."""

    def __init__(
        self,
        datasource,
        batch_size: int,
        n_batches: int,
        min_overlap: float,
        max_overlap: float,
        patch_locations_file_name="patch_locations_32_40.npy",
        shuffle=False,
        patch_locations_final_size=None,
        name=None,
    ):
        super().__init__(datasource)
        self.name = name
        self.datasource = datasource

        self.batch_size = batch_size
        self.n_batches = n_batches
        self.min_overlap = min_overlap
        self.max_overlap = max_overlap

        # if False, each epoch will be the same -- useful for tracking performance over time
        self.randomise_each_epoch = shuffle

        self.n_scenes = len(datasource.datasets)
        self.scene_names = [d.scene_name for d in datasource.datasets]

        self.img_overlaps = []
        self.patch_locations = []
        for scene_dataset in datasource.datasets:
            img_overlap_path = os.path.join(
                scene_dataset.data_root, scene_dataset.scene_name, "img_overlaps.npy"
            )
            patch_locations = os.path.join(
                scene_dataset.data_root,
                scene_dataset.scene_name,
                patch_locations_file_name,
            )
            used_img_mask = scene_dataset.used_img_mask
            img_overlaps = np.load(img_overlap_path)
            img_overlaps = img_overlaps[used_img_mask][:, used_img_mask]
            self.img_overlaps.append(img_overlaps)

            patch_locations = np.load(patch_locations)
            patch_locations = patch_locations[used_img_mask]
            patch_locations = rearrange(patch_locations, "b h w c -> b c h w")
            patch_locations = crop_to_aspect_ratio(
                patch_locations, scene_dataset.aspect_ratio
            )
            if patch_locations_final_size is not None:
                patch_locations = F.interpolate(
                    torch.from_numpy(patch_locations),
                    size=tuple(patch_locations_final_size),
                    mode="nearest-exact",
                )
            patch_locations = rearrange(patch_locations, "b c h w -> b h w c")
            self.patch_locations.append(patch_locations)

        self.current_epoch_batches = None
        self._make_epoch()

    def __len__(self):
        return self.n_batches

    def _make_epoch(self):
        """Samples a full set of batches for one epoch.
        Algorithm:
        1. Choose a random image to add to the batch
        2. While the batch is not yet full:
            - Choose an image that has minimum overlap with at least half of the images already in the batch
            - Update a list that gives the number of images in the batch that have overlap with each image
            - If I run out of images for which this is the case, reduce the fraction of images in the batch that I require overlap with
            - Filter out images in the batch, as well as images with a >max_overlap with another image in the batch (max_overlap should be quite stringent, eg 0.9)
        """

        self.current_epoch_batches = []

        for batch_idx in range(self.n_batches):
            if batch_idx % self.n_scenes == 0:
                scene_idxs = np.random.permutation(self.n_scenes)

            scene_idx = scene_idxs[batch_idx % self.n_scenes]

            img_overlaps = self.img_overlaps[scene_idx]
            n_scene_imgs = len(img_overlaps)

            assert n_scene_imgs >= self.batch_size

            batch = []
            allowed_images = np.ones(n_scene_imgs, dtype=bool)
            n_imgs_in_batch_with_overlap = np.zeros(n_scene_imgs, dtype=int)
            frac_batchimg_matches_threshold = 0.5
            used_batch_imgs = np.zeros(n_scene_imgs, dtype=bool)
            while len(batch) < self.batch_size:
                if len(batch) == 0:
                    img_idx = np.random.randint(len(img_overlaps))
                else:
                    potential_images = np.flatnonzero(
                        (
                            n_imgs_in_batch_with_overlap / len(batch)
                            >= frac_batchimg_matches_threshold
                        )
                        & allowed_images
                    )
                    if len(potential_images) == 0:
                        if allowed_images.sum() == 0:  # ignore max overlap
                            allowed_images = np.ones(n_scene_imgs, dtype=bool)
                            allowed_images[used_batch_imgs] = False
                        frac_batchimg_matches_threshold -= 0.1

                        continue

                    img_idx = np.random.choice(potential_images)

                # update various things
                allowed_images &= img_overlaps[img_idx] <= self.max_overlap
                allowed_images[img_idx] = False
                used_batch_imgs[img_idx] = True
                n_imgs_in_batch_with_overlap += img_overlaps[img_idx] > self.min_overlap

                batch.append(
                    {"scene_idx": scene_idx, "type": "img", "img_idx": img_idx}
                )
            batch.append(
                {
                    "type": "pass_through_information",
                    "img_patch_locations": self.patch_locations[scene_idx][
                        used_batch_imgs
                    ],
                }
            )

            self.current_epoch_batches.append(batch)

    def __iter__(self):
        if self.randomise_each_epoch:
            self._make_epoch()

        for batch in self.current_epoch_batches:
            yield batch
