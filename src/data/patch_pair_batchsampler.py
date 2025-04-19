import os
import random
from math import ceil

import faiss
import numpy as np
import torch
from einops import rearrange
from loguru import logger
from tqdm import tqdm

from src.utils.data import crop_to_aspect_ratio


class PatchPairBatchSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        datasource,
        batch_size,
        n_batches_per_epoch,
        max_anchor_pairs,
        patch_locations_file_name,
        pos_radius=0.2,
        neg_radius=5.0,
        shuffle=True,
        name="",
    ):
        super().__init__(datasource)

        self.datasource = datasource

        # settings for sampling
        self.n_batches_per_epoch = n_batches_per_epoch
        self.batch_size = batch_size  # max number of images to use
        self.max_anchor_pairs = max_anchor_pairs

        self.shuffle = shuffle

        # settings for the pair bank
        self.n_scenes = len(datasource.datasets)
        self.scene_names = [d.scene_name for d in datasource.datasets]

        self.pos_radius = pos_radius
        self.neg_radius = neg_radius

        self.patch_locations_file_name = patch_locations_file_name

        # indices for the patches making up each pair in the bank
        # indices run over all patches from all images in a particular scene
        self.pair_bank_size = ceil(
            self.max_anchor_pairs * self.n_batches_per_epoch / self.n_scenes
        )
        self.pair_bank = np.empty(
            (self.n_scenes, self.pair_bank_size, 2), dtype=np.int64
        )

        self.n_patches_per_image = 0
        (
            self.patch_location_faiss_indices,
            self.patch_locations,
        ) = self._make_location_indices()

        (
            self.n_total_positive_pairs_per_scene,
            self.n_total_negative_pairs_per_scene,
        ) = self._estimate_total_pairs()
        self.n_total_pairs_per_scene = (
            self.n_total_positive_pairs_per_scene
            + self.n_total_negative_pairs_per_scene
        )
        self.n_total_positive_pairs = np.sum(self.n_total_positive_pairs_per_scene)
        self.n_total_negative_pairs = np.sum(self.n_total_negative_pairs_per_scene)

        logger.info(
            f"Fraction of positive pairs for each scene: {self.n_total_positive_pairs_per_scene / self.n_total_pairs_per_scene}",
        )
        logger.info(
            f"Average fraction of positive pairs: {np.mean(self.n_total_positive_pairs_per_scene / self.n_total_pairs_per_scene)}"
        )
        logger.info(
            f"Total number of positive pairs for each scene: {self.n_total_positive_pairs_per_scene}"
        )
        logger.info(
            f"Total number of negative pairs for each scene: {self.n_total_negative_pairs_per_scene}"
        )
        logger.info(
            f"Total average number of positive pairs: {self.n_total_positive_pairs_per_scene.mean()}"
        )
        logger.info(
            f"Total average number of negative pairs: {self.n_total_negative_pairs_per_scene.mean()}"
        )

        self.name = name

    def _make_location_indices(self):
        """Set up the FAISS indices containing the patch locations for each scene."""
        faiss_location_indices = []
        all_patch_locations = []
        for scene_idx in tqdm(range(self.n_scenes), desc="setting up FAISS indices"):
            scene_dataset = self.datasource.datasets[scene_idx]

            quantizer = faiss.IndexFlatL2(3)
            nlist = 100  # Number of clusters
            patch_location_faiss_index = faiss.IndexIVFFlat(
                quantizer, 3, nlist, faiss.METRIC_L2
            )
            patch_location_faiss_index.nprobe = 50

            patch_location_path = os.path.join(
                scene_dataset.data_root,
                self.scene_names[scene_idx],
                self.patch_locations_file_name,
            )
            patch_locations = np.load(patch_location_path).astype(np.float32)
            # crop the patch locations to the correct aspect ratio
            patch_locations = rearrange(patch_locations, "n h w t -> n t h w")
            patch_locations = crop_to_aspect_ratio(
                patch_locations, scene_dataset.aspect_ratio
            )
            patch_locations = rearrange(patch_locations, "n t h w -> n h w t")
            # NOTE: all the patch locations from different dataset types _should_ be set up so that we now have the same shape for all

            self.n_patches_per_image = (
                patch_locations.shape[1] * patch_locations.shape[2]
            )

            # filter out patches belonging to filtered out images
            patch_locations = patch_locations[scene_dataset.used_img_mask]

            patch_locations = rearrange(patch_locations, "n h w t -> (n h w) t", t=3)

            patch_location_faiss_index.train(patch_locations)
            patch_location_faiss_index.add(patch_locations)

            faiss_location_indices.append(patch_location_faiss_index)
            all_patch_locations.append(patch_locations)

        return faiss_location_indices, all_patch_locations

    def _estimate_total_pairs(self, n_test_patches=2000):
        """Estimate for each scene the number of positive and the number of negative pairs in the environment."""

        total_pairs = []
        total_positive_pairs = []

        for scene_idx in range(self.n_scenes):
            patch_location_faiss_index = self.patch_location_faiss_indices[scene_idx]
            patch_locations = self.patch_locations[scene_idx]
            n_patches = patch_location_faiss_index.ntotal

            n_potential_pairs = n_patches**2
            patch_idx_permutation = np.random.permutation(n_patches)
            test_patch_idxs = patch_idx_permutation[:n_test_patches]

            test_patch_locations = patch_locations[test_patch_idxs]

            pos_lims, _, _ = patch_location_faiss_index.range_search(
                test_patch_locations, self.pos_radius**2
            )

            (
                pos_neg_lims,
                _,
                _,
            ) = patch_location_faiss_index.range_search(
                test_patch_locations, self.neg_radius**2
            )

            # I effectively check n_test_patches x n_patches pairs for each scene
            total_pairs.append(
                pos_neg_lims[-1] * n_potential_pairs / (n_test_patches * n_patches)
            )
            total_positive_pairs.append(
                pos_lims[-1] * n_potential_pairs / (n_test_patches * n_patches)
            )
        total_pairs = np.array(total_pairs)
        total_positive_pairs = np.array(total_positive_pairs)
        total_negative_pairs = total_pairs - total_positive_pairs

        return total_positive_pairs, total_negative_pairs

    def _fill_pair_bank(self):
        """Sample new pairs and fill the pair bank with them, for each scene."""

        # NOTE: Parallelising creates some race condition in the FAISS range search, so don't do that
        for scene_idx in tqdm(range(self.n_scenes), desc=f"Filling bank for one scene"):
            self.pair_bank[scene_idx] = self._sample_pairs(
                scene_idx, self.pair_bank_size
            )

    def _sample_pairs(self, scene_idx, n_pairs):
        """Sample n_pairs positive pairs from scene scene_idx to put into the pair bank.

        Parameters
        ----------
        scene_idx (int):
            The index of the scene to sample from.

        n_pairs (int):
            How many pairs to sample.

        Returns
        -------
        pair_idxs (n_pairs x 2 np.ndarray):
            the indices of the patches making up the pairs.
        """
        # NOTE: This is only a rough heuristic, valid only for "normal" values for self.pos_radius (ie pos_radius < 1.0m) and for n_pairs <~ 20000
        # In principle, the series implementation is faster when there are a lot of candidate values within pos radius that then need to get masked out and when n_pairs is large -- in that case masking out invalid patches requires re-assigning a very large array, which is slow.
        # With a bit more work, it might be possible to empirically derive a function to decide this, but for now I'll just use this heuristic.

        if self.pos_radius < 0.5:
            return self._sample_pairs_parallel(scene_idx, n_pairs)
        elif n_pairs < 500:
            return self._sample_pairs_parallel(scene_idx, n_pairs)
        else:
            return self._sample_pairs_series(scene_idx, n_pairs)

    def _sample_pairs_series(self, scene_idx, n_pairs):
        """Sample n_pairs positive pairs from scene scene_idx to put into the pair bank. Does so using for-loops.

        Parameters
        ----------
        scene_idx (int):
            The index of the scene to sample from.

        n_pairs (int):
            How many pairs to sample.

        Returns
        -------
        pair_idxs (n_pairs x 2 np.ndarray):
            the indices of the patches making up the pairs.
        """

        patch_location_faiss_index = self.patch_location_faiss_indices[scene_idx]
        patch_locations = self.patch_locations[scene_idx]

        n_patches = patch_location_faiss_index.ntotal

        patch_idx_permutation = np.random.permutation(n_patches)
        source_patch_idxs = patch_idx_permutation[:n_pairs]
        idx_of_last_source_patch = n_pairs - 1

        source_patch_locations = patch_locations[source_patch_idxs]

        lims, _, idxs = patch_location_faiss_index.range_search(
            source_patch_locations, self.pos_radius**2
        )
        mask = np.ones_like(idxs, dtype=bool)
        for i in range(n_pairs):
            mask[lims[i] : lims[i + 1]] = (
                idxs[lims[i] : lims[i + 1]] != source_patch_idxs[i]
            )

        pair_idxs = np.empty((n_pairs, 2), dtype=np.int64)
        pair_idxs[:, 0] = source_patch_idxs

        ## Sample second patch for each pair
        # NOTE: I tried vectorising this by not using np.random.choice in a for-loop but rather np.random.randint(low, high), but when I do that I have to mask the entire array of pos_neg_idxs with neg_mask (pos_neg_idxs[neg_mask]), and as it turns out this array is so large that the time spent allocating a new array for the masked-out idxs is larger than the speed gain from vectorising the sampling. In this implementation, I mask only chunks, and never assign the entire masked idxs array anywhere.

        for i in range(n_pairs):
            # filter out invalid patches
            current_mask = mask[lims[i] : lims[i + 1]]
            current_idxs = idxs[lims[i] : lims[i + 1]]
            current_idxs = current_idxs[current_mask]

            # sometimes but rarely all patches get filtered out, need to re-sample
            while len(current_idxs) == 0:
                idx_of_last_source_patch += 1
                idx_of_last_source_patch = idx_of_last_source_patch % n_patches
                new_source_patch_idx = patch_idx_permutation[idx_of_last_source_patch]
                pair_idxs[i, 0] = new_source_patch_idx

                _, _, new_idxs = patch_location_faiss_index.range_search(
                    patch_locations[new_source_patch_idx : new_source_patch_idx + 1],
                    self.pos_radius**2,
                )
                new_mask = new_idxs != new_source_patch_idx

                current_idxs = new_idxs[new_mask]

            pair_idxs[i, 1] = np.random.choice(current_idxs)

        return pair_idxs

    def _sample_pairs_parallel(self, scene_idx, n_pairs):
        """Sample n_pairs positive pairs from scene scene_idx to put into the pair bank. Does so in a vectorised manner.

        Parameters
        ----------
        scene_idx (int):
            The index of the scene to sample from.

        n_pairs (int):
            How many pairs to sample.

        Returns
        -------
        pair_idxs (n_pairs x 2 np.ndarray):
            the indices of the patches making up the pairs.
        """

        # TODO I think this might be faster vectorised (see the comment for _fill_single_scene_pair_bank above) -- I should try out which is faster for which values of n_pairs (I expect there is a cross-over point)
        patch_location_faiss_index = self.patch_location_faiss_indices[scene_idx]
        patch_locations = self.patch_locations[scene_idx]

        n_patches = patch_location_faiss_index.ntotal

        patch_idx_permutation = np.random.permutation(n_patches)
        source_patch_idxs = patch_idx_permutation[:n_pairs]
        idx_of_first_unused_source_patch = n_pairs

        source_patch_locations = patch_locations[source_patch_idxs]

        lims, _, idxs = patch_location_faiss_index.range_search(
            source_patch_locations, self.pos_radius**2
        )
        mask = np.ones_like(idxs, dtype=bool)
        for i in range(n_pairs):
            mask[lims[i] : lims[i + 1]] = (
                idxs[lims[i] : lims[i + 1]] != source_patch_idxs[i]
            )

        idxs = idxs[mask]
        # calculate new limits: count how many are masked out before each upper limit
        lims[1:] -= np.cumsum(~mask, dtype=np.uint)[lims[1:] - 1]

        # re-sample all the ones that got fully filtered out
        to_resample = np.where(lims[1:] == lims[:-1])[0]
        lims = np.delete(lims, to_resample)
        source_patch_idxs = np.delete(source_patch_idxs, to_resample)

        resampled_source_idxs = []
        resampled_lims = []
        resampled_idxs = []
        last_lim = lims[-1]

        while len(to_resample) > 0:
            new_source_idxs = patch_idx_permutation[
                idx_of_first_unused_source_patch : idx_of_first_unused_source_patch
                + len(to_resample)
            ]
            idx_of_first_unused_source_patch += len(to_resample)

            new_source_patch_locations = patch_locations[new_source_idxs]

            new_lims, _, new_idxs = patch_location_faiss_index.range_search(
                new_source_patch_locations, self.pos_radius**2
            )
            new_mask = np.ones_like(new_idxs, dtype=bool)
            for i in range(len(to_resample)):
                new_mask[new_lims[i] : new_lims[i + 1]] = (
                    new_idxs[new_lims[i] : new_lims[i + 1]] != new_source_idxs[i]
                )

            new_idxs = new_idxs[new_mask]
            new_lims[1:] -= np.cumsum(~new_mask, dtype=np.uint)[new_lims[1:] - 1]

            to_resample = np.where(new_lims[1:] == new_lims[:-1])[0]
            new_lims = np.delete(new_lims, to_resample)
            new_source_idxs = np.delete(new_source_idxs, to_resample)

            resampled_source_idxs.append(new_source_idxs)
            resampled_idxs.append(new_idxs)
            resampled_lims.append(new_lims[1:] + last_lim)
            last_lim += new_lims[-1]

        source_patch_idxs = np.concatenate((source_patch_idxs, *resampled_source_idxs))
        lims = np.concatenate((lims, *resampled_lims))
        idxs = np.concatenate((idxs, *resampled_idxs))

        pair_idxs = np.empty((n_pairs, 2), dtype=np.int64)
        pair_idxs[:, 0] = source_patch_idxs

        ## Sample second patch for each pair
        pair_idxs[:, 1] = idxs[np.random.randint(lims[:-1], lims[1:])]

        return pair_idxs

    def __len__(self):
        return self.n_batches_per_epoch

    def __iter__(self):
        self._fill_pair_bank()
        n_samples_from_each_scene = np.zeros(self.n_scenes, dtype=np.int64)

        # sample from each scene equally often, but in random order
        # depending on how many anchor pairs are actually needed, but on average each scene will be sampled from the same number of times
        scene_order = torch.tensor(
            list(range(self.n_scenes))
            * ceil(self.max_anchor_pairs * self.n_batches_per_epoch / self.n_scenes)
        )
        scene_order = scene_order[torch.randperm(len(scene_order))]
        current_scene_order_idx = 0

        for batch_idx in range(self.n_batches_per_epoch):

            anchor_pairs = []
            anchor_pair_scene_idxs = []
            image_idxs = set([])

            while (
                len(image_idxs) < self.batch_size
                and len(anchor_pairs) < self.max_anchor_pairs
            ):
                scene_idx = scene_order[current_scene_order_idx]
                current_scene_order_idx += 1

                new_anchor_pair = self.pair_bank[scene_idx][
                    n_samples_from_each_scene[scene_idx]
                ]
                anchor_pairs.append(new_anchor_pair[None])
                anchor_pair_scene_idxs.append(scene_idx)

                # add (scene_idx, img_idx_within_scene) to image_idxs
                image_idxs.add(
                    (scene_idx, new_anchor_pair[0] // self.n_patches_per_image)
                )
                image_idxs.add(
                    (scene_idx, new_anchor_pair[1] // self.n_patches_per_image)
                )

                n_samples_from_each_scene[scene_idx] += 1

            anchor_pairs = np.concatenate(anchor_pairs)  # n_anchor_pairs x 2
            anchor_pair_scene_idxs = np.array(anchor_pair_scene_idxs)  # n_anchor_pairs
            image_idxs = np.array(sorted(list(image_idxs)))  # n_imgs x 2

            # turn anchor pair patch indices from indices across all patches in the scene to indices across patches in the batch
            within_img_patch_idxs = anchor_pairs % self.n_patches_per_image
            anchor_img_env_idxs = anchor_pairs // self.n_patches_per_image

            matches = np.nonzero(
                # match scene idxs
                (anchor_pair_scene_idxs[:, None, None] == image_idxs[None, None, :, 0])
                # match img idxs within scene
                & (anchor_img_env_idxs[:, :, None] == image_idxs[None, None, :, 1])
            )  # n_anchor_pairs x 2 x n_imgs

            anchor_img_batch_idxs = np.empty_like(anchor_img_env_idxs)
            anchor_img_batch_idxs[matches[0], matches[1]] = matches[2]
            anchor_pairs = (
                anchor_img_batch_idxs * self.n_patches_per_image + within_img_patch_idxs
            )

            img_patch_locations = []
            for scene_idx, img_idx in image_idxs:
                img_patch_locations.append(
                    rearrange(
                        self.patch_locations[scene_idx],
                        "(n_imgs n_patches_per_img) three -> n_imgs n_patches_per_img three",
                        n_patches_per_img=self.n_patches_per_image,
                        three=3,
                    )[img_idx]
                )
            img_patch_locations = np.array(img_patch_locations)

            # make batch to yield
            # the PatchPairDataset handles this
            batch = [
                {
                    "anchor_pair_scene_idxs": anchor_pair_scene_idxs,
                    "type": "pass_through_information",
                    "anchor_patch_idxs": anchor_pairs,
                    "img_patch_locations": img_patch_locations,
                    "n_total_positive_pairs": self.n_total_positive_pairs,
                    "n_total_negative_pairs": self.n_total_negative_pairs,
                }
            ]

            for scene_idx, img_idx in image_idxs:
                batch.append(
                    {"scene_idx": scene_idx, "type": "img", "img_idx": img_idx}
                )

            yield batch
