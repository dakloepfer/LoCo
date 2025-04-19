import os
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from natsort import natsorted
from torch.utils.data import ConcatDataset, Dataset
from torchvision.transforms.functional import to_tensor

from src.utils.data import crop_to_aspect_ratio


class PatchPairDataset(Dataset):
    """Dataset that returns individual patch pairs from a single scene"""

    def __init__(
        self,
        data_root,
        scene_name,
        dataset_type="matterport3d",
        mode="train",
        augment_fn=None,
        **kwargs,
    ):
        super().__init__()

        self.data_root = data_root
        self.dataset_type = dataset_type.lower()
        self.scene_name = scene_name
        self.augment_fn = augment_fn if mode == "train" else None

        self.horizontal_only = kwargs.get("horizontal_only", False)
        self.max_iou = kwargs.get("max_iou", -1.0)

        if kwargs.get("normalize", "imagenet") == "imagenet":
            self.normalize = transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            )  # for backbones pretrained on ImageNet
        elif kwargs["normalize"] == "scannet":
            self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        else:
            raise NotImplementedError(
                f"Image normalization {kwargs['normalize']} not implemented."
            )

        self.img_height = kwargs.get("img_height", 1024)
        self.img_width = kwargs.get("img_width", 1024)
        self.aspect_ratio = self.img_width / self.img_height

        self.img_dir = None
        self.used_img_mask = None

        self.data_dict = self._create_data_dict()

    def __len__(self):
        return len(self.data_dict)

    def filter_img_files(self, img_dir):
        """Filter out images that aren't used for various reasons."""
        if self.dataset_type.startswith("matterport"):
            # need to keep order consistent with patch_locations
            all_img_files = sorted(os.listdir(img_dir))
        else:
            all_img_files = natsorted(os.listdir(img_dir))

        if self.max_iou > 0:
            img_overlaps = torch.from_numpy(
                np.load(
                    os.path.join(self.data_root, self.scene_name, "img_overlaps.npy")
                )
            )
            img_overlaps[torch.eye(len(all_img_files)).bool()] = (
                0.0  # ignore self-pairs
            )
            img_ious = (img_overlaps + img_overlaps.T) / 2

        used_img_files = []
        used_file_mask = torch.ones(len(all_img_files), dtype=torch.bool)

        for i, file_name in enumerate(all_img_files):
            if self.horizontal_only:
                # remove all the files that aren't looking horizontally
                if not file_name[-7] == "1":
                    used_file_mask[i] = False
                    continue
            if self.max_iou > 0:
                # make sure that images have at most a certain amount of IoU with each other
                if not used_file_mask[i]:
                    continue  # already removed this image
                used_file_mask[img_ious[i] > self.max_iou] = False

            used_img_files.append(file_name)

        self.used_img_mask = used_file_mask
        return used_img_files, used_file_mask

    def _create_data_dict(self):
        data_dict = {}
        scene_dir = os.path.join(self.data_root, self.scene_name)

        # ScanNet scenes start with "scene" and Matterport3D scenes are unique hashes
        img_dir_name = "color_npy" if self.dataset_type == "scannet" else "rgb"
        self.img_dir = os.path.join(scene_dir, f"{img_dir_name}")

        used_img_files, _ = self.filter_img_files(self.img_dir)

        for sample_idx, img_file_name in enumerate(used_img_files):
            data_dict[sample_idx] = {
                "img_path": os.path.join(self.img_dir, img_file_name),
            }

        return data_dict

    def _getimgitem(self, idx, scene_idx):
        img_path = self.data_dict[idx]["img_path"]

        img = np.load(img_path)
        img = to_tensor(img)

        img = crop_to_aspect_ratio(img, self.aspect_ratio)
        img = F.interpolate(
            img.unsqueeze(0),
            size=(self.img_height, self.img_width),
        ).squeeze(0)
        img = self.normalize(img)

        if self.augment_fn is not None:
            img = self.augment_fn(img)

        return {"img": img, "within_scene_img_idx": idx, "scene_idx": scene_idx}

    def __getitem__(self, idx):
        if idx["type"] == "img":
            return self._getimgitem(idx["img_idx"], idx["scene_idx"])
        elif idx["type"] == "pass_through_information":
            return idx
        else:
            raise ValueError(f"Invalid sample type {idx['type']}")


class PatchPairConcatDataset(ConcatDataset):
    r"""Dataset as a concatenation of multiple PatchPairDatasets..

    This class is useful to assemble different existing datasets, and is adapted from the built-in ConcatDataset to handle indices differently..

    Args:
        datasets (sequence): List of datasets to be concatenated
    """

    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__(datasets)

    def __getitem__(self, idx):
        if type(idx) != dict:
            return super().__getitem__(idx)

        if "type" in idx.keys():
            if idx["type"] == "pass_through_information":
                return idx

        if "dataset_idx" in idx.keys():
            dataset_idx = idx["dataset_idx"]
        elif "scene_idx" in idx.keys():
            dataset_idx = idx["scene_idx"]
        else:
            raise ValueError("Must provide dataset_idx or scene_idx as key in idx dict")

        return self.datasets[dataset_idx][idx]
