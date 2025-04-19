import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from natsort import natsorted
from torchvision.transforms.functional import to_tensor

from src.utils.data import crop_to_aspect_ratio

STUFF_CLASSES = {
    "void": 0,
    "wall": 1,
    "floor": 2,
    "curtain": 3,
    "stairs": 4,
    "ceiling": 5,
    "mirror": 6,
    "shower": 7,
    "column": 8,
    "beam": 9,
    "railing": 10,
    "shelving": 11,
    "blinds": 12,
    "board_panel": 13,
    "misc": 14,
    "unlabeled": 15,
}

# number of valid objects, excluding all "stuff" class objects
with open("data/matterport3d/n_valid_objects.json", "r") as f:
    MATTERPORT_N_OBJECTS_PER_SCENE = json.load(f)

with open("data/matterport3d/n_stuff_objects.json", "r") as f:
    MATTERPORT_N_STUFF_CLASS_OBJECTS = json.load(f)

with open("data/scannet/scans/n_valid_objects.json", "r") as f:
    SCANNET_N_OBJECTS_PER_SCENE = json.load(f)

with open("data/scannet/scans/n_stuff_objects.json", "r") as f:
    SCANNET_N_STUFF_CLASS_OBJECTS = json.load(f)


class SegmentationDataset(torch.utils.data.Dataset):
    """Dataset for a single Matterport3D or ScanNet scene, returning the image and the segmentation mask."""

    def __init__(
        self,
        data_root,
        scene_name,
        dataset_type="matterport3d",
        augment_fn=None,
        **kwargs,
    ):
        super().__init__()
        self.data_root = data_root
        self.dataset_type = dataset_type.lower()
        self.scene_name = scene_name

        self.augment_fn = augment_fn

        self.horizontal_only = kwargs.get("horizontal_only", False)

        if kwargs.get("normalize", "imagenet") == "imagenet":
            self.normalize = transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            )  # for backbones pretrained on ImageNet
        else:
            raise NotImplementedError(
                f"Image normalization {kwargs['normalize']} not implemented."
            )

        self.img_height = kwargs.get("img_height", 1024)
        self.img_width = kwargs.get("img_width", 1024)
        self.aspect_ratio = self.img_width / self.img_height

        self.img_dir = None
        self.depth_dir = None

        self.data_dict = self._create_data_dict()

    def filter_img_files(self, img_dir):
        """Filter out images that aren't used for various reasons."""
        if self.dataset_type.startswith("matterport"):
            # need to keep order consistent with patch_locations
            all_img_files = sorted(os.listdir(img_dir))
        else:
            all_img_files = natsorted(os.listdir(img_dir))

        used_img_files = []
        used_file_mask = torch.ones(len(all_img_files), dtype=torch.bool)

        for i, file_name in enumerate(all_img_files):
            if self.horizontal_only:
                # remove all the files that aren't looking horizontally
                if not file_name[-7] == "1":
                    used_file_mask[i] = False
                    continue

            used_img_files.append(file_name)

        return used_img_files, used_file_mask

    def _create_data_dict(self):
        data_dict = {}
        scene_dir = os.path.join(self.data_root, self.scene_name)

        # ScanNet scenes start with "scene" and Matterport3D scenes are unique hashes
        img_dir_name = "color_npy" if self.dataset_type == "scannet" else "rgb"
        if self.img_height >= 448:
            img_folder = os.path.join(scene_dir, f"{img_dir_name}_highres")
        else:
            img_folder = os.path.join(scene_dir, f"{img_dir_name}")
        segmentation_folder = os.path.join(
            scene_dir,
            "object_maps",
        )
        file_names, _ = self.filter_img_files(img_folder)

        sample_idx = 0
        for file_name in file_names:
            img_path = os.path.join(img_folder, file_name)
            segmentation_path = os.path.join(segmentation_folder, file_name)
            if not os.path.exists(segmentation_path):
                continue
            try:
                np.load(img_path)
                np.load(segmentation_path)
            except:
                continue
            data_dict[sample_idx] = {
                "img_path": img_path,
                "segmentation_path": segmentation_path,
            }
            sample_idx += 1

        return data_dict

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, index):
        img_path = self.data_dict[index]["img_path"]
        segmentation_path = self.data_dict[index]["segmentation_path"]

        img = to_tensor(np.load(img_path))  # already RGB

        img = crop_to_aspect_ratio(img, self.aspect_ratio)
        img = F.interpolate(
            img.unsqueeze(0),
            size=(self.img_height, self.img_width),
        ).squeeze(0)
        img = self.normalize(img)

        # should be a tensor that contains the segmentation categories as ints
        segmentation = torch.from_numpy(np.load(segmentation_path)).long()

        segmentation = crop_to_aspect_ratio(segmentation, self.aspect_ratio)
        segmentation = (
            F.interpolate(
                segmentation[None, None].float(),
                size=(self.img_height, self.img_width),
                mode="nearest-exact",
            )
            .squeeze()
            .long()
        )
        if self.augment_fn is not None:
            img = self.transforms(img)

        sample = {"img": img, "segmentation": segmentation}
        return sample
