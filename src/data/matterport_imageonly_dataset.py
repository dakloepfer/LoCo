import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_tensor


class MatterportImageOnlyDataset(torch.utils.data.Dataset):
    """Dataset for a single Matterport3D scene, returning the image only."""

    def __init__(self, data_root, scene_name, augment_fn=None, **kwargs):
        super().__init__()
        self.data_root = data_root
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
        self.downsampling_factor = self.img_height / 1024

        self.img_dir = None
        self.depth_dir = None

        self.data_dict = self._create_data_dict()

    def filter_img_files(self, img_dir):
        """Filter out images that aren't used for various reasons."""
        all_img_files = sorted(os.listdir(img_dir))
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

        if self.img_height >= 448:
            img_folder = os.path.join(scene_dir, "rgb_highres")
        else:
            img_folder = os.path.join(scene_dir, "rgb")

        file_names, _ = self.filter_img_files(img_folder)

        sample_idx = 0
        for file_name in file_names:
            img_path = os.path.join(img_folder, file_name)
            data_dict[sample_idx] = {
                "img_path": img_path,
            }
            sample_idx += 1

        return data_dict

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, index):
        img_path = self.data_dict[index]["img_path"]

        img = np.load(img_path)  # already RGB
        img = F.interpolate(
            to_tensor(img).unsqueeze(0),
            size=(self.img_height, self.img_width),
        ).squeeze(0)
        img = self.normalize(img)

        if self.augment_fn is not None:
            img = self.transforms(img)

        sample = {"img": img}
        return sample
