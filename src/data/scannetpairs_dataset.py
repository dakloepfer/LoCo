import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from einops import rearrange
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

from src.utils.data import crop_to_aspect_ratio


class ScanNetPairsDataset(Dataset):

    def __init__(
        self,
        data_root,
        mode="val",
        augment_fn=None,
        max_n_pairs=None,
        **kwargs,
    ):

        self.data_root = data_root
        self.mode = mode
        self.augment_fn = augment_fn if mode == "train" else None

        self.img_height = kwargs.get("img_height", 1024)
        self.img_width = kwargs.get("img_width", 1024)
        self.aspect_ratio = self.img_width / self.img_height

        self.initial_img_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((480, 640))]
        )

        self.normalize_mode = kwargs.get("normalize", "imagenet")
        if self.normalize_mode == "imagenet":
            self.normalize = transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            )
        elif self.normalize_mode == "scannet":
            self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        else:
            raise NotImplementedError(
                f"Image normalization {kwargs['normalize']} not implemented."
            )
        self.pairs = self.get_pairs()

        self.length = (
            len(self.pairs)
            if max_n_pairs is None
            else min(max_n_pairs, len(self.pairs))
        )

    def get_pairs(self):
        intrinsics = np.load(os.path.join(self.data_root, "intrinsics.npz"))
        data = np.load(os.path.join(self.data_root, "test.npz"))

        pairs = []

        for i in range(len(data["name"])):
            room_id, seq_id, ins_0, ins_1 = data["name"][i]
            scene_id = f"scene{room_id:04d}_{seq_id:02d}"
            K_i = torch.tensor(intrinsics[scene_id]).float()
            T_AtoB = torch.tensor(data["rel_pose"][i]).float().reshape(3, 4)
            pose_AtoB = torch.eye(4)
            pose_AtoB[:3, :4] = T_AtoB
            pairs.append((scene_id, ins_0, ins_1, K_i, pose_AtoB))

        return pairs

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        scene, instanceA, instanceB, intrinsics, pose_AtoB = self.pairs[index]

        scene_path = os.path.join(self.data_root, scene)

        img_pathA = os.path.join(scene_path, "color", f"{instanceA}.jpg")
        img_pathB = os.path.join(scene_path, "color", f"{instanceB}.jpg")
        imgA = cv2.cvtColor(cv2.imread(img_pathA), cv2.COLOR_BGR2RGB)
        imgB = cv2.cvtColor(cv2.imread(img_pathB), cv2.COLOR_BGR2RGB)

        depth_pathA = os.path.join(scene_path, "depth", f"{instanceA}.png")
        depth_pathB = os.path.join(scene_path, "depth", f"{instanceB}.png")
        depthA = (
            cv2.imread(depth_pathA, cv2.IMREAD_ANYDEPTH).astype(np.float32) / 1000.0
        )
        depthB = (
            cv2.imread(depth_pathB, cv2.IMREAD_ANYDEPTH).astype(np.float32) / 1000.0
        )

        imgA = self.initial_img_transform(imgA)
        imgB = self.initial_img_transform(imgB)

        imgA, crop_borders = crop_to_aspect_ratio(
            imgA, self.aspect_ratio, return_borders=True
        )
        imgB = crop_to_aspect_ratio(imgB, self.aspect_ratio)

        depthA = crop_to_aspect_ratio(depthA, self.aspect_ratio)
        depthB = crop_to_aspect_ratio(depthB, self.aspect_ratio)
        imgA = F.interpolate(
            imgA[None], (self.img_height, self.img_width), mode="bilinear"
        ).squeeze(0)
        imgB = F.interpolate(
            imgB[None], (self.img_height, self.img_width), mode="bilinear"
        ).squeeze(0)
        depthA = cv2.resize(depthA, (self.img_width, self.img_height))
        depthB = cv2.resize(depthB, (self.img_width, self.img_height))

        # fix intrinsics
        intrinsics[0, 2] -= crop_borders[0]
        intrinsics[1, 2] -= crop_borders[1]
        pixel_ratio = (
            self.img_width / 640 if crop_borders[0] == 0 else self.img_height / 480
        )
        intrinsics[:2] *= pixel_ratio
        if self.augment_fn:
            imgA, imgB = self.augment_fn(imgA, imgB)

        imgA = self.normalize(imgA)
        imgB = self.normalize(imgB)

        batch = {
            "imgA": imgA,
            "imgB": imgB,
            "intrinsicA": intrinsics,
            "intrinsicB": intrinsics,
            "depthA": depthA,
            "depthB": depthB,
            "pose_AtoB": pose_AtoB,
        }

        return batch
