import os
from glob import glob
from os.path import join

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset

from src.utils.data import crop_to_aspect_ratio

base_transform = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class MatterportVPRDataset(Dataset):

    def __init__(
        self,
        img_size,
        datasets_folder,
        scene_list_file="test_scenes_reduced.txt",
        dist_thresh=10.0,
    ):
        super().__init__()

        self.img_size = img_size
        self.aspect_ratio = img_size[1] / img_size[0]  # height / width

        self.horizontal_only = False
        self.min_overlap = 0.2
        self.max_iou = 0.4
        self.max_dist = dist_thresh

        self.dataset_name = "matterport3d_allimgs_iou"
        self.datasets_folder = datasets_folder
        self.dataset_folder = join(self.datasets_folder, "matterport3d")
        if not os.path.exists(scene_list_file):
            scene_list_file = join(self.dataset_folder, scene_list_file)

        with open(scene_list_file, "r") as f:
            self.scene_list = f.readlines()

        self.query_paths = []
        self.db_paths = []
        self.gt_indices = []

        self.scene_cum_n_query_images = [0]
        self.scene_cum_n_db_images = [0]

        self.gt_indices = []

        for scene in self.scene_list:
            scene = scene.strip()
            scene_folder = join(self.dataset_folder, scene)

            img_folder = join(scene_folder, "rgb")
            used_img_filepaths, used_file_mask = self.filter_img_files(img_folder)
            used_img_filepaths = [
                join(scene_folder, "rgb", filename) for filename in used_img_filepaths
            ]
            used_file_mask = used_file_mask.numpy()

            # img_overlap element i, j gives fraction of pixels in image i that are visible in image j
            overlap_file = join(scene_folder, "img_overlaps.npy")
            img_overlaps = np.load(overlap_file)
            img_overlaps = img_overlaps[used_file_mask][:, used_file_mask]
            img_overlaps[np.eye(img_overlaps.shape[0]).astype(bool)] = (
                0  # remove self-pairs
            )
            img_ious = (img_overlaps + img_overlaps.T) / 2

            cam_pose_file = join(
                scene_folder, "undistorted_camera_parameters", f"{scene}.conf"
            )
            with open(cam_pose_file, "r") as f:
                cam_pose_lines = f.readlines()

            scene_cam_pos = np.zeros((len(used_img_filepaths), 3))

            for line in cam_pose_lines:
                if line.startswith("scan"):
                    scan_line = line.strip().split(" ")[2:]
                    img_file_name = scan_line[0].replace(".jpg", ".npy")
                    img_file_path = join(scene_folder, "rgb", img_file_name)

                    if img_file_path not in used_img_filepaths:
                        continue
                    img_file_index = used_img_filepaths.index(img_file_path)
                    if scene_cam_pos[img_file_index].any():
                        raise ValueError(
                            f"Camera position already set for image {img_file_path}"
                        )

                    cam_pos = [
                        float(scan_line[4]),
                        float(scan_line[8]),
                        float(scan_line[12]),
                    ]
                    scene_cam_pos[img_file_index] = cam_pos

                else:
                    continue

            scene_query_filepaths = []
            forbidden_queries = set([])
            used_query_idxs = []
            scene_gt_indices = []
            available_db_idxs = set(list(range(len(used_img_filepaths))))

            for i in range(len(used_img_filepaths)):
                if i in forbidden_queries:
                    # already used as ground-truth image for something or it's too good a match, so don't use as query
                    continue
                query_cam_pos = scene_cam_pos[i]
                current_gt_indices = (img_overlaps[i] >= self.min_overlap) & (
                    img_ious[i] < self.max_iou
                )
                current_gt_indices = current_gt_indices & (
                    np.linalg.norm(scene_cam_pos - query_cam_pos[None], axis=1)
                    < self.max_dist
                )
                current_gt_indices = np.nonzero(current_gt_indices)[0]
                cleaned_current_gt_indices = []
                for j in current_gt_indices:
                    if j not in available_db_idxs:
                        continue
                    else:
                        cleaned_current_gt_indices.append(j)

                # require at least one valid gt image
                if len(cleaned_current_gt_indices) == 0:
                    continue

                # remove indices for images that are too similar to query from database
                too_good_indices = np.nonzero(
                    (img_ious[i] >= self.max_iou)
                    & (
                        np.linalg.norm(scene_cam_pos - query_cam_pos[None], axis=1)
                        < self.max_dist
                    )
                )[0]
                # check if any of the too good images are already used as Ground Truth images for a previous query image -- if so, skip this query image
                skip_current_image = False
                for j in too_good_indices:
                    for sampled_gt_idxs in scene_gt_indices:
                        if j in sampled_gt_idxs:
                            skip_current_image = True
                            break
                    if skip_current_image:
                        break
                if skip_current_image:
                    continue

                for j in too_good_indices:
                    if j in available_db_idxs:
                        available_db_idxs.remove(j)
                if i in available_db_idxs:
                    available_db_idxs.remove(i)

                used_query_idxs.append(i)
                forbidden_queries.update(cleaned_current_gt_indices)
                forbidden_queries.update(too_good_indices)
                scene_gt_indices.append(np.array(cleaned_current_gt_indices))

            used_img_filepaths = np.array(used_img_filepaths)
            db_idxs = np.array(sorted(list(available_db_idxs)))
            scene_db_filepaths = used_img_filepaths[db_idxs]
            scene_db_filepaths = [str(fp) for fp in scene_db_filepaths]
            used_query_idxs = np.array(used_query_idxs)
            scene_query_filepaths = used_img_filepaths[used_query_idxs]
            scene_query_filepaths = [str(fp) for fp in scene_query_filepaths]

            scene_gt_indices = [
                np.nonzero(gt_idxs[:, None] == db_idxs[None])[1]
                + self.scene_cum_n_db_images[-1]
                for gt_idxs in scene_gt_indices
            ]
            for gt_idxs in scene_gt_indices:
                assert len(gt_idxs) > 0

            self.gt_indices += scene_gt_indices
            self.db_paths += scene_db_filepaths
            self.query_paths += scene_query_filepaths

            self.scene_cum_n_db_images.append(len(self.db_paths))
            self.scene_cum_n_query_images.append(len(self.query_paths))

        self.db_num = len(self.db_paths)
        self.query_num = len(self.query_paths)
        self.images_paths = list(self.db_paths) + list(self.query_paths)
        self.db_indices = list(range(self.db_num))
        self.query_indices = list(range(self.db_num, self.db_num + self.query_num))

        for gt_idxs in self.gt_indices:
            assert max(gt_idxs) < self.db_num
        db_paths = set(self.db_paths)
        for q in self.query_paths:
            assert q not in db_paths

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

    def __getitem__(self, index):
        img = np.load(self.images_paths[index])
        img = base_transform(img)
        img = crop_to_aspect_ratio(img, self.aspect_ratio)
        img = torch.nn.functional.interpolate(
            img.unsqueeze(0), size=self.img_size, mode="bilinear", align_corners=False
        ).squeeze(0)

        return img

    def __len__(self):
        return len(self.images_paths)
