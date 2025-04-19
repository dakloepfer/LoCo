import os
from os.path import join

import numpy as np
import torch
import torchvision.transforms as T
from natsort import natsorted
from torch.utils.data import Dataset
from tqdm import tqdm

from src.utils.data import crop_to_aspect_ratio

base_transform = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class ScanNetVPRDataset(Dataset):

    def __init__(
        self,
        img_size,
        datasets_folder,
        scene_list_file="test_scenes_few.txt",
        dist_thresh=10.0,
    ):
        super().__init__()

        self.img_size = img_size
        self.aspect_ratio = img_size[1] / img_size[0]  # height / width

        self.min_overlap = 0.2
        self.max_iou = 0.4
        self.remove_from_db_iou = 0.7  # two images in database should have IoUs less than this -- to reduce database size
        self.max_dist = dist_thresh

        self.dataset_name = "scannet"
        self.datasets_folder = datasets_folder
        self.dataset_folder = join(self.datasets_folder, self.dataset_name)
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

        for scene in tqdm(self.scene_list, "making ScanNet VPR dataset..."):
            scene = scene.strip()
            scene_folder = join(self.dataset_folder, "scans", scene)

            img_folder = join(scene_folder, "color_npy")
            used_img_filepaths, used_file_mask = self.filter_img_files(img_folder)
            used_img_filepaths = [
                join(img_folder, filename) for filename in used_img_filepaths
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

            # sparsify the potential images
            # make sure that all the images in the database are reasonably different from each other
            sparse_img_filepaths = []

            available_idxs = set(list(range(len(used_img_filepaths))))
            for i in range(len(used_img_filepaths)):
                if i not in available_idxs:
                    continue

                sparse_img_filepaths.append(used_img_filepaths[i])

                remove_idxs = np.nonzero(img_ious[i] > self.remove_from_db_iou)[0]
                for j in remove_idxs:
                    if j in available_idxs:
                        available_idxs.remove(j)
            assert len(sparse_img_filepaths) == len(available_idxs)
            used_img_filepaths = sparse_img_filepaths
            used_idxs = np.array(sorted(list(available_idxs)), dtype=int)
            img_ious = img_ious[used_idxs][:, used_idxs]
            img_overlaps = img_overlaps[used_idxs][:, used_idxs]

            # load camera poses
            scene_cam_pos = np.zeros((len(used_img_filepaths), 3))
            cam_pos_folder = join(scene_folder, "pose")

            for i, img_file_path in enumerate(used_img_filepaths):
                file_name = os.path.basename(img_file_path).replace(".npy", ".txt")
                cam_pose_file = join(cam_pos_folder, file_name)
                cam_pos = np.loadtxt(cam_pose_file)[:3, 3]
                scene_cam_pos[i] = cam_pos

            forbidden_queries = set([])
            used_gt_idxs = set([])
            available_db_idxs = set(list(range(len(used_img_filepaths))))

            scene_gt_indices = []
            used_query_idxs = []

            for i in range(len(used_img_filepaths)):
                if i in forbidden_queries:
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
                cleaned_current_gt_indices = [
                    idx for idx in current_gt_indices if idx in available_db_idxs
                ]

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
                    if j in used_gt_idxs:
                        skip_current_image = True
                        break
                if skip_current_image:
                    continue

                # if we reach this, we use img i as a query image
                # remove too good images and the query image itself from the available database images
                for j in too_good_indices:
                    if j in available_db_idxs:
                        available_db_idxs.remove(j)
                if i in available_db_idxs:
                    available_db_idxs.remove(i)

                used_query_idxs.append(i)
                forbidden_queries.update(cleaned_current_gt_indices)
                used_gt_idxs.update(cleaned_current_gt_indices)
                scene_gt_indices.append(np.array(cleaned_current_gt_indices))

            if len(used_query_idxs) == 0:
                continue

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
            for i, gt_idxs in enumerate(scene_gt_indices):
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
        all_img_files = natsorted(os.listdir(img_dir))
        used_img_files = []
        used_file_mask = torch.ones(len(all_img_files), dtype=torch.bool)

        for i, file_name in enumerate(all_img_files):
            # Any filtering should be done here
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
