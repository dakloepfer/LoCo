import os

import numpy as np
import torch
import torchvision.transforms as T
from natsort import natsorted
from PIL import Image
from scipy.spatial.transform import Rotation
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset

from src.utils.data import crop_to_aspect_ratio


def get_cop_pose(file):
    """
    Takes in input of .camera file for baidu and outputs the cop numpy array [x y z] and 3x3 rotation matrix
    """
    with open(file) as f:
        lines = f.readlines()
        xyz_cop_line = lines[-2]
        # print(cop_line)
        xyz_cop = np.fromstring(xyz_cop_line, dtype=float, sep=" ")

        r1 = np.fromstring(lines[4], dtype=float, sep=" ")
        r2 = np.fromstring(lines[5], dtype=float, sep=" ")
        r3 = np.fromstring(lines[6], dtype=float, sep=" ")
        r = Rotation.from_matrix(np.array([r1, r2, r3]))
        # print(R)

        R_euler = r.as_euler("zyx", degrees=True)

    return xyz_cop, R_euler


base_transform = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class Baidu_Dataset(Dataset):
    """
    Return dataset class with images from database and queries for the Baidu dataset
    """

    def __init__(self, img_size, datasets_folder, dist_thresh=10):

        super().__init__()

        self.dataset_name = "baidu_datasets"
        self.datasets_folder = datasets_folder

        self.db_paths = natsorted(
            os.listdir(
                os.path.join(
                    self.datasets_folder, self.dataset_name, "training_images_undistort"
                )
            )
        )
        self.db_gt_paths = natsorted(
            os.listdir(
                os.path.join(self.datasets_folder, self.dataset_name, "training_gt")
            )
        )
        self.query_paths = natsorted(
            os.listdir(
                os.path.join(
                    self.datasets_folder, self.dataset_name, "query_images_undistort"
                )
            )
        )
        self.query_gt_paths = natsorted(
            os.listdir(
                os.path.join(self.datasets_folder, self.dataset_name, "query_gt")
            )
        )

        self.db_abs_paths = []
        self.query_abs_paths = []

        for p in self.db_paths:
            self.db_abs_paths.append(
                os.path.join(
                    self.datasets_folder,
                    self.dataset_name,
                    "training_images_undistort",
                    p,
                )
            )

        for q in self.query_paths:
            self.query_abs_paths.append(
                os.path.join(
                    self.datasets_folder, self.dataset_name, "query_images_undistort", q
                )
            )

        self.db_num = len(self.db_paths)
        self.query_num = len(self.query_paths)

        self.dist_thresh = dist_thresh

        self.img_size = img_size
        self.aspect_ratio = img_size[1] / img_size[0]  # height / width

        # form pose array from db_gt .camera files
        self.db_gt_arr = np.zeros((self.db_num, 3))  # for xyz
        self.db_gt_arr_euler = np.zeros((self.db_num, 3))  # for euler angles

        for idx, db_gt_file_rel in enumerate(self.db_gt_paths):
            db_gt_file = os.path.join(
                self.datasets_folder, self.dataset_name, "training_gt", db_gt_file_rel
            )

            with open(db_gt_file) as f:
                cop_pose, cop_R = get_cop_pose(db_gt_file)

            self.db_gt_arr[idx, :] = cop_pose
            self.db_gt_arr_euler[idx, :] = cop_R

        # form pose array from q_gt .camera files
        self.query_gt_arr = np.zeros((self.query_num, 3))  # for xyz
        self.query_gt_arr_euler = np.zeros((self.query_num, 3))  # for euler angles

        for idx, query_gt_file_rel in enumerate(self.query_gt_paths):
            query_gt_file = os.path.join(
                self.datasets_folder, self.dataset_name, "query_gt", query_gt_file_rel
            )

            with open(query_gt_file) as f:
                cop_pose, cop_R = get_cop_pose(query_gt_file)

            self.query_gt_arr[idx, :] = cop_pose
            self.query_gt_arr_euler[idx, :] = cop_R

        # Find soft_positives_per_query, which are within val_positive_dist_threshold only
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.db_gt_arr)
        self.dist, self.gt_indices = knn.radius_neighbors(
            self.query_gt_arr, radius=self.dist_thresh, return_distance=True
        )

        self.images_paths = list(self.db_abs_paths) + list(self.query_abs_paths)
        self.db_indices = list(range(self.db_num))
        self.query_indices = list(range(self.db_num, self.db_num + self.query_num))

    def __getitem__(self, index):
        img = Image.open(self.images_paths[index]).convert("RGB")

        img = base_transform(img)
        img = crop_to_aspect_ratio(img, self.aspect_ratio)
        img = T.functional.resize(img, self.img_size, antialias=True)

        return img

    def __len__(self):
        return len(self.images_paths)
