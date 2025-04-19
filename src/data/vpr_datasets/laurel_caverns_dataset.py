import os

import numpy as np
import torchvision.transforms as T
from natsort import natsorted
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


class Laurel_Caverns_Dataset(Dataset):
    """
    Returns dataset class with images from database and queries for the gardens dataset.
    """

    def __init__(self, img_size, datasets_folder, dist_thresh=8):
        super().__init__()

        self.img_size = img_size
        self.aspect_ratio = img_size[1] / img_size[0]  # height / width

        self.dataset_name = "laurel_caverns"
        self.datasets_folder = datasets_folder

        self.db_paths = natsorted(
            os.listdir(
                os.path.join(self.datasets_folder, self.dataset_name, "db_images")
            )
        )
        self.query_paths = natsorted(
            os.listdir(
                os.path.join(self.datasets_folder, self.dataset_name, "q_images")
            )
        )

        self.db_abs_paths = []
        self.query_abs_paths = []

        for p in self.db_paths:
            self.db_abs_paths.append(
                os.path.join(self.datasets_folder, self.dataset_name, "db_images", p)
            )

        for q in self.query_paths:
            self.ueryabs_paths.append(
                os.path.join(self.datasets_folder, self.dataset_name, "q_images", q)
            )

        self.db_num = len(self.db_abs_paths)
        self.query_num = len(self.query_abs_paths)

        self.gt_poses = np.load(
            os.path.join(
                self.datasets_folder, self.dataset_name, "pose_topic_list.npy"
            ),
            allow_pickle=True,
        )  # returns dictionary of Laurel Caverns poses

        self.db_gt_arr = self.gt_poses[:141, :2]
        self.query_gt_arr = self.gt_poses[229:341, :2]

        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.db_gt_arr)
        self.dist_thresh = dist_thresh
        self.dist, self.gt_indices = knn.radius_neighbors(
            self.query_gt_arr, radius=self.dist_thresh, return_distance=True
        )

        self.images_paths = list(self.db_abs_paths) + list(self.query_abs_paths)
        self.db_indices = list(range(self.db_num))
        self.query_indices = list(range(self.db_num, self.db_num + self.query_num))

    def __getitem__(self, index):
        img = Image.open(self.images_paths[index])

        img = base_transform(img)
        img = crop_to_aspect_ratio(img, self.aspect_ratio)
        img = T.functional.resize(img, self.img_size, antialias=True)
        return img

    def __len__(self):
        return len(self.images_paths)
