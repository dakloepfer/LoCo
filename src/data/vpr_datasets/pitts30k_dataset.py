import os
from glob import glob
from os.path import join

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


class Pitts30k_Dataset(Dataset):
    """Dataset with images from database and queries, used for inference (testing and building cache)."""

    def __init__(self, img_size, datasets_folder, dist_thresh=25):
        super().__init__()

        self.img_size = img_size
        self.aspect_ratio = img_size[1] / img_size[0]  # height / width

        self.dataset_name = "pitts30k"
        self.dist_thresh = dist_thresh

        self.datasets_folder = datasets_folder
        self.dataset_folder = join(self.datasets_folder, self.dataset_name)

        self.dataset_folder = join(self.dataset_folder, "images", "test")
        database_folder_name, query_folder_name = "database", "queries"
        if not os.path.exists(self.dataset_folder):
            raise FileNotFoundError(f"Folder {self.dataset_folder} does not exist")

        #### Read paths and UTM coordinates for all images.
        database_folder = join(self.dataset_folder, database_folder_name)
        query_folder = join(self.dataset_folder, query_folder_name)
        if not os.path.exists(database_folder):
            raise FileNotFoundError(f"Folder {database_folder} does not exist")
        if not os.path.exists(query_folder):
            raise FileNotFoundError(f"Folder {query_folder} does not exist")

        self.db_paths = natsorted(
            glob(join(database_folder, "**", "*.jpg"), recursive=True)
        )
        self.query_paths = natsorted(
            glob(join(query_folder, "**", "*.jpg"), recursive=True)
        )

        db_utms = np.array(
            [(path.split("@")[1], path.split("@")[2]) for path in self.db_paths],
            float,
        )
        query_utms = np.array(
            [(path.split("@")[1], path.split("@")[2]) for path in self.query_paths],
            float,
        )
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(db_utms)
        self.gt_indices = knn.radius_neighbors(
            query_utms, radius=self.dist_thresh, return_distance=False
        )

        self.images_paths = list(self.db_paths) + list(self.query_paths)

        self.db_num = len(self.db_paths)
        self.query_num = len(self.query_paths)

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
