import os

import numpy as np
import torch
import torchvision.transforms as T
from natsort import natsorted
from PIL import Image
from torch.utils.data import Dataset

from src.utils.data import crop_to_aspect_ratio

base_transform = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.Resize((405, 720)),  # Prevent OOM on GPU (with same aspect ratio)
    ]
)


class Eiffel_Dataset(Dataset):
    """
    Returns dataset class with images from database and queries for the gardens dataset.
    """

    def __init__(self, img_size, datasets_folder):
        super().__init__()

        self.img_size = img_size
        self.aspect_ratio = img_size[1] / img_size[0]  # height / width

        self.dataset_name = "eiffel"
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
            self.query_abs_paths.append(
                os.path.join(self.datasets_folder, self.dataset_name, "q_images", q)
            )

        self.db_num = len(self.db_abs_paths)
        self.query_num = len(self.query_abs_paths)

        # returns dictionary of eiffel dataset
        self.gt_positives = np.load(
            os.path.join(self.datasets_folder, self.dataset_name, "eiffel_gt.npy"),
            allow_pickle=True,
        )[101:]

        self.gt_indices = []
        for i in range(len(self.gt_positives)):
            self.gt_indices.append(
                self.gt_positives[i][1]
            )  # corresponds to index wise soft positives

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
