import os
from collections import namedtuple
from os.path import join

import torchvision.transforms as T
from PIL import Image
from scipy.io import loadmat
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset

from src.utils.data import crop_to_aspect_ratio


def path_to_pil_img(path):
    return Image.open(path).convert("RGB")


base_transform = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


dbStruct = namedtuple(
    "dbStruct",
    [
        "whichSet",
        "dataset",
        "dbImage",
        "locDb",
        "qImage",
        "locQ",
        "numDb",
        "numQ",
        "posDistThr",
        "posDistSqThr",
    ],
)


def parse_dbStruct(path):
    mat = loadmat(path)

    matStruct = mat["dbStruct"][0]

    dataset = "dataset"

    whichSet = "VPR"

    dbImage = matStruct[0]
    locDb = matStruct[1]

    qImage = matStruct[2]
    locQ = matStruct[3]

    numDb = matStruct[4].item()
    numQ = matStruct[5].item()

    posDistThr = matStruct[6].item()
    posDistSqThr = matStruct[7].item()

    return dbStruct(
        whichSet,
        dataset,
        dbImage,
        locDb,
        qImage,
        locQ,
        numDb,
        numQ,
        posDistThr,
        posDistSqThr,
    )


class Robotcar_Dataset(Dataset):
    def __init__(self, img_size, datasets_folder, dist_thresh=25):
        super().__init__()

        self.img_size = img_size
        self.aspect_ratio = img_size[1] / img_size[0]  # height / width

        self.datasets_folder = datasets_folder
        self.dataset_name = "Oxford_Robotcar"

        structFile = os.path.join(
            self.datasets_folder, self.dataset_name, "oxdatapart.mat"
        )
        root_dir = os.path.join(self.datasets_folder, self.dataset_name, "oxDataPart")

        self.dbStruct = parse_dbStruct(structFile)
        if dist_thresh is not None:  # Override localization radius
            self.dist_thresh = dist_thresh
        else:
            self.dist_thresh = self.dbStruct.posDistThr  # From file

        self.images_paths = [
            join(root_dir, dbIm.replace(" ", "")) for dbIm in self.dbStruct.dbImage
        ]

        self.images_paths += [
            join(root_dir, qIm.replace(" ", "")) for qIm in self.dbStruct.qImage
        ]

        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset

        self.db_num = self.dbStruct.numDb
        self.query_num = self.dbStruct.numQ

        self.soft_positives_per_query = None
        self.soft_positives_per_db = None
        self.distances = None

        knn = NearestNeighbors(n_jobs=1)
        knn.fit(self.dbStruct.locDb)

        self.distances, self.gt_indices = knn.radius_neighbors(
            self.dbStruct.locQ, radius=self.dist_thresh
        )
        self.db_indices = list(range(self.db_num))
        self.query_indices = list(range(self.db_num, self.db_num + self.query_num))

    def __getitem__(self, index):
        img = Image.open(self.images_paths[index])

        img = base_transform(img)
        img = crop_to_aspect_ratio(img, self.aspect_ratio)
        img = T.functional.resize(img, self.img_size, antialias=True)
        return img, index

    def __len__(self):
        return len(self.images_paths)
