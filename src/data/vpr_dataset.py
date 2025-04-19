"""Basically a wrapper for the different Visual Place Recognition Datasets."""

from torch.utils.data import BatchSampler, DataLoader

from src.data.vpr_datasets.baidu_dataset import Baidu_Dataset
from src.data.vpr_datasets.eiffel_dataset import Eiffel_Dataset
from src.data.vpr_datasets.gardens_dataset import Gardens_Dataset
from src.data.vpr_datasets.hawkins_dataset import Hawkins_Dataset
from src.data.vpr_datasets.laurel_caverns_dataset import Laurel_Caverns_Dataset
from src.data.vpr_datasets.matterport_vpr_dataset import MatterportVPRDataset
from src.data.vpr_datasets.pitts30k_dataset import Pitts30k_Dataset
from src.data.vpr_datasets.places17_dataset import Places17_Dataset
from src.data.vpr_datasets.robotcar_dataset import Robotcar_Dataset
from src.data.vpr_datasets.scannet_vpr_dataset import ScanNetVPRDataset
from src.data.vpr_datasets.st_lucia_dataset import St_Lucia_Dataset


class VPR_Dataset:

    def __init__(
        self, dataset_name, img_size, datasets_folder, batch_size=1, num_workers=1
    ):

        self.name = dataset_name
        self.img_size = img_size
        self.datasets_folder = datasets_folder

        self.batch_size = batch_size
        self.num_workers = num_workers

        if self.name == "baidu":  # indoors
            self.dataset = Baidu_Dataset(img_size, datasets_folder)
        elif self.name == "gardens":  # indoors
            self.dataset = Gardens_Dataset(img_size, datasets_folder)
        elif self.name == "17places":  # indoors
            self.dataset = Places17_Dataset(img_size, datasets_folder)
        elif self.name.startswith("matterport"):
            self.dataset = MatterportVPRDataset(img_size, datasets_folder)
        elif self.name.startswith("scannet"):
            self.dataset = ScanNetVPRDataset(img_size, datasets_folder)
        elif self.name.startswith("pitts"):  # outdoors
            self.dataset = Pitts30k_Dataset(img_size, datasets_folder)
        elif self.name == "st_lucia":  # outdoors
            self.dataset = St_Lucia_Dataset(img_size, datasets_folder)
        elif self.name == "robotcar":  # outdoors
            self.dataset = Robotcar_Dataset(img_size, datasets_folder)
        elif self.name.startswith("hawkins"):  # underground
            self.dataset = Hawkins_Dataset(img_size, datasets_folder)
        elif self.name == "laurel_caverns":  # undereground
            self.dataset = Laurel_Caverns_Dataset(img_size, datasets_folder)
        elif self.name == "eiffel":  # underwater
            self.dataset = Eiffel_Dataset(img_size, datasets_folder)
        else:
            raise ValueError(f"Unknown dataset name: {self.name}")

    def get_db_imgs(self):

        return DataLoader(
            self.dataset,
            batch_sampler=BatchSampler(
                self.dataset.db_indices, self.batch_size, drop_last=False
            ),
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def get_query_imgs(self):
        return DataLoader(
            self.dataset,
            batch_sampler=BatchSampler(
                self.dataset.query_indices, self.batch_size, drop_last=False
            ),
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def get_gt_indices(self):
        return self.dataset.gt_indices
