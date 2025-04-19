from bisect import bisect

from torch.utils.data import Dataset

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

DATASET_NAMES = {
    "indoor": ["baidu", "gardens", "17places"],
    "outdoor": ["pitts30k", "st_lucia", "robotcar"],
    "underground": ["hawkins", "laurel_caverns"],
    "underwater": ["eiffel"],
}


class VLAD_Vocab_Dataset(Dataset):

    def __init__(self, dataset_names, img_size, datasets_folder):
        super().__init__()

        self.dataset_names = dataset_names
        self.img_size = img_size
        self.datasets_folder = datasets_folder

        if type(dataset_names) is str:
            self.dataset_names = DATASET_NAMES[dataset_names]

        self.datasets = []

        for dataset_name in self.dataset_names:
            if dataset_name == "baidu":  # indoors
                self.datasets.append(Baidu_Dataset(img_size, datasets_folder))
            elif dataset_name == "gardens":  # indoors
                self.datasets.append(Gardens_Dataset(img_size, datasets_folder))
            elif dataset_name == "17places":  # indoors
                self.datasets.append(Places17_Dataset(img_size, datasets_folder))
            elif dataset_name.startswith("matterport"):
                self.datasets.append(MatterportVPRDataset(img_size, datasets_folder))
            elif dataset_name.startswith("scannet"):
                self.datasets.append(ScanNetVPRDataset(img_size, datasets_folder))
            elif dataset_name.startswith("pitts"):  # outdoors
                self.datasets.append(Pitts30k_Dataset(img_size, datasets_folder))
            elif dataset_name == "st_lucia":  # outdoors
                self.datasets.append(St_Lucia_Dataset(img_size, datasets_folder))
            elif dataset_name == "robotcar":  # outdoors
                self.datasets.append(Robotcar_Dataset(img_size, datasets_folder))
            elif dataset_name.startswith("hawkins"):  # underground
                self.datasets.append(Hawkins_Dataset(img_size, datasets_folder))
            elif dataset_name == "laurel_caverns":  # undereground
                self.datasets.append(Laurel_Caverns_Dataset(img_size, datasets_folder))
            elif dataset_name == "eiffel":  # underwater
                self.datasets.append(Eiffel_Dataset(img_size, datasets_folder))
            else:
                raise ValueError(f"Unknown dataset name: {dataset_name}")

        self.cum_lengths = [0]
        # cumulative lengths of the datasets
        for d in self.datasets:
            self.cum_lengths.append(self.cum_lengths[-1] + len(d))

    def __len__(self):
        return self.cum_lengths[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect(self.cum_lengths, idx) - 1
        return self.datasets[dataset_idx][idx - self.cum_lengths[dataset_idx]]
