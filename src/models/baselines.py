import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from sklearn.decomposition import PCA
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import tqdm

from src.data.matterport_dataset import MatterportDataset
from src.data.vlad_vocab_dataset import VLAD_Vocab_Dataset
from src.models.dino_network import DINO_Network
from src.models.dinov2_network import DINOv2_Network


class PCA_Baseline(nn.Module):
    """Base class for baseline models that handels all the PCA stuff to reduce the feature dimensionality"""

    def __init__(self, model_type, feature_dim, dataset_config=None):
        super().__init__()

        self.data_conf = dataset_config
        self.feature_dim = feature_dim
        self.network = None
        self.model_type = model_type

        self.pca_path = f"{model_type}_{feature_dim}features_pca.pkl"
        self.training_features_for_pca = []
        self.pca = None

    def _setup_dataset(
        self,
    ):
        """To make it a bit easier to set up different datasets"""
        raise NotImplementedError("Not Implemented yet for using Hydra")
        data_source = self.data_conf.TRAIN_DATA_SOURCE
        if data_source == "Matterport":
            data_root = self.data_conf.TRAIN_DATA_ROOT
            scene_list_path = self.data_conf.TRAIN_SCENE_LIST
            intrinsics_path = self.data_conf.TRAIN_INTRINSICS_PATH
            matterport_config = {
                "horizontal_only": self.data_conf.DATASET.MATTERPORT_HORIZONTAL_IMGS_ONLY,
                "normalize": self.data_conf.DATASET.MATTERPORT_NORMALIZE,
                "img_height": self.data_conf.DATASET.IMG_HEIGHT,
                "img_width": self.data_conf.DATASET.IMG_WIDTH,
            }
            mode = "train"

            with open(scene_list_path, "r") as f:
                scene_names = [name.split()[0] for name in f.readlines()]

            datasets = []
            for scene_name in tqdm(
                scene_names,
                desc=f"Loading {mode} datasets",
            ):

                datasets.append(
                    MatterportDataset(
                        data_root,
                        scene_name,
                        intrinsics_path,
                        mode=mode,
                        pose_dir=None,
                        augment_fn=None,
                        **matterport_config,
                    )
                )
            return ConcatDataset(datasets)

        elif data_source == "VPR":
            return VLAD_Vocab_Dataset(
                self.config.PLACE_RECOGNITION.VLAD_VOCAB_DATASET_NAMES[0],
                self.config.PLACE_RECOGNITION.IMG_SIZE,
                self.config.PLACE_RECOGNITION.DATASETS_FOLDER,
            )

        else:
            raise NotImplementedError(f"Data source {data_source} not implemented")

    def make_pca(self):
        if os.path.exists(self.pca_path):
            with open(self.pca_path, "rb") as f:
                self.load_pca(pickle.load(f))
        else:
            dataset = self._setup_dataset()
            loader = DataLoader(dataset, batch_size=4, num_workers=4, pin_memory=True)
            self.network = self.network.to(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
            with torch.no_grad():
                n_batches = 0
                for batch in tqdm(
                    loader, f"Getting the PCA for the {self.model_type} baseline"
                ):
                    if type(batch) is dict:
                        imgs = (
                            batch["img"].to("cuda:0")
                            if torch.cuda.is_available()
                            else batch["img"]
                        )
                    else:
                        imgs = (
                            batch.to("cuda:0") if torch.cuda.is_available() else batch
                        )

                    features = self.network(imgs)
                    self.training_features_for_pca.append(
                        rearrange(features.cpu(), "b c h w -> (b h w) c")
                    )
                    n_batches += 1
                    if n_batches >= 3200:
                        break
                self.calculate_pca()

            with open(
                self.pca_path,
                "wb+",
            ) as pca_file:
                pickle.dump(self.pca, pca_file)

    def calculate_pca(self):
        """Use the stored training features to calculate principal components"""
        if len(self.training_features_for_pca) == 0:
            raise ValueError("No training features stored")

        self.training_features_for_pca = torch.cat(
            self.training_features_for_pca, dim=0
        ).numpy()
        self.pca = PCA(self.feature_dim)
        self.pca.fit(self.training_features_for_pca)
        self.training_features_for_pca = None
        self.pca_mean = (
            torch.tensor(
                self.pca.mean_, dtype=torch.float, device=next(self.parameters()).device
            )
            if self.pca.mean_ is not None
            else torch.zeros(1, self.feature_dim, device=next(self.parameters()).device)
        )
        self.pca_components = torch.tensor(
            self.pca.components_,
            dtype=torch.float,
            device=next(self.parameters()).device,
        )

    def load_pca(self, pca):
        self.pca = pca
        self.training_features_for_pca = None
        self.pca_mean = (
            torch.tensor(
                self.pca.mean_, dtype=torch.float, device=next(self.parameters()).device
            )
            if self.pca.mean_ is not None
            else torch.zeros(1, self.feature_dim, device=next(self.parameters()).device)
        )
        self.pca_components = torch.tensor(
            self.pca.components_,
            dtype=torch.float,
            device=next(self.parameters()).device,
        )

    def forward(self, imgs):
        features = self.network(imgs)
        B, _, H, W = features.shape

        if self.feature_dim != self.out_dim:
            if self.pca is not None:
                features = rearrange(features, "b c h w -> (b h w) c")
                if self.pca_mean.device != features.device:
                    self.pca_mean = self.pca_mean.to(features.device)
                    self.pca_components = self.pca_components.to(features.device)
                features = features - self.pca_mean
                features = torch.matmul(features, self.pca_components.T)
                features = rearrange(
                    features, "(b h w) c -> b c h w", b=B, h=H, w=W, c=self.feature_dim
                )
            else:
                raise ValueError("PCA not calculated yet")

        return F.normalize(features, dim=1)  # normalize


class DINO_Baseline(PCA_Baseline):
    # Just a wrapper to keep the code consistent with the actual models
    def __init__(
        self,
        n_blocks: int = 11,
        model: str = "vitb8",
        use_value_facet: bool = True,
        feature_dim: int = 768,
        dataset_config=None,
    ):
        super().__init__(
            model_type="dino_baseline",
            feature_dim=feature_dim,
            dataset_config=dataset_config,
        )

        self.network = DINO_Network(
            n_blocks=n_blocks, model=model, use_value_facet=use_value_facet
        )
        self.out_dim = self.network.out_dim
        self.output_subsample = self.network.output_subsample

        if self.feature_dim != self.out_dim:
            self.make_pca()


class DINOv2_Baseline(PCA_Baseline):
    # Just a wrapper to keep the code consistent with the actual models
    def __init__(
        self,
        n_blocks: int = 11,
        model: str = "vitb14",
        use_value_facet: bool = True,
        use_layer_norm: bool = False,
        concat_class_token: bool = False,
        feature_dim: int = 768,
        dataset_config=None,
    ):
        super().__init__(
            model_type="dinov2_baseline",
            feature_dim=feature_dim,
            dataset_config=dataset_config,
        )

        self.network = DINOv2_Network(
            n_blocks=n_blocks,
            model=model,
            use_value_facet=use_value_facet,
            use_layer_norm=use_layer_norm,
            concat_class_token=concat_class_token,
        )
        self.out_dim = self.network.out_dim
        self.output_subsample = self.network.output_subsample

        if self.feature_dim != self.out_dim:
            self.make_pca()


class ResNet50Baseline(PCA_Baseline):
    def __init__(self, feature_dim: int, dataset_config=None):
        super().__init__(
            model_type="resnet50_baseline",
            feature_dim=feature_dim,
            dataset_config=dataset_config,
        )

        self.weights = ResNet50_Weights.DEFAULT
        resnet = resnet50(weights=self.weights)
        resnet.eval()
        self.network = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.network.eval()

        self.output_subsample = 32
        self.out_dim = 512

        if self.feature_dim != self.out_dim:
            self.make_pca()
