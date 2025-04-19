import hydra
import torch
from torch import nn
from yacs.config import CfgNode

from src.models.baselines import DINO_Baseline, DINOv2_Baseline, ResNet50Baseline
from src.models.dino_with_head import DINO_with_head
from src.models.pure_dino_feature_extractor import Pure_DINO_FeatureExtractor
from src.utils.misc import cfg_to_dict


class ConcatModel(nn.Module):
    """Model that takes multiple models and concatenates their outputs"""

    def __init__(self, network_configs: list, concat_dim=1):
        super().__init__()

        self.models = nn.ModuleList([])

        for config in network_configs:
            self.models.append(hydra.utils.instantiate(config))

        self.output_subsample = self.models[0].output_subsample
        for model in self.models[1:]:
            assert model.output_subsample == self.output_subsample

        self.concat_dim = concat_dim

    def forward(self, x):

        outputs = []
        for model in self.models:
            outputs.append(model(x))

        return torch.cat(outputs, dim=self.concat_dim)
