import logging
import os

import hydra
import torch
from omegaconf import DictConfig

from lightning import pytorch as pl
from src.test_scripts.location_consistency import test_location_consistency
from src.test_scripts.place_recognition import test_place_recognition
from src.test_scripts.segmentation import test_segmentation
from src.utils.profiler import build_profiler

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="main", version_base=None)
def main(config: DictConfig):
    logger.info(f"Number of available GPUs: {torch.cuda.device_count()}")

    pl.seed_everything(config.seed)  # reproducibility

    profiler = build_profiler(config)

    torch.autograd.set_grad_enabled(False)

    save_dir_stem = config.save_dir
    tested_location_consistency = False
    if type(config.test_tasks) == str:
        config.test_tasks = [config.test_tasks]

    for task in config.test_tasks:
        config.save_dir = os.path.join(save_dir_stem, task)
        if not os.path.exists(config.save_dir):
            os.makedirs(config.save_dir)

        if task in ["mean_average_precision", "pixel_correspondences"]:
            if not tested_location_consistency:
                test_location_consistency(config, profiler)
                tested_location_consistency = True
            else:
                continue
        elif task == "segmentation":
            test_segmentation(config, profiler)
        elif task == "place_recognition":
            test_place_recognition(config, profiler)

        else:
            raise NotImplementedError(f"Task {task} has not been implemented!")

    logger.info("All tasks have been tested!")


if __name__ == "__main__":
    main()
