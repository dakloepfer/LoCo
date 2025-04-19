from collections.abc import Iterable

import hydra
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
from tqdm import tqdm

import lightning.pytorch as pl
from src.data.patch_pair_dataset import PatchPairConcatDataset, PatchPairDataset
from src.data.scannetpairs_dataset import ScanNetPairsDataset
from src.utils.augment import build_augmentor
from src.utils.data import variable_keys_collate


class MultiSceneDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.augment_type = config.augmentation_type
        self.shuffle = config.shuffle

        self.val_tasks = config.val_tasks
        self.test_tasks = config.test_tasks

        self.matterport_config = {
            "horizontal_only": config.matterport_dataset.horizontal_imgs_only,
            "normalize": config.normalize,
            "img_height": config.matterport_dataset.img_height,
            "img_width": config.matterport_dataset.img_width,
        }
        self.scannet_config = {
            "normalize": config.normalize,
            "max_iou": config.scannet_dataset.max_iou,
            "img_height": config.scannet_dataset.img_height,
            "img_width": config.scannet_dataset.img_width,
        }
        self.scannetpairs_config = {
            "normalize": config.normalize,
            "img_height": config.scannetpairs_dataset.img_height,
            "img_width": config.scannetpairs_dataset.img_width,
            "max_n_pairs": config.scannetpairs_dataset.max_n_pairs,
        }

        self.disable_tqdm = config.tqdm_refresh_rate != 1

        self.batch_samplers = {"train": None, "val": None, "test": None}

    def setup(self, stage: str = None):
        """
        Setup train / val / test dataset. This method will be called by PL automatically.
        Args:
            stage (str): 'fit' in training phase, and 'test' in testing phase.
        """

        assert stage in ["fit", "test"], "stage must be either fit or test"

        if stage == "fit":
            self.augment_fn = build_augmentor(
                self.augment_type, data_source=self.config.train_data_source
            )

            self.train_dataset = self._setup_patchpair_dataset(
                self.config.train_data_source,
                self.config.train_data_root,
                self.config.train_scene_list,
                mode="train",
                augment_fn=self.augment_fn,
            )

            self.val_datasets = self._setup_eval_datasets(
                self.val_tasks,
                self.config.val_data_source,
                self.config.val_data_root,
                self.config.val_scene_list,
                mode="val",
            )
            logger.info("Train & Val Dataset loaded!")

        elif stage == "test":
            self.test_datasets = self._setup_eval_datasets(
                self.test_tasks,
                self.config.test_data_source,
                self.config.test_data_root,
                self.config.test_scene_list,
                mode="test",
            )
            logger.info("Test Dataset loaded!")

        else:
            raise NotImplementedError(f"Stage must be either fit or test, was {stage}")

    def _setup_patchpair_dataset(
        self,
        data_source,
        data_root,
        scene_list_path,
        mode="train",
        augment_fn=None,
    ):
        """To make it a bit easier to set up different datasets"""

        datasets = []

        if not isinstance(data_source, Iterable) or type(data_source) == str:
            data_source = [data_source]
        if not isinstance(data_root, Iterable) or type(data_root) == str:
            data_root = [data_root]
        if not isinstance(scene_list_path, Iterable) or type(scene_list_path) == str:
            scene_list_path = [scene_list_path]

        for ds, dr, slp in zip(data_source, data_root, scene_list_path):

            with open(slp, "r") as f:
                scene_names = [name.split()[0] for name in f.readlines()]

            if self.disable_tqdm and int(self.trainer.local_rank) == 0:
                print(f"Loading {mode} PatchPair datasets")

            for scene_name in tqdm(
                scene_names,
                desc=f"Loading {mode} patch pair datasets",
                disable=self.disable_tqdm,
            ):
                if ds == "Matterport":
                    datasets.append(
                        PatchPairDataset(
                            dr,
                            scene_name,
                            dataset_type=ds,
                            mode=mode,
                            augment_fn=augment_fn,
                            **self.matterport_config,
                        )
                    )
                elif ds == "ScanNet":
                    datasets.append(
                        PatchPairDataset(
                            dr,
                            scene_name,
                            dataset_type=ds,
                            mode=mode,
                            augment_fn=augment_fn,
                            **self.scannet_config,
                        )
                    )
                else:
                    raise NotImplementedError(
                        f"Data source {data_source} not implemented"
                    )

        if self.disable_tqdm and int(self.trainer.local_rank) == 0:
            print(f"Finished loading {mode} datasets.")

        return PatchPairConcatDataset(datasets)

    def _setup_pixelcorr_dataset(
        self, data_source, data_root, scene_list_path, mode="val", augment_fn=None
    ):
        datasets = []

        if not isinstance(data_source, Iterable) or type(data_source) == str:
            data_source = [data_source]
        if not isinstance(data_root, Iterable) or type(data_root) == str:
            data_root = [data_root]
        if not isinstance(scene_list_path, Iterable) or type(scene_list_path) == str:
            scene_list_path = [scene_list_path]

        for ds, dr, slp in zip(data_source, data_root, scene_list_path):

            if ds == "ScanNetPairs":
                datasets.append(
                    ScanNetPairsDataset(
                        dr, mode=mode, augment_fn=augment_fn, **self.scannetpairs_config
                    )
                )

            else:
                raise NotImplementedError(
                    f"Dataset of type {ds} not implemented as PixelCorrespondences Dataset."
                )

        return ConcatDataset(datasets)

    def _setup_eval_datasets(
        self, tasks, data_source, data_root, scene_list_path, mode="val"
    ):
        if not isinstance(data_source, Iterable) or type(data_source) == str:
            data_source = [data_source]
        if not isinstance(data_root, Iterable) or type(data_root) == str:
            data_root = [data_root]
        if not isinstance(scene_list_path, Iterable) or type(scene_list_path) == str:
            scene_list_path = [scene_list_path]

        # first dataset is always a mean AP dataset, ie a normal PatchPairDataset
        assert tasks[0] == "mean_average_precision"

        datasets = []
        for i, task in enumerate(tasks):
            if task == "mean_average_precision":
                # don't use augmentation for val / test
                dataset = self._setup_patchpair_dataset(
                    data_source[i],
                    data_root[i],
                    scene_list_path[i],
                    mode=mode,
                    augment_fn=None,
                )

            elif task == "pixel_correspondences":
                # don't use augmentation for val / test
                dataset = self._setup_pixelcorr_dataset(
                    data_source[i],
                    data_root[i],
                    scene_list_path[i],
                    mode=mode,
                    augment_fn=None,
                )
            else:
                raise NotImplementedError(
                    f"Creating appropriate datasets for task {task} not implemented yet"
                )
            datasets.append(dataset)

        return datasets

    def train_dataloader(self):
        batch_sampler = hydra.utils.instantiate(
            self.config.train_sampler,
            datasource=self.train_dataset,
            shuffle=self.shuffle,
            name="train",
        )
        self.batch_samplers["train"] = batch_sampler

        dataloader = DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=variable_keys_collate,
            num_workers=min(
                self.config.max_num_workers, self.config.train_sampler.batch_size
            ),
            pin_memory=self.config.pin_memory,
        )

        return dataloader

    def val_dataloader(self):
        dataloaders = []

        for task, val_dataset in zip(self.val_tasks, self.val_datasets):
            if task == "mean_average_precision":
                batch_sampler = hydra.utils.instantiate(
                    self.config.mean_ap_sampler,
                    datasource=val_dataset,
                    shuffle=False,
                    name="val",
                )
                dataloader = DataLoader(
                    val_dataset,
                    batch_sampler=batch_sampler,
                    collate_fn=variable_keys_collate,
                    num_workers=min(
                        self.config.max_num_workers,
                        self.config.mean_ap_sampler.batch_size,
                    ),
                    pin_memory=self.config.pin_memory,
                )
            elif task == "pixel_correspondences":
                dataloader = DataLoader(
                    val_dataset,
                    batch_size=self.pixel_corrs_sampler.batch_size,
                    shuffle=False,
                    num_workers=min(
                        self.config.max_num_workers,
                        self.config.pixel_corrs_sampler.batch_size,
                    ),
                    pin_memory=self.config.pin_memory,
                )

            dataloaders.append(dataloader)
        return dataloaders

    def test_dataloader(self):
        dataloaders = []

        for task, test_dataset in zip(self.test_tasks, self.test_datasets):
            if task == "mean_average_precision":
                batch_sampler = hydra.utils.instantiate(
                    self.config.mean_ap_sampler,
                    datasource=test_dataset,
                    shuffle=False,
                    name="test",
                )
                dataloader = DataLoader(
                    test_dataset,
                    batch_sampler=batch_sampler,
                    collate_fn=variable_keys_collate,
                    num_workers=min(
                        self.config.max_num_workers,
                        self.config.mean_ap_sampler.batch_size,
                    ),
                    pin_memory=self.config.pin_memory,
                )
            elif task == "pixel_correspondences":
                dataloader = DataLoader(
                    test_dataset,
                    batch_size=self.config.pixel_corrs_sampler.batch_size,
                    shuffle=False,
                    num_workers=min(
                        self.config.max_num_workers,
                        self.config.pixel_corrs_sampler.batch_size,
                    ),
                    pin_memory=self.config.pin_memory,
                )

            dataloaders.append(dataloader)
        return dataloaders
