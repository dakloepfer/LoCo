import os

import torch
from loguru import logger as loguru_logger
from omegaconf import OmegaConf
from tqdm import tqdm

from lightning import pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from src.data.segmentation_dataset import (
    MATTERPORT_N_OBJECTS_PER_SCENE,
    MATTERPORT_N_STUFF_CLASS_OBJECTS,
    SCANNET_N_OBJECTS_PER_SCENE,
    SCANNET_N_STUFF_CLASS_OBJECTS,
    SegmentationDataset,
)
from src.lightning.linear_probe import LinearProbeModule
from src.lightning.smoothap_module import PLModule
from src.utils.misc import get_rank_zero_only_logger

loguru_logger = get_rank_zero_only_logger(loguru_logger)


def precalculate_features(config, feature_extractor, dataset):
    # calculate the features, return a TensorDataset with the features and ground truth segmentations as labels
    # Don't downsample the gt_segmentations to the feature resolution
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.eval.segmentation.batch_size,
        shuffle=False,
        num_workers=config.data.max_num_workers,
        pin_memory=config.data.pin_memory,
    )
    feature_extractor.eval()

    features = []
    gt_segmentations = []

    for batch in tqdm(dataloader, "Pre-calculating Segmentation Features"):
        gt_segmentations.append(batch["segmentation"])
        with torch.no_grad():
            features.append(
                feature_extractor(batch["img"].to(config.device_list[0])).cpu()
            )

    features = torch.cat(features, dim=0)
    gt_segmentations = torch.cat(gt_segmentations, dim=0)
    feature_dataset = torch.utils.data.TensorDataset(features, gt_segmentations)

    return feature_dataset


def test_segmentation(config, profiler=None):
    # save config
    os.makedirs(os.path.join(config.save_dir, config.exp_name), exist_ok=True)
    OmegaConf.save(
        config, os.path.join(config.save_dir, config.exp_name, "config.yaml")
    )

    feature_extractor = PLModule(config, profiler=profiler)
    if len(config.device_list) == 1:
        feature_extractor = feature_extractor.to(config.device_list[0])

    with open(config.data.scene_list, "r") as f:
        scene_names = [name.split()[0] for name in f.readlines()]

    data_source = config.data.data_source
    data_root = config.data.data_root

    if data_source == "Matterport":
        n_valid_objects = MATTERPORT_N_OBJECTS_PER_SCENE
        n_stuff_class_objects = MATTERPORT_N_STUFF_CLASS_OBJECTS
    elif data_source == "ScanNet":
        n_valid_objects = SCANNET_N_OBJECTS_PER_SCENE
        n_stuff_class_objects = SCANNET_N_STUFF_CLASS_OBJECTS
    else:
        raise NotImplementedError(f"Data source {data_source} not implemented.")

    overall_results = {
        "mAP": [],
        "mIoU": [],
        "Jaccard": [],
        "Accuracy": [],
        "object_mAP": [],
        "object_mIoU": [],
        "object_Jaccard": [],
        "object_Accuracy": [],
        "stuff_mAP": [],
        "stuff_mIoU": [],
        "stuff_Jaccard": [],
        "stuff_Accuracy": [],
        "n_images": [],
    }
    output_file = os.path.join(
        config.save_dir, config.exp_name, "segmentation_results.txt"
    )
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    # write a header to the output_file
    with open(output_file, "w+") as f:
        f.write(
            "|\tScene Name\t|\tmAP\t\t|\tmIoU\t\t|\tJaccard\t\t|\tAccuracy\t|\tobject_mAP\t|\tobject_mIoU\t|\tobject_Jaccard\t|\tobject_Accuracy\t|\tstuff_mAP\t|\tstuff_mIoU\t|\tstuff_Jaccard\t|\tstuff_Accuracy\t|\tn_images\t|\n"
        )
        f.write(
            "_________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________\n"
        )

    for scene_name in scene_names:
        loguru_logger.info('Testing scene "{}"'.format(scene_name))

        segmentation_dataset = SegmentationDataset(
            data_root,
            scene_name,
            dataset_type=data_source,
            normalize=config.data.normalize,
            **config.data.dataset,
        )
        feature_dataset = precalculate_features(
            config, feature_extractor, segmentation_dataset
        )

        linear_probe_train_loader = torch.utils.data.DataLoader(
            feature_dataset,
            batch_size=config.eval.segmentation.batch_size,
            shuffle=True,
            num_workers=min(
                config.data.max_num_workers, config.eval.segmentation.batch_size
            ),
            pin_memory=True,
        )
        linear_probe = LinearProbeModule(
            config, n_valid_objects[scene_name], n_stuff_class_objects[scene_name]
        )

        logger = TensorBoardLogger(
            save_dir=os.path.join(config.save_dir, config.exp_name),
            name=f"linear_probe_training_scene{scene_name}",
            default_hp_metric=False,
        )

        trainer = pl.Trainer(
            max_epochs=config.eval.segmentation.linear_probe_max_epochs,
            logger=logger,
            enable_progress_bar=True,
            devices=1,
        )
        with torch.enable_grad():
            loguru_logger.info(
                'Start training linear probe for "{}"'.format(scene_name)
            )
            trainer.fit(linear_probe, linear_probe_train_loader)
            loguru_logger.info(f"Finished training linear probe for {scene_name}!")

        linear_probe_test_loader = torch.utils.data.DataLoader(
            feature_dataset,
            batch_size=config.eval.segmentation.batch_size,
            shuffle=False,
            num_workers=min(
                config.data.max_num_workers, config.eval.segmentation.batch_size
            ),
            pin_memory=True,
        )
        loguru_logger.info(
            f"Start calculating segmentation performance on scene {scene_name}!"
        )
        trainer.test(linear_probe, linear_probe_test_loader)

        scene_results = linear_probe.get_results()
        scene_results["n_images"] = len(feature_dataset)

        with open(output_file, "a") as f:
            f.write(
                f"|\t{scene_name}\t|\t{scene_results['mAP']:.5f}\t\t|\t{scene_results['mIoU']:.5f}\t\t|\t{scene_results['Jaccard']:.5f}\t\t|\t{scene_results['Accuracy']:.5f}\t\t|\t{scene_results['object_mAP']:.5f}\t\t|\t{scene_results['object_mIoU']:.5f}\t\t|\t{scene_results['object_Jaccard']:.5f}\t\t|\t{scene_results['object_Accuracy']:.5f}\t\t|\t{scene_results['stuff_mAP']:.5f}\t\t|\t{scene_results['stuff_mIoU']:.5f}\t\t|\t{scene_results['stuff_Jaccard']:.5f}\t\t|\t{scene_results['stuff_Accuracy']:.5f}\t\t|\t{scene_results['n_images']}\t\t|\n"
            )

        for key in overall_results.keys():
            if key == "n_images":
                overall_results[key].append(len(feature_dataset))
                continue
            overall_results[key].append(scene_results[key])
        loguru_logger.info(
            'Finished segmentation testing on scene "{}"'.format(scene_name)
        )
    mean = lambda x: sum(x) / len(x)
    with open(output_file, "a") as f:
        f.write(
            "_________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________\n"
        )
        f.write(
            f"|\tAverage\t\t|\t{mean(overall_results['mAP']):.5f}\t\t|\t{mean(overall_results['mIoU']):.5f}\t\t|\t{mean(overall_results['Jaccard']):.5f}\t\t|\t{mean(overall_results['Accuracy']):.5f}\t\t|\t{mean(overall_results['object_mAP']):.5f}\t\t|\t{mean(overall_results['object_mIoU']):.5f}\t\t|\t{mean(overall_results['object_Jaccard']):.5f}\t\t|\t{mean(overall_results['object_Accuracy']):.5f}\t\t|\t{mean(overall_results['stuff_mAP']):.5f}\t\t|\t{mean(overall_results['stuff_mIoU']):.5f}\t\t|\t{mean(overall_results['stuff_Jaccard']):.5f}\t\t|\t{mean(overall_results['stuff_Accuracy']):.5f}\t\t|\t{sum(overall_results['n_images'])}\t\t|\n"
        )
