import logging
import os

import hydra
import omegaconf
import torch
from omegaconf import DictConfig

import lightning.pytorch as pl
from lightning import Trainer
from lightning.pytorch.callbacks import (
    DeviceStatsMonitor,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from src.lightning.data import MultiSceneDataModule
from src.lightning.smoothap_module import PLModule
from src.utils.profiler import build_profiler

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="main", version_base=None)
def main(config: DictConfig):
    logger.info(f"Number of available GPUs: {torch.cuda.device_count()}")

    pl.seed_everything(config.seed)  # reproducibility

    # lightning data
    data_module = MultiSceneDataModule(config.data)
    logger.info(f"DataModule initialized!")

    # lightning module
    profiler = build_profiler(config)
    model = PLModule(config, profiler=profiler)
    logger.info(f"LightningModule initialized!")

    # Logger
    if config.logger_name == "wandb":
        exp_logger = WandbLogger(
            name=config.exp_name,
            project="locus",
            group=config.wandb_group,
            save_dir=config.save_dir,
        )
        exp_logger.experiment.config.update(omegaconf.OmegaConf.to_container(config))
        if config.log_gradients:
            # setting log="all" and log_graph=True results in some weird errors
            exp_logger.watch(model, log="gradients", log_graph=False)
    elif config.logger_name == "tensorboard":
        exp_logger = TensorBoardLogger(
            save_dir=config.save_dir,
            name=config.exp_name,
            default_hp_metric=False,
        )
    ckpt_dir = os.path.join(config.save_dir, "checkpoints", config.exp_name)
    if os.path.exists(ckpt_dir):
        version = len(os.listdir(ckpt_dir))
        ckpt_dir = os.path.join(ckpt_dir, f"version_{version}")
    else:
        ckpt_dir = os.path.join(ckpt_dir, "version_0")

    # Callbacks
    ckpt_callback = ModelCheckpoint(
        monitor="val/loss",
        verbose=True,
        save_top_k=10,
        mode="min",
        save_last=True,
        dirpath=ckpt_dir,
        auto_insert_metric_name=False,
        filename="epoch={epoch}-val_loss={val/loss:.3f}",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = ModelSummary(max_depth=10)
    tqdm_progress_bar = TQDMProgressBar(refresh_rate=config.tqdm_refresh_rate)
    callbacks = [
        lr_monitor,
        model_summary,
        tqdm_progress_bar,
    ]
    if not config.disable_ckpt:
        callbacks.append(ckpt_callback)
    if config.profiler_name is not None:
        device_stats_monitor = DeviceStatsMonitor(cpu_stats=True)
        callbacks.append(device_stats_monitor)

    # Lightning Trainer
    trainer = Trainer(
        max_epochs=config.max_epochs,
        gradient_clip_val=config.gradient_clipping,
        accumulate_grad_batches=config.accumulate_grad_batches,
        callbacks=callbacks,
        logger=exp_logger,
        reload_dataloaders_every_n_epochs=False,  # avoid repeated samples!
        profiler=profiler,
        detect_anomaly=config.detect_anomaly,
        devices=1,  # only use one GPU for the trainer
        fast_dev_run=config.fast_dev_run,
    )
    trainer.logger = exp_logger

    logger.info(f"Trainer initialized!")
    logger.info(f"Start training!")

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
