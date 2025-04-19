import omegaconf
from loguru import logger as loguru_logger
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from lightning import Trainer
from lightning.pytorch.callbacks import (
    DeviceStatsMonitor,
    ModelSummary,
    TQDMProgressBar,
)
from src.lightning.data import MultiSceneDataModule
from src.lightning.smoothap_module import PLModule
from src.utils.misc import get_rank_zero_only_logger

loguru_logger = get_rank_zero_only_logger(loguru_logger)


def test_location_consistency(config, profiler=None):
    # lightning data
    data_module = MultiSceneDataModule(config.data)
    loguru_logger.info(f"DataModule initialized!")

    # lightning module
    model = PLModule(config, profiler=profiler)
    loguru_logger.info(f"LightningModule initialized!")

    # Logger
    if config.logger_name == "wandb":
        logger = WandbLogger(
            name=config.exp_name,
            project="locus",
            group=config.wandb_group,
            save_dir=config.save_dir,
        )
        logger.experiment.config.update(
            omegaconf.OmegaConf.to_container(
                config, resolve=True, throw_on_missing=True
            )
        )
    elif config.logger_name == "tensorboard":
        logger = TensorBoardLogger(
            save_dir=config.save_dir,
            name=config.exp_name,
            default_hp_metric=False,
        )

    # Callbacks
    model_summary = ModelSummary(max_depth=10)
    tqdm_progress_bar = TQDMProgressBar(refresh_rate=config.tqdm_refresh_rate)
    callbacks = [
        model_summary,
        tqdm_progress_bar,
    ]
    if config.profiler_name is not None:
        device_stats_monitor = DeviceStatsMonitor(cpu_stats=True)
        callbacks.append(device_stats_monitor)

    # Lightning Trainer
    trainer = Trainer(
        callbacks=callbacks,
        logger=logger,
        profiler=profiler,
        devices=1,  # only use one GPU for the trainer
        detect_anomaly=config.detect_anomaly,
        reload_dataloaders_every_n_epochs=False,  # avoid repeated samples!
    )
    loguru_logger.info(f"Trainer initialized!")
    loguru_logger.info(f"Start testing!")

    trainer.test(model, datamodule=data_module)
