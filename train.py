#!/usr/bin/env python

# SPDX-FileCopyrightText: © 2024 Marten Lienen <m.lienen@tum.de> & Technical University of Munich
#
# SPDX-License-Identifier: MIT

import faulthandler
import logging
import math
import os
import warnings
from pathlib import Path

import hydra
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer.connectors.signal_connector import _SignalConnector

from turbdiff.config import instantiate_data_and_task
from turbdiff.plots import OpenFOAMPlots
from turbdiff.time_limit import TimeLimit
from turbdiff.utils import (
    WandbModelCheckpoint,
    WandbSummaries,
    filter_device_available,
    get_logger,
    log_hyperparameters,
    print_config,
    print_exceptions,
    set_seed,
)

# Log to traceback to stderr on segfault
faulthandler.enable(all_threads=False)

# Stop lightning from pestering us about things we already know
warnings.filterwarnings(
    "ignore",
    "There is a wandb run already in progress",
    module="pytorch_lightning.loggers.wandb",
)
warnings.filterwarnings(
    "ignore",
    "The dataloader, [^,]+, does not have many workers",
    module="pytorch_lightning",
)
logging.getLogger("pytorch_lightning.utilities.rank_zero").addFilter(
    filter_device_available
)


def if_eq(a, b, then, otherwise):
    """A conditional for OmegaConf interpolations."""
    if a == b:
        return then
    else:
        return otherwise


def resolve_eval(expr):
    """Resolve an arbitrary expression in OmegaConf interpolations."""
    # We trust our own configuration not to delete all our files, so just eval the
    # expression
    return eval(expr, {}, {"math": math})


OmegaConf.register_new_resolver("if_eq", if_eq)
OmegaConf.register_new_resolver("eval", resolve_eval)


log = get_logger()


def store_slurm_job_id(config: DictConfig):
    array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
    array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    job_id = os.environ.get("SLURM_JOB_ID")

    with open_dict(config):
        if array_job_id is not None and array_task_id is not None:
            config.slurm_job_id = f"{array_job_id}_{array_task_id}"
        elif job_id is not None:
            config.slurm_job_id = job_id


class Null_SignalConnector(_SignalConnector):
    def register_signal_handlers(self):
        pass


def get_callbacks(config):
    if config.monitor is not None:
        monitor = {"monitor": config.monitor, "mode": "min"}
    else:
        monitor = {}
    callbacks = [
        WandbModelCheckpoint(
            save_last=True, save_top_k=1, every_n_epochs=1, filename="best", **monitor
        ),
        TQDMProgressBar(refresh_rate=1),
        LearningRateMonitor(logging_interval="step"),
        OpenFOAMPlots(data_dir=Path(config.data.root) / "data"),
    ]
    if monitor != {}:
        callbacks.append(WandbSummaries(**monitor))
    if config.get("early_stopping") is not None and monitor != {}:
        stopper = EarlyStopping(
            patience=int(config.early_stopping),
            min_delta=0,
            strict=False,
            check_on_train_epoch_end=False,
            **monitor,
        )
        callbacks.append(stopper)
    if config.get("train_limit") is not None:
        callbacks.append(TimeLimit(config.train_limit))
    return callbacks


@hydra.main(config_path="config", config_name="train", version_base=None)
@print_exceptions
def main(config: DictConfig):
    set_seed(config)

    store_slurm_job_id(config)

    # Resolve interpolations to work around a bug:
    # https://github.com/omry/omegaconf/issues/862
    OmegaConf.resolve(config)
    wandb.init(**config.wandb, resume=(config.wandb.mode == "online") and "allow")
    print_config(config)

    torch.set_float32_matmul_precision(config.matmul_precision)
    # Indirect access to cuda module to avoid problems in pickling main function for
    # slurm submission
    if getattr(torch, "cuda").is_available():
        if config.matmul_precision == "medium":
            getattr(torch.backends, "cuda").matmul.allow_tf32 = True
            getattr(torch.backends, "cudnn").allow_tf32 = True
        elif config.matmul_precision == "high":
            getattr(torch.backends, "cuda").matmul.allow_tf32 = False
            getattr(torch.backends, "cudnn").allow_tf32 = True
        elif config.matmul_precision == "highest":
            getattr(torch.backends, "cuda").matmul.allow_tf32 = False
            getattr(torch.backends, "cudnn").allow_tf32 = False

    log.info("Instantiating data and system")
    datamodule, task = instantiate_data_and_task(config)

    if config.get("compile") is not None:
        log.info("Compiling model")
        task = torch.compile(task, mode=config.compile)

    logger = WandbLogger()
    log_hyperparameters(logger, config, task)

    log.info("Instantiating trainer")
    callbacks = get_callbacks(config)
    trainer: Trainer = instantiate(config.trainer, callbacks=callbacks, logger=logger)
    # submitit handles the requeuing, so we disable pytorch-lightning's SLURM feature
    trainer.signal_connector = Null_SignalConnector(trainer)

    if config.get("restart_from") is not None:
        log.info(f"Restarting training from {config.restart_from}")

        if ":" in config.restart_from:
            run_path, ckpt_name = config.restart_from.split(":")
        else:
            run_path, ckpt_name = config.restart_from, "last.ckpt"

        ckpt_root = Path(wandb.run.dir)
        f = wandb.restore(
            f"checkpoints/{ckpt_name}", run_path=run_path, root=str(ckpt_root)
        )
        if f is not None:
            f.close()
            downloaded_path = ckpt_root / "checkpoints" / ckpt_name
            ckpt_path = ckpt_root / "checkpoints" / "start.ckpt"
            downloaded_path.rename(ckpt_path)
        else:
            log.error("Could not download checkpoint!")
            return 1
    else:
        ckpt_path = None

    log.info("Starting training!")
    trainer.fit(task, datamodule=datamodule, ckpt_path=ckpt_path)

    if config.eval_testset:
        log.info("Starting testing!")
        trainer.test(ckpt_path="best", datamodule=datamodule)

    wandb.finish()
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    best_score = trainer.checkpoint_callback.best_model_score
    return float(best_score) if best_score is not None else None


if __name__ == "__main__":
    main()
