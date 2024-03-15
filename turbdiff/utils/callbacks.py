# SPDX-FileCopyrightText: © 2024 Marten Lienen <m.lienen@tum.de> & Technical University of Munich
#
# SPDX-License-Identifier: MIT

import math
import numbers
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint


class WandbModelCheckpoint(ModelCheckpoint):
    """Save checkpoints into the W&B run directory to sync them automatically."""

    def __init__(self, **kwargs):
        run_dir = Path(wandb.run.dir)
        cp_dir = run_dir / "checkpoints"

        super().__init__(**kwargs, dirpath=str(cp_dir))


class WandbSummaries(pl.Callback):
    """Set the W&B summaries of each metric to the values from the best epoch."""

    def __init__(self, monitor: str, mode: str):
        super().__init__()

        self.monitor = monitor
        self.mode = mode

        self.best_epoch = None
        self.best_metric = None
        self.best_metrics = None

        self.ready = True

    def on_sanity_check_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.ready = False

    def on_sanity_check_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.ready = True

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if not self.ready:
            return

        metrics = trainer.logged_metrics
        if self.monitor in metrics:
            metric = metrics[self.monitor]
            if torch.is_tensor(metric):
                metric = metric.item()

            if self._better(metric):
                self.best_epoch = trainer.current_epoch
                self.best_metric = metric
                self.best_metrics = self._copy_metrics(metrics)

        self._update_summaries()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._update_summaries()

    def on_exception(self, trainer: pl.Trainer, pl_module: pl.LightningModule, exception):
        self._update_summaries()

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._update_summaries()

    def state_dict(self):
        return {
            "monitor": self.monitor,
            "mode": self.mode,
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "best_metrics": self.best_metrics,
        }

    def load_state_dict(self, state_dict):
        self.monitor = state_dict["monitor"]
        self.mode = state_dict["mode"]
        self.best_epoch = state_dict["best_epoch"]
        self.best_metric = state_dict["best_metric"]
        self.best_metrics = state_dict["best_metrics"]

    def _better(self, metric):
        if self.best_metric is None or (
            not math.isfinite(self.best_metric) and math.isfinite(metric)
        ):
            # Moving from NaN or inf to something finite is always an improvement
            return True
        elif self.mode == "min" and metric < self.best_metric:
            return True
        elif self.mode == "max" and metric > self.best_metric:
            return True
        else:
            return False

    def _copy_metrics(self, metrics: dict):
        """Copy numeric values from `metrics` as python scalars."""

        copy = {}
        for key, value in metrics.items():
            if not isinstance(value, (numbers.Number, np.ndarray, torch.Tensor)):
                continue
            if isinstance(value, (np.ndarray, torch.Tensor)):
                value = value.tolist()
            copy[key] = value

        return copy

    def _update_summaries(self):
        # wandb is supposed not to update the summaries anymore once we set them manually,
        # but they are still getting updated, so we make sure to set them after logging
        if self.best_metrics is not None:
            wandb.run.summary.update({"best_epoch": self.best_epoch, **self.best_metrics})
