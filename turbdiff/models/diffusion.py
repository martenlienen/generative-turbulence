# SPDX-FileCopyrightText: © 2024 Marten Lienen <m.lienen@tum.de> & Technical University of Munich
#
# SPDX-License-Identifier: MIT

import math
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn

from ..data.ofles import OpenFOAMBatch
from ..data.ofles import Variable as V
from ..utils import get_logger
from .cell_type_embeddings import CellTypeEmbedding
from .conditioning import Conditioning
from .ddpm import DenoisingModel, GaussianDiffusion
from .metrics import (
    MaxMeanTKEPositionMetric,
    SampleMetricsCollection,
    SampleStore,
    WassersteinMetric,
    WassersteinTKE,
)
from .normalization import Normalization

log = get_logger()


def actfn_from_str(name: str):
    actfns = {
        "silu": nn.SiLU,
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "softplus": nn.Softplus,
        "tanh": nn.Tanh,
    }
    return actfns[name]


class DiffusionTraining(pl.LightningModule):
    def __init__(
        self,
        data_dir: Path,
        samples_root: Path,
        dim: int = 32,
        cell_type_embedding_type: str = "learned",
        cell_type_embedding_dim: int = 4,
        normalization_mode: str = "mean-std",
        variables: tuple[V, ...] = tuple(V),
        beta_schedule: str = "sigmoid",
        timesteps: int = 100,
        learning_rate: float = 1e-3,
        min_learning_rate: float = 1e-3,
        lr_decay: float | None = None,
        max_train_steps: int = 1_000,
        loss: str = "l1",
        cell_type_features: bool = True,
        cell_pos_features: bool = False,
        clip_denoised: bool = False,
        noise_bcs: bool = False,
        learned_variances: bool = False,
        elbo_weight: float | None = None,
        detach_elbo_mean: bool = True,
        time_embedding: str = "nyquist",
        actfn: str = "silu",
        optimizer: str = "adam",
        norm_type: str = "instance",
        with_geometry_embedding: bool = True,
    ):
        super().__init__()

        self.variables = variables

        self.normalization_mode = normalization_mode
        self.normalization = Normalization(self.variables, normalization_mode)

        self.dim = dim
        self.cell_type_embedding_type = cell_type_embedding_type
        self.cell_type_embedding_dim = cell_type_embedding_dim
        self.cell_type_features = cell_type_features
        if self.cell_type_features:
            self.cell_type_embedding = CellTypeEmbedding.create(
                cell_type_embedding_type, cell_type_embedding_dim
            )
        else:
            self.cell_type_embedding = None
        self.cell_pos_features = cell_pos_features
        self.conditioning = Conditioning(
            self.variables,
            self.cell_type_embedding,
            self.cell_pos_features,
        )

        self.beta_schedule = beta_schedule
        self.timesteps = timesteps
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.lr_decay = lr_decay
        self.max_train_steps = max_train_steps
        self.loss = loss
        self.clip_denoised = clip_denoised
        self.noise_bcs = noise_bcs
        self.learned_variances = learned_variances
        self.elbo_weight = elbo_weight
        self.detach_elbo_mean = detach_elbo_mean
        self.actfn = actfn
        self.optimizer = optimizer

        vars_dim = sum(v.dims for v in self.variables)

        model = DenoisingModel(
            in_features=vars_dim,
            out_features=vars_dim * (2 if learned_variances else 1),
            c_local_features=self.conditioning.local_conditioning_dim,
            c_global_features=self.conditioning.global_conditioning_dim,
            timesteps=timesteps,
            dim=dim,
            # TODO: Decrease to 3?
            u_net_levels=4,
            actfn=actfn_from_str(actfn),
            norm_type=norm_type,
            with_geometry_embedding=with_geometry_embedding,
        )
        self.model = GaussianDiffusion(
            model,
            timesteps=timesteps,
            beta_schedule=beta_schedule,
            loss_type=loss,
            clip_denoised=clip_denoised,
            noise_bcs=noise_bcs,
            learned_variances=learned_variances,
            elbo_weight=elbo_weight,
            detach_elbo_mean=detach_elbo_mean,
        )

        assert V.U in self.variables
        self.val_sample_store = self._sample_store("val", samples_root)
        self.val_sample_metrics = self._sample_metrics("val", data_dir)
        self.test_sample_store = self._sample_store("test", samples_root)
        self.test_sample_metrics = self._sample_metrics("test", data_dir)

        self.stats = None

    def _sample_store(self, phase: str, samples_root: Path):
        return SampleStore(samples_root / f"{phase}-samples.h5", self.variables)

    def _sample_metrics(self, phase: str, data_dir: Path):
        metrics = [WassersteinTKE(), WassersteinMetric(), MaxMeanTKEPositionMetric()]
        return SampleMetricsCollection(phase, data_dir, metrics)

    def sample(self, batch: OpenFOAMBatch, start_from=None):
        x, C = self._model_input(batch)
        x = self.model.p_sample_loop(
            x, C, batch.data.cell_idx, pbar=True, start_from=start_from
        )
        x = self.normalization.denormalize_grid(x, batch.stats)
        return x

    def training_step(self, batch: OpenFOAMBatch, batch_idx):
        x, C = self._model_input(batch)
        loss, t = self.model(x, C, batch.data.metadata, self.variables)

        self.log("train/loss", loss, batch_size=batch.data.n_samples, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch: OpenFOAMBatch, batch_idx):
        if self.stats is None:
            self.stats = batch.stats

        x = self.sample(batch)
        self.val_sample_store.add_samples(x, batch.data.metadata)

        return {}

    def on_validation_epoch_start(self):
        self.val_sample_store.reset()

    def on_validation_epoch_end(self):
        final_validation = self.trainer.should_stop or (
            self.trainer.max_epochs is not None
            and self.current_epoch == self.trainer.max_epochs - 1
        )
        metrics = self.val_sample_metrics.compute(
            self.val_sample_store,
            self.stats,
            self.device,
            expensive_metrics=final_validation,
        )
        self.log_dict(metrics)

    def test_step(self, batch: OpenFOAMBatch, batch_idx):
        if self.stats is None:
            self.stats = batch.stats

        x = self.sample(batch)
        self.test_sample_store.add_samples(x, batch.data.metadata)

        return {}

    def on_test_epoch_start(self):
        self.test_sample_store.reset()

    def on_test_epoch_end(self):
        metrics = self.test_sample_metrics.compute(
            self.test_sample_store, self.stats, self.device, expensive_metrics=True
        )
        self.log_dict(metrics)

    def configure_optimizers(self):
        if self.optimizer == "adam":
            opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == "adamw":
            opt = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == "radam":
            opt = torch.optim.RAdam(self.parameters(), lr=self.learning_rate)
        else:
            raise RuntimeError(f"Unknown optimizer {self.optimizer}")

        # Decay exponentially to the min learning rate over the course of the timesteps
        def decay(step):
            decay_step = (
                math.log(self.min_learning_rate / self.learning_rate)
                / self.max_train_steps
            )
            return math.exp(decay_step * min(step, self.max_train_steps))

        config = {"optimizer": opt}
        if self.lr_decay == "exp":
            config["lr_scheduler"] = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(opt, decay),
                "interval": "step",
            }

        return config

    def _model_input(self, batch: OpenFOAMBatch):
        x = batch.data.grid_embedding(self.variables)
        x = self.normalization.normalize_grid(x, batch.stats)
        C = self.conditioning(batch.data)

        return x, C
