# SPDX-FileCopyrightText: © 2024 Marten Lienen <m.lienen@tum.de> & Technical University of Munich
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn

from ..data.ofles import OpenFOAMBatch
from ..data.ofles import Variable as V
from ..data.ofles import split_channels
from .cell_type_embeddings import CellTypeEmbedding
from .conditioning import Conditioning
from .metrics import (
    MaxMeanTKEPositionMetric,
    SampleMetricsCollection,
    SampleStore,
    WassersteinMetric,
    WassersteinTKE,
)
from .normalization import Normalization
from .utils import select_cells


class RegressionTraining(pl.LightningModule):
    def __init__(
        self,
        data_dir: Path,
        samples_root: Path,
        *,
        variables: tuple[V, ...],
        context_window: int,
        unroll_steps: int,
        eval_unroll_steps: int,
        sample_steps: list[int],
        main_sample_step: int,
        normalization_mode: str,
        cell_type_features: bool,
        cell_type_embedding_type: str,
        cell_type_embedding_dim: int,
        cell_pos_features: bool,
        compute_expensive_sample_metrics: bool,
    ):
        super().__init__()

        assert V.U in variables
        self.variables = variables
        self.context_window = context_window
        self.unroll_steps = unroll_steps
        self.eval_unroll_steps = eval_unroll_steps

        self.cell_type_features = cell_type_features
        self.cell_type_embedding_type = cell_type_embedding_type
        self.cell_type_embedding_dim = cell_type_embedding_dim
        if self.cell_type_features:
            self.cell_type_embedding = CellTypeEmbedding.create(
                cell_type_embedding_type, cell_type_embedding_dim
            )
        else:
            self.cell_type_embedding = None
        self.cell_pos_features = cell_pos_features
        self.conditioning = Conditioning(
            self.variables, self.cell_type_embedding, self.cell_pos_features
        )

        self.normalization_mode = normalization_mode
        self.normalization = Normalization(variables, normalization_mode)

        self.sample_steps = sample_steps
        self.compute_expensive_sample_metrics = compute_expensive_sample_metrics
        self.main_sample_step = main_sample_step
        self.val_sample_metrics = nn.ModuleList(
            [self._sample_metrics(f"val/{s}", data_dir) for s in self.sample_steps]
        )
        self.test_sample_metrics = nn.ModuleList(
            [self._sample_metrics(f"test/{s}", data_dir) for s in self.sample_steps]
        )
        self.val_sample_stores = []
        self.test_sample_stores = []
        for s in self.sample_steps:
            self.val_sample_stores.append(self._sample_store(f"val-{s}", samples_root))
            self.test_sample_stores.append(self._sample_store(f"test-{s}", samples_root))

        if len(self.sample_steps) > 0:
            assert self.eval_unroll_steps >= max(self.sample_steps)

        self.model = None
        self.loss = nn.MSELoss()

        self.stats = None

    def _sample_store(self, phase: str, samples_root: Path):
        return SampleStore(samples_root / f"{phase}-samples.h5", self.variables)

    def _sample_metrics(self, phase: str, data_dir: Path):
        metrics = [WassersteinTKE(), WassersteinMetric(), MaxMeanTKEPositionMetric()]
        return SampleMetricsCollection(phase, data_dir, metrics)

    def unroll_samples(
        self, batch: OpenFOAMBatch, sample_steps: list[int], block_size: int
    ):
        assert block_size >= self.context_window
        x_context, C = self._model_input(batch)
        # Unroll in blocks to save memory for smaller GPUs
        x_sample = []
        for i in range(0, max(sample_steps) + 1, block_size):
            x_hat = self._predict_x(
                x_context, C, batch.data.inside_mask, unroll_steps=block_size
            )
            x_context = x_hat[:, -self.context_window :]
            idxs = [j - i for j in sample_steps if i <= j < i + block_size]
            x_sample.append(x_hat[:, idxs])
        x_sample = torch.cat(x_sample, dim=1)
        x_sample = self.normalization.denormalize_grid(x_sample, batch.stats)

        return x_sample

    def _unroll_predict(self, batch: OpenFOAMBatch):
        x, C = self._model_input(batch)
        x_context, x_target = self._split_x(x)
        n_steps = x_target.shape[1]
        x_hat = self._predict_x(
            x_context, C, batch.data.inside_mask, unroll_steps=n_steps
        )

        return x_hat, x_target

    def training_step(self, batch: OpenFOAMBatch, batch_idx):
        x_hat, x_target = self._unroll_predict(batch)
        loss = self.loss(x_hat, x_target)

        self.log("train/loss", loss, batch_size=batch.data.n_samples, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch: OpenFOAMBatch, batch_idx):
        if self.stats is None:
            self.stats = batch.stats

        x_hat, x_target = self._unroll_predict(batch)
        x_sample = self.normalization.denormalize_grid(x_hat, batch.stats)
        x_target_denorm = self.normalization.denormalize_grid(x_target, batch.stats)

        for s, store in zip(self.sample_steps, self.val_sample_stores):
            store.add_samples(x_sample[:, s - 1], batch.data.metadata)

        loss = self.loss(x_hat[:, : self.unroll_steps], x_target[:, : self.unroll_steps])
        self.log("val/loss", loss, batch_size=batch.data.n_samples, prog_bar=True)
        self._log_unroll_metrics(x_sample, x_target_denorm, batch, phase="val")

        return {"loss": loss}

    def on_validation_epoch_start(self):
        for store in self.val_sample_stores:
            store.reset()

    def on_validation_epoch_end(self):
        final_validation = self.trainer.should_stop or (
            self.trainer.max_epochs is not None
            and self.current_epoch == self.trainer.max_epochs - 1
        )

        metrics = {}
        for s, sample_metrics, store in zip(
            self.sample_steps, self.val_sample_metrics, self.val_sample_stores
        ):
            step_metrics = sample_metrics.compute(
                store,
                self.stats,
                self.device,
                expensive_metrics=(
                    self.compute_expensive_sample_metrics and final_validation
                ),
            )
            metrics.update(step_metrics)

            # Log the main step samples as global metrics
            if s == self.main_sample_step:
                sample_metrics = {
                    "/".join([(parts := key.split("/"))[0], *parts[2:]]): value
                    for key, value in step_metrics.items()
                }
                metrics.update(sample_metrics)

        self.log_dict(metrics)

    def test_step(self, batch: OpenFOAMBatch, batch_idx):
        if self.stats is None:
            self.stats = batch.stats

        x_hat, x_target = self._unroll_predict(batch)
        x_sample = self.normalization.denormalize_grid(x_hat, batch.stats)
        x_target_denorm = self.normalization.denormalize_grid(x_target, batch.stats)

        for s, store in zip(self.sample_steps, self.test_sample_stores):
            store.add_samples(x_sample[:, s - 1], batch.data.metadata)

        loss = self.loss(x_hat[:, : self.unroll_steps], x_target[:, : self.unroll_steps])
        self.log("test/loss", loss, batch_size=batch.data.n_samples, prog_bar=True)
        self._log_unroll_metrics(x_sample, x_target_denorm, batch, phase="test")

        return {"loss": loss}

    def on_test_epoch_start(self):
        for store in self.test_sample_stores:
            store.reset()

    def on_test_epoch_end(self):
        metrics = {}
        for s, sample_metrics, store in zip(
            self.sample_steps, self.test_sample_metrics, self.test_sample_stores
        ):
            step_metrics = sample_metrics.compute(
                store,
                self.stats,
                self.device,
                expensive_metrics=self.compute_expensive_sample_metrics,
            )
            metrics.update(step_metrics)

            # Log the main step samples as global metrics
            if s == self.main_sample_step:
                sample_metrics = {
                    "/".join([(parts := key.split("/"))[0], *parts[2:]]): value
                    for key, value in step_metrics.items()
                }
                metrics.update(sample_metrics)

        self.log_dict(metrics)

    def _split_x(self, x: torch.Tensor):
        return x[:, : self.context_window], x[:, self.context_window :]

    def _predict_x(self, x_context, C, inside_mask, *, unroll_steps: int):
        x_hat = []
        for i in range(unroll_steps):
            # Forecast one step but keep values on the boundaries constant
            x_hat_i = torch.where(inside_mask, self.model(x_context, C), x_context[:, -1])

            # Keep to construct complete forecast in the end
            x_hat.append(x_hat_i)

            # Move context one step forward
            if x_context.shape[1] == 1:
                x_context = x_hat_i
            else:
                x_context = torch.cat((x_context[:, 1:], x_hat_i.unsqueeze(dim=1)), dim=1)

        return torch.stack(x_hat, dim=1)

    def _model_input(self, batch: OpenFOAMBatch):
        x = batch.data.grid_embedding(self.variables)
        x = self.normalization.normalize_grid(x, batch.stats)
        C = self.conditioning(batch.data)

        return x, C

    def _log_unroll_metrics(
        self, x_sample, x_target, batch: OpenFOAMBatch, *, phase: str
    ):
        x_sample_cells = select_cells(x_sample, batch.data.cell_idx)
        x_target_cells = select_cells(x_target, batch.data.cell_idx)

        x_sample_cells_v = split_channels(x_sample_cells, self.variables, dim=-2)
        x_target_cells_v = split_channels(x_target_cells, self.variables, dim=-2)

        unroll_metrics = {}
        for v in self.variables:
            mse = (
                ((x_sample_cells_v[v] - x_target_cells_v[v]) ** 2)
                .sum(dim=-2)
                .mean(dim=-1)
            )
            v_metrics = {
                f"{phase}/unroll/mse-{v.name.lower()}-{i + 1}": mse[:, i].mean()
                for i in range(mse.shape[1])
            }
            unroll_metrics.update(v_metrics)
        self.log_dict(unroll_metrics, batch_size=batch.data.n_samples)
