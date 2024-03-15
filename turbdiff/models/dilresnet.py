# SPDX-FileCopyrightText: © 2024 Marten Lienen <m.lienen@tum.de> & Technical University of Munich
#
# SPDX-License-Identifier: MIT

import itertools as it
import math
from pathlib import Path

import einops as eo
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.ofles import OpenFOAMData
from ..data.ofles import Variable as V
from .conditioning import Conditioning, global_conditioning, local_conditioning
from .regression import RegressionTraining
from .utils import ravel_cells


class DilatedCNNBlock(nn.Module):
    def __init__(self, dim: int, dilations: list[int]):
        super().__init__()

        self.dim = dim
        self.dilations = dilations
        self.layers = nn.ModuleList(
            [
                nn.Conv3d(
                    dim,
                    dim,
                    kernel_size=3,
                    dilation=d,
                    padding=d,
                    padding_mode="replicate",
                )
                for d in it.chain(dilations, reversed(dilations[:-1]))
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x), inplace=True)
        return x


class DilResNet(nn.Module):
    def __init__(
        self,
        n_features: int,
        c_local_features: int,
        c_global_features: int,
        N: int = 4,
        hidden_dim: int = 48,
    ):
        super().__init__()

        self.n_features = n_features
        self.c_local_features = c_local_features
        self.c_global_features = c_global_features
        self.N = N
        self.hidden_dim = hidden_dim
        self.dilations = [1, 2, 4, 8]

        self.encode = nn.Conv3d(
            n_features, hidden_dim, kernel_size=3, padding=1, padding_mode="replicate"
        )
        self.encode_c_local = nn.Conv3d(
            c_local_features,
            hidden_dim,
            kernel_size=3,
            padding=1,
            padding_mode="replicate",
        )
        self.blocks = nn.ModuleList(
            [DilatedCNNBlock(hidden_dim, self.dilations) for _ in range(N)]
        )
        self.decode = nn.Conv3d(
            hidden_dim, n_features, kernel_size=3, padding=1, padding_mode="replicate"
        )

    def forward(self, x, C: dict[Conditioning.Type, torch.Tensor]):
        c_local = local_conditioning(C)
        if c_local is not None:
            c_local = self.encode_c_local(c_local)
        c_global = global_conditioning(C)
        if c_global is not None:
            raise RuntimeError("Global conditioning not implemented in DilResNet")
        x = self.encode(x)
        for block in self.blocks:
            if c_local is not None:
                x = x + c_local
            x = x + block(x)
        return self.decode(x)


class DilResNetTraining(RegressionTraining):
    def __init__(
        self,
        data_dir: Path,
        samples_root: Path,
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
        learning_rate: float,
        min_learning_rate: float,
        max_train_steps: int,
        N: int = 4,
        hidden_dim: int = 48,
        training_noise_std: float | None = None,
        compute_expensive_sample_metrics: bool = True,
    ):
        super().__init__(
            data_dir=data_dir,
            samples_root=samples_root,
            variables=variables,
            context_window=context_window,
            unroll_steps=unroll_steps,
            eval_unroll_steps=eval_unroll_steps,
            sample_steps=sample_steps,
            main_sample_step=main_sample_step,
            normalization_mode=normalization_mode,
            cell_type_features=cell_type_features,
            cell_type_embedding_type=cell_type_embedding_type,
            cell_type_embedding_dim=cell_type_embedding_dim,
            cell_pos_features=cell_pos_features,
            compute_expensive_sample_metrics=compute_expensive_sample_metrics,
        )

        assert unroll_steps == 1, "DilResNet training only uses unroll_steps=1"

        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.max_train_steps = max_train_steps
        self.training_noise_std = training_noise_std

        self.model = DilResNet(
            n_features=sum([v.dims for v in variables]),
            c_local_features=self.conditioning.local_conditioning_dim,
            c_global_features=self.conditioning.global_conditioning_dim,
            N=N,
            hidden_dim=hidden_dim,
        )

        n_features = sum([v.dims for v in variables])
        self.register_buffer("dx_mean", torch.zeros(n_features))
        self.register_buffer("dx_var", torch.ones(n_features))
        self.register_buffer("n_train_batches_tracked", torch.tensor(0, dtype=torch.long))

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # Decay exponentially to the min learning rate over the course of the timesteps
        def decay(step):
            decay_step = (
                math.log(self.min_learning_rate / self.learning_rate)
                / self.max_train_steps
            )
            return math.exp(decay_step * min(step, self.max_train_steps))

        lr_scheduler_config = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(opt, decay),
            "interval": "step",
        }
        return {"optimizer": opt, "lr_scheduler": lr_scheduler_config}

    def training_step(self, batch: OpenFOAMData, batch_idx):
        x, C = self._model_input(batch)

        x0 = x[:, self.context_window - 1]
        if self.training_noise_std is not None:
            x0 = x0 + self.training_noise_std * torch.randn_like(x0)

        x1 = x[:, self.context_window]
        dx = x1 - x0

        dx_hat_normed = self.model(x0, C)

        loss = self.loss(
            ravel_cells(dx_hat_normed)[..., batch.data.cell_idx],
            F.batch_norm(
                ravel_cells(dx)[..., batch.data.cell_idx],
                self.dx_mean,
                self.dx_var,
                training=self.training and (self.n_train_batches_tracked).item() < 1000,
            ),
        )

        if self.training:
            self.n_train_batches_tracked.add_(1)

        self.log("train/loss", loss, batch_size=batch.data.n_samples, prog_bar=True)
        return {"loss": loss}

    def _predict_x(self, x_context, C, inside_mask, *, unroll_steps: int):
        dx_mean = eo.rearrange(self.dx_mean, "f -> f 1 1 1")
        dx_std = eo.rearrange(self.dx_var.sqrt(), "f -> f 1 1 1")

        x_next = x_context[:, -1]
        x_hat = []
        for i in range(unroll_steps):
            dx = torch.addcmul(dx_mean, dx_std, self.model(x_next, C))
            x_next = torch.where(inside_mask, x_next + dx, x_next)
            x_hat.append(x_next)
        return torch.stack(x_hat, dim=1)
