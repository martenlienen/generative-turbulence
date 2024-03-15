# SPDX-FileCopyrightText: © 2024 Marten Lienen <m.lienen@tum.de> & Technical University of Munich
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import wandb
from omegaconf import DictConfig

from .data.ofles import OpenFOAMDataModule, Variable
from .data.ofles_seq import OpenFOAMSequenceDataModule
from .models.diffusion import DiffusionTraining
from .models.dilresnet import DilResNetTraining
from .models.tfnet import TFNetTraining


def instantiate_datamodule(config: DictConfig):
    if config.variables is None:
        variables = tuple(Variable)
    else:
        variables = tuple(Variable.from_str(var) for var in config.variables.split(","))

    if config.name in ("ofles", "shapes"):
        return OpenFOAMDataModule(
            Path(config.root),
            discard_first_seconds=config.discard_first_seconds,
            num_workers=config.num_workers,
            batch_size=config.batch_size,
            eval_batch_size=config.eval_batch_size,
            val_samples=config.val_samples,
            test_samples=config.test_samples,
            pin_memory=config.pin_memory,
            variables=variables,
        )
    elif config.name in ("shapes-seq", "regression"):
        return OpenFOAMSequenceDataModule(
            Path(config.root),
            discard_first_seconds=config.discard_first_seconds,
            num_workers=config.num_workers,
            batch_size=config.batch_size,
            seq_len=config.context_window + config.unroll_steps,
            eval_batch_size=config.eval_batch_size,
            eval_seq_len=config.context_window + config.eval_unroll_steps,
            val_samples=config.val_samples,
            test_samples=config.test_samples,
            pin_memory=config.pin_memory,
            variables=variables,
            stride=config.stride,
        )
    else:
        raise RuntimeError(f"Unknown datamodule {config.name}")


def instantiate_data_and_task(config: DictConfig):
    datamodule = instantiate_datamodule(config.data)

    if wandb.run is None:
        samples_root = Path(config.samples_root) / "explore" / "debug"
    else:
        samples_root = (
            Path(config.samples_root) / (wandb.run.group or "explore") / wandb.run.id
        )

    if config.model.name.startswith("diffusion"):
        if config.model.variables is None:
            variables = tuple(Variable)
        else:
            variables = tuple(
                Variable.from_str(var) for var in config.model.variables.split(",")
            )

        datamodule.setup("fit")
        max_train_steps = config.model.max_epochs * len(datamodule.train_dataloader())
        task = DiffusionTraining(
            data_dir=Path(config.data.root) / "data",
            samples_root=samples_root,
            dim=config.model.dim,
            cell_type_embedding_type=config.model.cell_type_embedding_type,
            cell_type_embedding_dim=config.model.cell_type_embedding_dim,
            normalization_mode=config.model.normalization_mode,
            variables=variables,
            beta_schedule=config.model.beta_schedule,
            timesteps=config.model.timesteps,
            learning_rate=config.model.learning_rate,
            min_learning_rate=config.model.min_learning_rate,
            lr_decay=config.model.lr_decay,
            max_train_steps=max_train_steps,
            loss=config.model.loss,
            cell_type_features=config.model.cell_type_features,
            cell_pos_features=config.model.get("cell_pos_features", False),
            clip_denoised=config.model.clip_denoised,
            noise_bcs=config.model.noise_bcs,
            learned_variances=config.model.learned_variances,
            elbo_weight=config.model.elbo_weight,
            detach_elbo_mean=config.model.detach_elbo_mean,
            time_embedding=config.model.time_embedding,
            actfn=config.model.actfn,
            optimizer=config.model.optimizer,
            norm_type=config.model.norm_type,
            with_geometry_embedding=config.model.with_geometry_embedding,
        )
    elif config.model.name == "tfnet":
        if config.model.variables is None:
            variables = tuple(Variable)
        else:
            variables = tuple(
                Variable.from_str(var) for var in config.model.variables.split(",")
            )

        task = TFNetTraining(
            data_dir=Path(config.data.root) / "data",
            samples_root=samples_root,
            normalization_mode=config.model.normalization_mode,
            variables=variables,
            cell_type_features=config.model.cell_type_features,
            cell_type_embedding_type=config.model.cell_type_embedding_type,
            cell_type_embedding_dim=config.model.cell_type_embedding_dim,
            cell_pos_features=config.model.get("cell_pos_features", False),
            temporal_filtering_length=config.model.temporal_filtering_length,
            context_window=config.model.context_window,
            unroll_steps=config.model.unroll_steps,
            eval_unroll_steps=config.model.eval_unroll_steps,
            sample_steps=config.model.sample_steps,
            main_sample_step=config.model.main_sample_step,
            learning_rate=config.model.learning_rate,
            dropout_rate=config.model.dropout_rate,
            kernel_size=config.model.kernel_size,
            compute_expensive_sample_metrics=config.model.compute_expensive_sample_metrics,
        )
    elif config.model.name == "dilresnet":
        if config.model.variables is None:
            variables = tuple(Variable)
        else:
            variables = tuple(
                Variable.from_str(var) for var in config.model.variables.split(",")
            )

        datamodule.setup("fit")
        max_train_steps = config.model.max_epochs * len(datamodule.train_dataloader())
        task = DilResNetTraining(
            data_dir=Path(config.data.root) / "data",
            samples_root=samples_root,
            variables=variables,
            context_window=config.model.context_window,
            unroll_steps=config.model.unroll_steps,
            eval_unroll_steps=config.model.eval_unroll_steps,
            sample_steps=config.model.sample_steps,
            main_sample_step=config.model.main_sample_step,
            normalization_mode=config.model.normalization_mode,
            cell_type_features=config.model.cell_type_features,
            cell_type_embedding_type=config.model.cell_type_embedding_type,
            cell_type_embedding_dim=config.model.cell_type_embedding_dim,
            cell_pos_features=config.model.get("cell_pos_features", False),
            learning_rate=config.model.learning_rate,
            min_learning_rate=config.model.min_learning_rate,
            max_train_steps=max_train_steps,
            N=config.model.N,
            hidden_dim=config.model.hidden_dim,
            training_noise_std=config.model.training_noise_std,
            compute_expensive_sample_metrics=config.model.compute_expensive_sample_metrics,
        )
    else:
        raise Exception(f"Unknown model {config.model.name}")

    return datamodule, task
