#!/usr/bin/env python

# SPDX-FileCopyrightText: © 2024 Marten Lienen <m.lienen@tum.de> & Technical University of Munich
#
# SPDX-License-Identifier: MIT

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf
from pytorch_lightning.utilities import move_data_to_device
from tqdm import tqdm

from turbdiff.config import instantiate_data_and_task
from turbdiff.models.metrics import SampleStore
from turbdiff.utils.seed import manual_seed
from turbdiff.utils.wandb import load_checkpoint, load_config, wandb_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", default="cuda")
    parser.add_argument("-s", "--seed", type=int, default=817598073042842)
    parser.add_argument("precision", help="Matmul precision")
    parser.add_argument("path", help="W&B run path")
    parser.add_argument("samples_path", help=".h5 file for storing samples")
    parser.add_argument("data_dir", help="Data directory")
    args = parser.parse_args()

    device = torch.device(args.device)
    seed = args.seed
    precision = args.precision
    run_path = args.path
    samples_path = Path(args.samples_path)
    data_dir = Path(args.data_dir)

    assert samples_path.suffix == ".h5"
    assert not samples_path.exists()
    assert data_dir.is_dir()

    manual_seed(seed)
    torch.set_float32_matmul_precision(precision)

    run = wandb_run(run_path)

    config = OmegaConf.create(load_config(run))
    datamodule, task = instantiate_data_and_task(config)

    checkpoint = load_checkpoint(run, "checkpoints/best.ckpt")
    task.load_state_dict(checkpoint["state_dict"])

    task = task.to(device)

    datamodule.setup("validate")

    sample_store = SampleStore(samples_path, task.variables)
    task.eval()
    with torch.no_grad():
        for batch in tqdm(datamodule.val_dataloader(), desc="Cases"):
            batch = move_data_to_device(batch, device)

            x_sample = task.sample(batch)

            sample_store.add_samples(x_sample, batch.data.metadata)

    metrics = task._sample_metrics("val", data_dir).to(device)
    stats = move_data_to_device(datamodule.stats, device)
    log_metrics = metrics.compute(sample_store, stats, device, expensive_metrics=False)
    log_metrics = {key: float(value.item()) for key, value in log_metrics.items()}
    for key in sorted(log_metrics.keys()):
        value = log_metrics[key]
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
