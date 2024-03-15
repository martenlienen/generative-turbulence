#!/usr/bin/env python

# SPDX-FileCopyrightText: © 2024 Marten Lienen <m.lienen@tum.de> & Technical University of Munich
#
# SPDX-License-Identifier: MIT

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf
from pytorch_lightning.utilities import move_data_to_device
from tqdm import tqdm

from turbdiff.config import instantiate_data_and_task
from turbdiff.data.ofles import Variable as V
from turbdiff.models.metrics import SampleStore
from turbdiff.utils.wandb import load_checkpoint, load_config, wandb_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", default="cuda")
    parser.add_argument(
        "-f", "--first", type=int, default=199, help="How many steps to unroll"
    )
    parser.add_argument("-n", "--samples", type=int, default=16, help="Number of samples")
    parser.add_argument("-b", "--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("path", help="W&B run path")
    parser.add_argument("samples_dir", help="Directory for sample .h5 files")
    parser.add_argument("data_dir", help="Data directory")
    args = parser.parse_args()

    device = torch.device(args.device)
    first = args.first
    n_samples = args.samples
    batch_size = args.batch_size
    run_path = args.path
    samples_dir = Path(args.samples_dir)
    data_dir = Path(args.data_dir)

    assert n_samples % batch_size == 0
    assert data_dir.is_dir()

    run = wandb_run(run_path)

    samples_path = samples_dir / f"{run.id}.h5"

    config = OmegaConf.create(load_config(run))
    datamodule, task = instantiate_data_and_task(config)

    checkpoint = load_checkpoint(run, "checkpoints/best.ckpt")
    task.load_state_dict(checkpoint["state_dict"])

    task = task.to(device)

    datamodule.setup("test")
    dataset = datamodule.test_dataset

    sample_store = SampleStore(samples_path, task.variables)
    if samples_path.exists():
        print("Samples already exist, moving directly to evaluation")
    else:
        task.eval()
        with torch.no_grad():
            sample_steps = [first]
            for sample_idxs in tqdm(dataset.sample_idxs_by_file(), desc="Cases"):
                for _ in tqdm(range(n_samples // batch_size), desc="Batches", position=1):
                    # Start from the first sample for each case
                    batch = dataset[[sample_idxs[0]] * batch_size]
                    batch.data.t = batch.data.t[:, : task.context_window]
                    batch.data.samples = {
                        v: sample[:, : task.context_window]
                        for v, sample in batch.data.samples.items()
                    }
                    batch = move_data_to_device(batch, device)

                    # Add small amounts of noise to the initial velocity field
                    u = batch.data.samples[V.U]
                    batch.data.samples[V.U] = u + 0.01 * torch.randn_like(u)

                    x_hat = task.unroll_samples(batch, sample_steps, block_size=25)

                    sample_store.add_samples(x_hat[:, 0], batch.data.metadata)

    metrics = task._sample_metrics("test/initial", data_dir).to(device)
    stats = move_data_to_device(datamodule.stats, device)
    log_metrics = metrics.compute(sample_store, stats, device)
    log_metrics = {key: float(value.item()) for key, value in log_metrics.items()}
    (samples_path.parent / f"{run.id}-metrics.json").write_text(json.dumps(log_metrics))
    for key in sorted(log_metrics.keys()):
        value = log_metrics[key]
        print(f"{key}: {value}")
        run.summary[key] = value
    run.update()


if __name__ == "__main__":
    main()
