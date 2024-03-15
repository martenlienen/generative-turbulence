#!/usr/bin/env python

# SPDX-FileCopyrightText: © 2024 Marten Lienen <m.lienen@tum.de> & Technical University of Munich
#
# SPDX-License-Identifier: MIT

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from pytorch_lightning.utilities import move_data_to_device
from tqdm import tqdm

from turbdiff.config import instantiate_data_and_task
from turbdiff.data.ofles import Variable as V
from turbdiff.models.diffusion import DiffusionTraining
from turbdiff.models.regression import RegressionTraining
from turbdiff.utils.wandb import load_checkpoint, load_config, wandb_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode")
    parser.add_argument("-d", "--device", default="cuda")
    parser.add_argument("path", help="W&B run path")
    parser.add_argument("times_dir", help="Times directory")
    args = parser.parse_args()

    mode = args.mode
    device = torch.device(args.device)
    run_path = args.path
    times_dir = Path(args.times_dir)

    assert times_dir.is_dir()

    run = wandb_run(run_path)

    config = OmegaConf.create(load_config(run))
    datamodule, task = instantiate_data_and_task(config)

    checkpoint = load_checkpoint(run, "checkpoints/best.ckpt")
    task.load_state_dict(checkpoint["state_dict"])

    task = task.to(device)

    datamodule.setup("test")
    dataset = datamodule.test_dataset

    torch.set_float32_matmul_precision("medium")

    times = []
    task.eval()
    with torch.no_grad():
        bar = tqdm(dataset.sample_idxs_by_file(), desc="Cases")
        for sample_idxs in bar:
            # Start from the first sample for each case
            batch = dataset[[sample_idxs[0]]]
            batch = move_data_to_device(batch, device)

            # Measure CUDA runtime
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            if isinstance(task, DiffusionTraining):
                samples = task.sample(batch)
            elif isinstance(task, RegressionTraining):
                batch.data.t = batch.data.t[:, : task.context_window]
                batch.data.samples = {
                    v: sample[:, : task.context_window]
                    for v, sample in batch.data.samples.items()
                }
                # Add small amounts of noise to the initial velocity field
                u = batch.data.samples[V.U]
                batch.data.samples[V.U] = u + 0.01 * torch.randn_like(u)
                if mode == "init":
                    samples = task.unroll_samples(batch, [199], block_size=25)
                else:
                    samples = task.unroll_samples(batch, [21], block_size=25)
            torch.cuda.synchronize()
            end = time.perf_counter_ns()
            diff = end - start
            times.append(diff)
            bar.set_postfix({"time": diff / 10**9})
    times = np.array(times)
    # Convert to seconds
    times = times.astype(float) / 10**9

    file_name = f"{run.id}.txt" if mode is None else f"{run.id}-init.txt"
    np.savetxt(times_dir / file_name, times)
    print(times)
    print(times.min())
    key = "sample_time" if mode is None else "sample_time_init"
    run.summary[key] = times.min()
    run.update()


if __name__ == "__main__":
    main()
