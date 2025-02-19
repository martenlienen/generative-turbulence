#!/usr/bin/env python

# SPDX-FileCopyrightText: © 2024 Marten Lienen <m.lienen@tum.de> & Technical University of Munich
#
# SPDX-License-Identifier: MIT

import argparse
import logging
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from pytorch_lightning.utilities import move_data_to_device
from tqdm import tqdm

from turbdiff.config import instantiate_data_and_task
from turbdiff.models.metrics import SampleStore
from turbdiff.utils.logging import print_config
from turbdiff.utils.seed import manual_seed

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint with overrides")
    parser.add_argument("-d", "--device", default="cuda")
    parser.add_argument("-s", "--seed", default=2883413570083077179, type=int)
    parser.add_argument("ckpt", help="Path to .ckpt file")
    parser.add_argument("samples_path", help=".h5 file for storing samples")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    device = torch.device(args.device)
    seed = args.seed
    ckpt_path = args.ckpt
    samples_path = Path(args.samples_path)
    overrides = args.overrides

    assert samples_path.suffix == ".h5"
    assert not samples_path.exists()

    ckpt = torch.load(ckpt_path) if ckpt_path is not None else {}
    if "config" in ckpt:
        log.info("Load config from checkpoint")
        run_config = OmegaConf.create(ckpt["config"])
    else:
        log.error("Checkpoint has no config")
        sys.exit(1)

    config = OmegaConf.merge(run_config, OmegaConf.from_cli(overrides))

    print_config(config)

    manual_seed(seed)
    torch.set_float32_matmul_precision(config.matmul_precision)

    datamodule, task = instantiate_data_and_task(config)
    task.load_state_dict(ckpt["state_dict"])
    task = task.to(device)

    datamodule.setup("validate")

    sample_store = SampleStore(samples_path, task.variables)
    task.eval()
    with torch.no_grad():
        for batch in tqdm(datamodule.val_dataloader(), desc="Cases"):
            batch = move_data_to_device(batch, device)

            x_sample = task.sample(batch)

            sample_store.add_samples(x_sample, batch.data.metadata)

    metrics = task._sample_metrics("val", config.data.root).to(device)
    stats = move_data_to_device(datamodule.stats, device)
    log_metrics = metrics.compute(sample_store, stats, device, expensive_metrics=False)
    log_metrics = {key: float(value.item()) for key, value in log_metrics.items()}
    for key in sorted(log_metrics.keys()):
        value = log_metrics[key]
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
