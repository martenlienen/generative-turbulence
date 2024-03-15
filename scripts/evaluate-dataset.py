#!/usr/bin/env python

# SPDX-FileCopyrightText: © 2024 Marten Lienen <m.lienen@tum.de> & Technical University of Munich
#
# SPDX-License-Identifier: MIT

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from pytorch_lightning.utilities import move_data_to_device
from tqdm import tqdm

from turbdiff.data.ofles import OpenFOAMDataModule
from turbdiff.data.ofles import Variable as V
from turbdiff.models.metrics import (
    MaxMeanTKEPositionMetric,
    SampleMetricsCollection,
    SampleStore,
    WassersteinMetric,
    WassersteinTKE,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expensive", action="store_true", default=False)
    parser.add_argument("-d", "--device", default="cuda")
    parser.add_argument("dataset_dir", help="Data directory")
    parser.add_argument("samples_path")
    args = parser.parse_args()

    expensive_metrics = args.expensive
    device = torch.device(args.device)
    samples_path = Path(args.samples_path)
    dataset_dir = Path(args.dataset_dir)
    n_samples = 16

    assert samples_path.suffix == ".h5"
    assert not samples_path.exists()

    variables = (V.U, V.P)
    datamodule = OpenFOAMDataModule(
        dataset_dir, discard_first_seconds=0.025, variables=variables
    )
    datamodule.setup("test")

    repo = datamodule.test_dataset.repo
    sample_store = SampleStore(samples_path, variables)

    for i in tqdm(range(repo.n_cases), desc="Store data"):
        n_data = len(repo.times[i])
        data_idx = np.round(np.linspace(250, n_data // 2 - 50, num=n_samples)).astype(int)
        data = repo.read(i, data_idx)

        sample_store.add_samples(data.grid_embedding(variables), data.metadata)

    metrics = SampleMetricsCollection(
        "data",
        dataset_dir / "data",
        [WassersteinTKE(), WassersteinMetric(), MaxMeanTKEPositionMetric()],
    ).to(device)
    stats = move_data_to_device(datamodule.stats, device)
    log_metrics = metrics.compute(
        sample_store, stats, device, expensive_metrics=expensive_metrics
    )
    log_metrics = {key: float(value.item()) for key, value in log_metrics.items()}
    for key in sorted(log_metrics.keys()):
        value = log_metrics[key]
        print(f"{key}: {value}")

    Path("metrics.json").write_text(json.dumps(log_metrics))


if __name__ == "__main__":
    main()
