#!/usr/bin/env python

# SPDX-FileCopyrightText: © 2024 Marten Lienen <m.lienen@tum.de> & Technical University of Munich
#
# SPDX-License-Identifier: MIT

import argparse
from pathlib import Path

import torch

from turbdiff.data.ofles import OpenFOAMStats
from turbdiff.data.ofles import Variable as V
from turbdiff.models.metrics import (
    MaxMeanTKEPositionMetric,
    SampleMetricsCollection,
    SampleStore,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", default="cuda")
    parser.add_argument("stats", help="Stats file")
    parser.add_argument("data", help="Case data directory")
    parser.add_argument("samples", help="Samples file")
    args = parser.parse_args()

    device = torch.device(args.device)
    stats_file = Path(args.stats)
    data_dir = Path(args.data)
    samples_file = Path(args.samples)

    stats = OpenFOAMStats.from_file(stats_file)
    store = SampleStore(samples_file, (V.U, V.P))
    metrics = SampleMetricsCollection("x", data_dir, [MaxMeanTKEPositionMetric()])

    print(metrics.compute(store, stats, device))


if __name__ == "__main__":
    main()
