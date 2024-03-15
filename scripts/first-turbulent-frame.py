#!/usr/bin/env python

# SPDX-FileCopyrightText: © 2024 Marten Lienen <m.lienen@tum.de> & Technical University of Munich
#
# SPDX-License-Identifier: MIT

import argparse
from pathlib import Path

import numpy as np
import torch
from pytorch_lightning.utilities import move_data_to_device

from turbdiff.data.ofles import OpenFOAMDataRepository
from turbdiff.data.ofles import Variable as V
from turbdiff.models.metrics import (
    LogTKESpectrumL2Distance,
    TurbulentKineticEnergySpectrum,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--step", default=10, type=int)
    parser.add_argument("-d", "--device", default="cuda")
    parser.add_argument("case", help="Case directory")
    args = parser.parse_args()

    step = args.step
    device = torch.device(args.device)
    case_dir = Path(args.case)

    repo = OpenFOAMDataRepository([case_dir / "data.h5"], (V.U,))
    time = repo.times[0]
    n_frames = len(time)
    beginning = repo.read(0, list(range(0, n_frames // 2, step)))
    end = repo.read(0, list(range(n_frames // 2, n_frames, 25)))

    beginning = move_data_to_device(beginning, device)
    end = move_data_to_device(end, device)

    u_beg = beginning.grid_embedding((V.U,))
    u_end = end.grid_embedding((V.U,))

    # Cut off padding layers
    u_beg = u_beg[..., 1:-1, 1:-1, 1:-1]
    u_end = u_end[..., 1:-1, 1:-1, 1:-1]

    # Select last block
    u_beg = u_beg[..., 144:, :, :]
    u_end = u_end[..., 144:, :, :]

    dist = LogTKESpectrumL2Distance(TurbulentKineticEnergySpectrum())
    dist = dist.to(device)

    u_mean = u_end.mean(dim=0)
    D_end, *_ = dist(u_end, u_end, u_mean)
    D_end = D_end.cpu().numpy()
    np.fill_diagonal(D_end, D_end.max())
    limit = 2 * D_end.min(axis=1).max()

    D, *_ = dist(u_beg, u_end, u_mean)
    D_min = D.amin(dim=1).cpu().numpy()

    larger = np.cumsum(D_min > limit)
    first_turbulent = step * np.searchsorted(larger, larger.max(), side="left")

    np.save((case_dir / "first-turbulent-frame.npy"), first_turbulent)
    print(
        f"{case_dir.name}: {first_turbulent} (time={1000 * time[first_turbulent]:.1f}ms, "
        f"limit={limit:.2f})"
    )


if __name__ == "__main__":
    main()
