#!/usr/bin/env python

# SPDX-FileCopyrightText: © 2024 Marten Lienen <m.lienen@tum.de> & Technical University of Munich
#
# SPDX-License-Identifier: MIT

import argparse
from pathlib import Path

import h5py as h5
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--discard-first", default=0.025, type=float)
    parser.add_argument("case")
    args = parser.parse_args()

    discard_first = args.discard_first
    case_root = Path(args.case)

    with h5.File(case_root / "data.h5", mode="r") as f:
        t = np.array(f["data/times"])
        u = np.array(f["data/u"][t > discard_first])

    # Smooth the velocities with a Gaussian kernel along the first axis
    mses = []
    for kernel_width in np.linspace(1, 32, 32):
        kernel = np.exp(-np.linspace(-50, 50, 101) ** 2 / (2 * kernel_width**2))
        kernel /= kernel.sum()
        u_smooth = np.apply_along_axis(
            lambda x: np.convolve(x, kernel, mode="valid"), 0, u
        )
        # Compute the mean squared error between the smoothed and unsmoothed velocities
        mse = ((u[50:-50] - u_smooth) ** 2).sum(axis=-1).mean(axis=-1).mean().item()

        print(mse)
        mses.append(mse)

    np.savetxt(case_root / "gaussian-smoothing-error.txt", mses)


if __name__ == "__main__":
    main()
