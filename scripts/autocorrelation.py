#!/usr/bin/env python

# SPDX-FileCopyrightText: © 2024 Marten Lienen <m.lienen@tum.de> & Technical University of Munich
#
# SPDX-License-Identifier: MIT

import argparse
from pathlib import Path

import h5py as h5
import numpy as np
from tqdm import tqdm

from turbdiff.data.ofles import OpenFOAMDataRepository
from turbdiff.utils.index import unravel_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case", help="Case directory")
    args = parser.parse_args()

    case_root = Path(args.case)

    # Select the cells in the last quarter of the domain
    repo = OpenFOAMDataRepository([case_root / "data.h5"], ())
    metadata = repo.read_metadata(0)
    idx = unravel_index(metadata.unpadded_cell_idx, tuple(metadata.unpadded_cell_counts))
    cell_counts = metadata.unpadded_cell_counts
    back = (idx[:, 0] >= (cell_counts[0] - max(cell_counts[1:]))).numpy()

    # Load the mean flow
    with h5.File(case_root / "mean-flow.h5", "r") as f:
        u_mean = np.array(f["data/u"])

    # Load the velocities
    with h5.File(case_root / "data.h5", "r") as f:
        u_dataset = f["data/u"]
        n_steps = u_dataset.shape[0]
        u = np.array(u_dataset[n_steps // 2 :])

    # Compute the fluctuating velocities in the back quarter
    u_fluc = u[:, back] - u_mean[back]

    # Compute the autocorrelation up to T steps
    T = 200
    n_steps = u_fluc.shape[0]
    corrcoeff = np.zeros(T + 1)
    for i in tqdm(range(len(corrcoeff))):
        corrcoeff[i] = np.einsum("ijk,ijk->", u_fluc[i:], u_fluc[: n_steps - i])
    corrcoeff /= corrcoeff[0]

    max_decorrelated_coeff = np.abs(corrcoeff[-100:]).max()
    print(f"Max decorrelated coefficient: {max_decorrelated_coeff}")
    for i in range(len(corrcoeff)):
        if np.abs(corrcoeff[i]) <= max_decorrelated_coeff:
            decorrelation_steps = i + 1
            print(f"Decorrelation steps: {decorrelation_steps}")
            break

    np.savez(
        case_root / "autocorrelation.npz",
        decorrelation_steps=decorrelation_steps,
        corrcoeff=corrcoeff,
    )


if __name__ == "__main__":
    main()
