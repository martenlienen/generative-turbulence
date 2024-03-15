#!/usr/bin/env python

# SPDX-FileCopyrightText: © 2024 Marten Lienen <m.lienen@tum.de> & Technical University of Munich
#
# SPDX-License-Identifier: MIT

import argparse
from pathlib import Path

import numpy as np

from turbdiff.data.ofles import OpenFOAMDataRepository
from turbdiff.data.ofles import Variable as V


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case", help="Case directory")
    args = parser.parse_args()

    case_root = Path(args.case)
    repo = OpenFOAMDataRepository([case_root / "data.h5"], (V.U,))
    data = repo.read(0, list(range(2500, 5000, 10)))
    u = data.grid_embedding((V.U,))
    u_mean = u.mean(dim=0)
    u_fluc = u - u_mean
    # Ensure that we only look after the object in the flow
    u_fluc = u_fluc[..., 24:, :, :]
    tke = 0.5 * (u_fluc**2).sum(dim=-4)
    tke_mean_profile = tke.mean(dim=(-1, -2))
    max_tke_point = tke_mean_profile.argmax(dim=1).float().mean() + 24
    print(f"{case_root.name}: {max_tke_point}")
    np.save(case_root / "max-mean-tke.npy", float(max_tke_point))


if __name__ == "__main__":
    main()
