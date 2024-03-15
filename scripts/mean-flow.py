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
    parser.add_argument("dir", help="Data directory")
    args = parser.parse_args()

    discard_first = args.discard_first
    dir = Path(args.dir)

    with h5.File(dir / "data.h5", "r") as f:
        t = np.array(f["data"]["times"])
        t_select = np.nonzero(t > discard_first)[0]
        u_mean = np.array(f["data/u"][t_select]).mean(axis=0)
        p_mean = np.array(f["data/p"][t_select]).mean(axis=0)

    with h5.File(dir / "mean-flow.h5", "w") as f:
        data_group = f.require_group("data")
        data_group.create_dataset("u", data=u_mean)
        data_group.create_dataset("p", data=p_mean)


if __name__ == "__main__":
    main()
