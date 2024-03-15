#!/usr/bin/env python

# SPDX-FileCopyrightText: © 2024 Marten Lienen <m.lienen@tum.de> & Technical University of Munich
#
# SPDX-License-Identifier: MIT

import argparse
import json
from pathlib import Path

import h5py as h5
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("out")
    args = parser.parse_args()

    root = Path(args.root)
    out_path = Path(args.out)

    if out_path.exists():
        mses = json.loads(out_path.read_text())
    else:
        mses = {}
    for inflow_root in root.glob("inflow-*/*"):
        inflow = inflow_root.parent.name[len("inflow-") :]
        dims = int(inflow_root.name[0])

        train_data_path = inflow_root / "train" / "high-step" / "mean-flow.h5"
        with h5.File(train_data_path, mode="r") as f:
            u_mean = np.array(f["data/u"])
            p_mean = np.array(f["data/p"])

        test_data_path = inflow_root / "test" / "high-step" / "data.h5"
        with h5.File(test_data_path, mode="r") as f:
            u = np.array(f["data/u"])
            p = np.array(f["data/p"])

        mse_u = ((u - u_mean) ** 2).sum(axis=-1).mean(axis=-1).mean().item()
        mse_p = ((p - p_mean) ** 2).sum(axis=-1).mean(axis=-1).mean().item()

        print(inflow, f"{dims}D", f"{mse_u=}", f"{mse_p=}")
        if inflow not in mses:
            mses[inflow] = {}
        mses[inflow][dims] = {"u": mse_u, "p": mse_p}
        out_path.write_text(json.dumps(mses))


if __name__ == "__main__":
    main()
