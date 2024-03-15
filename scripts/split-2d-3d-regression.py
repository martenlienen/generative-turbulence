#!/usr/bin/env python

# SPDX-FileCopyrightText: © 2024 Marten Lienen <m.lienen@tum.de> & Technical University of Munich
#
# SPDX-License-Identifier: MIT

import argparse
import subprocess
from pathlib import Path

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    args = parser.parse_args()

    root = Path(args.root)

    bar = tqdm(root.glob("inflow-*/*"), position=1)
    for dataset in bar:
        bar.set_description(dataset.parent.name + " " + dataset.name)
        cmd = [
            "scripts/split-hdf5.py",
            "-p",
            "0.1",
            str(dataset / "data" / "high-step" / "data.h5"),
            *[
                str(dataset / phase / "high-step" / "data.h5")
                for phase in ["train", "val", "test"]
            ],
        ]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
