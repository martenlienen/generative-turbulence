#!/usr/bin/env python

# SPDX-FileCopyrightText: © 2024 Marten Lienen <m.lienen@tum.de> & Technical University of Munich
#
# SPDX-License-Identifier: MIT

import argparse
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def split_data(
    file: Path,
    split_files: list[Path],
    proportions: list[float | None],
    compression: str | None,
):
    with h5py.File(file, "r") as f:
        # Compute how many steps should go in which file
        assert len([p for p in proportions if p is None]) <= 1
        n_steps = len(f["data/times"])
        split_steps = [round(n_steps * p) if p is not None else None for p in proportions]
        assigned_steps = sum([steps for steps in split_steps if steps is not None])
        assert assigned_steps <= n_steps
        split_steps = [
            s if s is not None else n_steps - assigned_steps for s in split_steps
        ]
        split_ranges = np.split(np.arange(n_steps), np.cumsum(split_steps)[:-1])

        for split_file, split_range in tqdm(
            list(zip(split_files, split_ranges)), desc="Splits"
        ):
            if split_file.name == "-":
                print(f"Skipping file {split_file}")
                continue

            assert not split_file.is_file()

            split_file.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(split_file, "w") as out_f:
                # Copy global attributes
                for name, value in f.attrs.items():
                    out_f.attrs[name] = value

                # Copy non-data datasets and groups
                for key in f.keys():
                    if key != "data":
                        f.copy(key, out_f)

                # Copy the requested subset of the data
                f_data = f["data"]
                out_data = out_f.require_group("data")
                for name, value in f_data.attrs.items():
                    out_data.attrs[name] = value
                out_data["times"] = f_data["times"][split_range]
                for key in f_data:
                    if key != "times":
                        out_data.create_dataset(
                            key, data=f_data[key][split_range], compression=compression
                        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--compression", choices=["gzip"], help="Compress the .h5 file"
    )
    parser.add_argument(
        "-p", default=0.2, type=float, help="Proportion of data to put into each split"
    )
    parser.add_argument("file", help="File to split")
    parser.add_argument("splits", nargs="+", help="File to save the splits into")
    args = parser.parse_args()

    compression = args.compression
    proportion = args.p

    file = Path(args.file)
    split_files = [Path(f) for f in args.splits]
    proportions = [None] + [proportion] * (len(split_files) - 1)

    split_data(file, split_files, proportions, compression)


if __name__ == "__main__":
    main()
