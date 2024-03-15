#!/usr/bin/env python

# SPDX-FileCopyrightText: © 2024 Marten Lienen <m.lienen@tum.de> & Technical University of Munich
#
# SPDX-License-Identifier: MIT

import argparse
import pickle
from pathlib import Path

import numgrid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    args = parser.parse_args()

    out = Path(args.out)

    grids = {}
    # numgrid only accepts these n
    for n in [
        6, 14, 26, 38, 50, 74, 86, 110, 146, 170, 194, 230, 266, 302, 350, 434,
        590, 770, 974, 1202, 1454, 1730, 2030, 2354, 2702, 3074, 3470, 3890, 4334,
        4802, 5294, 5810
    ]:  # fmt: skip
        grids[n] = numgrid.get_angular_grid(n)

    out.write_bytes(pickle.dumps(grids))


if __name__ == "__main__":
    main()
