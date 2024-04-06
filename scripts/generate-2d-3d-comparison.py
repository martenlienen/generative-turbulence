#!/usr/bin/env python

# SPDX-FileCopyrightText: © 2024 Marten Lienen <m.lienen@tum.de> & Technical University of Munich
#
# SPDX-License-Identifier: MIT

import argparse
from pathlib import Path

from generate_utils import ChannelConfig, generate_case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Directory to generate the cases into")
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(exist_ok=True, parents=True)

    base_config = ChannelConfig(
        n=(128, 32, 32), inflow=1.0, end_time=0.6, write_interval=1e-2
    )
    # Add the double step
    base_config = base_config.add_basic_step(height=18, width=26, offset=15)
    base_config = base_config.add_basic_step(height=9, width=26, offset=15 + 26)

    for scale in [0.5, 1, 1.5, 2, 4]:
        scale_config = base_config.refine(scale)
        n = scale_config.n[-1]
        generate_case(root / str(n) / "3d" / "case", scale_config)
        generate_case(root / str(n) / "2d" / "case", scale_config.to_2d())


if __name__ == "__main__":
    main()
