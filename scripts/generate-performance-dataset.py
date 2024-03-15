#!/usr/bin/env python

# SPDX-FileCopyrightText: © 2024 Marten Lienen <m.lienen@tum.de> & Technical University of Munich
#
# SPDX-License-Identifier: MIT

import argparse
from dataclasses import replace
from pathlib import Path

from generate_utils import ChannelConfig, generate_case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Directory to generate the cases into")
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(exist_ok=True, parents=True)

    # We only care about the log files, so we don't need to write results
    base_config = ChannelConfig(inflow=20.0, end_time=0.1, write_interval=1.0)
    # wide-pillar case
    base_config = base_config.add_hole(x=12, y=16, z=0, width=12, depth=16, height=32)

    for parallel in [1, 2, 4, 8, 16]:
        case_config = replace(base_config, parallel=parallel)
        generate_case(root / f"parallel-{parallel:02d}", case_config)


if __name__ == "__main__":
    main()
