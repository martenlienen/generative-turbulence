#!/usr/bin/env python

# SPDX-FileCopyrightText: © 2024 Marten Lienen <m.lienen@tum.de> & Technical University of Munich
#
# SPDX-License-Identifier: MIT

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
from generate_utils import ChannelConfig, generate_case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Directory to generate the cases into")
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(exist_ok=True, parents=True)

    base_config = ChannelConfig(end_time=0.5, write_interval=1e-4)
    base_config = base_config.add_basic_step(height=28, width=24, offset=16)

    for inflow in np.linspace(0.5, 20.0, num=16):
        case_root = root / f"inflow-{inflow:.03f}"
        inflow_config = replace(base_config, inflow=inflow)
        generate_case(case_root / "3d" / "data" / "high-step" / "case", inflow_config)
        generate_case(
            case_root / "2d" / "data" / "high-step" / "case", inflow_config.to_2d()
        )


if __name__ == "__main__":
    main()
