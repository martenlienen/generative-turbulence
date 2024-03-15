#!/usr/bin/env python

# SPDX-FileCopyrightText: © 2024 Marten Lienen <m.lienen@tum.de> & Technical University of Munich
#
# SPDX-License-Identifier: MIT

import argparse
import shutil
from pathlib import Path

from turbdiff.openfoam import edit_openfoam_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inflow",
        nargs=3,
        type=float,
        default=[10.0, 0.0, 0.0],
        help="Inflow velocity",
    )
    parser.add_argument("--end-time", type=float, default=0.1, help="End time")
    parser.add_argument("--delta-t", type=float, default=1e-5, help="Initial time step")
    parser.add_argument(
        "--write-interval", type=float, default=0.001, help="Write interval"
    )
    parser.add_argument(
        "-p", "--parallel", type=int, default=1, help="Number of parallel processes"
    )
    parser.add_argument("case", help="Path to generate the case at")
    args = parser.parse_args()

    inflow_velocity = args.inflow
    end_time = args.end_time
    delta_t = args.delta_t
    write_interval = args.write_interval
    parallel = args.parallel
    case_dir = Path(args.case)

    template_dir = Path(__file__).parent / "les-template"
    shutil.copytree(template_dir, case_dir, dirs_exist_ok=True)

    with edit_openfoam_dict(case_dir / "initial-conditions" / "U") as config:
        config.assignments["boundaryField"]["inlets"]["value"].value = inflow_velocity

    with edit_openfoam_dict(case_dir / "system" / "controlDict") as config:
        config.assignments["endTime"] = end_time
        config.assignments["deltaT"] = delta_t
        config.assignments["writeInterval"] = write_interval

    with edit_openfoam_dict(case_dir / "system" / "decomposeParDict") as config:
        config.assignments["numberOfSubdomains"] = max(parallel, 1)

    # Copy initial conditions into the first time step to be overwritten by
    # potentialFoam
    shutil.copytree(case_dir / "initial-conditions", case_dir / "0.00000")


if __name__ == "__main__":
    main()
