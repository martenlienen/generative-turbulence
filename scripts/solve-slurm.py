#!/usr/bin/env python

# SPDX-FileCopyrightText: © 2024 Marten Lienen <m.lienen@tum.de> & Technical University of Munich
#
# SPDX-License-Identifier: MIT

import argparse
import os
import shlex
import subprocess
from pathlib import Path

from turbdiff.openfoam import parse_openfoam_dict

SLURM_JOB = """
#!{shell}
{sbatch_options}

echo "Running on ${{SLURMD_NODENAME}} ($(hostname))"

cases=({cases})
case=${{cases[(($SLURM_ARRAY_TASK_ID - 1))]}}

echo "Solving case ${{case}}"

udocker run --volume=$case:/home/openfoam \\
            --entrypoint "/home/openfoam/entrypoint.sh" of
""".strip()


def submit_job(job):
    return subprocess.run(["sbatch"], input=job.encode("utf-8"), check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--time", default="7-0", help="Maximum time [Default 7 days]"
    )
    parser.add_argument("-q", "--qos", help="QOS to submit the job under")
    parser.add_argument("-p", "--partition", help="Partition to run on")
    parser.add_argument("-m", "--mem", default="16G", help="Memory to reserve")
    parser.add_argument("-l", "--log", default="/tmp", help="Directory for SLURM logs")
    parser.add_argument("--shell", default=os.environ["SHELL"], help="Shell for the job")
    parser.add_argument("cases", nargs="+")
    args = parser.parse_args()

    case_paths = [Path(c).resolve() for c in args.cases]

    max_parallel = max(
        parse_openfoam_dict(c / "system" / "decomposeParDict").assignments[
            "numberOfSubdomains"
        ]
        for c in case_paths
    )

    time = args.time
    qos = args.qos
    mem = args.mem
    partition = args.partition
    log_dir = Path(args.log)
    shell = args.shell

    options = {
        "job-name": "openfoam",
        "cpus-per-task": max_parallel,
        "mem": mem,
        "time": time,
    }

    if qos is not None:
        options["qos"] = qos

    if partition is not None:
        options["partition"] = partition

    if str(log_dir) == "/dev/null":
        options["error"] = "/dev/null"
        options["output"] = "/dev/null"
    else:
        # If the log directory does not exist, the job will die immediately
        log_dir.mkdir(exist_ok=True, parents=True)

        options["error"] = log_dir / "slurm-%A_%a.out"
        options["output"] = log_dir / "slurm-%A_%a.out"

    # Use SLURM's array mechanism to start multiple jobs
    options["array"] = f"1-{len(case_paths)}"

    sbatch_lines = [
        f"#SBATCH --{shlex.quote(name)}={shlex.quote(str(value))}"
        for name, value in options.items()
    ]
    sbatch = "\n".join(sbatch_lines)

    case_array = " ".join(str(c) for c in case_paths)
    job = SLURM_JOB.format(shell=shell, sbatch_options=sbatch, cases=case_array)
    submit_job(job)


if __name__ == "__main__":
    main()
