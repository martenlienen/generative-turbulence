#!/usr/bin/env python

# SPDX-FileCopyrightText: © 2024 Marten Lienen <m.lienen@tum.de> & Technical University of Munich
#
# SPDX-License-Identifier: MIT

import argparse
import json
from pathlib import Path

import fluidfoam as ff
import h5py as h5
import numpy as np
from joblib import Parallel, delayed, parallel_backend
from tqdm import tqdm

from turbdiff.openfoam import parse_openfoam_dict


def isfloat(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def parallel_read(case_dir: Path, time_dirs: list[str], field_name: str, dtype: np.dtype):
    def read_and_convert(time_dir: str):
        return ff.readfield(
            case_dir, time_dir, field_name, verbose=False, precision=100
        ).astype(dtype)

    return Parallel(return_as="generator")(
        delayed(read_and_convert)(d) for d in time_dirs
    )


def stream_to_h5(
    data: h5.Group,
    name: str,
    case_dir: Path,
    time_dirs: list[str],
    compression: str | None,
):
    n_times = len(time_dirs)
    dataset = None
    for i, field in tqdm(
        enumerate(parallel_read(case_dir, time_dirs, name, np.float32)),
        total=n_times,
        desc=name,
    ):
        if name == "U":
            field = field.T
        if dataset is None:
            dataset = data.create_dataset(
                name.lower(), (n_times, *field.shape), compression=compression
            )
        dataset[i] = field


def main():
    parser = argparse.ArgumentParser(
        description="Convert OpenFOAM simulation results into an .h5 file"
    )
    parser.add_argument(
        "-c", "--compression", choices=["gzip"], help="Compress the .h5 file"
    )
    parser.add_argument("data", help="OpenFOAM data directory")
    args = parser.parse_args()

    # Use "gzip" to save about 50% space but significantly slower writing and reading
    compression = args.compression
    data_dir = Path(args.data)

    case_dir = data_dir / "case"
    assert data_dir.is_dir()
    assert case_dir.is_dir()

    ###################################
    # Load OpenFOAM's polymesh format #
    ###################################

    polymesh = case_dir / "constant" / "polyMesh"

    points_file = ff.OpenFoamFile(polymesh, name="points", verbose=False)
    faces_file = ff.OpenFoamFile(polymesh, name="faces", verbose=False)
    owner_file = ff.OpenFoamFile(polymesh, name="owner", verbose=False)
    neighbor_file = ff.OpenFoamFile(polymesh, name="neighbour", verbose=False)
    boundary_file = ff.OpenFoamFile(polymesh, name="boundary", verbose=False)

    points = points_file.values.reshape((-1, 3)).astype(np.float32)

    faces = np.empty((len(faces_file.faces), 4), dtype=np.int64)
    for idx, face in faces_file.faces.items():
        assert face["npts"] == 4, "Only quad faces are supported"
        faces[idx] = face["id_pts"]

    cells = [[] for _ in range(owner_file.nb_cell)]
    face2cell = owner_file.values
    for face, cell in enumerate(face2cell):
        cells[cell].append(face)
    for face, cell in enumerate(neighbor_file.values):
        cells[cell].append(face)
    cells = np.array(cells, dtype=np.int64)

    boundaries = {
        name.decode(): {
            "type": desc[b"type"].decode(),
            "start": int(desc[b"startFace"]),
            "n": int(desc[b"nFaces"]),
        }
        for name, desc in boundary_file.boundaryface.items()
    }

    #############################
    # Load the simulation times #
    #############################

    # Select the time directories and order them in ascending order
    time_dirs = [d.name for d in case_dir.iterdir() if isfloat(d.name)]
    time_dirs.sort(key=float)

    # Drop the first time directory because potentialFoam does not initialize the k and
    # nut fields, which messes up our data reading
    time_dirs.pop(0)

    times = np.array(list(map(float, time_dirs)), dtype=np.float32)

    ################################
    # Read the boundary conditions #
    ################################

    boundary_conditions = {}
    for var in ["p", "U", "k", "nut"]:
        config = parse_openfoam_dict(case_dir / "initial-conditions" / var)
        bc = {}
        for name, desc in config.assignments["boundaryField"].items():
            if desc["type"] == "zeroGradient":
                bc[name] = {"type": "zero-gradient"}
            elif desc["type"] == "fixedValue":
                bc[name] = {"type": "fixed-value", "value": desc["value"].value}
            elif desc["type"] == "inletOutlet":
                # Disregard the values as we only choose 0s
                bc[name] = {"type": "inlet-outlet"}
            elif desc["type"] == "noSlip":
                bc[name] = {"type": "fixed-value", "value": [0, 0, 0]}
            else:
                raise RuntimeError(
                    f"Unknown boundary condition {desc} for boundary {name}"
                )
        boundary_conditions[var] = bc

    ################################
    # Read the physical parameters #
    ################################

    config = parse_openfoam_dict(case_dir / "constant" / "physicalProperties")
    nu = config.assignments["nu"].value

    ######################
    # Store data in HDF5 #
    ######################

    with h5.File(data_dir / "data.h5", mode="w") as f:
        physical = f.require_group("physical")
        physical.attrs["nu"] = nu
        domain = f.require_group("domain")
        domain["points"] = points
        domain["faces"] = faces
        domain["face2cell"] = face2cell
        domain["cells"] = cells
        domain.attrs["boundaries"] = json.dumps(boundaries)
        bcs = f.require_group("boundary-conditions")
        for var, bc in boundary_conditions.items():
            bc_group = bcs.require_group(var.lower())
            for name, desc in bc.items():
                boundary_group = bc_group.require_group(name)
                boundary_group.attrs["type"] = desc["type"]
                if "value" in desc:
                    boundary_group["value"] = np.array(desc["value"], dtype=np.float32)

        data = f.require_group("data")
        data["times"] = times

        # Stream the field data from OpenFOAM directly into the h5 file
        with parallel_backend("loky", n_jobs=-1):
            stream_to_h5(data, "p", case_dir, time_dirs, compression)
            stream_to_h5(data, "U", case_dir, time_dirs, compression)
            stream_to_h5(data, "k", case_dir, time_dirs, compression)
            stream_to_h5(data, "nut", case_dir, time_dirs, compression)


if __name__ == "__main__":
    main()
