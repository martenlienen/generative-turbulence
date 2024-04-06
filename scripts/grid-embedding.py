#!/usr/bin/env python

# SPDX-FileCopyrightText: © 2024 Marten Lienen <m.lienen@tum.de> & Technical University of Munich
#
# SPDX-License-Identifier: MIT

import argparse
import copy
import json
from pathlib import Path

import h5py as h5
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="OpenFOAM data directory")
    args = parser.parse_args()

    data_dir = Path(args.data)

    with h5.File(data_dir / "data.h5", mode="r") as f:
        boundaries = json.loads(f["domain"].attrs["boundaries"])
        points = np.array(f["domain/points"])
        faces = np.array(f["domain/faces"])
        face2cell = np.array(f["domain/face2cell"])
        cells = np.array(f["domain/cells"])

    mesh_params = json.loads((data_dir / "case" / "mesh-params.json").read_text())
    bounding_box = np.array(mesh_params["bounding_box"])
    unpadded_cell_counts = np.array(mesh_params["cell_counts"], dtype=int)

    #################################################
    # Construct mapping from mesh to 3D grid format #
    #################################################

    face_pos = points[faces].mean(axis=1)
    cell_pos = face_pos[cells].mean(axis=1)

    # Compute the index of every cell in the 3D grid
    dx = bounding_box / unpadded_cell_counts
    cell_idx = np.round((cell_pos - dx / 2) / dx).astype(int)

    # Adjust the indices for the extra padding layer (for boundary conditions)
    cell_idx += 1

    # Compute the orientation of each face of a cell w.r.t. to its owning cell:
    #
    # 0 -> +x, 1 -> +y, 2 -> +z, 3 -> -x, 4 -> -y, 5 -> -z
    axes_3d = np.identity(3, dtype=int)
    directions = np.concatenate((axes_3d, -axes_3d), axis=0)
    c2f_vec = face_pos - cell_pos[face2cell]
    face_dir = np.inner(c2f_vec, directions).argmax(axis=-1)

    # Embed the boundary conditions into the grid
    grid_boundaries = copy.deepcopy(boundaries)
    for desc in grid_boundaries.values():
        boundary_faces = np.arange(desc["start"], desc["start"] + desc["n"])
        desc["idx"] = (
            cell_idx[face2cell[boundary_faces]] + directions[face_dir[boundary_faces]]
        )

    ######################
    # Store mapping HDF5 #
    ######################

    # There are two extra cells in each dimension because of the padding layer
    cell_counts = unpadded_cell_counts + 2

    def ravel_idx(idx):
        return np.ravel_multi_index(idx.T, cell_counts)

    with h5.File(data_dir / "data.h5", mode="r+") as f:
        geometry_group = f.require_group("geometry")
        geometry_group["bounding_box"] = bounding_box
        geometry_group["cell_counts"] = unpadded_cell_counts
        holes = mesh_params["holes"]
        geometry_group["holes/positions"] = np.array([hole["position"] for hole in holes])
        geometry_group["holes/sizes"] = np.array([hole["size"] for hole in holes])

        grid_group = f.require_group("grid")
        grid_group["cell_counts"] = cell_counts
        grid_group["cell_idx"] = ravel_idx(cell_idx)
        boundaries_group = grid_group.require_group("boundaries")
        for name, desc in grid_boundaries.items():
            ds = boundaries_group.create_dataset(name, data=ravel_idx(desc["idx"]))
            ds.attrs["type"] = desc["type"]
            ds.attrs["start"] = desc["start"]
            ds.attrs["n"] = desc["n"]


if __name__ == "__main__":
    main()
