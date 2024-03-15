#!/usr/bin/env python

# SPDX-FileCopyrightText: © 2024 Marten Lienen <m.lienen@tum.de> & Technical University of Munich
#
# SPDX-License-Identifier: MIT

import argparse
import itertools as it
import json
import math
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Self

import numpy as np
from ofblockmeshdicthelper import BlockMeshDict
from tqdm import tqdm


@dataclass(frozen=True)
class Cuboid:
    """An axis-aligned cuboid in an arbitrary number of dimensions."""

    pos: tuple[int, ...]
    size: tuple[int, ...]
    is_boundary_face: tuple[tuple[bool, bool], ...] = (*([(True, True)] * 3),)

    def intersection(self, other: Self) -> Self | None:
        intersection_pos = []
        intersection_size = []

        # Iterate over each dimension
        for p1, s1, p2, s2 in zip(self.pos, self.size, other.pos, other.size):
            # Compute the start and end points of the intersection in this dimension
            start = max(p1, p2)
            end = min(p1 + s1, p2 + s2)

            # If there's no overlap in this dimension, return None
            if start > end:
                return None

            intersection_pos.append(start)
            intersection_size.append(end - start)

        return Cuboid(tuple(intersection_pos), tuple(intersection_size))

    def project(self, dim: int):
        """Project away one dimension."""

        n = len(self.pos)
        return Cuboid(
            tuple(self.pos[i] for i in range(n) if i != dim),
            tuple(self.size[i] for i in range(n) if i != dim),
            tuple(self.is_boundary_face[i] for i in range(n) if i != dim),
        )

    def surface_intersection(
        self, other: Self, *, dim: int, direction: int, other_direction: int
    ) -> Self | None:
        if direction == -1 and other_direction == -1:
            if self.pos[dim] != other.pos[dim]:
                return None
        elif direction == -1 and other_direction == +1:
            if self.pos[dim] != other.pos[dim] + other.size[dim]:
                return None
        elif direction == +1 and other_direction == -1:
            if self.pos[dim] + self.size[dim] != other.pos[dim]:
                return None
        elif direction == +1 and other_direction == +1:
            if self.pos[dim] + self.size[dim] != other.pos[dim] + other.size[dim]:
                return None
        else:
            raise RuntimeError(f"Invalid directions {direction}, {other_direction}")

        return self.project(dim).intersection(other.project(dim))

    @property
    def volume(self):
        return math.prod(self.size)

    def difference(self, other: Self) -> list[Self]:
        intersection = self.intersection(other)
        # If there is no intersection, the difference is just this cuboid
        if intersection is None:
            return [self]

        # Compute the lower, intersection and upper parts in each dimension
        dim_parts = [
            (
                (p, p_int - p, "lower"),
                (p_int, s_int, "inter"),
                (p_int + s_int, (p + s) - (p_int + s_int), "upper"),
            )
            for p, s, p_int, s_int in zip(
                self.pos, self.size, intersection.pos, intersection.size
            )
        ]

        parts = []
        for pos_size_types in it.product(*dim_parts):
            pos, size, types = list(map(tuple, zip(*pos_size_types)))
            cuboid = Cuboid(pos, size)
            if cuboid.volume == 0 or cuboid == intersection:
                continue
            # A face is a boundary face if it is shared with the a boundary face of
            # `self` or if it is shared with the intersection
            is_boundary_face = [
                (
                    (
                        self.is_boundary_face[i][0]
                        and (
                            common := cuboid.surface_intersection(
                                self, dim=i, direction=-1, other_direction=-1
                            )
                        )
                        is not None
                        and common.volume > 0
                    )
                    or (
                        t == "upper"
                        and (
                            common := cuboid.surface_intersection(
                                intersection, dim=i, direction=-1, other_direction=+1
                            )
                        )
                        is not None
                        and common.volume > 0
                    ),
                    (
                        self.is_boundary_face[i][1]
                        and (
                            common := cuboid.surface_intersection(
                                self, dim=i, direction=+1, other_direction=+1
                            )
                        )
                        is not None
                        and common.volume > 0
                    )
                    or (
                        t == "lower"
                        and (
                            common := cuboid.surface_intersection(
                                intersection, dim=i, direction=+1, other_direction=-1
                            )
                        )
                        is not None
                        and common.volume > 0
                    ),
                )
                for i, t in enumerate(types)
            ]
            cuboid = replace(cuboid, is_boundary_face=is_boundary_face)
            parts.append(cuboid)

        return parts

    @property
    def faces(self) -> list[tuple[int, int]]:
        n = len(self.pos)
        # Add or subtract 1/2 to decode which face of a cell is meant
        lower_faces = [(i, self.pos[i] - 0.5) for i in range(n)]
        upper_faces = [(i, self.pos[i] + self.size[i] - 1 + 0.5) for i in range(n)]
        return lower_faces + upper_faces

    def cut(self, plane: tuple[int, int]) -> list[Self]:
        """Cut this cuboid along a plane."""

        dim, value = plane
        # If the plane does not intersect the cuboid, there is no cut
        if self.pos[dim] >= value or self.pos[dim] + self.size[dim] - 1 <= value:
            return [self]

        upper_pos = math.ceil(value)
        lower_size = upper_pos - self.pos[dim]
        upper_size = self.size[dim] - lower_size

        lower_part = replace(
            self,
            size=tuple(
                [(lower_size if i == dim else s) for i, s in enumerate(self.size)]
            ),
            # The new internal face can never be a boundary face
            is_boundary_face=tuple(
                [
                    ((l, False) if i == dim else (l, u))
                    for i, (l, u) in enumerate(self.is_boundary_face)
                ]
            ),
        )
        upper_part = replace(
            self,
            pos=tuple([(upper_pos if i == dim else p) for i, p in enumerate(self.pos)]),
            size=tuple(
                [(upper_size if i == dim else s) for i, s in enumerate(self.size)]
            ),
            # The new internal face can never be a boundary face
            is_boundary_face=tuple(
                [
                    ((False, u) if i == dim else (l, u))
                    for i, (l, u) in enumerate(self.is_boundary_face)
                ]
            ),
        )
        return [lower_part, upper_part]


def create_block(bmd: BlockMeshDict, name: str, ncells, offset, scale):
    # Coordinates of the unit cube in OpenFOAM vertex ordering
    block_base = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ]
    )

    vertex_coordinates = block_base @ np.diag(scale) + np.array(offset)
    vertices = [
        bmd.add_vertex(*coords, f"{name}-{i}")
        for i, coords in enumerate(vertex_coordinates)
    ]
    return bmd.add_hexblock([v.name for v in vertices], ncells, name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--2d", action=argparse.BooleanOptionalAction, help="Generate a 2D channel"
    )
    parser.add_argument(
        "-H",
        nargs=3,
        type=float,
        default=[5.0, 1.0, 1.0],
        help="Bounding box dimensions",
    )
    parser.add_argument(
        "-n", nargs=3, type=int, default=[50, 10, 10], help="Number of cells"
    )
    parser.add_argument(
        "--hole",
        nargs=6,
        type=int,
        action="append",
        help="Walled off holes to cut into the bounding box",
    )
    parser.add_argument("case", help="OpenFOAM case directory")
    args = parser.parse_args()

    two_dimensional = getattr(args, "2d")
    hx, hy, hz = args.H
    nx, ny, nz = args.n
    holes = args.hole
    case_dir = Path(args.case)

    if two_dimensional:
        assert ny == 1

    # Create the domain geometry
    bounding_box = Cuboid((0, 0, 0), (nx, ny, nz))
    holes = [Cuboid((i, j, k), (w, d, h)) for i, j, k, w, d, h in holes]

    # Cut holes into the domain
    domain = [bounding_box]
    for hole in holes:
        domain = [sub_part for part in domain for sub_part in part.difference(hole)]

    # Cut blocks along the boundary surfaces of all other blocks, because OpenFOAM's
    # blockMesh does not support two blocks sharing a partial face.
    faces = [face for cuboid in domain for face in cuboid.faces]
    for face in list(set(faces)):
        domain = [slice for cuboid in domain for slice in cuboid.cut(face)]

    # Convert the domain cuboid parts into a blockmeshdict configuration
    bmd = BlockMeshDict()
    bmd.set_metric("m")

    dx, dy, dz = hx / nx, hy / ny, hz / nz

    walls = []
    inlets = []
    outlets = []
    empties = []
    for block_idx, cuboid in enumerate(domain):
        i, j, k = cuboid.pos
        w, d, h = cuboid.size
        position = (i * dx, j * dy, k * dz)
        size = (w * dx, d * dy, h * dz)
        block = create_block(bmd, f"block-{block_idx}", cuboid.size, position, size)

        if i == 0:
            inlets.append(block.face("w"))
        elif cuboid.is_boundary_face[0][0]:
            walls.append(block.face("w"))
        if i + w == nx:
            outlets.append(block.face("e"))
        elif cuboid.is_boundary_face[0][1]:
            walls.append(block.face("e"))

        # Track the front and back walls separately in case we are generating a 2D channel
        if cuboid.is_boundary_face[1][0]:
            empties.append(block.face("s"))
        if cuboid.is_boundary_face[1][1]:
            empties.append(block.face("n"))

        if cuboid.is_boundary_face[2][0]:
            walls.append(block.face("b"))
        if cuboid.is_boundary_face[2][1]:
            walls.append(block.face("t"))
    if not two_dimensional:
        walls.extend(empties)
        empties = []

    # Mark boundaries
    bmd.add_boundary("wall", "walls", walls)
    bmd.add_boundary("patch", "inlets", inlets)
    bmd.add_boundary("patch", "outlets", outlets)
    if len(empties) > 0:
        bmd.add_boundary("empty", "empties", empties)

    # Merge all duplicate vertices
    aliases = defaultdict(list)
    for v in bmd.vertices.values():
        # Round to multiples of dx/dy/dz to avoid vertices not being merged because of
        # round-off errors
        aliases[(round(v.x / dx), round(v.y / dy), round(v.z / dz))].append(v)
    for vs in aliases.values():
        if len(vs) <= 1:
            continue
        bmd.reduce_vertex(vs[0].name, *[v.name for v in vs[1:]])

    # Generate blockMeshDict
    bmd.assign_vertexid()
    bmd_path = case_dir / "system" / "blockMeshDict"
    bmd_path.parent.mkdir(exist_ok=True, parents=True)
    bmd_path.write_text(bmd.format())

    mesh_params = {
        "bounding_box": [hx, hy, hz],
        "cell_counts": [nx, ny, nz],
        "holes": [{"position": hole.pos, "size": hole.size} for hole in holes],
    }
    (case_dir / "mesh-params.json").write_text(json.dumps(mesh_params))


if __name__ == "__main__":
    main()
