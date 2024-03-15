#!/usr/bin/env python

# SPDX-FileCopyrightText: © 2024 Marten Lienen <m.lienen@tum.de> & Technical University of Munich
#
# SPDX-License-Identifier: MIT

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from generate_utils import ChannelConfig, generate_case
from tqdm import tqdm


@dataclass
class Rect:
    pos: tuple[int, ...]
    size: tuple[int, ...]

    def __post_init__(self):
        x, y = self.pos
        w, h = self.size

        for corner in [x, y, x + w - 1, y + h - 1]:
            assert 0 <= corner < 48

    @property
    def volume(self):
        return math.prod(self.size)

    @property
    def diameter(self):
        return min(*self.size)

    @property
    def boundary_distance(self):
        x, y = self.pos
        w, h = self.size
        return (min(x, 47 - (x + w - 1)), min(y, 47 - (y + h - 1)))


@dataclass
class Shape:
    name: str
    rects: list[Rect]

    def render(self) -> np.ndarray:
        bitmap = np.zeros((48, 48), dtype=bool)
        for rect in self.rects:
            x, y = rect.pos
            w, h = rect.size
            bitmap[x : x + w, y : y + h] = 1
        return bitmap

    @property
    def symmetries(self) -> list[str]:
        syms = []
        bitmap = self.render()
        if np.all(bitmap == np.rot90(bitmap, k=1)):
            syms.append("rot-90")
        if np.all(bitmap == np.rot90(bitmap, k=2)):
            syms.append("rot-180")
        if np.all(bitmap == np.flip(bitmap, axis=0)):
            syms.append("flip-x")
        if np.all(bitmap == np.flip(bitmap, axis=1)):
            syms.append("flip-y")
        return syms


def get_cases():
    cases = [
        Shape("step-higher", [Rect((0, 0), (48, 21))]),
        Shape("step-lower", [Rect((0, 0), (48, 10))]),
        Shape("corner", [Rect((0, 0), (18, 18))]),
        Shape("opp-corners-sym", [Rect((0, 0), (15, 15)), Rect((33, 33), (15, 15))]),
        Shape("opp-corners-asym", [Rect((0, 0), (11, 11)), Rect((31, 31), (17, 17))]),
        Shape("neighbor-corners", [Rect((0, 0), (15, 15)), Rect((0, 33), (15, 15))]),
        Shape(
            "corners",
            [Rect(pos, (12, 12)) for pos in [(0, 0), (0, 36), (36, 0), (36, 36)]],
        ),
        Shape("pillar", [Rect((20, 0), (8, 32))]),
        Shape("wide-pillar", [Rect((16, 0), (16, 32))]),
        Shape("offset-pillar", [Rect((10, 0), (10, 32))]),
        Shape("double-pillar", [Rect((10, 0), (8, 32)), Rect((30, 0), (8, 32))]),
        Shape("opp-pillar", [Rect((10, 0), (10, 32)), Rect((30, 16), (10, 32))]),
        Shape("bar", [Rect((18, 0), (12, 48))]),
        Shape("double-bar", [Rect((10, 0), (9, 48)), Rect((30, 0), (9, 48))]),
        Shape("offset-bar", [Rect((27, 0), (12, 48))]),
        Shape("teeth", [Rect((18, 0), (12, 16)), Rect((18, 28), (12, 20))]),
        Shape("wide-teeth", [Rect((14, 0), (20, 16)), Rect((14, 28), (20, 20))]),
        Shape("offset-teeth", [Rect((10, 0), (12, 16)), Rect((22, 28), (12, 20))]),
        Shape("elbow", [Rect((20, 0), (8, 28)), Rect((20, 20), (28, 8))]),
        Shape("wide-elbow", [Rect((20, 0), (13, 28)), Rect((20, 20), (28, 13))]),
        Shape("elbow-asym", [Rect((20, 0), (16, 28)), Rect((20, 20), (28, 11))]),
        Shape("elbow-snug", [Rect((0, 0), (16, 28)), Rect((0, 20), (48, 11))]),
        Shape("open-elbow", [Rect((15, 0), (10, 16)), Rect((32, 22), (16, 10))]),
        Shape(
            "donut",
            [
                Rect((13, 13), (6, 16)),
                Rect((13, 29), (16, 6)),
                Rect((29, 19), (6, 16)),
                Rect((19, 13), (16, 6)),
            ],
        ),
        Shape(
            "U",
            [Rect((13, 13), (6, 22)), Rect((19, 13), (10, 6)), Rect((29, 13), (6, 22))],
        ),
        Shape(
            "H",
            [Rect((13, 13), (6, 22)), Rect((19, 21), (10, 6)), Rect((29, 13), (6, 22))],
        ),
        Shape("T", [Rect((19, 10), (10, 20)), Rect((9, 30), (30, 8))]),
        Shape("disjoint-T", [Rect((19, 10), (10, 13)), Rect((9, 30), (30, 8))]),
        Shape("plus", [Rect((19, 10), (10, 28)), Rect((10, 19), (28, 10))]),
        Shape("minus", [Rect((10, 13), (28, 11))]),
        Shape("square", [Rect((16, 16), (16, 16))]),
        Shape("square-large", [Rect((13, 13), (22, 22))]),
        Shape("square-offset", [Rect((10, 10), (19, 19))]),
        Shape("2x2", [Rect((12, 12), (12, 12)), Rect((24, 24), (12, 12))]),
        Shape("2x2-large", [Rect((7, 7), (17, 17)), Rect((24, 24), (17, 17))]),
        Shape(
            "3x3",
            [
                Rect((9 + i * 10, 9 + j * 10), (10, 10))
                for i in range(3)
                for j in range(3)
                if (i - j) % 2 == 0
            ],
        ),
        Shape(
            "3x3-inv",
            [
                Rect((9 + i * 10, 9 + j * 10), (10, 10))
                for i in range(3)
                for j in range(3)
                if (i - j) % 2 == 1
            ],
        ),
        Shape("cross", [Rect((0, 19), (48, 10)), Rect((19, 0), (10, 48))]),
        Shape("cross-wide", [Rect((0, 11), (48, 17)), Rect((19, 0), (10, 48))]),
        Shape("cross-offset", [Rect((0, 28), (48, 10)), Rect((14, 0), (10, 48))]),
        Shape("platform", [Rect((6, 0), (36, 10))]),
        Shape("step-low", [Rect((0, 0), (48, 14))]),
        Shape("high-platform", [Rect((10, 0), (28, 18))]),
        Shape("step-high", [Rect((0, 0), (48, 18))]),
        Shape("altar", [Rect((10, 0), (28, 14)), Rect((18, 14), (12, 14))]),
    ]

    min_fill = 0.0
    max_fill = 0.5
    min_diameter = 6
    min_boundary_distance = 6
    for case in cases:
        bitmap = case.render()
        fill_ratio = bitmap.sum() / bitmap.size
        assert (
            min_fill <= fill_ratio <= max_fill
        ), f"{case.name} has fill ratio {fill_ratio}"

        for rect in case.rects:
            assert (
                rect.diameter >= min_diameter
            ), f"{case.name} has a rect with diameter {rect.diameter}"
            for dist in rect.boundary_distance:
                assert (
                    dist == 0 or dist >= min_boundary_distance
                ), f"{case.name}: {rect} is too close to the boundary"
    print(f"All cases have a fill ratio between {min_fill} and {max_fill}")
    print(f"All rects have a minimum diameter of {min_diameter}")
    print(
        f"All rects are either snug or at least {min_boundary_distance} cells from the boundary"
    )

    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Directory to generate the cases into")
    parser.add_argument(
        "--depth", type=int, default=12, help="How deep the shapes should be"
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=12,
        help="How far away to place the shapes from the inlet",
    )
    parser.add_argument(
        "--scale",
        type=float,
        help="Scale the resolution of the simulation by this factor",
    )
    args = parser.parse_args()

    depth = args.depth
    offset = args.offset
    scale = args.scale

    root = Path(args.root)
    root.mkdir(exist_ok=True, parents=True)

    base_config = ChannelConfig(inflow=20.0, end_time=0.5, write_interval=1e-4)

    cases = get_cases()
    bar = tqdm(desc="Cases", total=len(cases))
    for case in cases:
        config = base_config
        for rect in case.rects:
            x, y = rect.pos
            w, h = rect.size
            config = config.add_hole(x=offset, y=x, z=y, width=depth, depth=w, height=h)
        if scale is not None:
            config = config.refine(scale)
        case_root = root / "data" / case.name
        generate_case(case_root, config)
        (case_root / "shape.json").write_text(json.dumps(asdict(case)))
        bar.update()

    val_cases = set(
        [
            "disjoint-T",
            "square",
            "step-low",
            "platform",
            "offset-bar",
            "elbow",
            "offset-pillar",
            "neighbor-corners",
            "3x3-inv",
        ]
    )
    test_cases = set(
        [
            "step-high",
            "altar",
            "3x3",
            "cross-offset",
            "square-large",
            "double-pillar",
            "opp-corners-sym",
            "wide-elbow",
            "U",
        ]
    )
    train_cases = set([c.name for c in cases]) - val_cases - test_cases
    assert len(val_cases & test_cases) == 0
    for dataset, cases in [
        ("train", train_cases),
        ("val", val_cases),
        ("test", test_cases),
    ]:
        (root / dataset).mkdir(exist_ok=True, parents=True)
        for case in list(cases):
            (root / dataset / case).symlink_to(f"../data/{case}")


if __name__ == "__main__":
    main()
