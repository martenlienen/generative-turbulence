# SPDX-FileCopyrightText: © 2024 Marten Lienen <m.lienen@tum.de> & Technical University of Munich
#
# SPDX-License-Identifier: MIT

import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path


@dataclass
class ChannelConfig:
    h: tuple[float, float, float] = (0.4, 0.1, 0.1)
    n: tuple[int, int, int] = (192, 48, 48)
    inflow: float = 20.0
    holes: list[tuple[int, int, int, int, int, int]] = field(default_factory=list)
    two_dimensional: bool = False
    parallel: int = 1
    delta_t: float = 1e-5
    end_time: float = 1.0
    write_interval: float = 1e-3

    def add_basic_step(self, *, height: int, width: int, offset: int):
        hole = (offset, 0, 0, width, self.n[1], height)
        return replace(self, holes=self.holes + [hole])

    def add_top_step(self, *, height: int, width: int, offset: int):
        hole = (offset, 0, self.n[2] - height, width, self.n[1], height)
        return replace(self, holes=self.holes + [hole])

    def add_hole(self, *, x: int, y: int, z: int, width: int, depth: int, height: int):
        hole = (x, y, z, width, depth, height)
        return replace(self, holes=self.holes + [hole])

    def to_2d(self):
        hx, hy, hz = self.h
        nx, ny, nz = self.n
        return replace(
            self,
            h=(hx, hy / ny, hz),
            n=(nx, 1, nz),
            holes=[(x, 0, z, w, 1, h) for x, y, z, w, d, h in self.holes],
            two_dimensional=True,
        )

    def refine(self, scale: float):
        def scale_int(n: int) -> int:
            return round(n * scale)

        return replace(
            self,
            n=tuple(list(map(scale_int, self.n))),
            holes=[tuple(list(map(scale_int, hole))) for hole in self.holes],
        )


def generate_case(path: Path, config: ChannelConfig):
    les_cmd = [
        "scripts/les-case.py",
        "--inflow",
        str(config.inflow),
        "0",
        "0",
        "--end-time",
        str(config.end_time),
        "--delta-t",
        str(config.delta_t),
        "--write-interval",
        str(config.write_interval),
        "--parallel",
        str(config.parallel),
        str(path),
    ]
    subprocess.run(les_cmd, check=True)

    channel_cmd = [
        "scripts/channel-3d.py",
        "-H",
        *map(str, config.h),
        "-n",
        *map(str, config.n),
    ]
    for hole in config.holes:
        assert len(hole) == 6
        channel_cmd.append("--hole")
        channel_cmd.extend(map(str, hole))
    if config.two_dimensional:
        channel_cmd.append("--2d")
    channel_cmd.append(str(path))
    subprocess.run(channel_cmd, check=True)
