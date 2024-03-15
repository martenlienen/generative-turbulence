# SPDX-FileCopyrightText: © 2024 Marten Lienen <m.lienen@tum.de> & Technical University of Munich
#
# SPDX-License-Identifier: MIT

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lark


@dataclass
class Config:
    header: dict
    assignments: dict


@dataclass
class Units:
    powers: list[int]


@dataclass
class Uniform:
    value: list[float | int] | float | int


@dataclass
class Nonuniform:
    values: list[float | int]


@dataclass
class UnitValue:
    units: Units
    value: Any


class DictTransformer(lark.Transformer):
    def units(self, values):
        return Units(values)

    def simple_list(self, values):
        return values

    def assignment(self, data):
        ident, *values = data
        if len(values) == 1:
            values = values[0]
        return ident, values

    def dict(self, assignments):
        return {k: v for k, v in assignments}

    def config(self, values):
        header, *assignments = values
        return Config(header, self.dict(assignments))

    def header(self, dict):
        return dict[0]

    def value(self, values):
        if len(values) == 1:
            return values[0]
        else:
            return values

    def field(self, values):
        uniformity, values = values
        if uniformity == "uniform":
            return Uniform(values)
        elif uniformity == "nonuniform":
            return Nonuniform(values)
        else:
            raise RuntimeError(f"Unknown uniformity {uniformity}")

    def unit_value(self, values):
        units, value = values

        return UnitValue(units, value)

    def SIGNED_INT(self, token):
        return int(token)

    def SIGNED_NUMBER(self, token):
        # Disambiguate between ints and floats here to make the grammar unambiguous
        if token.isdigit():
            return int(token)
        else:
            return float(token)

    def IDENTIFIER(self, token):
        return str(token)

    def LINE_COMMENT(self, token):
        return lark.Discard

    def BLOCK_COMMENT(self, token):
        return lark.Discard

    def UNIFORMITY(self, token):
        return str(token)

    def ESCAPED_STRING(self, token):
        return str(token)


PARSER, TRANSFORMER = None, None


def parse_openfoam_dict(path: Path, *, debug=False) -> Config:
    global PARSER, TRANSFORMER

    if PARSER is None:
        grammar = (Path(__file__).parent / "openfoam.lark").read_text()
        PARSER = lark.Lark(grammar, start="config")
        TRANSFORMER = DictTransformer()

    tree = PARSER.parse(path.read_text())
    if debug:
        print(tree.pretty())
    return TRANSFORMER.transform(tree)


HEADER = r"""
/*--------------------------------*- C++ -*----------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Version:  10
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/
FoamFile""".strip()

TOP_SEPARATOR = (
    "\n// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n"
)
FOOTER = (
    "\n// ************************************************************************* //\n"
)


def serialize_value(value, indent=0):
    if isinstance(value, str):
        return value
    elif isinstance(value, int) or isinstance(value, float):
        return str(value)
    elif isinstance(value, list):
        return "".join(["(", " ".join(serialize_value(v, indent) for v in value), ")"])
    elif isinstance(value, dict):
        out = ["\n", " " * indent, "{\n"]
        for k, v in value.items():
            out.append(" " * (indent + 2))
            out.append(k)
            out.append(" ")
            out.append(serialize_value(v, indent + 2))
            if not isinstance(v, dict):
                out.append(";")
            out.append("\n")
        out.append(" " * indent)
        out.append("}")
        return "".join(out)
    elif isinstance(value, Units):
        return "[" + " ".join(map(str, value.powers)) + "]"
    elif isinstance(value, Uniform):
        return "uniform " + serialize_value(value.value)
    elif isinstance(value, Nonuniform):
        return "nonuniform " + serialize_value(value.values)
    elif isinstance(value, UnitValue):
        return " ".join([serialize_value(value.units), serialize_value(value.value)])
    else:
        raise RuntimeError(f"Unknown node {type(value)}: {value}")


def serialize_openfoam_dict(config: Config) -> str:
    out = [HEADER]
    out.append(serialize_value(config.header))
    out.append(TOP_SEPARATOR)
    for name, value in config.assignments.items():
        out.append("\n")
        out.append(name)
        out.append(" ")
        out.append(serialize_value(value))
        if not isinstance(value, dict):
            out.append(";\n")
    out.append("\n")
    out.append(FOOTER)

    return "".join(out)


@contextmanager
def edit_openfoam_dict(path: Path):
    config = parse_openfoam_dict(path)
    yield config
    path.write_text(serialize_openfoam_dict(config))
