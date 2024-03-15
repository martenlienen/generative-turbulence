# SPDX-FileCopyrightText: © 2024 Marten Lienen <m.lienen@tum.de> & Technical University of Munich
#
# SPDX-License-Identifier: MIT

import enum

import torch
import torch.nn as nn

from ..data.ofles import OpenFOAMData
from ..data.ofles import Variable as V
from .cell_type_embeddings import CellTypeEmbedding


class Conditioning(nn.Module):
    class Type(enum.Enum):
        CELL_TYPE = enum.auto()
        CELL_POS = enum.auto()

        @property
        def local(self):
            match self:
                case self.CELL_TYPE | self.CELL_POS:
                    return True
                case _:
                    raise NotImplementedError()

        @property
        def global_(self):
            match self:
                case self.CELL_TYPE | self.CELL_POS:
                    return False
                case _:
                    raise NotImplementedError()

    def __init__(
        self,
        variables: tuple[V, ...],
        cell_type_embedding: CellTypeEmbedding | None,
        cell_pos: bool,
    ):
        super().__init__()

        self.variables = variables
        self.cell_type_embedding = cell_type_embedding
        self.cell_pos = cell_pos

    def forward(self, data: OpenFOAMData):
        C = {}
        if self.cell_type_embedding is not None:
            C[Conditioning.Type.CELL_TYPE] = self.cell_type_embedding(data)
        if self.cell_pos:
            C[Conditioning.Type.CELL_POS] = torch.stack(
                torch.meshgrid(
                    *[
                        torch.linspace(0, 1, c, device=data.device)
                        for c in data.metadata.cell_counts
                    ],
                    indexing="ij",
                )
            )

        return C

    @property
    def local_conditioning_dim(self):
        dim = 0
        if self.cell_type_embedding is not None:
            dim += self.cell_type_embedding.out_dim
        if self.cell_pos:
            dim += 3
        return dim

    @property
    def global_conditioning_dim(self):
        dim = 0
        return dim


def local_conditioning(C: dict[Conditioning.Type, torch.Tensor]) -> torch.Tensor | None:
    c_local = [values for t, values in C.items() if t.local]
    if len(c_local) == 0:
        return None
    else:
        return torch.cat(c_local, dim=0)


def global_conditioning(C: dict[Conditioning.Type, torch.Tensor]) -> torch.Tensor | None:
    c_global = [values for t, values in C.items() if t.global_]
    if len(c_global) == 0:
        return None
    else:
        return torch.cat(c_global, dim=0)
