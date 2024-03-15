# SPDX-FileCopyrightText: © 2024 Marten Lienen <m.lienen@tum.de> & Technical University of Munich
#
# SPDX-License-Identifier: MIT

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.ofles import OpenFOAMData


class CellTypeEmbedding(nn.Module):
    """Mark each cell in a 3D grid with an embedding denoting its type."""

    @staticmethod
    def create(type: Literal["learned", "onehot"], dim: int):
        if type == "learned":
            return CellTypeLearnedEmbedding(dim)
        elif type == "onehot":
            return CellTypeOneHotEmbedding()
        else:
            raise RuntimeError(f"Unknown cell type embedding {type}")

    def __init__(self):
        super().__init__()

        self.boundary_types = {
            "inside": 0,
            "outside": 1,
            "walls": 2,
            "inlets": 3,
            "outlets": 4,
            # Cells to marked as empty for 2D simulations in OpenFOAM
            "empties": 5,
        }

    @property
    def n_types(self):
        return len(self.boundary_types)

    @property
    def out_dim(self):
        raise NotImplementedError()

    def cell_types(self, data: OpenFOAMData) -> torch.Tensor:
        cell_type = torch.full(
            (data.cell_counts.prod(),),
            self.boundary_types["outside"],
            dtype=torch.long,
            device=data.device,
        )
        cell_type[data.cell_idx] = self.boundary_types["inside"]
        for name, desc in data.boundaries.items():
            cell_type[desc["idx"]] = self.boundary_types[name]

        return cell_type.reshape(tuple(data.cell_counts))


class CellTypeLearnedEmbedding(CellTypeEmbedding):
    def __init__(self, dim: int):
        super().__init__()

        self.dim = dim
        self.embedding = nn.Embedding(self.n_types, embedding_dim=dim)

    def forward(self, data: OpenFOAMData):
        return torch.movedim(self.embedding(self.cell_types(data)), -1, 0)

    @property
    def out_dim(self):
        return self.dim


class CellTypeOneHotEmbedding(CellTypeEmbedding):
    def forward(self, data: OpenFOAMData):
        return torch.movedim(
            F.one_hot(self.cell_types(data), num_classes=self.n_types), -1, 0
        )

    @property
    def out_dim(self):
        return self.n_types
