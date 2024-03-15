# SPDX-FileCopyrightText: © 2024 Marten Lienen <m.lienen@tum.de> & Technical University of Munich
#
# SPDX-License-Identifier: MIT

import einops as eo
import torch
import torch.nn as nn

from ..data.ofles import OpenFOAMStats
from ..data.ofles import Variable as V


class Normalization(nn.Module):
    def __init__(self, variables: tuple[V, ...], mode: str):
        super().__init__()

        self.variables = variables
        self.mode = mode

    def normalize_grid(self, x: torch.Tensor, stats: OpenFOAMStats):
        """Normalize a spatial feature tensor with 3 spatial dimensions at the end."""

        mean, std = self._mean_and_std_3d(stats)
        return torch.addcmul(-mean / std, torch.reciprocal(std), x)

    def denormalize_grid(self, x: torch.Tensor, stats: OpenFOAMStats):
        """De-normalize a spatial feature tensor with 3 spatial dimensions at the end."""

        mean, std = self._mean_and_std_3d(stats)
        return torch.addcmul(mean, std, x)

    def _mean_and_std_3d(self, stats: OpenFOAMStats):
        """Return normalization mean and standard deviation reshaped to broadcast over 3D tensors."""

        mean, std = stats.normalizers(self.variables, self.mode)

        # Broadcast over the spatial dimensions
        mean = eo.rearrange(mean, "f -> f 1 1 1")
        std = eo.rearrange(std, "f -> f 1 1 1")

        return mean, std
