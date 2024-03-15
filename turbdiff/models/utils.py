# SPDX-FileCopyrightText: © 2024 Marten Lienen <m.lienen@tum.de> & Technical University of Munich
#
# SPDX-License-Identifier: MIT

import torch


def broadcast_right(x: torch.Tensor, other: torch.Tensor):
    """Unsqueeze `x` to the right so that it broadcasts against `other`."""
    assert other.ndim >= x.ndim
    return x.reshape(*((-1,) * x.ndim), *((1,) * (other.ndim - x.ndim)))


def select_cells(x: torch.Tensor, cell_idx: torch.Tensor):
    return x.flatten(start_dim=-3)[..., cell_idx]


def ravel_cells(x: torch.Tensor):
    return x.flatten(start_dim=-3)


def where_cells(cell_idx, cell_values, other: torch.Tensor | None = None):
    if other is None:
        x = torch.zeros_like(cell_values)
    else:
        x = other.clone()
    ravel_cells(x)[..., cell_idx] = ravel_cells(cell_values)[..., cell_idx]
    return x
