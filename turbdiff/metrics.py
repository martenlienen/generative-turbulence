# SPDX-FileCopyrightText: © 2024 Marten Lienen <m.lienen@tum.de> & Technical University of Munich
#
# SPDX-License-Identifier: MIT

import numpy as np
import torch


def centered_difference_derivative(x: torch.Tensor, *, dim: int, h: float):
    """Approximate the first derivative of `x` along `dim` with centered differences.

    Note that the result will be shorter by 2 along the dimension that you take
    derivative with respect to.
    """
    n = x.shape[dim]
    return (x.narrow(dim, 2, n - 2) - x.narrow(dim, 0, n - 2)) / (2 * h)


def unpadded_derivative(x: torch.Tensor, h: tuple[float, float, float], *, dim: int):
    """Take the derivative of `x` along `dim` and cut off the padding layer."""
    assert dim < 0
    # First, cut off the padding in the directions that we do not take the derivative
    # with respect to
    for i in range(-3, 0, 1):
        if i != dim:
            x = x.narrow(i, 1, x.shape[i] - 2)
    return centered_difference_derivative(x, dim=dim, h=h[dim])


def divergence(u: torch.Tensor, h: tuple[float, float, float]):
    """Compute the divergence at the inner cells of the velocity field `u`."""
    ux, uy, uz = torch.unbind(u, dim=-4)
    ux_x = unpadded_derivative(ux, h, dim=-3)
    uy_y = unpadded_derivative(uy, h, dim=-2)
    uz_z = unpadded_derivative(uz, h, dim=-1)
    return (ux_x + uy_y + uz_z).unsqueeze(dim=-4)


def curl(u: torch.Tensor, h: tuple[float, float, float]):
    """Compute the curl at the inner cells of the velocity field `u`."""
    ux, uy, uz = torch.unbind(u, dim=-4)
    ux_y = unpadded_derivative(ux, h, dim=-2)
    ux_z = unpadded_derivative(ux, h, dim=-1)
    uy_x = unpadded_derivative(uy, h, dim=-3)
    uy_z = unpadded_derivative(uy, h, dim=-1)
    uz_x = unpadded_derivative(uz, h, dim=-3)
    uz_y = unpadded_derivative(uz, h, dim=-2)

    return torch.stack((uz_y - uy_z, ux_z - uz_x, uy_x - ux_y), dim=-4)


def vector_gradient(u: torch.Tensor, h: tuple[float, float, float]):
    """Compute the gradient of the 3D vector field `u`."""

    n = u.shape[-4]

    def narrow(x: torch.Tensor, j: int):
        # Cut off dimensions other than j to align all the derivatives
        for i in range(3):
            if i == j:
                continue
            x = x.narrow(i - 3, 1, x.shape[i - 3] - 2)
        return x

    return torch.stack(
        [
            torch.stack(
                [
                    narrow(
                        centered_difference_derivative(
                            u.select(dim=-4, index=i), dim=j - 3, h=h[j]
                        ),
                        j,
                    )
                    for j in range(3)
                ],
                dim=-4,
            )
            for i in range(n)
        ],
        dim=-5,
    )


def enstrophy(u: torch.Tensor, h: tuple[float, float, float]):
    """Compute the enstrophy of the velocity field `u`.

    We integrate the squared vorticity over each cell, so you get one value per cell.
    """

    if isinstance(h, torch.Tensor):
        dx = h.prod()
    else:
        dx = np.prod(h)
    return (torch.linalg.norm(curl(u, h), dim=-4, keepdim=True) ** 2) * dx
