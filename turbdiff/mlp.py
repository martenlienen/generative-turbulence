# SPDX-FileCopyrightText: © 2024 Marten Lienen <m.lienen@tum.de> & Technical University of Munich
#
# SPDX-License-Identifier: MIT

from typing import Callable

import torch.nn as nn


class MLP(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        hidden_features: int | list[int],
        hidden_layers: int | None = None,
        actfn: Callable[[], nn.Module] = nn.Identity,
    ):
        if hidden_layers is None:
            assert isinstance(hidden_features, list)
            hidden_layers = len(hidden_features)
        elif isinstance(hidden_features, list):
            assert len(hidden_features) == hidden_layers
        else:
            assert hidden_layers >= 0
            hidden_features = [hidden_features] * hidden_layers

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.actfn = actfn

        if hidden_layers == 0:
            modules = [nn.Linear(in_features, out_features)]
        else:
            modules = [nn.Linear(in_features, hidden_features[0]), actfn()]
            for i in range(1, hidden_layers):
                modules.append(nn.Linear(hidden_features[i - 1], hidden_features[i]))
                modules.append(actfn())
            modules.append(nn.Linear(hidden_features[-1], out_features))

        super().__init__(*modules)
