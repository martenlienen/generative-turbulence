# SPDX-FileCopyrightText: © 2024 Marten Lienen <m.lienen@tum.de> & Technical University of Munich
#
# SPDX-License-Identifier: MIT

import inspect

import torch.nn as nn


class KwargsSequential(nn.Sequential):
    """
    A variant of nn.Sequential that passes keyword arguments only to modules that expect
    them.
    """

    def __init__(self, *modules):
        super().__init__(*modules)

        self.module_info = []
        for module in modules:
            # Get the signature of the forward method of the module
            sig = inspect.signature(module.forward)
            # Check if the forward method accepts **kwargs
            has_var_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
            self.module_info.append((has_var_kwargs, set(sig.parameters.keys())))

    def forward(self, input, *args, **kwargs):
        for module, (has_var_kwargs, accepted_args) in zip(self, self.module_info):
            # Filter kwargs to only include those that the module's forward method accepts
            filtered_kwargs = (
                kwargs
                if has_var_kwargs
                else {k: v for k, v in kwargs.items() if k in accepted_args}
            )
            input = module(input, *args, **filtered_kwargs)

        return input
