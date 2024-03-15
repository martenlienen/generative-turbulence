# SPDX-FileCopyrightText: © 2024 Marten Lienen <m.lienen@tum.de> & Technical University of Munich
#
# SPDX-License-Identifier: MIT

import tempfile
from pathlib import Path
from typing import Optional, Union

import torch
import wandb
from wandb.apis.public import Run

API = None

RunLike = Union[str, Run]


def _api_object():
    global API
    if API is None:
        API = wandb.Api()
    return API


def _to_run_object(run: RunLike) -> Run:
    if isinstance(run, str):
        return _api_object().run(run)
    else:
        return run


def wandb_run(run: RunLike):
    return _to_run_object(run)


def deepen(config: dict):
    deep = {}
    for key, value in config.items():
        parts = key.split("/")
        node = deep
        for part in parts[:-1]:
            if part not in node:
                node[part] = {}
            node = node[part]
        if value == "None":
            node[parts[-1]] = None
        else:
            node[parts[-1]] = value
    return deep


def load_config(run: RunLike):
    return deepen(_to_run_object(run).config)


def download_file(
    run: RunLike, filename: str, to: Optional[Union[str, Path]] = None
) -> Path:
    if to is None:
        root = Path()
    else:
        root = Path(to)
    file = _to_run_object(run).file(filename)
    f = file.download(root=str(root))
    f.close()
    return root / filename


def load_checkpoint(run: RunLike, filename: str):
    with tempfile.TemporaryDirectory() as d:
        path = download_file(run, filename, to=d)
        return torch.load(path)
