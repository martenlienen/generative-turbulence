# SPDX-FileCopyrightText: © 2024 Marten Lienen <m.lienen@tum.de> & Technical University of Munich
#
# SPDX-License-Identifier: MIT

import math
import pickle
import random
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import h5py as h5
import numpy as np
import pytorch_lightning as pl
import torch
from cachetools import cachedmethod
from lightning_utilities.core.apply_func import apply_to_collection
from more_itertools import chunked
from torch.utils.data import DataLoader, Dataset, Sampler

from ..utils.index import ravel_multi_index, unravel_index


class Variable(Enum):
    U = 0
    P = 1
    K = 2
    NUT = 3

    # Some derived variables
    CURL = 10
    ENSTROPHY = 11
    DIVERGENCE = 12
    GRAD = 13

    @property
    def dims(self):
        match self:
            case Variable.U | Variable.CURL:
                return 3
            case Variable.P | Variable.K | Variable.NUT:
                return 1
            case Variable.ENSTROPHY | Variable.DIVERGENCE:
                return 1
            case Variable.GRAD:
                return 9

    @staticmethod
    def from_str(name: str) -> "Variable":
        name_lower = name.lower()
        for v in Variable:
            if v.name.lower() == name_lower:
                return v
        raise RuntimeError(f"Unknown variable {name}")


@dataclass
class BoundaryCondition:
    class Type(Enum):
        FIXED_VALUE = 0
        ZERO_GRADIENT = 1
        INLET_OUTLET = 2

    type: Type
    value: torch.Tensor | None = None

    @staticmethod
    def from_h5(group: h5.Group) -> "BoundaryCondition":
        match group.attrs["type"]:
            case "fixed-value":
                return BoundaryCondition(
                    BoundaryCondition.Type.FIXED_VALUE,
                    # Convert to numpy array first to deal with scalar values
                    torch.tensor(np.array(group["value"])),
                )
            case "zero-gradient":
                return BoundaryCondition(BoundaryCondition.Type.ZERO_GRADIENT)
            case "inlet-outlet":
                return BoundaryCondition(BoundaryCondition.Type.INLET_OUTLET)
            case _:
                raise RuntimeError(f"Unknown boundary condition {group}")


def split_channels(
    x: torch.Tensor, variables: tuple[Variable, ...], *, dim=-4
) -> dict[Variable, torch.Tensor]:
    """Split a grid embedding into chunks corresponding to the variables."""
    return {
        v: chunk
        for v, chunk in zip(
            variables, torch.split(x, [v.dims for v in variables], dim=dim)
        )
    }


@dataclass
class ChannelHole:
    pos: np.ndarray
    size: np.ndarray


@dataclass
class OpenFOAMMetadata:
    file: Path
    nu: float
    h: np.ndarray
    cell_counts: np.ndarray
    cell_idx: torch.Tensor
    boundaries: dict[str, dict]
    boundary_conditions: dict[Variable, dict[str, BoundaryCondition]]
    holes: list[ChannelHole]

    _unpadded_cell_idx: torch.Tensor | None = None
    _inside_mask: torch.Tensor | None = None

    @property
    def device(self):
        return self.cell_idx.device

    @property
    def two_dimensional(self):
        # Because of the padding, 1 cell is actually 3 cells
        return self.cell_counts.min() == 3

    @property
    def unpadded_cell_counts(self):
        PADDING = 1
        return self.cell_counts - 2 * PADDING

    @property
    def unpadded_cell_idx(self):
        if self._unpadded_cell_idx is None:
            PADDING = 1
            self._unpadded_cell_idx = ravel_multi_index(
                unravel_index(self.cell_idx, tuple(self.cell_counts)) - PADDING,
                tuple(self.unpadded_cell_counts),
            )
        return self._unpadded_cell_idx

    @property
    def inside_mask(self):
        if self._inside_mask is None:
            self._inside_mask = torch.zeros(
                tuple(self.cell_counts), device=self.device, dtype=torch.bool
            )
            self._inside_mask.flatten()[self.cell_idx] = True
        return self._inside_mask

    @property
    def n_cells(self):
        return len(self.cell_idx)

    @property
    def unpadded_cell_idx_except_next_to_outlet(self):
        cell_idx = unravel_index(self.unpadded_cell_idx, tuple(self.unpadded_cell_counts))
        cell_idx = cell_idx[cell_idx[:, 0] < self.unpadded_cell_counts[0] - 1]
        nx, ny, nz = tuple(self.unpadded_cell_counts)
        return ravel_multi_index(cell_idx, (nx - 1, ny, nz))

    @property
    def cell_idx_except_next_to_outlet(self):
        nx, ny, nz = n = tuple(self.cell_counts)
        cell_idx = unravel_index(self.cell_idx, n)
        cell_idx = cell_idx[cell_idx[:, 0] < nx - 2]
        return ravel_multi_index(cell_idx, (nx - 2, ny, nz))

    @property
    def hydraulic_diameter(self):
        """Compute the hydraulic diameter of the channel [1].

        [1] https://en.wikipedia.org/wiki/Hydraulic_diameter
        """
        nx, ny, nz = self.unpadded_cell_counts
        hx, hy, hz = self.h
        cross_section_over_step = (ny * hy) * (nz * hz)
        perimeter_over_step = 2 * (ny * hy + nz * hz)

        return 4 * cross_section_over_step / perimeter_over_step

    @property
    def case_name(self):
        return self.file.parent.name


@dataclass
class OpenFOAMData:
    metadata: OpenFOAMMetadata
    t: torch.Tensor
    samples: dict[Variable, torch.Tensor]

    _grid_embeddings: dict = field(default_factory=dict)

    def __getattr__(self, name):
        # The object might not be fully initialized when this is called, so there might
        # not be a `metadata` attribute
        if "metadata" in self.__dict__:
            # Allow to access any metadata attribute directly on the batch object
            return getattr(self.metadata, name)
        else:
            raise AttributeError()

    @property
    def n_samples(self):
        return next(iter(self.samples.values())).shape[0]

    @property
    def device(self):
        return self.cell_idx.device

    @property
    def variables(self):
        return tuple(self.samples.keys())

    def pin_memory(self):
        return apply_to_collection(
            self, dtype=torch.Tensor, function=lambda t: t.pin_memory()
        )

    @cachedmethod(lambda self: self._grid_embeddings)
    def grid_embedding(self, variables: tuple[Variable, ...]):
        dims = [v.dims for v in variables]
        batch_dims = self.samples[variables[0]].shape[:-2]
        x = torch.zeros((*batch_dims, sum(dims), *self.cell_counts), device=self.device)
        x_v = {
            # Create a transposed view, so that the values broadcast more easily from
            # the channels-last format in the HDF5 files
            v: chunk.flatten(start_dim=-len(self.cell_counts)).transpose(-1, -2)
            for v, chunk in split_channels(x, variables).items()
        }
        for v in variables:
            x_v[v][..., self.cell_idx, :] = self.samples[v]

        # Fix the variable values for FIXED_VALUE boundary conditions
        for v in variables:
            for name, desc in self.boundary_conditions.get(v, {}).items():
                if desc.type is BoundaryCondition.Type.FIXED_VALUE:
                    x_v[v][..., self.boundaries[name]["idx"], :] = desc.value

        return x


@dataclass
class OpenFOAMStats:
    stats: dict[str, dict[str, torch.Tensor]]

    _normalizers: dict = field(default_factory=dict)

    @cachedmethod(lambda self: self._normalizers)
    def normalizers(self, variables: tuple[Variable, ...], mode: str):
        if ":" in mode:
            mode = {
                Variable.from_str((pair := config.split(":"))[0]): pair[1]
                for config in mode.split(";")
            }
        else:
            orig_mode = mode
            mode = defaultdict(lambda: orig_mode)

        any_tensor = self.stats[variables[0].name.lower()]["mean"]
        dims = [v.dims for v in variables]
        mean = any_tensor.new_zeros(sum(dims))
        std = any_tensor.new_ones(sum(dims))
        for v, mean_v, std_v in zip(
            variables, torch.split(mean, dims), torch.split(std, dims)
        ):
            v_mode = mode[v]
            if "norm" in v_mode:
                stats = self.stats[f"norm({v.name.lower()})"]
                if v_mode == "norm":
                    std_v[:] = stats["mean"]
                elif v_mode == "norm-std":
                    mean_v[:] = stats["mean"]
                    std_v[:] = stats["std"]
                elif v_mode == "norm-max":
                    std_v[:] = stats["max"]
                else:
                    raise RuntimeError(f"Unknown normalization mode {v_mode}")
            else:
                stats = self.stats[v.name.lower()]
                if v_mode == "abs-max":
                    std_v[:] = torch.maximum(stats["min"].abs(), stats["max"].abs())
                elif v_mode == "mean-std":
                    mean_v[:] = stats["mean"]
                    std_v[:] = stats["std"]
                elif v_mode == "std":
                    std_v[:] = stats["std"]
                else:
                    raise RuntimeError(f"Unknown normalization mode {v_mode}")

        # Avoid division by 0 during normalization
        std = torch.where(std >= 1e-8, std, 1.0)

        return mean, std

    @staticmethod
    def from_file(file: Path):
        stats = pickle.loads(file.read_bytes())
        tensor_stats = {
            v: {name: torch.tensor(value) for name, value in stats.items()}
            for v, stats in stats.items()
        }
        return OpenFOAMStats(tensor_stats)


@dataclass
class OpenFOAMBatch:
    data: OpenFOAMData
    stats: OpenFOAMStats


def tensor_size(x: torch.Tensor):
    return x.element_size() * x.nelement()


def tensors_size_megabytes(xs: list[torch.Tensor]):
    return int(np.ceil(sum(map(tensor_size, xs)) / 2**20))


class OpenFOAMDataRepository:
    def __init__(self, files: list[Path], variables: tuple[Variable, ...]):
        super().__init__()

        self.files = files
        self.variables = variables

        self.reset_caches()

    def reset_caches(self):
        self.metadata = {}
        self._times = None

    @property
    def n_cases(self):
        return len(self.files)

    @property
    def times(self):
        if self._times is None:
            self._times = []
            for file in self.files:
                with h5.File(file, "r") as f:
                    self._times.append(np.array(f["data/times"]).copy())
        return self._times

    def read(self, file_idx: int, samples: list[int]):
        metadata = self.read_metadata(file_idx)
        t = torch.tensor(self.times[file_idx][samples])
        data = self.read_data(file_idx, samples)

        return OpenFOAMData(metadata, t, data)

    @cachedmethod(lambda self: self.metadata)
    def read_metadata(self, file_idx: int):
        with h5.File(self.files[file_idx], mode="r") as f:
            bounding_box = torch.tensor(np.array(f["geometry/bounding_box"]))
            bb_cell_counts = torch.tensor(np.array(f["geometry/cell_counts"]))
            h = bounding_box / bb_cell_counts

            nu = f["physical"].attrs["nu"]

            hole_pos = np.array(f["geometry/holes/positions"]).copy()
            hole_sizes = np.array(f["geometry/holes/sizes"]).copy()
            holes = [
                ChannelHole(hole_pos[i], hole_sizes[i]) for i in range(len(hole_pos))
            ]

            cell_counts = np.array(f["grid/cell_counts"]).copy()
            cell_idx = torch.tensor(np.array(f["grid/cell_idx"]))
            boundaries = {}
            for name in f["grid/boundaries"].keys():
                grp = f["grid/boundaries"][name]
                boundaries[name] = {
                    "type": grp.attrs["type"],
                    "idx": torch.tensor(np.array(grp)),
                }
            boundary_conditions = {
                Variable.from_str(var): {
                    boundary: BoundaryCondition.from_h5(condition)
                    for boundary, condition in bcs.items()
                }
                for var, bcs in f["boundary-conditions"].items()
            }

        return OpenFOAMMetadata(
            file=self.files[file_idx],
            nu=nu,
            h=h,
            cell_counts=cell_counts,
            cell_idx=cell_idx,
            boundaries=boundaries,
            boundary_conditions=boundary_conditions,
            holes=holes,
        )

    def read_data(self, file_idx: int, sample_idxs: list[int]):
        # h5py requires the indices to be sorted and unique when reading
        sample_idxs = np.asarray(sample_idxs)
        unique_sorted_idxs, inverse_idx = np.unique(sample_idxs, return_inverse=True)

        with h5.File(self.files[file_idx], mode="r") as f:
            data_group = f["data"]

            data = {}
            for v in self.variables:
                dataset = data_group[v.name.lower()]
                field = torch.tensor(np.array(dataset[unique_sorted_idxs]))

                # Extend the scalar fields with a feature dimension
                if field.ndim == 2:
                    field = field[..., None]

                # Undo the sorting and uniquification
                field = field[inverse_idx]

                data[v] = field

        return data


class OpenFOAMDataset(Dataset):
    def __init__(
        self,
        repo: OpenFOAMDataRepository,
        stats: OpenFOAMStats,
        discard_first_seconds: float = -1.0,
    ):
        super().__init__()

        self.repo = repo
        self.stats = stats
        self.discard_first_seconds = discard_first_seconds

        self.reset_caches()

    def reset_caches(self):
        # Somehow training with multiple data workers keeps crashing, so recreate all
        # state post forking
        self.repo.reset_caches()
        self.valid_steps = []
        for times in self.repo.times:
            self.valid_steps.append(np.nonzero(times > self.discard_first_seconds)[0])

    def sample_idxs_by_file(self):
        i = 0
        indices = []
        for steps in self.valid_steps:
            n = len(steps)
            indices.append(list(range(i, i + n)))
            i += n

        return indices

    def __len__(self):
        return sum(len(vs) for vs in self.valid_steps)

    def __getitem__(self, index: int | list[int]):
        if isinstance(index, int):
            index = [index]
        index = np.array(index)

        file_idx = 0
        while index.min() >= len(self.valid_steps[file_idx]):
            index -= len(self.valid_steps[file_idx])
            file_idx += 1
        assert index.max() < len(
            self.valid_steps[file_idx]
        ), "All samples have to be from the same geometry"

        data = self.repo.read(
            file_idx, [self.valid_steps[file_idx][idx] for idx in index]
        )
        return OpenFOAMBatch(data, self.stats)

    def get_times(self, file_idx: int, times: list[float]):
        # Compare times exactly in tenth of milliseconds
        t = np.round(self.repo.times[file_idx] * 10_000).astype(int).tolist()
        idxs = [t.index(round(t_ * 10_000)) for t_ in times]

        return OpenFOAMBatch(self.repo.read(file_idx, idxs), self.stats)


class OpenFOAMSampler(Sampler):
    def __init__(self, dataset: OpenFOAMDataset, *, batch_size: int, shuffle: bool):
        super().__init__(dataset)

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return sum(
            math.ceil(len(steps) / self.batch_size) for steps in self.dataset.valid_steps
        )

    def __iter__(self):
        indices = self.dataset.sample_idxs_by_file()

        if self.shuffle:
            for idxs in indices:
                random.shuffle(idxs)

        batches = []
        for idxs in indices:
            file_batches = list(chunked(idxs, self.batch_size))
            batches.extend(file_batches)

        if self.shuffle:
            random.shuffle(batches)

        yield from batches


class OpenFOAMEvaluationSampler(Sampler):
    def __init__(
        self, dataset: OpenFOAMDataset, *, batch_size: int, samples_per_file: int
    ):
        super().__init__(dataset)

        self.dataset = dataset
        self.batch_size = batch_size
        self.samples_per_file = samples_per_file

    def __len__(self):
        batches_per_case = math.ceil(self.samples_per_file / self.batch_size)
        return self.dataset.repo.n_cases * batches_per_case

    def __iter__(self):
        indices = self.dataset.sample_idxs_by_file()

        # Select evenly distributed samples from each file
        indices = [
            [
                idxs[i]
                for i in np.round(
                    np.linspace(0, len(idxs) - 1, num=self.samples_per_file)
                ).astype(int)
            ]
            for idxs in indices
        ]

        # Split the indices into batches
        batches = []
        for idxs in indices:
            file_batches = list(chunked(idxs, self.batch_size))
            batches.extend(file_batches)

        yield from batches


def find_data_files(cases_root: Path):
    # Construct paths directly to avoid traversing these directories with all the data
    # directories which can be very slow over NFS
    return [path for dir in cases_root.iterdir() if (path := dir / "data.h5").is_file()]


def reset_dataset_caches(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        dataset.reset_caches()


class OpenFOAMDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: Path,
        discard_first_seconds: float,
        num_workers: int = 2,
        batch_size: int = 1,
        eval_batch_size: int = 8,
        val_samples: int = 8,
        test_samples: int = 32,
        pin_memory: bool = True,
        variables: tuple[Variable, ...] = tuple(Variable),
    ):
        super().__init__()

        self.root = root
        self.discard_first_seconds = discard_first_seconds
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.val_samples = val_samples
        self.test_samples = test_samples
        self.pin_memory = pin_memory
        self.variables = variables

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        self.stats = OpenFOAMStats.from_file(self.root / "stats.pickle")
        if stage in ("fit",) and self.train_dataset is None:
            self.train_dataset = self._dataset("train")
        if stage in ("fit", "validate") and self.val_dataset is None:
            self.val_dataset = self._dataset("val")
        if stage in ("test",) and self.test_dataset is None:
            self.test_dataset = self._dataset("test")

    def _dataset(self, phase: str):
        files = find_data_files(self.root / phase)
        return OpenFOAMDataset(
            OpenFOAMDataRepository(files, self.variables),
            stats=self.stats,
            discard_first_seconds=self.discard_first_seconds,
        )

    def train_dataloader(self):
        sampler = OpenFOAMSampler(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )
        return self._dataloader(self.train_dataset, sampler)

    def val_dataloader(self):
        sampler = OpenFOAMEvaluationSampler(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            samples_per_file=self.val_samples,
        )
        return self._dataloader(self.val_dataset, sampler)

    def test_dataloader(self):
        sampler = OpenFOAMEvaluationSampler(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            samples_per_file=self.test_samples,
        )
        return self._dataloader(self.test_dataset, sampler)

    def _dataloader(self, dataset, sampler):
        return DataLoader(
            dataset,
            sampler=sampler,
            worker_init_fn=reset_dataset_caches,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
