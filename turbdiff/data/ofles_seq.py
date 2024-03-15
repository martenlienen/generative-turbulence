# SPDX-FileCopyrightText: © 2024 Marten Lienen <m.lienen@tum.de> & Technical University of Munich
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import einops as eo
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from .ofles import (
    OpenFOAMBatch,
    OpenFOAMData,
    OpenFOAMDataRepository,
    OpenFOAMEvaluationSampler,
    OpenFOAMSampler,
    OpenFOAMStats,
)
from .ofles import Variable as V
from .ofles import find_data_files, reset_dataset_caches


class OpenFOAMSequenceDataset(Dataset):
    def __init__(
        self,
        repo: OpenFOAMDataRepository,
        stats: OpenFOAMStats,
        *,
        sequence_length: int = 8,
        stride: int = 1,
        discard_first_seconds: float = -1.0,
    ):
        super().__init__()

        self.repo = repo
        self.stats = stats
        self.sequence_length = sequence_length
        self.stride = stride
        self.discard_first_seconds = discard_first_seconds

        assert self.sequence_length >= 1
        assert self.stride >= 1

        self.reset_caches()

    def reset_caches(self):
        # Somehow training with multiple data workers keeps crashing, so recreate all
        # state post forking
        self.repo.reset_caches()
        self.valid_steps = []
        for times in self.repo.times:
            select = times > self.discard_first_seconds
            idxs = np.nonzero(select)[0]

            # Only keep indices from which we can start a sequence of the requested
            # length
            idxs = idxs[: -(self.sequence_length * self.stride - 1)]

            assert np.all(np.diff(idxs) == 1), "All steps should be consecutive"
            self.valid_steps.append(idxs)

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

        idxs = [
            step
            for idx in index
            for step in range(
                self.valid_steps[file_idx][idx],
                self.valid_steps[file_idx][idx] + self.sequence_length * self.stride,
                self.stride,
            )
        ]
        return self._read_idxs(file_idx, idxs)

    def _read_idxs(self, file_idx: int, idxs: list[int]):
        data = self.repo.read(file_idx, idxs)
        t = eo.rearrange(data.t, "(b t) ... -> b t ...", t=self.sequence_length)
        samples = {
            variable: eo.rearrange(sample, "(b t) ... -> b t ...", t=self.sequence_length)
            for variable, sample in data.samples.items()
        }

        seq_data = OpenFOAMData(data.metadata, t, samples)
        return OpenFOAMBatch(seq_data, self.stats)


class OpenFOAMSequenceDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: Path,
        discard_first_seconds: float,
        num_workers: int = 2,
        batch_size: int = 1,
        seq_len: int = 2,
        eval_batch_size: int = 8,
        eval_seq_len: int = 100,
        val_samples: int = 8,
        test_samples: int = 32,
        pin_memory: bool = True,
        variables: tuple[V, ...] = tuple(V),
        stride: int = 1,
    ):
        super().__init__()

        self.root = root
        self.discard_first_seconds = discard_first_seconds
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.val_samples = val_samples
        self.eval_batch_size = eval_batch_size
        self.eval_seq_len = eval_seq_len
        self.test_samples = test_samples
        self.pin_memory = pin_memory
        self.variables = variables
        self.stride = stride

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        self.stats = OpenFOAMStats.from_file(self.root / "stats.pickle")
        if stage in ("fit",) and self.train_dataset is None:
            self.train_dataset = self._dataset("train", self.seq_len)
        if stage in ("fit", "validate") and self.val_dataset is None:
            self.val_dataset = self._dataset("val", self.eval_seq_len)
        if stage in ("test",) and self.test_dataset is None:
            self.test_dataset = self._dataset("test", self.eval_seq_len)

    def _dataset(self, phase: str, seq_len: int):
        files = find_data_files(self.root / phase)
        return OpenFOAMSequenceDataset(
            OpenFOAMDataRepository(files, self.variables),
            sequence_length=seq_len,
            stride=self.stride,
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
