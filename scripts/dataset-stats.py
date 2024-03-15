#!/usr/bin/env python

# SPDX-FileCopyrightText: © 2024 Marten Lienen <m.lienen@tum.de> & Technical University of Munich
#
# SPDX-License-Identifier: MIT

import argparse
import pickle
from pathlib import Path

import h5py as h5
import numpy as np
from joblib import Parallel, delayed, parallel_backend
from tqdm import tqdm

from turbdiff.data.ofles import OpenFOAMDataRepository
from turbdiff.data.ofles import Variable as V
from turbdiff.data.ofles import find_data_files
from turbdiff.metrics import curl as curl_grid
from turbdiff.models.utils import select_cells


def compute_stats(field: np.ndarray):
    # Compute statistics over all timesteps and nodes
    return (
        field.min(axis=(0, 1)),
        field.max(axis=(0, 1)),
        np.prod(field.shape[:2]),
        field.sum(axis=(0, 1), dtype=np.float128),
        (field**2).sum(axis=(0, 1), dtype=np.float128),
    )


def process_chunk(file: Path, idxs: np.ndarray):
    repo = OpenFOAMDataRepository([file], (V.U, V.P, V.K, V.NUT))
    data = repo.read(0, idxs.tolist())

    u = data.samples[V.U].numpy()
    p = data.samples[V.P].numpy()
    k = data.samples[V.K].numpy()
    nut = data.samples[V.NUT].numpy()

    u_grid = data.grid_embedding((V.U,))
    curl = select_cells(curl_grid(u_grid, data.h), data.unpadded_cell_idx).numpy()
    curl = np.swapaxes(curl, -1, -2)

    return {
        "p": compute_stats(p),
        "u": compute_stats(u),
        "norm(u)": compute_stats(np.linalg.norm(u, axis=-1, keepdims=True)),
        "norm(curl)": compute_stats(np.linalg.norm(curl, axis=-1, keepdims=True)),
        "k": compute_stats(k),
        "nut": compute_stats(nut),
    }


def chunk_time_steps(file: list[Path], chunk_size: int):
    with h5.File(file, "r") as f:
        t = np.array(f["data"]["times"])

    idxs = np.arange(len(t))
    for chunk_idxs in np.array_split(idxs, int(np.ceil(len(idxs) / chunk_size))):
        yield chunk_idxs


def reduce_stats(mins, maxs, counts, sums, sums_of_squares):
    total = counts.sum()
    mean = sums.sum(axis=0) / total
    std = np.sqrt(sums_of_squares.sum(axis=0) / total - mean**2)

    return {
        "min": mins.min(axis=0).astype(np.float32),
        "max": maxs.max(axis=0).astype(np.float32),
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
    }


def parallel_stats(files: list[Path], chunk_size: int):
    chunks = [
        (file, chunk)
        for file in tqdm(files, desc="Chunking")
        for chunk in chunk_time_steps(file, chunk_size)
    ]
    chunk_stats = [
        stats
        for stats in tqdm(
            Parallel(return_as="generator")(
                delayed(process_chunk)(file, chunk) for file, chunk in chunks
            ),
            desc="Chunks",
            total=len(chunks),
        )
    ]

    fields = list(chunk_stats[0].keys())
    joint_stats = {
        field: (np.array(x) for x in zip(*[stats[field] for stats in chunk_stats]))
        for field in fields
    }

    return {field: reduce_stats(*data) for field, data in joint_stats.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chunk-size",
        default=10,
        type=int,
        help="Size of chunks to read at once from each file",
    )
    parser.add_argument(
        "--parallel", default=-1, type=int, help="How many processes to run in parallel"
    )
    parser.add_argument("dir", help="Dataset directory")
    args = parser.parse_args()

    chunk_size = args.chunk_size
    parallel_jobs = args.parallel
    root = Path(args.dir)

    train_dir = root / "train"
    data_files = find_data_files(train_dir)

    with parallel_backend("loky", n_jobs=parallel_jobs):
        stats = parallel_stats(data_files, chunk_size)

    (root / "stats.pickle").write_bytes(pickle.dumps(stats))


if __name__ == "__main__":
    main()
