#!/usr/bin/env python

# SPDX-FileCopyrightText: © 2024 Marten Lienen <m.lienen@tum.de> & Technical University of Munich
#
# SPDX-License-Identifier: MIT

import argparse
import math
from pathlib import Path

import h5py as h5
import numpy as np
from tqdm import tqdm


def wasserstein2_normal(a_mean, a_cov, b_mean, b_cov):
    """2-Wasserstein distance between two diagonal Normal distributions."""
    return np.sqrt(
        (
            np.linalg.norm(a_mean[:, None] - b_mean[None, :], axis=-1, ord=2) ** 2
            + a_cov.sum(axis=-1)[:, None]
            + b_cov.sum(axis=-1)[None, :]
            - 2 * np.sqrt(a_cov[:, None] * b_cov[None, :]).sum(axis=-1)
        ).clip(min=0)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--discard-first", default=0.025, type=float)
    parser.add_argument("--seed", default=713879, type=int)
    parser.add_argument("-k", default=32, type=int)
    parser.add_argument("--epsilon", default=1e-3, type=float)
    parser.add_argument("--max-iter", default=100, type=int)
    parser.add_argument("--max-cluster-size", default=None, type=int)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("dir", help="Data directory")
    args = parser.parse_args()

    discard_first = args.discard_first
    seed = args.seed
    k = args.k
    epsilon = args.epsilon
    max_iter = args.max_iter
    max_cluster_size = args.max_cluster_size
    verbose = args.verbose
    dir = Path(args.dir)

    rng = np.random.default_rng(seed)

    with h5.File(dir / "data.h5", "r") as f:
        t = np.array(f["data"]["times"])
        t_select = np.nonzero(t > discard_first)[0]
        u_mean = []
        u_cov = []
        u_dataset = f["data"]["u"]
        n_cells = u_dataset.shape[1]
        BLOCK_SIZE = 10_000
        # Read data in blocks to avoid having to read in 50GB at once
        for idx in tqdm(range(0, n_cells, BLOCK_SIZE), desc="Reading data"):
            u = np.array(u_dataset[t_select, idx : idx + BLOCK_SIZE])
            u_mean.append(np.mean(u, axis=0))
            u_cov.append(np.var(u, axis=0))
        u_mean = np.concatenate(u_mean)
        u_cov = np.concatenate(u_cov)

    # Hand-rolled K-means with k-means++ initialization based on the 2-Wasserstein
    # distance between diagonal Normal distributions.
    cluster_mean = np.empty((k, 3))
    cluster_cov = np.empty((k, 3))

    # Randomly choose a point as the first cluster center.
    initial_centroids = np.empty(k, dtype=int)
    idx = rng.choice(n_cells)
    initial_centroids[0] = idx
    cluster_mean[0] = u_mean[idx]
    cluster_cov[0] = u_cov[idx]

    for i in tqdm(range(1, k), desc="K-means++"):
        # Compute distance to already selected cluster centers
        D = wasserstein2_normal(cluster_mean[:i], cluster_cov[:i], u_mean, u_cov)

        # Select the distance to the closest cluster center for each point
        D = D.min(axis=0)

        # Never select the same node twice
        D[initial_centroids[:i]] = 0

        # Choose the next cluster center with probability proportional to D**2.
        D_sq = D**2
        p = D_sq / D_sq.sum()
        idx = rng.choice(n_cells, p=p)

        initial_centroids[i] = idx
        cluster_mean[i] = u_mean[idx]
        cluster_cov[i] = u_cov[idx]

    pbar = tqdm(range(max_iter), desc="K-means")
    for i in pbar:
        # Compute distance to cluster centers
        D = wasserstein2_normal(cluster_mean, cluster_cov, u_mean, u_cov)

        # Save old clusters for convergence and statistics
        old_cluster_mean = cluster_mean.copy()
        old_cluster_cov = cluster_cov.copy()

        if i > 0:
            old_assignments = assignments.copy()

        # Assign each point to the closest cluster
        assignments = D.argmin(axis=0)

        assignments_changed = (assignments != old_assignments).sum() if i > 0 else 0

        # Update cluster centers
        for i in range(k):
            assignment_filter = assignments == i
            cluster_mean[i] = u_mean[assignment_filter].mean(axis=0)
            cluster_cov[i] = u_cov[assignment_filter].mean(axis=0)

        # Check for convergence
        D = wasserstein2_normal(
            cluster_mean, cluster_cov, old_cluster_mean, old_cluster_cov
        )
        delta = np.trace(D) / k
        pbar.set_postfix({"delta": delta, "changed": assignments_changed})
        if delta < epsilon:
            print(f"Converged after {i + 1} iterations")
            break

    D = wasserstein2_normal(cluster_mean, cluster_cov, u_mean, u_cov)
    assignments = D.argmin(axis=0)
    counts = np.bincount(assignments)
    cluster_dist = np.array([D[i, assignments == i].mean() for i in range(k)])

    result = {"assignments": assignments}

    if verbose:
        print("Cluster ranking by size with average distance to cluster center:")
        for i, idx in enumerate(np.argsort(-counts)):
            print(f"{i + 1}: {counts[idx]} - {cluster_dist[idx]}")

    if max_cluster_size is not None:
        split_assignments = assignments.copy()
        next_cluster = k
        for i in tqdm(range(k), desc="Splitting"):
            if counts[i] <= max_cluster_size:
                continue

            assigned = np.nonzero(split_assignments == i)[0]
            n_splits = math.ceil(counts[i] / max_cluster_size)
            for split_idx, split_assignment in enumerate(
                np.array_split(assigned, n_splits)
            ):
                if split_idx == 0:
                    # We leave the first split at the original index
                    continue
                split_assignments[split_assignment] = next_cluster
                next_cluster += 1

        if verbose:
            print("Cluster ranking by size post splitting:")
            split_counts = np.bincount(split_assignments)
            for i, idx in enumerate(np.argsort(-split_counts)):
                print(f"{i + 1}: {split_counts[idx]}")

        result = {"assignments": split_assignments, "raw_assignments": assignments}

    np.savez(dir / "regions.npz", **result)
    print(f"Saved assignments to {dir}")


if __name__ == "__main__":
    main()
