# SPDX-FileCopyrightText: © 2024 Marten Lienen <m.lienen@tum.de> & Technical University of Munich
#
# SPDX-License-Identifier: MIT

import pickle
from collections import defaultdict
from pathlib import Path

import einops as eo
import h5py as h5
import numpy as np
import ot
import torch
import torch.nn as nn
import torchmetrics as tm
from deadpool import Deadpool
from pytorch_lightning.utilities import move_data_to_device
from scipy.special import roots_legendre
from tqdm import tqdm

from ..data.ofles import (
    OpenFOAMData,
    OpenFOAMDataRepository,
    OpenFOAMMetadata,
    OpenFOAMStats,
)
from ..data.ofles import Variable as V
from ..data.ofles import split_channels
from ..metrics import curl
from ..utils import get_logger
from .utils import select_cells

log = get_logger()


class SampleStore:
    def __init__(self, samples_file: Path, variables: list[V]):
        super().__init__()

        # This class cannot be used in distributed mode because of the HDF5 file writing
        assert not (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )

        self.samples_file = samples_file
        self.variables = variables

        samples_file.parent.mkdir(parents=True, exist_ok=True)

    def add_samples(self, x: torch.Tensor, metadata: OpenFOAMMetadata):
        # Prepare data for writing to HDF5
        x_v = split_channels(
            # Select values at in-domain cells and convert to channels-last format
            eo.rearrange(select_cells(x, metadata.cell_idx), "b f c -> b c f").cpu(),
            self.variables,
            dim=-1,
        )

        with h5.File(self.samples_file, "a") as f:
            case_group = f.require_group(metadata.case_name)
            data_group = case_group.require_group("data")

            # Store the sample data in the HDF5 file
            n_prev_samples = data_group.attrs.get("n_samples", 0)
            n_samples = x.shape[0]
            for v in self.variables:
                dataset_name = v.name.lower()
                v_data = x_v[v].numpy()
                if dataset_name not in data_group:
                    # Create a resizeable dataset
                    dataset = data_group.create_dataset(
                        dataset_name,
                        data=v_data,
                        # Make one chunk one sample
                        chunks=v_data[:1].shape,
                        # Allow resizing the the number-of-samples dimension
                        maxshape=(None, *v_data.shape[1:]),
                    )
                else:
                    # Append the data to the existing samples
                    dataset = data_group[dataset_name]
                    sample_capacity = dataset.shape[0]
                    if sample_capacity < n_prev_samples + n_samples:
                        dataset.resize(n_prev_samples + n_samples, axis=0)
                    dataset[n_prev_samples : n_prev_samples + n_samples] = v_data

            # Increase sample count
            data_group.attrs["n_samples"] = n_prev_samples + n_samples

    @property
    def case_names(self):
        with h5.File(self.samples_file, "r") as f:
            return list(f.keys())

    def load_samples(self, metadata: OpenFOAMMetadata, *, range=None) -> OpenFOAMData:
        """Load all samples for the case described by the metadata."""

        with h5.File(self.samples_file, "r") as f:
            data_group = f[metadata.case_name]["data"]
            samples_v = {}
            for v in self.variables:
                dataset = data_group[v.name.lower()]
                if range is not None:
                    dataset = dataset[range]
                samples_v[v] = torch.tensor(np.array(dataset))
                if samples_v[v].ndim == 2:
                    # Ensure that we have a batch dimension even if we are loading a
                    # single sample
                    samples_v[v] = samples_v[v][None]
        return OpenFOAMData(metadata, torch.tensor([]), samples_v)

    def reset(self):
        if not self.samples_file.is_file():
            return
        with h5.File(self.samples_file, "a") as f:
            # Reset the number of samples for each dataset without deleting the data
            # (better safe than sorry)
            for case_name in f.keys():
                f[case_name]["data"].attrs["n_samples"] = 0


class SampleMetricsCollection(nn.Module):
    """Collect samples in an HDF5 file and compute metrics over them.

    # Note

    Don't call the `forward` function to avoid computing expensive sample metrics on
    every batch. Instead, call `update` for each batch and then `compute` at the end.
    """

    def __init__(self, prefix: str, data_dir: Path, metrics: list[nn.Module]):
        super().__init__()

        self.prefix = prefix
        self.data_dir = data_dir
        self.metrics = nn.ModuleList(metrics)

    def compute(
        self,
        sample_store: SampleStore,
        stats: OpenFOAMStats,
        device,
        *,
        expensive_metrics: bool = True,
    ):
        values = {}
        case_names = sample_store.case_names
        metric_names = set()
        for case_name in case_names:
            repo = OpenFOAMDataRepository(
                [self.data_dir / case_name / "data.h5"], sample_store.variables
            )
            samples = sample_store.load_samples(repo.read_metadata(0))

            if samples.n_samples == 0:
                continue

            # Load ground-truth data samples for the current case evenly distributed
            # over the second half of the simulation, assuming that the second half is
            # surely fully-developed turbulence
            n_data = len(repo.times[0])
            data_idx = np.round(
                np.linspace(n_data // 2, n_data - 1, num=samples.n_samples)
            ).astype(int)
            data = repo.read(0, data_idx)

            # Move all data to the compute device
            data = move_data_to_device(data, device)
            samples = move_data_to_device(samples, device)

            # Evaluate the metrics
            for metric in self.metrics:
                if not expensive_metrics and metric.is_expensive():
                    continue
                case_values = metric(samples, data, stats)
                for name, value in case_values.items():
                    values[self.log_name(case_name, name)] = value
                    metric_names.add(name)

        # Take the mean of each metric over all cases
        for metric_name in metric_names:
            values[f"{self.prefix}/{metric_name}"] = torch.mean(
                torch.stack(
                    [
                        values[self.log_name(case_name, metric_name)]
                        for case_name in case_names
                        if self.log_name(case_name, metric_name) in values
                    ]
                )
            )

        return values

    def log_name(self, case: str, metric: str):
        return f"{self.prefix}/{case}/{metric}"


class MeanMetric(tm.Metric):
    full_state_update = False

    def __init__(self):
        super().__init__()

        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0, dtype=int), dist_reduce_fx="sum")

    def update(self, *args, **kwargs):
        sum, count = self.metric(*args, **kwargs)
        self.sum += sum
        self.count += count

    def metric(self, *args, **kwargs):
        raise NotImplementedError()

    def compute(self):
        if self.count == 0:
            return np.nan
        else:
            return self.sum / self.count


def interp3(grid: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Tri-linearly interpolate data from regular 3D grids onto arbitrary points.

    Arguments
    ---------
    grid
        F-dimensional features at integer grid coordinates with shape (..., F, X, Y, Z)
    points
        3D points with shape (N, 3)

    Returns
    -------
    Interpolation values with shape (..., N, F)
    """

    # Find the 8 neighboring grid point coordinates of each query point
    p0 = torch.floor(points).long()
    p1 = p0 + 1
    x0, y0, z0 = torch.unbind(torch.floor(points).long(), dim=-1)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    # Ensure indices are within grid bounds
    lower_bounds = grid.new_zeros(3, dtype=torch.long)
    upper_bounds = grid.new_tensor(grid.shape[-3:], dtype=torch.long) - 1
    p0 = torch.clamp(p0, lower_bounds, upper_bounds)
    p1 = torch.clamp(p1, lower_bounds, upper_bounds)

    x0, y0, z0 = torch.unbind(p0, dim=-1)
    x1, y1, z1 = torch.unbind(p1, dim=-1)

    # Compute the interpolation weights
    wx, wy, wz = torch.unbind(points - p0, dim=-1)

    # Use the weights to compute the interpolated value
    return (
        (1 - wx) * (1 - wy) * (1 - wz) * grid[..., x0, y0, z0]
        + (1 - wx) * (1 - wy) * wz * grid[..., x0, y0, z1]
        + (1 - wx) * wy * (1 - wz) * grid[..., x0, y1, z0]
        + (1 - wx) * wy * wz * grid[..., x0, y1, z1]
        + wx * (1 - wy) * (1 - wz) * grid[..., x1, y0, z0]
        + wx * (1 - wy) * wz * grid[..., x1, y0, z1]
        + wx * wy * (1 - wz) * grid[..., x1, y1, z0]
        + wx * wy * wz * grid[..., x1, y1, z1]
    )


class TurbulentKineticEnergySpectrum(nn.Module):
    """Estimate the turbulent kinetic energy spectrum of a 3D flow field."""

    def __init__(self, n: int = 5810):
        """
        Arguments
        ---------
        n: Number of grid points for Lebedev quadrature
        """

        super().__init__()

        numgrids_file = Path(__file__).parent / "numgrids.pickle"
        numgrids = pickle.loads(numgrids_file.read_bytes())

        self.n = n
        if n not in numgrids:
            raise RuntimeError(f"n={n} is not supported by numgrid.")

        x, y, z, w = numgrids[n]
        p = torch.tensor([x, y, z]).T
        w = torch.tensor(w)

        self.register_buffer("p", p.float())
        self.register_buffer("w", w.float())

    def forward(self, u_perturbation: torch.Tensor, k: torch.Tensor):
        # Compute the TKE at each point (Pope, p. 88)
        tke = 0.5 * (u_perturbation**2).sum(dim=-4)

        # Fourier-transform the TKE
        tke_fft = torch.fft.fftn(tke, dim=(-3, -2, -1))
        tke_fft = torch.fft.fftshift(tke_fft, dim=(-3, -2, -1))

        # Construct the query points for interpolation by centering a sphere on the zero
        # frequency in the shifted FFT and scaling it to radius k
        center = k.new_tensor([s // 2 for s in u_perturbation.shape[-3:]])
        p_query = k[:, None, None] * self.p + center

        # Integrate the directional squared frequency amplitudes of the spectrum over a
        # sphere of radius exactly by interpolating the values from the grid points onto
        # the sphere in the log-domain. We interpolate in the log-domain, because the
        # magnitudes decay exponentially with increasing k which is badly approximated
        # by a linear function and leads to overestimation of the energies, i.e. a shift
        # up in a TKE log-log plot.
        tke_fft_interp = interp3((tke_fft.abs() ** 2).log(), p_query).exp().float()
        # Weights sum up to 1, so scale by the surface area of the radius-k sphere
        E_k = torch.matmul(tke_fft_interp, self.w) * (4 * torch.pi * k**2)

        # Sum the energies over all three dimensions to get the total energy
        return E_k


class LogTKESpectrumL2Distance(nn.Module):
    """
    Estimate the L2 distance between the log-TKE spectrum functions E(k) of two
    turbulent flows with Gauss-Legendre integration.
    """

    def __init__(self, tke_spectrum: nn.Module, n: int = 64):
        """
        Arguments
        ---------
        tke_spectrum: Module to compute the TKE spectrum of a velocity field
        n: Number of nodes for Gauss-Legendre integration
        """

        super().__init__()

        self.tke_spectrum = tke_spectrum
        self.n = n

        legendre_nodes, legendre_weights = roots_legendre(n)
        legendre_nodes = torch.tensor(legendre_nodes)
        legendre_weights = torch.tensor(legendre_weights)

        self.register_buffer("legendre_nodes", legendre_nodes.float())
        self.register_buffer("legendre_weights", legendre_weights.float())

    def forward(self, u_a: torch.Tensor, u_b: torch.Tensor, u_mean: torch.Tensor):
        # Ensure that we don't also pass the pressure by accident
        assert u_a.shape[-4] == 3
        assert u_b.shape[-4] == 3
        assert u_mean.shape[-4] == 3

        # Ensure that all velocity fields have the same spatial dimensions
        assert u_a.shape[-3:] == u_b.shape[-3:]
        assert u_a.shape[-3:] == u_mean.shape[-3:]

        # Linearly transform the Legendre nodes from [-1, 1] to the valid range of
        # frequencies k
        k_min = 1.0
        k_max = float((min(u_a.shape[-3:]) - 1) // 2)
        slope = (k_max - k_min) / 2
        k = slope * self.legendre_nodes + ((k_max - k_min) / 2 + k_min)

        # Compute the log-TKE-spectra
        log_tke_a = self.tke_spectrum(u_a - u_mean, k).log()
        log_tke_b = self.tke_spectrum(u_b - u_mean, k).log()

        # Compute the pairwise distances between any two log-TKE-spectra
        D = slope * torch.einsum(
            "ijk, k -> ij",
            (log_tke_a[:, None] - log_tke_b[None]) ** 2,
            self.legendre_weights,
        )
        D = torch.sqrt(D)

        return D, log_tke_a, log_tke_b, k


class WassersteinTKE(nn.Module):
    def __init__(self):
        super().__init__()

        self.distance = LogTKESpectrumL2Distance(TurbulentKineticEnergySpectrum())
        self.case_data = defaultdict(dict)

    def is_expensive(self):
        return False

    def forward(self, samples: OpenFOAMData, data: OpenFOAMData, stats: OpenFOAMStats):
        if samples.two_dimensional:
            # Handling the 2D case everywhere is too much of a hassle
            return {}

        u_sample = samples.grid_embedding((V.U,))
        u_data = data.grid_embedding((V.U,))

        mean_flow_file = data.metadata.file.parent / "mean-flow.h5"
        if mean_flow_file.is_file():
            with h5.File(mean_flow_file, "r") as f:
                u_mean_data = u_sample.new_tensor(np.array(f["data/u"]))
                u_mean = OpenFOAMData(data.metadata, samples.t[:1], {V.U: u_mean_data})
                u_mean = u_mean.grid_embedding((V.U,))
        else:
            log.warning(
                f"Mean flow file {mean_flow_file} is missing! "
                "Estimating mean from data samples."
            )

            # Estimate the mean flow as the mean of the data samples as a fallback
            if u_data.shape[0] == 1:
                log.warning(
                    f"Only a single data sample for case {data.case_name}. "
                    "Mean flow estimate will be useless for TKE!"
                )
            u_mean = u_data.mean(dim=0)

        # Cut off synthetic boundary cells
        u_sample = u_sample[..., 1:-1, 1:-1, 1:-1]
        u_data = u_data[..., 1:-1, 1:-1, 1:-1]
        u_mean = u_mean[..., 1:-1, 1:-1, 1:-1]

        offset_multiplier = {"front": 3, "middle": 2, "back": 1}
        channel_width = min(u_sample.shape[-2:])
        channel_length = u_sample.shape[-3]
        D_regions = []
        distances = {}
        for region in ["front", "middle", "back"]:
            # Restrict the velocity field to a cube region in the channel
            n = offset_multiplier[region]
            u_sample_region = torch.narrow(
                u_sample,
                dim=-3,
                start=channel_length - n * channel_width,
                length=channel_width,
            )
            u_data_region = torch.narrow(
                u_data,
                dim=-3,
                start=channel_length - n * channel_width,
                length=channel_width,
            )
            u_mean_region = torch.narrow(
                u_mean,
                dim=-3,
                start=channel_length - n * channel_width,
                length=channel_width,
            )

            # Compute the log-TKE-spectra and the pairwise distances between them
            D_region, log_tke_sample, log_tke_data, k = self.distance(
                u_sample_region, u_data_region, u_mean_region
            )
            D_region = D_region.cpu().numpy()

            # Store the log-TKE-spectra for later plotting
            self.case_data[region][data.case_name] = (
                log_tke_sample.cpu(),
                log_tke_data.cpu(),
                k.cpu(),
            )

            distances[f"tke-{region}"] = torch.tensor(self.wasserstein2(D_region))

            D_regions.append(D_region)
        D_regions = np.stack(D_regions)

        # Combine the three regional distances and compute a joint Wasserstein distance
        D_combined = np.sqrt((D_regions**2).sum(axis=0))
        distances["tke"] = torch.tensor(self.wasserstein2(D_combined))

        return distances

    def wasserstein2(self, D: np.ndarray):
        return np.sqrt(ot.emd2([], [], D**2))


def wasserstein2_squared(D: np.ndarray):
    return ot.emd2([], [], D**2)


class WassersteinMetric(nn.Module):
    def is_expensive(self):
        return True

    def forward(self, samples: OpenFOAMData, data: OpenFOAMData, stats: OpenFOAMStats):
        regions_file = data.metadata.file.parent / "regions.npz"
        if not regions_file.is_file():
            log.warning(
                f"Regions file {regions_file} is missing, "
                "can't compute Wasserstein metric!"
            )
            return {}
        regions_data = np.load(regions_file)["assignments"]
        regions = torch.tensor(regions_data, device=samples.device, dtype=torch.long)
        region_counts = np.bincount(regions_data)
        region_weights = region_counts.astype(float) / region_counts.sum()

        sample_features = self.features(samples, stats)
        data_features = self.features(data, stats)

        # PythonOT's EMD solver is single-threaded, so we queue up all jobs in parallel
        n = samples.n_samples
        m = data.n_samples
        pbar_desc = f"Wasserstein Regions {data.case_name}"
        pbar = tqdm(total=n * m * len(region_counts), desc=pbar_desc)
        with Deadpool(max_backlog=100) as executor:
            wasserstein_jobs = {}
            for k in range(len(region_counts)):
                region_filter = regions == k
                samples_region = sample_features[:, region_filter]
                data_region = data_features[:, region_filter]
                for i in range(n):
                    for j in range(m):
                        D_region = torch.linalg.norm(
                            samples_region[i][:, None] - data_region[j][None, :], dim=-1
                        )
                        wasserstein_jobs[(i, j, k)] = executor.submit(
                            wasserstein2_squared, D_region.cpu().numpy()
                        )
                        pbar.update()

            pbar.close()

            D = np.zeros((n, m, len(region_counts)))
            for idxs, job in wasserstein_jobs.items():
                i, j, k = idxs
                D[i, j, k] = job.result()

        D = np.einsum("ijk,k -> ij", D, region_weights)
        D = np.sqrt(D)

        return {"wasserstein": torch.tensor(self.wasserstein2(D))}

    def wasserstein2(self, D: np.ndarray):
        return np.sqrt(ot.emd2([], [], D**2))

    def features(self, data: OpenFOAMData, stats: OpenFOAMStats):
        """Compute normalized features for each cell for pairwise comparison."""

        u = data.samples[V.U]
        p = data.samples[V.P]

        u_grid = data.grid_embedding((V.U,))
        vorticity = select_cells(curl(u_grid, data.h), data.unpadded_cell_idx)
        vorticity = torch.swapaxes(vorticity, -1, -2)

        features = torch.cat((u, vorticity, p), dim=-1)

        _, std = stats.normalizers(
            (V.U, V.CURL, V.P), mode="u:norm-std;curl:norm-std;p:mean-std"
        )
        return features / std


class MaxMeanTKEPositionMetric(nn.Module):
    def is_expensive(self):
        return False

    def forward(self, samples: OpenFOAMData, data: OpenFOAMData, stats: OpenFOAMStats):
        gt_path = data.metadata.file.parent / "max-mean-tke.npy"
        if not gt_path.is_file():
            log.warning(f"Ground-truth max-mean-TKE file {gt_path} is missing!")
            return {}
        gt = float(np.load(gt_path))

        u_sample = samples.grid_embedding((V.U,))

        # Estimating the mean flow is part of the task, so we estimate it from the
        # generated samples instead of using the ground-truth
        u_mean = u_sample.mean(dim=0)

        u_fluc = u_sample - u_mean
        # Ensure that we only look after the object in the flow
        u_fluc = u_fluc[..., 24:, :, :]
        tke = 0.5 * (u_fluc**2).sum(dim=-4)
        tke_mean_profile = tke.mean(dim=(-1, -2))
        estimate = tke_mean_profile.argmax(dim=1).float().mean() + 24

        return {"max-mean-tke-pos": (gt - estimate) ** 2}
