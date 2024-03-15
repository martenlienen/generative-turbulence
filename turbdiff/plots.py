# SPDX-FileCopyrightText: © 2024 Marten Lienen <m.lienen@tum.de> & Technical University of Munich
#
# SPDX-License-Identifier: MIT

import concurrent.futures as cf
import functools
import math
from io import BytesIO
from pathlib import Path

import matplotlib
import matplotlib.colors as mc
import matplotlib.pyplot as pp
import numpy as np
import PIL
import pytorch_lightning as pl
import wandb
from loky import get_reusable_executor

from .data.ofles import OpenFOAMData, OpenFOAMDataRepository, split_channels
from .data.ofles import Variable as V
from .models.diffusion import DiffusionTraining
from .models.metrics import WassersteinTKE
from .models.regression import RegressionTraining
from .utils import get_logger

log = get_logger()

module_dir = Path(__file__).absolute().parent
pp.style.use(module_dir / "turbdiff.mplstyle")


def render_figure(fig: pp.Figure) -> PIL.Image:
    """Render a matplotlib figure into a Pillow image."""
    buf = BytesIO()
    fig.savefig(buf, **{"format": "rgba"})
    return PIL.Image.frombuffer(
        "RGBA", fig.canvas.get_width_height(), buf.getbuffer(), "raw", "RGBA", 0, 1
    )


def render_and_close(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        # Use non-interative backend to avoid errors in multi-process rendering. We
        # initialize it here, to ensure that it is set in every subprocess.
        matplotlib.use("Agg")

        fig = f(*args, **kwargs)
        if fig is None:
            return None
        img = render_figure(fig)
        pp.close(fig)
        return img

    return wrapper


@render_and_close
def plot_slice(sample: OpenFOAMData, data: OpenFOAMData, *, dim: int = -2):
    sample_vars = sample.variables
    sample_v = split_channels(sample.grid_embedding(sample_vars), sample_vars)
    data_v = split_channels(data.grid_embedding(sample_vars), sample_vars)

    domain_shape = sample_v[sample_vars[0]].shape
    width = 6
    aspect_ratio = domain_shape[-3] / max(domain_shape[-2:])
    height = 1.5 * width / aspect_ratio
    fig = pp.figure(figsize=(width, height), dpi=300, constrained_layout=True)
    axes = fig.subplots(ncols=2, nrows=len(sample_vars), squeeze=False)
    axes[0, 0].set_title("Sample")
    axes[0, 1].set_title("Data")
    for i, v in enumerate(sample_vars):
        slices = []
        for x in [sample_v[v], data_v[v]]:
            slice = x.select(dim, x.shape[dim] // 2).cpu().numpy()[0]
            if v.dims == 1:
                slice = slice[0]
            else:
                slice = np.linalg.norm(slice, axis=0)
            slices.append(slice)

        if v in (V.DIVERGENCE, V.CURL, V.ENSTROPHY):
            # These derived variables are based on finite difference estimates of the
            # first derivative of the velocity along the x-axis, i.e. in the outlet
            # direction of the flow, but there is no prescribed value for the velocity
            # at the outlet and the finite difference derivative estimates are invalid.
            # Therefore, we cut off the slice of cells in front of the outlet.
            slices = [s[:-1] for s in slices]

        if v in (V.P, V.DIVERGENCE):
            norm = mc.CenteredNorm(vcenter=0, halfrange=np.abs(slices[-1]).max())
            cmap = "coolwarm"
        else:
            norm = mc.Normalize(vmin=slices[-1].min(), vmax=slices[-1].max())
            cmap = "cividis"

        axes[i, 0].set_ylabel(v.name)
        for j, slice in enumerate(slices):
            axes[i, j].imshow(
                slice.T, origin="lower", interpolation="none", norm=norm, cmap=cmap
            )

    return fig


@render_and_close
def plot_tke_spectrum(case_data):
    n_cases = len(case_data)
    n_cols = 3
    n_rows = math.ceil(n_cases / n_cols)
    fig = pp.figure(figsize=(2 * n_cols, 2 * n_rows), constrained_layout=True, dpi=300)
    axes = fig.subplots(n_rows, n_cols, squeeze=False, sharex=True)

    color_sample = "b"
    color_data = "g"
    for case_idx, ax, case_name in zip(
        range(n_cases), axes.ravel(), sorted(case_data.keys())
    ):
        log_tke_sample, log_tke_data, k = case_data[case_name]
        tke_mean_sample = log_tke_sample.mean(dim=0).exp()
        tke_mean_data = log_tke_data.mean(dim=0).exp()

        ax.plot(k, tke_mean_sample, c=color_sample, label="Sample")
        ax.plot(k, tke_mean_data, c=color_data, label="Data")

        for i in range(len(log_tke_sample)):
            ax.plot(
                k, log_tke_sample[i].exp(), lw=0.5, ls="--", c=color_sample, alpha=0.5
            )
        for i in range(len(log_tke_data)):
            ax.plot(k, log_tke_data[i].exp(), lw=0.5, ls="--", c=color_data, alpha=0.5)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$k$")
        ax.set_ylabel(r"$E(k)$")

        ax.set_title(case_name)
        if case_idx == 0:
            ax.legend(loc="lower left")

    for ax in axes.ravel()[-(n_cols * n_rows - n_cases) :]:
        ax.set_axis_off()

    return fig


class OpenFOAMPlots(pl.Callback):
    def __init__(self, data_dir: Path):
        super().__init__()

        self.data_dir = data_dir

    def on_validation_end(self, trainer, task):
        metrics = self.generate_plots(task, "val")
        trainer.logger.log_metrics(metrics, step=trainer.global_step)

    def on_test_end(self, trainer, task):
        metrics = self.generate_plots(task, "test")
        trainer.logger.log_metrics(metrics, step=trainer.global_step)

    def generate_plots(self, task, phase: str):
        metrics = {}
        if isinstance(task, DiffusionTraining):
            metrics.update(
                self.plot_tke_spectra(
                    getattr(task, f"{phase}_sample_metrics"), phase=phase
                )
            )
            metrics.update(
                self.plot_slices(getattr(task, f"{phase}_sample_store"), phase=phase)
            )
        elif isinstance(task, RegressionTraining):
            for step, sample_metrics, sample_store in zip(
                task.sample_steps,
                getattr(task, f"{phase}_sample_metrics"),
                getattr(task, f"{phase}_sample_stores"),
            ):
                metrics.update(
                    self.plot_tke_spectra(sample_metrics, phase=phase, step=step)
                )
                metrics.update(self.plot_slices(sample_store, phase=phase, step=step))

        metrics = {
            key: wandb.Image(value.result()) if isinstance(value, cf.Future) else value
            for key, value in metrics.items()
        }
        return metrics

    def plot_tke_spectra(self, metrics_collection, *, phase: str, step=None):
        prefix = "" if step is None else f"{step}/"
        spectra = {}
        executor = get_reusable_executor()
        for metric in metrics_collection.metrics:
            if not isinstance(metric, WassersteinTKE):
                continue

            for region, case_data in metric.case_data.items():
                if len(case_data) > 0:
                    spectra[f"{phase}/{prefix}tke-{region}-plot"] = executor.submit(
                        plot_tke_spectrum, case_data
                    )
        return spectra

    def plot_slices(self, sample_store, *, phase: str, step=None):
        prefix = "" if step is None else f"{step}/"
        slices = {}
        case_names = sample_store.case_names
        executor = get_reusable_executor()
        for case_name in case_names:
            repo = OpenFOAMDataRepository(
                [self.data_dir / case_name / "data.h5"], sample_store.variables
            )
            data = repo.read(0, [-1])
            sample = sample_store.load_samples(data.metadata, range=0)

            slices[f"{phase}/{prefix}{case_name}/y-slice"] = executor.submit(
                plot_slice, sample, data, dim=-2
            )
            if not data.two_dimensional:
                slices[f"{phase}/{prefix}{case_name}/z-slice"] = executor.submit(
                    plot_slice, sample, data, dim=-1
                )
        return slices
