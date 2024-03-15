# SPDX-FileCopyrightText: © 2024 Marten Lienen <m.lienen@tum.de> & Technical University of Munich
#
# SPDX-License-Identifier: MIT

import math
from dataclasses import dataclass
from functools import partial

import numpy as np
import scipy.optimize as so
import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from torch import nn
from tqdm.auto import tqdm

from turbdiff.metrics import curl, divergence

from ..data.ofles import Variable as V
from ..data.ofles import split_channels
from ..sequential import KwargsSequential
from ..utils import get_logger
from .attention import fused_attention
from .conditioning import Conditioning, global_conditioning, local_conditioning
from .utils import broadcast_right, ravel_cells, where_cells

log = get_logger()


@dataclass
class ModelPrediction:
    noise: torch.Tensor
    x_start: torch.Tensor
    mean: torch.Tensor
    log_var: torch.Tensor


# small helper modules


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()

        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


def pad_to_multiple_of(x: torch.Tensor, n: int, *, mode: str):
    h, w, d = x.shape[-3:]
    h_pad = n - h % n
    w_pad = n - w % n
    d_pad = n - d % n
    if min(h_pad, w_pad, d_pad) > 0:
        return F.pad(x, (0, d_pad, 0, w_pad, 0, h_pad), mode=mode), (
            h_pad,
            w_pad,
            d_pad,
        )
    else:
        return x, (0, 0, 0)


def unpad(x: torch.Tensor, padding: tuple[int, int, int]):
    if min(padding) > 0:
        h_pad, w_pad, d_pad = padding
        return x[..., :-h_pad, :-w_pad, :-d_pad]
    else:
        return x


class PreNorm(nn.Module):
    def __init__(self, norm: nn.Module, fn: nn.Module, enabled: bool = True):
        super().__init__()

        self.norm = norm
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


# sinusoidal positional embeds


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class NyquistFrequencyEmbedding(nn.Module):
    """Sine-cosine embedding for timesteps that scales from 1/8 to a (< 1) multiple of
    the Nyquist frequency.

    We choose 1/8 as the slowest frequency so that the slowest-varying embedding varies
    roughly lineary across [0, 2pi] as the relative error between x and sin(x) on [0,
    2pi / 8] is at most 2.5%. The Nyquist frequency is the largest frequency that one
    can sample at T steps without aliasing, so one could assume that to be a great
    choice for the highest frequency but sampling sine and cosine at the Nyquist
    frequency would result in constant (and therefore uninformative) 1 and 0 features,
    so we Nyquist/2 is a better choice. However, Nyquist/2 (which is T/2) leads to the
    evaluation points of the fastest varying points to overlap, so that those features
    would only take a small number of values, such as 2 or 4. In combination with the
    other points, these embeddings would of course still be distinguishable but by
    choosing an irrational fastest frequency, we can get unique embeddings also in the
    fastest-varying dimension for all timepoints. We choose arbitrarily 1/phi where phi
    is the golden ratio.
    """

    def __init__(self, dim: int, timesteps: int):
        super().__init__()

        assert dim % 2 == 0

        T = timesteps
        k = dim // 2

        # Nyquist frequency for T samples per cycle
        nyquist_frequency = T / 2

        golden_ratio = (1 + np.sqrt(5)) / 2
        frequencies = np.geomspace(1 / 8, nyquist_frequency / (2 * golden_ratio), num=k)

        # Sample every frequency twice, once shifted by pi/2 to get cosine
        scale = np.repeat(2 * np.pi * frequencies / timesteps, 2)
        bias = np.tile(np.array([0, np.pi / 2]), k)

        self.register_buffer(
            "scale", torch.tensor(scale, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "bias", torch.tensor(bias, dtype=torch.float32), persistent=False
        )

    def forward(self, t):
        return torch.addcmul(self.bias, self.scale, t[..., None]).sin()


# building block modules


class Block(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        actfn,
        norm_klass=None,
    ):
        super().__init__()

        self.conv = nn.Conv3d(dim, dim_out, 3, padding=1, padding_mode="replicate")
        self.norm = norm_klass(dim_out)
        self.act = actfn()

    def forward(self, x, scale_shift=None):
        x = self.conv(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = torch.addcmul(shift, scale + 1, x)

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim_in, dim_out, *, c_dim: int, actfn, norm_klass):
        super().__init__()

        self.project_onto_scale_shift = nn.Linear(c_dim, dim_out * 2)

        self.block1 = Block(dim_in, dim_out, actfn=actfn, norm_klass=norm_klass)
        self.block2 = Block(dim_out, dim_out, actfn=actfn, norm_klass=norm_klass)
        self.conv = nn.Conv3d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()

    def forward(self, x, c):
        c = rearrange(self.project_onto_scale_shift(c), "... c -> ... c 1 1 1")
        scale_shift = c.chunk(2, dim=-4)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)

        return h + self.conv(x)


class LinearAttention(nn.Module):
    """Linear attention as proposed in [1].

    [1] "Efficient Attention: Attention with Linear Complexities", Zhuoran et al.
    """

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()

        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv3d(dim, 3 * hidden_dim, 1, bias=False)

        self.combine_heads = nn.Sequential(*[nn.Conv3d(hidden_dim, dim, 1)])

    def forward(self, x):
        b, c, h, w, d = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(
            qkv, "b (qkv h c) x y z -> qkv b h c (x y z)", h=self.heads, qkv=3
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        out = torch.einsum("bhci, bhdi, bhck -> bhdk", k, v, q)
        out = rearrange(
            out, "b h c (x y z) -> b (h c) x y z", h=self.heads, x=h, y=w, z=d
        )
        return self.combine_heads(out)


class LocalAttention(nn.Module):
    def __init__(self, dim: int, window_size: int, heads: int = 4, dim_head: int = 32):
        super().__init__()

        self.dim = dim
        self.window_size = window_size
        self.heads = heads
        self.dim_head = dim_head

        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, 1, bias=False)

        self.merge_heads = nn.Sequential(*[nn.Conv3d(hidden_dim, dim, 1)])

    def forward(self, x):
        qkv = self.to_qkv(x)

        needs_padding = any(qkv.shape[i] % self.window_size != 0 for i in range(-3, 0))
        if needs_padding:
            # "constant" reduces the impact of the padded cells in softmax compared to
            # "replicate"
            qkv, padding = pad_to_multiple_of(qkv, self.window_size, mode="constant")

        wx = wy = wz = self.window_size
        q, k, v = rearrange(
            qkv,
            "b (qkv h c) (x wx) (y wy) (z wz) -> qkv (b x y z) h (wx wy wz) c",
            h=self.heads,
            wx=wx,
            wy=wy,
            wz=wz,
            qkv=3,
        )
        q, k, v = map(lambda t: t.contiguous(), [q, k, v])

        out = fused_attention(q, k, v)

        out = rearrange(
            out,
            "(b x y z) h (wx wy wz) c -> b (h c) (x wx) (y wy) (z wz)",
            x=qkv.shape[-3] // wx,
            y=qkv.shape[-2] // wy,
            z=qkv.shape[-1] // wz,
            wx=wx,
            wy=wy,
            wz=wz,
        )

        if needs_padding:
            out = unpad(out, padding)

        return self.merge_heads(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv3d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w, d = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(
                t, "b (h c) x y z -> b h (x y z) c", h=self.heads
            ).contiguous(),
            qkv,
        )

        out = fused_attention(q, k, v)

        out = rearrange(out, "b h (x y z) d -> b (h d) x y z", x=h, y=w, z=d)
        return self.to_out(out)


# model


def expand_as(x: torch.Tensor, y: torch.Tensor, dim: int):
    """Expand `y` to the same shape as `x` except in dimension `dim`.

    Any existing dimension in `y` has to equal the corresponding one in `x`.
    """

    assert x.ndim >= y.ndim
    shape = list(x.shape)
    shape[dim] = -1
    return y.expand(shape)


class UNet(nn.Module):
    """
    A general U-Net structure with interpolation instead of max-pool and transposed
    convolutions.
    """

    def __init__(
        self,
        downsampling_blocks: list[nn.Module],
        upsampling_blocks: list[nn.Module],
        center_block: nn.Module,
        *,
        downsampling_factor: float = 2.0,
    ):
        super().__init__()

        assert len(downsampling_blocks) == len(upsampling_blocks)

        self.downsampling_blocks = nn.ModuleList(downsampling_blocks)
        self.upsampling_blocks = nn.ModuleList(upsampling_blocks)
        self.center_block = center_block
        self.downsampling_factor = downsampling_factor

        self.scale_factor = 1 / downsampling_factor

    def forward(self, x: torch.Tensor, *args, **kwargs):
        skips = []

        for block in self.downsampling_blocks:
            x = block(x, *args, **kwargs)
            skips.append(x)
            # Ensure that no dimension is reduced below the kernel size of 3
            downsample_size = [max(int(s * self.scale_factor), 3) for s in x.shape[-3:]]
            x = nn.functional.interpolate(
                x, size=downsample_size, mode="trilinear", align_corners=True
            )

        x = self.center_block(x, *args, **kwargs)

        for block in self.upsampling_blocks:
            x_skip = skips.pop()
            x = nn.functional.interpolate(
                x, size=x_skip.shape[-3:], mode="trilinear", align_corners=True
            )
            x = block(torch.cat((x, x_skip), dim=-4), *args, **kwargs)

        return x


class GeometryEmbedding(nn.Module):
    def __init__(self, in_features, out_features, actfn):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.actfn = actfn

        self.extract_features = nn.Sequential(
            nn.Conv3d(in_features, out_features, kernel_size=5, stride=5),
            actfn(),
            nn.Conv3d(out_features, out_features, kernel_size=5, stride=1),
            actfn(),
            nn.Conv3d(out_features, out_features, kernel_size=5, stride=5),
        )

    def forward(self, c_local: torch.Tensor):
        # Select the front slice containing the object in the flow
        c_local = torch.narrow(c_local, dim=-3, start=0, length=50)

        return self.extract_features(c_local).mean(dim=(-3, -2, -1))


class DenoisingModel(nn.Module):
    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        c_local_features: int,
        c_global_features: int,
        timesteps: int,
        dim: int,
        u_net_levels: int,
        actfn=nn.SiLU,
        norm_type: str = "instance",
        with_geometry_embedding: bool = False,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.c_local_features = c_local_features
        self.c_global_features = c_global_features
        self.dim = dim
        self.timesteps = timesteps
        self.u_net_levels = u_net_levels
        self.with_geometry_embedding = with_geometry_embedding

        if norm_type == "instance":
            norm_klass = lambda dim: nn.GroupNorm(dim, dim)
        elif norm_type == "layer":
            norm_klass = lambda dim: nn.GroupNorm(1, dim)
        elif norm_type == "group":
            norm_klass = lambda dim: nn.GroupNorm(8, dim)
        else:
            raise RuntimeError(f"Unknown norm type {norm_type}")

        self.encode_x = nn.Conv3d(in_features, dim, 1)
        c_local_dim = 0
        if c_local_features > 0:
            self.encode_c_local = nn.Conv3d(c_local_features, dim, 1)
            c_local_dim += dim
        c_dim = dim
        self.encode_t = NyquistFrequencyEmbedding(dim, timesteps)
        if c_global_features > 0:
            self.encode_c_global = nn.Linear(c_global_features, dim)
            c_dim += dim
        if with_geometry_embedding and c_local_features > 0:
            self.geometry_embedding = GeometryEmbedding(c_local_features, dim, actfn)
            c_dim += dim

        self.process_c = nn.Sequential(
            nn.Linear(c_dim, 4 * c_dim),
            actfn(),
            nn.Linear(4 * c_dim, c_dim),
            actfn(),
        )

        resnet_block = partial(
            ResnetBlock, c_dim=c_dim, actfn=actfn, norm_klass=norm_klass
        )

        self.decode = KwargsSequential(
            resnet_block(dim, dim), nn.Conv3d(dim, out_features, 1)
        )

        downsampling_blocks = [resnet_block(dim + c_local_dim, dim * 2)] + [
            resnet_block(dim * 2**i, dim * 2 ** (i + 1)) for i in range(1, u_net_levels)
        ]
        upsampling_blocks = [
            resnet_block(2 * dim * 2 ** (i + 1), dim * 2**i)
            for i in reversed(range(u_net_levels))
        ]
        center_dim = dim * 2**u_net_levels
        center_block = KwargsSequential(
            resnet_block(center_dim, center_dim),
            Residual(PreNorm(norm_klass(center_dim), Attention(center_dim))),
            resnet_block(center_dim, center_dim),
        )
        self.u_net = UNet(downsampling_blocks, upsampling_blocks, center_block)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, C: dict[Conditioning.Type, torch.Tensor]
    ):
        # TODO: Encode c_local and c_global only once during sampling
        c_local = local_conditioning(C)

        c_parts = [self.encode_t(t)]
        c_global = global_conditioning(C)
        if c_global is not None:
            c_parts.append(self.encode_c_global(c_global))
        if self.with_geometry_embedding:
            if c_local is None:
                log.warning("Geometry embedding requested but no local conditioning")
            else:
                batch_size = x.shape[0]
                c_parts.append(self.geometry_embedding(c_local).expand((batch_size, -1)))
        c = self.process_c(torch.cat(c_parts, dim=-1))

        x = self.encode_x(x)
        if c_local is not None:
            batch_size = x.shape[0]
            x = torch.cat(
                (x, self.encode_c_local(c_local).expand((batch_size, -1, -1, -1, -1))),
                dim=-4,
            )

        x = self.u_net(x, c=c)

        return self.decode(x, c=c)


# gaussian diffusion trainer class


def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def log_linear_beta_schedule(timesteps):
    """A version of the linear beta schedule that works for arbitrary timesteps."""

    log_alphas_cumprod_T = np.log(1e-6)
    T, log_T = timesteps, np.log(timesteps)
    one_to_T = np.arange(1, T + 1)

    def f(alpha_T):
        return (
            np.log(T + one_to_T * (alpha_T - 1)).sum() - T * log_T - log_alphas_cumprod_T
        )

    alpha_T = so.bisect(f, 1e-10, 1.0)
    alphas = (T + one_to_T * (alpha_T - 1)) / T
    betas = 1 - alphas
    return torch.tensor(betas)


def log_snr_linear_beta_schedule(timesteps, snr_1=1e3, snr_T=1e-5):
    """A beta schedule that decays the log-SNR linearly."""

    T = timesteps
    log_snr_1 = np.log(snr_1)
    log_snr_T = np.log(snr_T)

    alpha_cumprods = []
    for t in range(1, T + 1):

        def f(alpha_cumprod):
            return (
                np.log(alpha_cumprod)
                - np.log1p(-alpha_cumprod)
                - ((T - t) * log_snr_1 + (t - 1) * log_snr_T) / (T - 1)
            )

        alpha_cumprods.append(so.bisect(f, 1e-8, 1.0 - 1e-8))
    alpha_cumprods = np.array(alpha_cumprods)

    alphas = np.concatenate(
        (alpha_cumprods[:1], alpha_cumprods[1:] / alpha_cumprods[:-1])
    )
    betas = 1 - alphas
    return torch.tensor(betas)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (
        v_end - v_start
    )
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def normal_log_lk(x, mean, log_var):
    """Log-likelihood of `x` under the given normal distribution."""
    log_2pi = math.log(2 * math.pi)
    return -0.5 * (log_var + log_2pi + (x - mean) ** 2 * torch.exp(-log_var))


def batch_mean(x: torch.Tensor):
    return reduce(x, "b ... -> b", "mean")


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        timesteps: int = 1000,
        loss_type: str = "l2",
        beta_schedule: str = "sigmoid",
        clip_denoised: bool = False,
        noise_bcs: bool = False,
        learned_variances: bool = False,
        elbo_weight: float | None = None,
        detach_elbo_mean: bool = True,
    ):
        super().__init__()

        self.model = model
        self.clip_denoised = clip_denoised
        self.noise_bcs = noise_bcs
        self.learned_variances = learned_variances
        self.elbo_weight = elbo_weight
        self.detach_elbo_mean = detach_elbo_mean

        if beta_schedule == "linear":
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == "log-linear":
            beta_schedule_fn = log_linear_beta_schedule
        elif beta_schedule == "log-snr-linear":
            beta_schedule_fn = log_snr_linear_beta_schedule
        elif beta_schedule == "cosine":
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == "sigmoid":
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f"unknown beta schedule {beta_schedule}")

        betas = beta_schedule_fn(timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.num_timesteps = timesteps
        self.loss_type = loss_type

        # sampling related parameters

        def register_buffer(name, val):
            self.register_buffer(name, val.to(torch.float32), persistent=False)

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.rsqrt(alphas_cumprod))
        register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        register_buffer("log_betas", torch.log(betas))
        # Numerically stable version of
        #
        #     log(betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        posterior_log_var = (
            self.log_betas
            + torch.log1p(-alphas_cumprod_prev)
            - torch.log1p(-alphas_cumprod)
        )

        # Predict a distribution for the noiseless data with a variance that shrinks at
        # the same rate as for the diffusion process. Otherwise one interpolation point
        # (posterior_log_var[0]) will be at -inf.
        posterior_log_var[0] = self.log_betas[0] * (
            posterior_log_var[1] / self.log_betas[1]
        )
        register_buffer("posterior_log_var", posterior_log_var)

        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            broadcast_right(self.sqrt_recip_alphas_cumprod[t], x_t) * x_t
            - broadcast_right(self.sqrt_recipm1_alphas_cumprod[t], x_t) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            broadcast_right(self.sqrt_recip_alphas_cumprod[t], x_t) * x_t - x0
        ) / broadcast_right(self.sqrt_recipm1_alphas_cumprod[t], x_t)

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            broadcast_right(self.posterior_mean_coef1[t], x_t) * x_start
            + broadcast_right(self.posterior_mean_coef2[t], x_t) * x_t
        )
        posterior_log_var = broadcast_right(self.posterior_log_var[t], x_t)
        return posterior_mean, posterior_log_var

    def model_predictions(self, x_t, t, C, cell_idx, clip_x_start=False):
        model_output = self.model(x_t, t, C)
        if self.learned_variances:
            pred_noise, variance_weights = model_output.chunk(2, dim=1)

            log_betas = broadcast_right(self.log_betas[t], variance_weights)
            posterior_log_var = broadcast_right(
                self.posterior_log_var[t], variance_weights
            )
            log_var = torch.lerp(
                log_betas, posterior_log_var, torch.sigmoid(variance_weights)
            )
        else:
            pred_noise, log_var = model_output, self.log_betas[t]

        x_start = self.predict_start_from_noise(x_t, t, pred_noise)
        if not self.noise_bcs:
            x_start = where_cells(cell_idx, x_start, x_t)

        if clip_x_start:
            x_start = torch.clamp(x_start, min=-1.0, max=1.0)

        mean, _ = self.q_posterior(x_start, x_t, t)

        return ModelPrediction(
            noise=pred_noise, x_start=x_start, mean=mean, log_var=log_var
        )

    @torch.no_grad()
    def p_sample(self, x_t, t: int, C, cell_idx):
        batch_size = x_t.shape[0]
        batched_times = x_t.new_tensor(t, dtype=torch.long).expand(batch_size)
        pred = self.model_predictions(
            x_t, batched_times, C, cell_idx, clip_x_start=self.clip_denoised
        )
        return pred.mean, pred.log_var

    @torch.no_grad()
    def p_sample_loop(
        self,
        x_bcs,
        C,
        cell_idx,
        pbar=False,
        start_from: int | None = None,
    ):
        if start_from is None:
            x_t = torch.randn_like(x_bcs)
        else:
            start_from_ = x_bcs.new_tensor(start_from - 1, dtype=torch.long).expand(
                x_bcs.shape[0]
            )
            x_t = self.q_sample(x_bcs, start_from_, torch.randn_like(x_bcs))
        if not self.noise_bcs:
            x_t = where_cells(cell_idx, x_t, x_bcs)

        if start_from is None:
            T = self.num_timesteps
        else:
            T = start_from
        ts = reversed(range(0, T))
        if pbar:
            ts = tqdm(ts, desc="sampling loop time step", total=T, position=1)
        batch_size = x_t.shape[0]
        for t in ts:
            mean_tm1, log_var_tm1 = self.p_sample(x_t, t, C, cell_idx)

            if t == 0:
                # Return the mean of the predicted distribution
                x_t = mean_tm1
            else:
                noise = torch.randn_like(x_t)
                if not self.noise_bcs:
                    noise = where_cells(cell_idx, noise)
                std_tm1 = (log_var_tm1 / 2).exp()
                x_t = mean_tm1 + broadcast_right(std_tm1, noise) * noise

                if self.noise_bcs:
                    t_ = x_t.new_tensor(t, dtype=torch.long).expand(batch_size)
                    x_t = where_cells(
                        cell_idx, x_t, self.q_sample(x_bcs, t_, torch.randn_like(x_bcs))
                    )

        # Fix boundary condition values in the end regardless of if we denoised them
        x_t = where_cells(cell_idx, x_t, x_bcs)

        return x_t

    def q_sample(self, x_start, t, noise):
        return (
            broadcast_right(self.sqrt_alphas_cumprod[t], x_start) * x_start
            + broadcast_right(self.sqrt_one_minus_alphas_cumprod[t], x_start) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == "l1":
            return F.l1_loss
        elif self.loss_type == "l2":
            return F.mse_loss
        else:
            raise ValueError(f"invalid loss type {self.loss_type}")

    def p_losses(self, x_start, t, C, metadata, variables):
        # Generate noise inside the simulation domain
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
        if not self.noise_bcs:
            x_t = where_cells(metadata.cell_idx, x_t, x_start)

        # predict and take gradient step

        pred = self.model_predictions(
            x_t, t, C, metadata.cell_idx, clip_x_start=self.clip_denoised
        )
        simple_loss = self.loss_fn(pred.noise, noise, reduction="none")

        # Consider loss only for in-domain cells
        simple_loss = ravel_cells(simple_loss)[..., metadata.cell_idx]

        simple_loss = batch_mean(simple_loss)

        loss = simple_loss.mean()
        if self.elbo_weight is not None and self.learned_variances:
            true_mean, true_log_var = self.q_posterior(x_start, x_t, t)

            model_mean = pred.mean
            if self.detach_elbo_mean:
                # Only learn the variances through the ELBO as in the improved-DDPM
                # paper
                model_mean = model_mean.detach()

            kl = normal_kl(true_mean, true_log_var, model_mean, pred.log_var)
            log_lk = normal_log_lk(x_t, model_mean, pred.log_var)

            # Restrict to in-domain cells
            kl = ravel_cells(kl)[..., metadata.cell_idx]
            log_lk = ravel_cells(log_lk)[..., metadata.cell_idx]

            elbo = torch.where(t == 0, -batch_mean(log_lk), batch_mean(kl))
            loss = loss + self.elbo_weight * elbo.mean()

        return loss, t

    def forward(self, x, *args, **kwargs):
        batch_size = x.shape[0]
        t = torch.randint(
            0, self.num_timesteps, (batch_size,), device=x.device, dtype=torch.long
        )

        loss, t = self.p_losses(x, t, *args, **kwargs)

        return loss, t
