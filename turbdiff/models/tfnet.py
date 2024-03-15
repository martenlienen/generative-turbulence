# SPDX-FileCopyrightText: © 2024 Marten Lienen <m.lienen@tum.de> & Technical University of Munich
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import einops as eo
import torch
import torch.nn as nn

from ..data.ofles import Variable as V
from .conditioning import Conditioning, global_conditioning, local_conditioning
from .regression import RegressionTraining


class MagnitudeLoss(nn.Module):
    def __init__(self, loss):
        super(MagnitudeLoss, self).__init__()
        self.loss = loss

    def forward(self, w):
        return self.loss(w, w.detach() * 0)


class SmoothnessLoss(nn.Module):
    """From Back to Basics:
    Unsupervised Learning of Optical Flow
    via Brightness Constancy and Motion Smoothness"""

    def __init__(self, loss, delta=1):
        super(SmoothnessLoss, self).__init__()
        self.loss = loss
        self.delta = delta

    def forward(self, w):
        ldudx = self.loss(
            (w[:, 0, 1:, :] - w[:, 0, :-1, :]) / self.delta, w[:, 0, 1:, :].detach() * 0
        )
        ldudy = self.loss(
            (w[:, 0, :, 1:] - w[:, 0, :, :-1]) / self.delta, w[:, 0, :, 1:].detach() * 0
        )
        ldvdx = self.loss(
            (w[:, 1, 1:, :] - w[:, 1, :-1, :]) / self.delta, w[:, 1, 1:, :].detach() * 0
        )
        ldvdy = self.loss(
            (w[:, 1, :, 1:] - w[:, 1, :, :-1]) / self.delta, w[:, 1, :, 1:].detach() * 0
        )
        return ldudx + ldudy + ldvdx + ldvdy


class WeightedSpatialMSELoss(nn.Module):
    def __init__(self, weights):
        super(WeightedSpatialMSELoss, self).__init__()
        self.loss = nn.MSELoss(reduce=False, size_average=False)
        self.weights = weights

    def forward(self, preds, trues):
        print(self.loss(preds, trues).shape, self.weights.shape)
        return self.loss(preds, trues).mean(4).mean(3).mean(2).mean(0) * self.weights


class DivergenceLoss(nn.Module):
    def __init__(self, loss, delta=1):
        super(DivergenceLoss, self).__init__()
        self.delta = delta
        self.loss = loss

    def forward(self, preds):
        # preds: bs*2*H*W

        u = preds[:, :1]
        v = preds[:, -1:]
        u_x = field_grad(u, 0)
        v_y = field_grad(v, 1)
        div = v_y + u_x
        return self.loss(div, div.detach() * 0)


class DivergenceLoss2(nn.Module):
    def __init__(self, loss, delta=1):
        super(DivergenceLoss2, self).__init__()
        self.delta = delta
        self.loss = loss

    def forward(self, preds, trues):
        # preds: bs*steps*2*H*W
        u = preds[:, :1]
        v = preds[:, -1:]
        u_x = field_grad(u, 0)
        v_y = field_grad(v, 1)
        div_pred = v_y + u_x

        u = trues[:, :1]
        v = trues[:, -1:]
        u_x = field_grad(u, 0)
        v_y = field_grad(v, 1)
        div_true = v_y + u_x
        return self.loss(div_pred, div_true)


def vorticity(u, v):
    return field_grad(v, 0) - field_grad(u, 1)


class VorticityLoss(nn.Module):
    def __init__(self, loss):
        super(VorticityLoss, self).__init__()
        self.loss = loss

    def forward(self, preds, trues):
        u, v = trues[:, :1], trues[:, -1:]
        u_pred, v_pred = preds[:, :1], preds[:, -1:]
        return self.loss(vorticity(u, v), vorticity(u_pred, v_pred))


def field_grad(f, dim):
    # dim = 1: derivative to x direction, dim = 2: derivative to y direction
    dx = 1
    dim += 1
    N = len(f.shape)
    out = torch.zeros(f.shape)
    slice1 = [slice(None)] * N
    slice2 = [slice(None)] * N
    slice3 = [slice(None)] * N
    slice4 = [slice(None)] * N

    # 2nd order interior
    slice1[-dim] = slice(1, -1)
    slice2[-dim] = slice(None, -2)
    slice3[-dim] = slice(1, -1)
    slice4[-dim] = slice(2, None)
    out[tuple(slice1)] = (f[tuple(slice4)] - f[tuple(slice2)]) / (2 * dx)

    # 2nd order edges
    slice1[-dim] = 0
    slice2[-dim] = 0
    slice3[-dim] = 1
    slice4[-dim] = 2
    a = -1.5 / dx
    b = 2.0 / dx
    c = -0.5 / dx
    out[tuple(slice1)] = (
        a * f[tuple(slice2)] + b * f[tuple(slice3)] + c * f[tuple(slice4)]
    )
    slice1[-dim] = -1
    slice2[-dim] = -3
    slice3[-dim] = -2
    slice4[-dim] = -1
    a = 0.5 / dx
    b = -2.0 / dx
    c = 1.5 / dx

    out[tuple(slice1)] = (
        a * f[tuple(slice2)] + b * f[tuple(slice3)] + c * f[tuple(slice4)]
    )
    return out


class SpectrumLoss(nn.Module):
    def __init__(self, loss):
        super(SpectrumLoss, self).__init__()
        self.loss = loss

    def forward(self, preds, trues):
        return self.loss(tke2spectrum(trues), tke2spectrum(preds))


def TKE(preds):
    preds = preds.reshape(preds.shape[0], -1, 2, 64, 64)
    mean_flow = torch.mean(preds, dim=1).unsqueeze(1)
    tur_preds = torch.mean((preds - mean_flow) ** 2, dim=1)
    tke = (tur_preds[:, 0] + tur_preds[:, 1]) / 2
    return tke


class TKELoss(nn.Module):
    def __init__(self, loss):
        super(TKELoss, self).__init__()
        self.loss = loss

    def forward(self, preds, trues):
        return self.loss(TKE(trues), TKE(preds))


def conv(input_channels, output_channels, kernel_size, stride, dropout_rate):
    layer = nn.Sequential(
        nn.Conv3d(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
        ),
        nn.BatchNorm3d(output_channels),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Dropout(dropout_rate),
    )
    return layer


def deconv(input_channels, output_channels):
    layer = nn.Sequential(
        nn.ConvTranspose3d(
            input_channels, output_channels, kernel_size=4, stride=2, padding=1
        ),
        nn.LeakyReLU(0.1, inplace=True),
    )
    return layer


class Encoder(nn.Module):
    def __init__(self, input_channels, c_local_channels, kernel_size, dropout_rate):
        super(Encoder, self).__init__()
        self.conv1 = conv(
            input_channels,
            64,
            kernel_size=kernel_size,
            stride=2,
            dropout_rate=dropout_rate,
        )
        self.conv1_local = conv(
            c_local_channels,
            64,
            kernel_size=kernel_size,
            stride=2,
            dropout_rate=dropout_rate,
        )
        self.conv2 = conv(
            64, 128, kernel_size=kernel_size, stride=2, dropout_rate=dropout_rate
        )
        self.conv3 = conv(
            128, 256, kernel_size=kernel_size, stride=2, dropout_rate=dropout_rate
        )
        self.conv4 = conv(
            256, 512, kernel_size=kernel_size, stride=2, dropout_rate=dropout_rate
        )
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.002 / n)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, c_local):
        out_conv1 = self.conv1(x)
        if c_local is not None:
            out_conv1 = out_conv1 + self.conv1_local(c_local[None])
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        return out_conv1, out_conv2, out_conv3, out_conv4


def clip_spatial(a: torch.Tensor, b: torch.Tensor):
    return a[..., : b.shape[-3], : b.shape[-2], : b.shape[-1]]


class LES(nn.Module):
    def __init__(
        self,
        n_features: int,
        c_local_features: int,
        c_global_features: int,
        context_window: int,
        kernel_size,
        dropout_rate,
        temporal_filtering_length,
    ):
        super(LES, self).__init__()

        self.n_features = n_features
        self.c_local_features = c_local_features
        self.c_global_features = c_global_features
        self.context_window = context_window
        self.spatial_filter = nn.Conv3d(1, 1, kernel_size=3, padding=1, bias=False)
        self.temporal_filter = nn.Conv3d(
            temporal_filtering_length, 1, kernel_size=1, padding=0, bias=False
        )
        self.temporal_filtering_length = temporal_filtering_length

        filtered_dim = n_features * (context_window - temporal_filtering_length + 1)
        self.encoder_bar = Encoder(
            filtered_dim, c_local_features, kernel_size, dropout_rate
        )
        self.encoder_tilde = Encoder(
            filtered_dim, c_local_features, kernel_size, dropout_rate
        )
        self.encoder_prime = Encoder(
            filtered_dim, c_local_features, kernel_size, dropout_rate
        )

        self.deconv3 = deconv(512, 256)
        self.deconv2 = deconv(256, 128)
        self.deconv1 = deconv(128, 64)
        self.deconv0 = deconv(64, 32)
        self.output_layer = nn.Conv3d(
            32, n_features, kernel_size=kernel_size, padding=(kernel_size - 1) // 2
        )

    def forward(self, xx, C: dict[Conditioning.Type, torch.Tensor]):
        n_features = xx.shape[-4]
        # u = u_mean + u_tilde + u_prime
        # 1. Spatial filtering
        u_star = (
            self.spatial_filter(xx.flatten(end_dim=-4).unsqueeze(dim=-4))
            .squeeze(dim=-4)
            .unflatten(dim=0, sizes=xx.shape[:-3])
        )
        # 2. Residual after spatial filtering
        u_prime = xx - u_star

        # 3. Temporal filtering
        u_bar = eo.rearrange(
            self.temporal_filter(
                eo.rearrange(
                    u_star.unfold(1, size=self.temporal_filtering_length, step=1),
                    "b t f ... slice -> (b t f) slice ...",
                )
            ),
            "(b t f) 1 ... -> b t f ...",
            b=u_star.shape[0],
            f=n_features,
        )

        # 4. Residual after temporal filtering
        u_tilde = u_star[:, -u_bar.shape[1] :] - u_bar

        # Throw away u' features that don't match the u_bar and u_tilde
        u_prime = u_prime[:, -u_bar.shape[1] :]

        # Stack all time windows
        u_bar = eo.rearrange(u_bar, "b t f ... -> b (t f) ...")
        u_tilde = eo.rearrange(u_tilde, "b t f ... -> b (t f) ...")
        u_prime = eo.rearrange(u_prime, "b t f ... -> b (t f) ...")

        # U-Net
        c_local = local_conditioning(C)
        c_global = global_conditioning(C)
        if c_global is not None:
            raise RuntimeError("Global conditioning not implemented for TFNet")
        out_conv1_bar, out_conv2_bar, out_conv3_bar, out_conv4_bar = self.encoder_bar(
            u_bar, c_local
        )
        (
            out_conv1_tilde,
            out_conv2_tilde,
            out_conv3_tilde,
            out_conv4_tilde,
        ) = self.encoder_tilde(u_tilde, c_local)
        (
            out_conv1_prime,
            out_conv2_prime,
            out_conv3_prime,
            out_conv4_prime,
        ) = self.encoder_prime(u_prime, c_local)

        out_deconv3 = self.deconv3(out_conv4_bar + out_conv4_tilde + out_conv4_prime)
        out_conv3 = out_conv3_bar + out_conv3_tilde + out_conv3_prime
        out_deconv2 = self.deconv2(out_conv3 + clip_spatial(out_deconv3, out_conv3))
        out_conv2 = out_conv2_bar + out_conv2_tilde + out_conv2_prime
        out_deconv1 = self.deconv1(out_conv2 + clip_spatial(out_deconv2, out_conv2))
        out_conv1 = out_conv1_bar + out_conv1_tilde + out_conv1_prime
        out_deconv0 = self.deconv0(out_conv1 + clip_spatial(out_deconv1, out_conv1))
        out = self.output_layer(clip_spatial(out_deconv0, xx))
        return out


class TFNetTraining(RegressionTraining):
    def __init__(
        self,
        data_dir: Path,
        samples_root: Path,
        normalization_mode: str,
        variables: tuple[V, ...],
        temporal_filtering_length: int,
        context_window: int,
        unroll_steps: int,
        eval_unroll_steps: int,
        sample_steps: list[int],
        main_sample_step: int,
        learning_rate=0.001,
        dropout_rate=0,
        kernel_size=3,
        cell_type_features: bool = True,
        cell_type_embedding_type: str = "learned",
        cell_type_embedding_dim: int = 8,
        cell_pos_features: bool = False,
        compute_expensive_sample_metrics: bool = True,
    ):
        super().__init__(
            data_dir=data_dir,
            samples_root=samples_root,
            variables=variables,
            context_window=context_window,
            unroll_steps=unroll_steps,
            eval_unroll_steps=eval_unroll_steps,
            sample_steps=sample_steps,
            main_sample_step=main_sample_step,
            normalization_mode=normalization_mode,
            cell_type_features=cell_type_features,
            cell_type_embedding_type=cell_type_embedding_type,
            cell_type_embedding_dim=cell_type_embedding_dim,
            cell_pos_features=cell_pos_features,
            compute_expensive_sample_metrics=compute_expensive_sample_metrics,
        )

        self.temporal_filtering_length = temporal_filtering_length
        self.context_window = context_window
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.kernel_size = kernel_size

        self.model = LES(
            n_features=sum([v.dims for v in variables]),
            c_local_features=self.conditioning.local_conditioning_dim,
            c_global_features=self.conditioning.global_conditioning_dim,
            context_window=self.context_window,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
            temporal_filtering_length=temporal_filtering_length,
        )

        self.loss = nn.MSELoss()

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return opt
