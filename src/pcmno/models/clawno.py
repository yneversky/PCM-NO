from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fno import SpectralConv2d


class MLP2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int):
        super().__init__()
        self.first = nn.Conv2d(in_channels, hidden_channels, 1)
        self.second = nn.Conv2d(hidden_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.second(F.gelu(self.first(x)))


class PeriodicClawFNO2d(nn.Module):
    """Matched ClawNO-style divergence-free parameterization for P-NS."""

    def __init__(
        self,
        resolution: int,
        length: float = 2 * math.pi,
        modes: int = 16,
        width: int = 48,
        layers: int = 4,
    ) -> None:
        super().__init__()
        self.resolution = int(resolution)
        self.length = float(length)
        self.fc0 = nn.Conv2d(5, width, 1)
        self.spectral_layers = nn.ModuleList(
            [SpectralConv2d(width, width, modes, modes) for _ in range(layers)]
        )
        self.mlp_layers = nn.ModuleList(
            [MLP2d(width, width, width) for _ in range(layers)]
        )
        self.pointwise_layers = nn.ModuleList(
            [nn.Conv2d(width, width, 1) for _ in range(layers)]
        )
        self.norm = nn.InstanceNorm2d(width)
        self.q = MLP2d(width, 1, width * 4)
        freq = torch.fft.fftfreq(self.resolution, d=self.length / self.resolution)
        freq = freq * (2 * math.pi)
        self.register_buffer("kx", freq.view(self.resolution, 1))
        self.register_buffer("ky", freq.view(1, self.resolution))

    @staticmethod
    def grid(batch: int, height: int, width: int, device, dtype):
        gx = torch.linspace(0, 1, height, device=device, dtype=dtype)
        gy = torch.linspace(0, 1, width, device=device, dtype=dtype)
        return (
            gx.view(1, 1, height, 1).expand(batch, 1, height, width),
            gy.view(1, 1, 1, width).expand(batch, 1, height, width),
        )

    def fixed_divergence_free_layer(self, potential: torch.Tensor) -> torch.Tensor:
        potential_hat = torch.fft.fft2(potential[:, 0], norm="ortho")
        ux = torch.fft.ifft2(1j * self.ky * potential_hat, norm="ortho").real
        uy = torch.fft.ifft2(-1j * self.kx * potential_hat, norm="ortho").real
        return torch.stack((ux, uy), dim=1)

    def forward(self, state: torch.Tensor, viscosity: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = state.shape
        if height != self.resolution or width != self.resolution:
            raise ValueError(
                f"ClawNO was built for {self.resolution}x{self.resolution}, "
                f"received {height}x{width}."
            )
        viscosity_map = viscosity.to(state.device, state.dtype).view(batch, 1, 1, 1)
        viscosity_map = viscosity_map.expand(batch, 1, height, width)
        gx, gy = self.grid(batch, height, width, state.device, state.dtype)
        hidden = self.fc0(torch.cat((state, viscosity_map, gx, gy), dim=1))
        for spectral, mlp, pointwise in zip(
            self.spectral_layers, self.mlp_layers, self.pointwise_layers
        ):
            filtered = self.norm(spectral(self.norm(hidden)))
            hidden = F.gelu(mlp(filtered) + pointwise(hidden))
        return self.fixed_divergence_free_layer(self.q(hidden))
