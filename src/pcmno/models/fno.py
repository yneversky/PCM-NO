from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)
        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale
            * torch.randn(
                in_channels, out_channels, modes1, modes2, dtype=torch.cfloat
            )
        )
        self.weights2 = nn.Parameter(
            scale
            * torch.randn(
                in_channels, out_channels, modes1, modes2, dtype=torch.cfloat
            )
        )

    @staticmethod
    def complex_multiply(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", x, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(
            batch,
            self.out_channels,
            height,
            width // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )
        modes1 = min(self.modes1, height // 2)
        modes2 = min(self.modes2, width // 2 + 1)
        out_ft[:, :, :modes1, :modes2] = self.complex_multiply(
            x_ft[:, :, :modes1, :modes2], self.weights1[:, :, :modes1, :modes2]
        )
        out_ft[:, :, -modes1:, :modes2] = self.complex_multiply(
            x_ft[:, :, -modes1:, :modes2], self.weights2[:, :, :modes1, :modes2]
        )
        return torch.fft.irfft2(out_ft, s=(height, width), norm="ortho")


class PNSFNO2d(nn.Module):
    """FNO backbone used for P-NS experiments."""

    def __init__(self, modes: int = 16, width: int = 48, layers: int = 4):
        super().__init__()
        self.modes = int(modes)
        self.width = int(width)
        self.layers = int(layers)
        self.fc0 = nn.Conv2d(5, width, 1)
        self.spectral_layers = nn.ModuleList(
            [SpectralConv2d(width, width, modes, modes) for _ in range(layers)]
        )
        self.pointwise_layers = nn.ModuleList(
            [nn.Conv2d(width, width, 1) for _ in range(layers)]
        )
        self.q = nn.Sequential(
            nn.Conv2d(width, 128, 1),
            nn.GELU(),
            nn.Conv2d(128, 2, 1),
        )

    @staticmethod
    def grid(batch: int, height: int, width: int, device, dtype):
        gx = torch.linspace(0, 1, height, device=device, dtype=dtype)
        gy = torch.linspace(0, 1, width, device=device, dtype=dtype)
        gx = gx.view(1, 1, height, 1).expand(batch, 1, height, width)
        gy = gy.view(1, 1, 1, width).expand(batch, 1, height, width)
        return gx, gy

    def forward(self, state: torch.Tensor, viscosity: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = state.shape
        viscosity_map = viscosity.to(state.device, state.dtype).view(batch, 1, 1, 1)
        viscosity_map = viscosity_map.expand(batch, 1, height, width)
        gx, gy = self.grid(batch, height, width, state.device, state.dtype)
        hidden = self.fc0(torch.cat((state, viscosity_map, gx, gy), dim=1))
        for spectral, pointwise in zip(self.spectral_layers, self.pointwise_layers):
            hidden = F.gelu(spectral(hidden) + pointwise(hidden))
        return self.q(hidden)


class LCFNO2d(nn.Module):
    """FNO backbone used for the MAC-stored LC experiments."""

    def __init__(
        self,
        stored_size: int,
        modes: int = 16,
        width: int = 48,
        layers: int = 4,
    ) -> None:
        super().__init__()
        self.stored_size = int(stored_size)
        self.modes = int(modes)
        self.width = int(width)
        self.layers = int(layers)
        self.fc0 = nn.Conv2d(6, width, 1)
        self.convs = nn.ModuleList(
            [SpectralConv2d(width, width, modes, modes) for _ in range(layers)]
        )
        self.ws = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.fc1 = nn.Conv2d(width, width, 1)
        self.fc2 = nn.Conv2d(width, 2, 1)
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, stored_size),
            torch.linspace(0, 1, stored_size),
            indexing="ij",
        )
        self.register_buffer("grid", torch.stack((xx, yy), dim=0)[None])

    def forward(
        self, state: torch.Tensor, lid: torch.Tensor, viscosity: torch.Tensor
    ) -> torch.Tensor:
        batch, _, height, width = state.shape
        if height != self.stored_size or width != self.stored_size:
            raise ValueError(
                f"LCFNO2d was built for {self.stored_size}x{self.stored_size}, "
                f"received {height}x{width}."
            )
        grid = self.grid.expand(batch, -1, -1, -1).to(state.device, state.dtype)
        lid_map = lid.to(state.device, state.dtype).view(batch, 1, 1, 1)
        lid_map = lid_map.expand(batch, 1, height, width)
        reynolds_map = torch.log10(
            (1.0 / viscosity.to(state.device, state.dtype)).clamp_min(1.0)
        ).view(batch, 1, 1, 1)
        reynolds_map = reynolds_map.expand(batch, 1, height, width)
        hidden = self.fc0(torch.cat((state, grid, lid_map, reynolds_map), dim=1))
        for spectral, pointwise in zip(self.convs, self.ws):
            hidden = F.gelu(spectral(hidden) + pointwise(hidden))
        hidden = F.gelu(self.fc1(hidden))
        return self.fc2(hidden)
