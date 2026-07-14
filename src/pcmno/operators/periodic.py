from __future__ import annotations

import math

import torch


class PeriodicSpectralOps:
    """Spectral operators on the periodic square ``[0, L]^2``.

    Velocity tensors have shape ``[batch, 2, nx, ny]``. Channel zero is
    horizontal velocity and channel one is vertical velocity.
    """

    def __init__(
        self,
        n: int,
        length: float = 2 * math.pi,
        forcing_wavenumber: int = 4,
        forcing_amplitude: float = 0.1,
        drag: float = 0.1,
        device: str | torch.device = "cpu",
    ) -> None:
        self.n = int(n)
        self.length = float(length)
        self.forcing_wavenumber = int(forcing_wavenumber)
        self.forcing_amplitude = float(forcing_amplitude)
        self.drag = float(drag)
        self.device = torch.device(device)

        freq = torch.fft.fftfreq(self.n, d=self.length / self.n, device=self.device)
        freq = freq * (2 * math.pi)
        self.kx = freq.view(self.n, 1)
        self.ky = freq.view(1, self.n)
        self.k2 = self.kx.square() + self.ky.square()
        self.k2_safe = self.k2.clone()
        self.k2_safe[0, 0] = 1.0

        integer_modes = torch.fft.fftfreq(self.n, device=self.device) * self.n
        kix = integer_modes.view(self.n, 1)
        kiy = integer_modes.view(1, self.n)
        cutoff = self.n // 3
        self.dealias = ((kix.abs() <= cutoff) & (kiy.abs() <= cutoff)).float()

        x = torch.linspace(0, self.length, self.n + 1, device=self.device)[:-1]
        y = torch.linspace(0, self.length, self.n + 1, device=self.device)[:-1]
        self.x, self.y = torch.meshgrid(x, y, indexing="ij")
        self.forcing_vorticity = (
            -self.forcing_amplitude
            * self.forcing_wavenumber
            * torch.cos(self.forcing_wavenumber * self.y)
        )

    def fft2(self, value: torch.Tensor) -> torch.Tensor:
        return torch.fft.fft2(value, norm="ortho")

    def ifft2_real(self, value_hat: torch.Tensor) -> torch.Tensor:
        return torch.fft.ifft2(value_hat, norm="ortho").real

    def velocity_from_vorticity(self, vorticity: torch.Tensor) -> torch.Tensor:
        omega_hat = self.fft2(vorticity) * self.dealias
        psi_hat = omega_hat / self.k2_safe
        psi_hat[:, 0, 0] = 0.0
        ux = self.ifft2_real(1j * self.ky * psi_hat)
        uy = self.ifft2_real(-1j * self.kx * psi_hat)
        return torch.stack((ux, uy), dim=1)

    def velocity_from_potential(self, potential: torch.Tensor) -> torch.Tensor:
        """Fixed divergence-free layer used by the ClawNO baseline."""
        if potential.ndim == 4:
            potential = potential[:, 0]
        potential_hat = self.fft2(potential)
        ux = self.ifft2_real(1j * self.ky * potential_hat)
        uy = self.ifft2_real(-1j * self.kx * potential_hat)
        return torch.stack((ux, uy), dim=1)

    def project_velocity(self, velocity: torch.Tensor) -> torch.Tensor:
        """Dealiased spectral Helmholtz projection."""
        ux_hat = self.fft2(velocity[:, 0]) * self.dealias
        uy_hat = self.fft2(velocity[:, 1]) * self.dealias
        div_hat = 1j * self.kx * ux_hat + 1j * self.ky * uy_hat
        ux_hat = (ux_hat + 1j * self.kx * div_hat / self.k2_safe) * self.dealias
        uy_hat = (uy_hat + 1j * self.ky * div_hat / self.k2_safe) * self.dealias
        return torch.stack((self.ifft2_real(ux_hat), self.ifft2_real(uy_hat)), dim=1)

    def divergence(self, velocity: torch.Tensor) -> torch.Tensor:
        ux_hat = self.fft2(velocity[:, 0])
        uy_hat = self.fft2(velocity[:, 1])
        return self.ifft2_real(1j * self.kx * ux_hat + 1j * self.ky * uy_hat)

    def vorticity(self, velocity: torch.Tensor) -> torch.Tensor:
        ux_hat = self.fft2(velocity[:, 0])
        uy_hat = self.fft2(velocity[:, 1])
        return self.ifft2_real(1j * self.kx * uy_hat - 1j * self.ky * ux_hat)

    def grad_scalar(self, scalar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scalar_hat = self.fft2(scalar)
        return (
            self.ifft2_real(1j * self.kx * scalar_hat),
            self.ifft2_real(1j * self.ky * scalar_hat),
        )

    def laplacian_scalar(self, scalar: torch.Tensor) -> torch.Tensor:
        return self.ifft2_real(-self.k2 * self.fft2(scalar))

    def vorticity_rhs(self, vorticity: torch.Tensor, viscosity: torch.Tensor) -> torch.Tensor:
        omega_hat = self.fft2(vorticity) * self.dealias
        psi_hat = omega_hat / self.k2_safe
        psi_hat[:, 0, 0] = 0.0
        ux = self.ifft2_real(1j * self.ky * psi_hat)
        uy = self.ifft2_real(-1j * self.kx * psi_hat)
        omega_x = self.ifft2_real(1j * self.kx * omega_hat)
        omega_y = self.ifft2_real(1j * self.ky * omega_hat)
        advection = ux * omega_x + uy * omega_y
        advection = self.ifft2_real(self.fft2(advection) * self.dealias)
        laplacian = self.ifft2_real(-self.k2 * omega_hat)
        return (
            -advection
            + viscosity.view(-1, 1, 1) * laplacian
            + self.forcing_vorticity.view(1, self.n, self.n)
            - self.drag * vorticity
        )

    def rk4_step(
        self, vorticity: torch.Tensor, viscosity: torch.Tensor, dt: float
    ) -> torch.Tensor:
        k1 = self.vorticity_rhs(vorticity, viscosity)
        k2 = self.vorticity_rhs(vorticity + 0.5 * dt * k1, viscosity)
        k3 = self.vorticity_rhs(vorticity + 0.5 * dt * k2, viscosity)
        k4 = self.vorticity_rhs(vorticity + dt * k3, viscosity)
        updated = vorticity + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        updated = self.ifft2_real(self.fft2(updated) * self.dealias)
        return updated - updated.mean(dim=(-2, -1), keepdim=True)

    def random_vorticity(self, batch_size: int, initial_velocity_rms: float = 1.0) -> torch.Tensor:
        noise = torch.randn(batch_size, self.n, self.n, device=self.device)
        radius = torch.sqrt(self.k2)
        filt = torch.exp(-0.5 * (radius / max(2.0, self.n / 10.0)).pow(4))
        vorticity = self.ifft2_real(self.fft2(noise) * filt * self.dealias)
        vorticity = vorticity - vorticity.mean(dim=(-2, -1), keepdim=True)
        vorticity = vorticity + 0.25 * self.forcing_vorticity.view(1, self.n, self.n)
        velocity = self.velocity_from_vorticity(vorticity)
        rms = torch.sqrt(velocity.square().mean(dim=(1, 2, 3), keepdim=True) + 1e-12)
        return vorticity / rms.view(-1, 1, 1) * float(initial_velocity_rms)
