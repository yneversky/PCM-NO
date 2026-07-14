from __future__ import annotations

from functools import lru_cache

import torch


def mac_grid_size(state: torch.Tensor) -> int:
    if state.ndim != 4 or state.shape[1] != 2 or state.shape[-1] != state.shape[-2]:
        raise ValueError(f"Expected [B,2,S,S] MAC state, received {tuple(state.shape)}")
    return int(state.shape[-1] - 2)


def mac_spacing(n: int) -> float:
    return 1.0 / float(n)


def apply_mac_boundary(
    state: torch.Tensor,
    lid: torch.Tensor | None,
    zero_boundary_update: bool = False,
) -> torch.Tensor:
    """Apply the boundary/ghost-value convention used by the released LC data."""
    x = state.clone()
    batch, _, size, _ = x.shape
    n = size - 2
    if lid is None:
        lid_value = torch.zeros(batch, device=x.device, dtype=x.dtype)
    else:
        lid_value = lid.to(device=x.device, dtype=x.dtype).reshape(batch)
    if zero_boundary_update:
        lid_value = torch.zeros_like(lid_value)

    u, v = x[:, 0], x[:, 1]
    u[:, :, n + 1] = 0.0
    v[:, n + 1, :] = 0.0
    u[:, 1 : n + 1, 0] = 0.0
    u[:, 1 : n + 1, n] = 0.0
    v[:, 0, 1 : n + 1] = 0.0
    v[:, n, 1 : n + 1] = 0.0
    u[:, 0, 1:n] = 0.0
    u[:, n + 1, 1:n] = lid_value[:, None]
    v[:, 1:n, 0] = 0.0
    v[:, 1:n, n + 1] = 0.0

    u[:, 0, 0] = u[:, 0, n] = u[:, n + 1, 0] = u[:, n + 1, n] = 0.0
    v[:, 0, 0] = v[:, 0, n + 1] = v[:, n, 0] = v[:, n, n + 1] = 0.0
    x[:, 0], x[:, 1] = u, v
    return x


def mac_divergence(state: torch.Tensor) -> torch.Tensor:
    n = mac_grid_size(state)
    h = mac_spacing(n)
    u, v = state[:, 0], state[:, 1]
    div = (u[:, 1 : n + 1, 1 : n + 1] - u[:, 1 : n + 1, 0:n]) / h
    div = div + (v[:, 1 : n + 1, 1 : n + 1] - v[:, 0:n, 1 : n + 1]) / h
    return div


def mac_adjoint_divergence(lam: torch.Tensor, stored_size: int | None = None) -> torch.Tensor:
    batch, n, _ = lam.shape
    size = stored_size or n + 2
    h = mac_spacing(n)
    correction = torch.zeros(batch, 2, size, size, device=lam.device, dtype=lam.dtype)
    if n > 1:
        correction[:, 0, 1 : n + 1, 1:n] = (lam[:, :, 0 : n - 1] - lam[:, :, 1:n]) / h
        correction[:, 1, 1:n, 1 : n + 1] = (lam[:, 0 : n - 1, :] - lam[:, 1:n, :]) / h
    return correction


def mac_boundary_residual(state: torch.Tensor, lid: torch.Tensor | None) -> torch.Tensor:
    batch, _, size, _ = state.shape
    n = size - 2
    lid_value = (
        torch.zeros(batch, device=state.device, dtype=state.dtype)
        if lid is None
        else lid.to(state.device, state.dtype).reshape(batch)
    )
    u, v = state[:, 0], state[:, 1]
    values = [
        u[:, 1 : n + 1, 0].reshape(batch, -1),
        u[:, 1 : n + 1, n].reshape(batch, -1),
        v[:, 0, 1 : n + 1].reshape(batch, -1),
        v[:, n, 1 : n + 1].reshape(batch, -1),
        u[:, 0, 1:n].reshape(batch, -1),
        (u[:, n + 1, 1:n] - lid_value[:, None]).reshape(batch, -1),
        v[:, 1:n, 0].reshape(batch, -1),
        v[:, 1:n, n + 1].reshape(batch, -1),
    ]
    residual = torch.cat(values, dim=1)
    return torch.sqrt(residual.square().mean(dim=1) + 1e-12)


def mac_active_mask(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    size = n + 2
    mask = torch.zeros(1, 2, size, size, device=device, dtype=dtype)
    mask[:, 0, 1 : n + 1, 1:n] = 1.0
    mask[:, 1, 1:n, 1 : n + 1] = 1.0
    return mask


def mac_laplacian(state: torch.Tensor) -> torch.Tensor:
    n = mac_grid_size(state)
    h = mac_spacing(n)
    lap = torch.zeros_like(state)
    lap[:, :, 1:-1, 1:-1] = (
        state[:, :, 1:-1, 2:]
        - 2.0 * state[:, :, 1:-1, 1:-1]
        + state[:, :, 1:-1, :-2]
        + state[:, :, 2:, 1:-1]
        - 2.0 * state[:, :, 1:-1, 1:-1]
        + state[:, :, :-2, 1:-1]
    ) / (h * h)
    return lap


def cg_solve_batched(
    apply_operator,
    rhs: torch.Tensor,
    max_iterations: int = 80,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed-budget differentiable batched conjugate gradients."""
    x = torch.zeros_like(rhs)
    residual = rhs - apply_operator(x)
    direction = residual.clone()
    residual_sq = residual.square().sum(dim=(1, 2))
    rhs_norm = torch.sqrt(rhs.square().sum(dim=(1, 2)) + 1e-30)
    for _ in range(int(max_iterations)):
        applied = apply_operator(direction)
        denominator = (direction * applied).sum(dim=(1, 2)).clamp_min(1e-30)
        alpha = residual_sq / denominator
        x = x + alpha[:, None, None] * direction
        residual = residual - alpha[:, None, None] * applied
        new_residual_sq = residual.square().sum(dim=(1, 2))
        beta = new_residual_sq / residual_sq.clamp_min(1e-30)
        direction = residual + beta[:, None, None] * direction
        residual_sq = new_residual_sq
    relative_residual = torch.sqrt(residual_sq + 1e-30) / rhs_norm.clamp_min(1e-30)
    return x, relative_residual


_DCT_CACHE: dict[tuple[int, str, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}


def _dct_basis(n: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    key = (n, str(device), dtype)
    if key in _DCT_CACHE:
        return _DCT_CACHE[key]
    i = torch.arange(n, device=device, dtype=dtype).view(n, 1)
    k = torch.arange(n, device=device, dtype=dtype).view(1, n)
    basis = torch.cos(torch.pi * (i + 0.5) * k / n)
    basis[:, 0] *= (1.0 / n) ** 0.5
    if n > 1:
        basis[:, 1:] *= (2.0 / n) ** 0.5
    h = mac_spacing(n)
    eigenvalues = 4.0 * torch.sin(torch.pi * torch.arange(n, device=device, dtype=dtype) / (2 * n)).square() / (h * h)
    _DCT_CACHE[key] = basis, eigenvalues
    return basis, eigenvalues


def dct_neumann_solve(rhs: torch.Tensor, damping: float = 1e-8) -> torch.Tensor:
    """Exact separable solve for the regular-grid Neumann normal equation."""
    _, n, _ = rhs.shape
    basis, eigenvalues = _dct_basis(n, rhs.device, rhs.dtype)
    coeff = torch.matmul(basis.T, rhs)
    coeff = torch.matmul(coeff, basis)
    denominator = eigenvalues.view(n, 1) + eigenvalues.view(1, n) + float(damping)
    coeff = coeff / denominator
    coeff[:, 0, 0] = 0.0
    lam = torch.matmul(basis, coeff)
    return torch.matmul(lam, basis.T)


def _project_free_faces(
    state: torch.Tensor,
    iterations: int,
    damping: float,
    solver: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, _, size, _ = state.shape
    rhs = mac_divergence(state)
    rhs = rhs - rhs.mean(dim=(1, 2), keepdim=True)

    def apply_operator(lam: torch.Tensor) -> torch.Tensor:
        correction = mac_adjoint_divergence(lam, stored_size=size)
        return mac_divergence(correction) + float(damping) * lam

    solver = solver.lower()
    if solver == "cg":
        lam, relative_residual = cg_solve_batched(apply_operator, rhs, iterations)
    elif solver == "dct":
        lam = dct_neumann_solve(rhs, damping=damping)
        residual = rhs - apply_operator(lam)
        relative_residual = torch.sqrt(residual.square().sum(dim=(1, 2)) + 1e-30)
        relative_residual = relative_residual / torch.sqrt(rhs.square().sum(dim=(1, 2)) + 1e-30)
    else:
        raise ValueError(f"Unknown MAC projection solver: {solver}")
    projected = state - mac_adjoint_divergence(lam, stored_size=size)
    if projected.shape[0] != batch:
        raise RuntimeError("MAC projection changed batch size.")
    return projected, relative_residual


def mac_project(
    state: torch.Tensor,
    lid: torch.Tensor | None = None,
    zero_boundary_update: bool = False,
    iterations: int = 80,
    tolerance: float = 1e-6,
    damping: float = 1e-8,
    solver: str = "cg",
    return_info: bool = False,
):
    """Boundary-aware tangent projection or affine state retraction.

    ``tolerance`` is recorded for protocol completeness. The paper implementation
    uses a fixed iteration budget, so the value does not terminate CG early.
    """
    del tolerance
    bounded = apply_mac_boundary(state, lid=lid, zero_boundary_update=zero_boundary_update)
    projected, relative_residual = _project_free_faces(
        bounded, iterations=iterations, damping=damping, solver=solver
    )
    projected = apply_mac_boundary(
        projected, lid=lid, zero_boundary_update=zero_boundary_update
    )
    if return_info:
        return projected, {
            "iterations": int(iterations),
            "relative_residual_mean": float(relative_residual.detach().mean().cpu()),
            "relative_residual_max": float(relative_residual.detach().max().cpu()),
            "solver": solver,
        }
    return projected


def mac_project_div_only(
    state: torch.Tensor,
    iterations: int = 80,
    tolerance: float = 1e-6,
    damping: float = 1e-8,
    solver: str = "cg",
    return_info: bool = False,
):
    """Divergence-only ablation without prescribed-boundary lifting."""
    del tolerance
    projected, relative_residual = _project_free_faces(
        state, iterations=iterations, damping=damping, solver=solver
    )
    if return_info:
        return projected, {
            "iterations": int(iterations),
            "relative_residual_mean": float(relative_residual.detach().mean().cpu()),
            "relative_residual_max": float(relative_residual.detach().max().cpu()),
            "solver": solver,
        }
    return projected


def lc_solver_step(
    state: torch.Tensor,
    viscosity: torch.Tensor,
    lid: torch.Tensor,
    dt: float,
    drag: float,
    projection_iterations: int,
    projection_tolerance: float,
    projection_damping: float,
) -> torch.Tensor:
    """One micro-step of the released LC diffusion-drag projected solver."""
    bounded = apply_mac_boundary(state, lid=lid, zero_boundary_update=False)
    n = mac_grid_size(bounded)
    active = mac_active_mask(n, bounded.device, bounded.dtype)
    laplacian = mac_laplacian(bounded)
    viscosity_map = viscosity.to(bounded.device, bounded.dtype).view(-1, 1, 1, 1)
    provisional = bounded + float(dt) * active * (viscosity_map * laplacian - float(drag) * bounded)
    provisional = apply_mac_boundary(provisional, lid=lid, zero_boundary_update=False)
    return mac_project(
        provisional,
        lid=lid,
        zero_boundary_update=False,
        iterations=projection_iterations,
        tolerance=projection_tolerance,
        damping=projection_damping,
        solver="cg",
    )
