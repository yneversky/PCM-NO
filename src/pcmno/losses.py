from __future__ import annotations

import torch
import torch.nn.functional as F

from pcmno.operators.mac import (
    apply_mac_boundary,
    mac_active_mask,
    mac_boundary_residual,
    mac_divergence,
    mac_grid_size,
    mac_laplacian,
)
from pcmno.transitions import LCDynamics, PNSDynamics


def pns_divergence_loss(velocity: torch.Tensor, dynamics: PNSDynamics) -> torch.Tensor:
    return dynamics.operators.divergence(velocity).square().mean()


def pns_vorticity_residual_loss(
    previous: torch.Tensor,
    prediction: torch.Tensor,
    viscosity: torch.Tensor,
    dynamics: PNSDynamics,
) -> torch.Tensor:
    operators = dynamics.operators
    omega_previous = operators.vorticity(previous)
    omega_prediction = operators.vorticity(prediction)
    time_derivative = (omega_prediction - omega_previous) / dynamics.saved_interval
    omega_x, omega_y = operators.grad_scalar(omega_prediction)
    advection = prediction[:, 0] * omega_x + prediction[:, 1] * omega_y
    laplacian = operators.laplacian_scalar(omega_prediction)
    residual = (
        time_derivative
        + advection
        - viscosity.view(-1, 1, 1) * laplacian
        - operators.forcing_vorticity.view(1, operators.n, operators.n)
        + operators.drag * omega_prediction
    )
    return residual.square().mean()


def pns_rollout_loss(
    method: str,
    model: torch.nn.Module,
    sequence: torch.Tensor,
    viscosity: torch.Tensor,
    dynamics: PNSDynamics,
    weights: dict,
) -> tuple[torch.Tensor, dict[str, float]]:
    horizon = sequence.shape[1] - 1
    prediction = sequence[:, 0]
    data_loss = prediction.new_zeros(())
    div_loss = prediction.new_zeros(())
    pde_loss = prediction.new_zeros(())
    update_loss = prediction.new_zeros(())
    normalized = {"divloss": "divreg", "pdeloss": "pino", "pcm_fno": "pcmno"}.get(
        method, method
    )
    for step in range(1, horizon + 1):
        previous_prediction = prediction
        target = sequence[:, step]
        true_previous = sequence[:, step - 1]
        prediction, auxiliary = dynamics.step(
            normalized, model, previous_prediction, viscosity, training=True
        )
        data_loss = data_loss + F.mse_loss(prediction, target)
        if normalized == "divreg":
            div_loss = div_loss + pns_divergence_loss(prediction, dynamics)
        elif normalized == "pino":
            div_loss = div_loss + pns_divergence_loss(prediction, dynamics)
            pde_loss = pde_loss + pns_vorticity_residual_loss(
                previous_prediction, prediction, viscosity, dynamics
            )
        elif normalized == "pcmno":
            target_update = dynamics.operators.project_velocity(target - true_previous)
            predicted_update = dynamics.saved_interval * auxiliary["tangent"]
            update_loss = update_loss + F.mse_loss(predicted_update, target_update)
    data_loss = data_loss / horizon
    total = data_loss
    parts = {"data": float(data_loss.detach().cpu())}
    if normalized == "divreg":
        div_loss = div_loss / horizon
        total = total + float(weights.get("lambda_div", 0.1)) * div_loss
        parts["div"] = float(div_loss.detach().cpu())
    elif normalized == "pino":
        div_loss = div_loss / horizon
        pde_loss = pde_loss / horizon
        total = (
            total
            + float(weights.get("lambda_div", 0.1)) * div_loss
            + float(weights.get("lambda_pde", 1e-5)) * pde_loss
        )
        parts.update(div=float(div_loss.detach().cpu()), pde=float(pde_loss.detach().cpu()))
    elif normalized == "pcmno":
        update_loss = update_loss / horizon
        total = total + float(weights.get("lambda_bb", 0.0)) * update_loss
        parts["update"] = float(update_loss.detach().cpu())
    parts["total"] = float(total.detach().cpu())
    return total, parts


def lc_soft_terms(
    prediction: torch.Tensor,
    previous: torch.Tensor,
    lid: torch.Tensor,
    viscosity: torch.Tensor,
    saved_interval: float,
    drag: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    divergence_loss = mac_divergence(prediction).square().mean()
    boundary_loss = mac_boundary_residual(prediction, lid).square().mean()
    n = mac_grid_size(prediction)
    active = mac_active_mask(n, prediction.device, prediction.dtype)
    derivative = (prediction - previous) / float(saved_interval)
    laplacian = mac_laplacian(apply_mac_boundary(prediction, lid))
    viscosity_map = viscosity.to(prediction.device, prediction.dtype).view(-1, 1, 1, 1)
    residual = active * (derivative - viscosity_map * laplacian + float(drag) * prediction)
    return divergence_loss, boundary_loss, residual.square().mean()


def lc_rollout_loss(
    method: str,
    model: torch.nn.Module,
    sequence: torch.Tensor,
    lid: torch.Tensor,
    viscosity: torch.Tensor,
    dynamics: LCDynamics,
    weights: dict,
    saved_interval: float,
    drag: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    horizon = sequence.shape[1] - 1
    prediction = sequence[:, 0]
    data_loss = prediction.new_zeros(())
    divergence_loss = prediction.new_zeros(())
    boundary_loss = prediction.new_zeros(())
    pde_loss = prediction.new_zeros(())
    update_loss = prediction.new_zeros(())
    for step in range(1, horizon + 1):
        previous = prediction
        target = sequence[:, step]
        prediction, auxiliary = dynamics.step(
            method, model, previous, lid, viscosity, training=True
        )
        data_loss = data_loss + F.mse_loss(prediction, target)
        if method == "divreg":
            div, boundary, _ = lc_soft_terms(
                prediction, previous, lid, viscosity, saved_interval, drag
            )
            divergence_loss += div
            boundary_loss += boundary
        elif method == "pino":
            div, boundary, pde = lc_soft_terms(
                prediction, previous, lid, viscosity, saved_interval, drag
            )
            divergence_loss += div
            boundary_loss += boundary
            pde_loss += pde
        elif method == "pcmno" and float(weights.get("lambda_bb", 0.0)) > 0:
            update_loss += F.mse_loss(auxiliary["raw"], target - previous)
    data_loss = data_loss / horizon
    total = data_loss
    parts = {"data": float(data_loss.detach().cpu())}
    if method in {"divreg", "pino"}:
        divergence_loss /= horizon
        boundary_loss /= horizon
        total = (
            total
            + float(weights.get("lambda_div", 0.05)) * divergence_loss
            + float(weights.get("lambda_bc", 0.1)) * boundary_loss
        )
        parts.update(
            div=float(divergence_loss.detach().cpu()),
            bc=float(boundary_loss.detach().cpu()),
        )
    if method == "pino":
        pde_loss /= horizon
        total = total + float(weights.get("lambda_pde", 0.01)) * pde_loss
        parts["pde"] = float(pde_loss.detach().cpu())
    if method == "pcmno" and float(weights.get("lambda_bb", 0.0)) > 0:
        update_loss /= horizon
        total = total + float(weights["lambda_bb"]) * update_loss
        parts["update"] = float(update_loss.detach().cpu())
    parts["total"] = float(total.detach().cpu())
    return total, parts
