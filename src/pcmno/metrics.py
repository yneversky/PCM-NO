from __future__ import annotations

import torch

from pcmno.operators.mac import mac_boundary_residual, mac_divergence
from pcmno.operators.periodic import PeriodicSpectralOps


def relative_l2(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-12):
    numerator = torch.linalg.vector_norm(prediction - target, dim=(1, 2, 3))
    denominator = torch.linalg.vector_norm(target, dim=(1, 2, 3)) + eps
    return numerator / denominator


def pns_normalized_divergence(
    velocity: torch.Tensor, operators: PeriodicSpectralOps, eps: float = 1e-12
):
    divergence_rms = torch.sqrt(operators.divergence(velocity).square().mean(dim=(1, 2)))
    velocity_rms = torch.sqrt(velocity.square().mean(dim=(1, 2, 3))) + eps
    return divergence_rms / velocity_rms


def energy_relative_error(
    prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-12
):
    predicted_energy = 0.5 * prediction.square().sum(dim=(1, 2, 3))
    target_energy = 0.5 * target.square().sum(dim=(1, 2, 3))
    return (predicted_energy - target_energy).abs() / (target_energy.abs() + eps)


def lc_divergence_log10(state: torch.Tensor, margin: int = 2):
    divergence = mac_divergence(state)
    if margin > 0:
        divergence = divergence[:, margin:-margin, margin:-margin]
    rms = torch.sqrt(divergence.square().mean(dim=(1, 2)) + 1e-30)
    return torch.log10(rms + 1e-12)


def lc_boundary_log10(state: torch.Tensor, lid: torch.Tensor):
    return torch.log10(mac_boundary_residual(state, lid) + 1e-12)


def lc_centerline_profile_error(
    prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-12
):
    """Relative error in the two standard cavity centerline profiles.

    The stored arrays include MAC ghost and boundary entries. The implementation
    selects the nearest active centerline indices, matching the paper protocol's
    discrete centerline comparison.
    """
    size = prediction.shape[-1]
    n = size - 2
    center = n // 2
    # u_x on the vertical centerline and u_y on the horizontal centerline.
    pred_ux = prediction[:, 0, 1 : n + 1, center]
    true_ux = target[:, 0, 1 : n + 1, center]
    pred_uy = prediction[:, 1, center, 1 : n + 1]
    true_uy = target[:, 1, center, 1 : n + 1]
    error_x = torch.linalg.vector_norm(pred_ux - true_ux, dim=1)
    error_x = error_x / (torch.linalg.vector_norm(true_ux, dim=1) + eps)
    error_y = torch.linalg.vector_norm(pred_uy - true_uy, dim=1)
    error_y = error_y / (torch.linalg.vector_norm(true_uy, dim=1) + eps)
    return 0.5 * (error_x + error_y)
