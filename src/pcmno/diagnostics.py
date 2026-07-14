from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from pcmno.data import LCFullTrajectoryDataset, PNSFullTrajectoryDataset
from pcmno.factory import build_dynamics, dataset_name, dataset_paths
from pcmno.metrics import lc_boundary_log10, lc_divergence_log10, relative_l2
from pcmno.operators.mac import mac_divergence, mac_project
from pcmno.utils import cuda_sync


@torch.no_grad()
def tangent_filtering_diagnostic(
    cfg: dict,
    model: torch.nn.Module,
    method: str,
    device: torch.device,
    max_batches: int | None = None,
) -> pd.DataFrame:
    """Evaluate the local projection geometry on realized candidate updates."""
    dataset = dataset_name(cfg)
    paths = dataset_paths(cfg)
    batch_size = int(cfg["evaluation"]["batch_size"])
    if dataset == "pns":
        loader = DataLoader(
            PNSFullTrajectoryDataset(paths["test"]),
            batch_size=batch_size,
            shuffle=False,
        )
        dynamics = build_dynamics(cfg, device)
    else:
        loader = DataLoader(
            LCFullTrajectoryDataset(paths["archive"], "test"),
            batch_size=batch_size,
            shuffle=False,
        )
        dynamics = build_dynamics(cfg, device)

    rows = []
    for batch_index, batch in enumerate(tqdm(loader, desc="tangent diagnostic", leave=False)):
        if max_batches is not None and batch_index >= int(max_batches):
            break
        if dataset == "pns":
            trajectory, viscosity, _ = batch
            source = trajectory[:, 0].to(device)
            target = trajectory[:, 1].to(device)
            viscosity = viscosity.to(device)
            raw = model(source, viscosity)
            projected = dynamics.operators.project_velocity(raw)
            target_update = dynamics.operators.project_velocity(target - source)
        else:
            trajectory, lid, viscosity, _ = batch
            source = trajectory[:, 0].to(device)
            target = trajectory[:, 1].to(device)
            lid = lid.to(device)
            viscosity = viscosity.to(device)
            raw = model(source, lid, viscosity)
            projected = dynamics._project(raw, None, True, training=False)
            target_update = dynamics._project(target - source, None, True, training=False)

        raw_error = relative_l2(raw, target_update)
        projected_error = relative_l2(projected, target_update)
        normal = raw - projected
        raw_energy = raw.square().sum(dim=(1, 2, 3)).clamp_min(1e-30)
        normal_fraction = 100.0 * normal.square().sum(dim=(1, 2, 3)) / raw_energy
        lhs = (raw - target_update).square().sum(dim=(1, 2, 3))
        rhs = (projected - target_update).square().sum(dim=(1, 2, 3))
        rhs = rhs + normal.square().sum(dim=(1, 2, 3))
        pythagorean_gap = (lhs - rhs).abs() / lhs.clamp_min(1e-30)
        for index in range(source.shape[0]):
            rows.append(
                {
                    "batch": batch_index,
                    "sample": index,
                    "RawRelErr": float(raw_error[index].cpu()),
                    "ProjectedRelErr": float(projected_error[index].cpu()),
                    "NormalFractionPct": float(normal_fraction[index].cpu()),
                    "PythagoreanGap": float(pythagorean_gap[index].cpu()),
                }
            )
    return pd.DataFrame(rows)


@torch.no_grad()
def residual_accumulation_reset(
    cfg: dict,
    model: torch.nn.Module,
    device: torch.device,
    max_trajectories: int | None = None,
) -> pd.DataFrame:
    """Controlled LC replay with identical projected updates in both branches."""
    if dataset_name(cfg) != "lc":
        raise ValueError("Residual accumulation/reset is defined for the LC configuration.")
    archive = dataset_paths(cfg)["archive"]
    dataset = LCFullTrajectoryDataset(archive, "test")
    if max_trajectories is not None:
        indices = range(min(len(dataset), int(max_trajectories)))
    else:
        indices = range(len(dataset))
    dynamics = build_dynamics(cfg, device)
    horizon = min(int(cfg["evaluation"]["rollout_horizon"]), dataset.states.shape[1] - 1)
    rows = []
    for trajectory_index in tqdm(indices, desc="residual replay", leave=False):
        trajectory, lid, viscosity, _ = dataset[trajectory_index]
        trajectory = trajectory.unsqueeze(0).to(device)
        lid = lid.view(1).to(device)
        viscosity = viscosity.view(1).to(device)
        full_state = dynamics._project(trajectory[:, 0], lid, False, training=False)
        tangent_state = full_state.clone()
        cumulative = torch.zeros(1, device=device)
        for step in range(1, horizon + 1):
            raw = model(full_state, lid, viscosity)
            tangent = dynamics._project(raw, None, True, training=False)
            realized_update_residual = torch.sqrt(
                mac_divergence(tangent).square().mean(dim=(1, 2)) + 1e-30
            )
            cumulative = cumulative + realized_update_residual
            tangent_state = tangent_state + tangent
            full_state = dynamics._project(full_state + tangent, lid, False, training=False)
            rows.extend(
                [
                    {
                        "trajectory": trajectory_index,
                        "step": step,
                        "branch": "TangentOnlyReplay",
                        "LogDiv": float(lc_divergence_log10(tangent_state, 2).cpu()),
                        "LogBC": float(lc_boundary_log10(tangent_state, lid).cpu()),
                        "CumulativeProjectedUpdateResidual": float(cumulative.cpu()),
                    },
                    {
                        "trajectory": trajectory_index,
                        "step": step,
                        "branch": "FullReplay",
                        "LogDiv": float(lc_divergence_log10(full_state, 2).cpu()),
                        "LogBC": float(lc_boundary_log10(full_state, lid).cpu()),
                        "CumulativeProjectedUpdateResidual": float(cumulative.cpu()),
                    },
                ]
            )
    return pd.DataFrame(rows)


@torch.no_grad()
def lc_projection_sensitivity(
    states: torch.Tensor,
    lids: torch.Tensor,
    iterations: tuple[int, ...] = (20, 80, 160),
    tolerances: tuple[float, ...] = (1e-3, 1e-5, 1e-7),
    damping: float = 1e-8,
) -> pd.DataFrame:
    rows = []
    for tolerance in tolerances:
        for iteration_count in iterations:
            cuda_sync()
            start = time.perf_counter()
            projected, info = mac_project(
                states,
                lid=lids,
                zero_boundary_update=False,
                iterations=iteration_count,
                tolerance=tolerance,
                damping=damping,
                return_info=True,
            )
            cuda_sync()
            elapsed = 1000.0 * (time.perf_counter() - start)
            rows.append(
                {
                    "iterations": iteration_count,
                    "tolerance": tolerance,
                    "LogDiv": float(lc_divergence_log10(projected, 2).mean().cpu()),
                    "LogBC": float(lc_boundary_log10(projected, lids).mean().cpu()),
                    "LinearRelativeResidual": info["relative_residual_mean"],
                    "RuntimeMsPerBatch": elapsed,
                }
            )
    return pd.DataFrame(rows)
