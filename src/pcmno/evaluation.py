from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from pcmno.data import LCFullTrajectoryDataset, PNSFullTrajectoryDataset
from pcmno.factory import build_dynamics, build_model, dataset_name, dataset_paths
from pcmno.metrics import (
    energy_relative_error,
    lc_boundary_log10,
    lc_centerline_profile_error,
    lc_divergence_log10,
    pns_normalized_divergence,
    relative_l2,
)
from pcmno.utils import choose_device, cuda_sync, safe_torch_load


def load_checkpoint_model(
    checkpoint: str | Path,
    cfg: dict,
    method: str,
    device: torch.device,
    data_path: str | Path | None = None,
):
    model = build_model(cfg, method, device=device, data_path=data_path)
    obj = safe_torch_load(checkpoint, map_location=device)
    state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    model.load_state_dict(state)
    model.eval()
    return model, obj


@torch.no_grad()
def evaluate_pns(
    model: torch.nn.Module,
    method: str,
    cfg: dict,
    split_path: str | Path,
    device: torch.device,
    horizons: Iterable[int] | None = None,
    batch_size: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    dataset = PNSFullTrajectoryDataset(split_path)
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size or cfg["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    resolution = int(dataset.velocity.shape[-1])
    dynamics = build_dynamics(cfg, device, resolution=resolution)
    configured_max = min(int(cfg["evaluation"]["rollout_horizon"]), dataset.velocity.shape[1] - 1)
    max_horizon = min(configured_max, max(int(x) for x in horizons)) if horizons else configured_max
    requested = sorted(set(int(x) for x in (horizons or range(1, max_horizon + 1))))
    requested = [x for x in requested if 1 <= x <= max_horizon]
    accumulators = {
        step: {"rel": [], "div": [], "energy": []} for step in requested
    }
    elapsed = 0.0
    sample_steps = 0
    for trajectory, viscosity, _ in tqdm(loader, desc=f"evaluate {method}", leave=False):
        trajectory = trajectory.to(device)
        viscosity = viscosity.to(device)
        prediction = trajectory[:, 0]
        for step in range(1, max_horizon + 1):
            cuda_sync()
            start = time.perf_counter()
            prediction, _ = dynamics.step(method, model, prediction, viscosity)
            cuda_sync()
            elapsed += time.perf_counter() - start
            sample_steps += prediction.shape[0]
            if step in accumulators:
                target = trajectory[:, step]
                accumulators[step]["rel"].append(relative_l2(prediction, target).cpu())
                accumulators[step]["div"].append(
                    pns_normalized_divergence(prediction, dynamics.operators).cpu()
                )
                accumulators[step]["energy"].append(
                    energy_relative_error(prediction, target).cpu()
                )
    rows = []
    for step in requested:
        rel = torch.cat(accumulators[step]["rel"])
        div = torch.cat(accumulators[step]["div"])
        energy = torch.cat(accumulators[step]["energy"])
        rows.append(
            {
                "step": step,
                "RelL2": float(rel.mean()),
                "LogDiv": float(torch.log10(div.mean() + 1e-12)),
                "EnergyRelErr": float(energy.mean()),
            }
        )
    runtime = 1000.0 * elapsed / max(sample_steps, 1)
    frame = pd.DataFrame(rows)
    summary = _summary_from_curve(frame, runtime, dataset="pns", method=method)
    return frame, summary


@torch.no_grad()
def evaluate_lc(
    model: torch.nn.Module,
    method: str,
    cfg: dict,
    archive_path: str | Path,
    split: str,
    device: torch.device,
    horizons: Iterable[int] | None = None,
    batch_size: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    dataset = LCFullTrajectoryDataset(archive_path, split)
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size or cfg["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    dynamics = build_dynamics(cfg, device)
    configured_max = min(int(cfg["evaluation"]["rollout_horizon"]), dataset.states.shape[1] - 1)
    max_horizon = min(configured_max, max(int(x) for x in horizons)) if horizons else configured_max
    requested = sorted(set(int(x) for x in (horizons or range(1, max_horizon + 1))))
    requested = [x for x in requested if 1 <= x <= max_horizon]
    accumulators = {
        step: {"rel": [], "div": [], "bc": [], "profile": []}
        for step in requested
    }
    elapsed = 0.0
    sample_steps = 0
    for trajectory, lid, viscosity, _ in tqdm(loader, desc=f"evaluate {method}", leave=False):
        trajectory = trajectory.to(device)
        lid = lid.to(device)
        viscosity = viscosity.to(device)
        prediction = trajectory[:, 0]
        for step in range(1, max_horizon + 1):
            cuda_sync()
            start = time.perf_counter()
            prediction, _ = dynamics.step(
                method, model, prediction, lid, viscosity, training=False
            )
            cuda_sync()
            elapsed += time.perf_counter() - start
            sample_steps += prediction.shape[0]
            if step in accumulators:
                target = trajectory[:, step]
                accumulators[step]["rel"].append(relative_l2(prediction, target).cpu())
                accumulators[step]["div"].append(lc_divergence_log10(prediction, 2).cpu())
                accumulators[step]["bc"].append(lc_boundary_log10(prediction, lid).cpu())
                accumulators[step]["profile"].append(
                    lc_centerline_profile_error(prediction, target).cpu()
                )
    rows = []
    for step in requested:
        rows.append(
            {
                "step": step,
                "RelL2": float(torch.cat(accumulators[step]["rel"]).mean()),
                "LogDiv": float(torch.cat(accumulators[step]["div"]).mean()),
                "LogBC": float(torch.cat(accumulators[step]["bc"]).mean()),
                "ProfileErr": float(torch.cat(accumulators[step]["profile"]).mean()),
            }
        )
    runtime = 1000.0 * elapsed / max(sample_steps, 1)
    frame = pd.DataFrame(rows)
    summary = _summary_from_curve(frame, runtime, dataset="lc", method=method)
    return frame, summary


def _summary_from_curve(frame: pd.DataFrame, runtime: float, dataset: str, method: str) -> dict:
    summary = {"dataset": dataset, "method": method, "TimeMsPerTrajectoryStep": runtime}
    for horizon in (1, 20, 50):
        row = frame[frame["step"] == horizon]
        if row.empty:
            continue
        for column in frame.columns:
            if column != "step":
                summary[f"{column}_{horizon}"] = float(row.iloc[0][column])
    return summary


def evaluate_checkpoint(
    cfg: dict,
    method: str,
    checkpoint: str | Path,
    split: str = "test",
    data_override: str | Path | None = None,
    horizons: Iterable[int] | None = None,
    device_name: str | None = None,
):
    device = choose_device(device_name or cfg.get("runtime", {}).get("device", "auto"))
    dataset = dataset_name(cfg)
    paths = dataset_paths(cfg)
    if dataset == "pns":
        data_path = Path(data_override) if data_override else paths[split]
        model, _ = load_checkpoint_model(checkpoint, cfg, method, device, data_path)
        return evaluate_pns(model, method, cfg, data_path, device, horizons)
    data_path = Path(data_override) if data_override else paths["archive"]
    model, _ = load_checkpoint_model(checkpoint, cfg, method, device, data_path)
    return evaluate_lc(model, method, cfg, data_path, split, device, horizons)
