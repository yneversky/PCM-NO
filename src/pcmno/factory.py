from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from pcmno.models import LCFNO2d, PNSFNO2d, PeriodicClawFNO2d
from pcmno.operators.periodic import PeriodicSpectralOps
from pcmno.transitions import LCDynamics, PNSDynamics, ProjectionSettings
from pcmno.config import resolve_path


def dataset_name(cfg: dict) -> str:
    name = str(cfg["dataset"]["name"]).lower()
    if name not in {"pns", "lc"}:
        raise ValueError(f"dataset.name must be pns or lc, received {name}")
    return name


def dataset_paths(cfg: dict, repo_root: str | Path | None = None) -> dict[str, Path]:
    dataset = dataset_name(cfg)
    data = cfg["dataset"]
    if dataset == "pns":
        return {
            "train": resolve_path(data["train_path"], repo_root),
            "val": resolve_path(data["val_path"], repo_root),
            "test": resolve_path(data["test_path"], repo_root),
        }
    return {"archive": resolve_path(data["archive_path"], repo_root)}


def infer_lc_stored_size(archive: str | Path) -> int:
    with np.load(archive, allow_pickle=False) as data:
        return int(data["states"].shape[-1])


def build_model(
    cfg: dict,
    method: str,
    device: torch.device,
    data_path: str | Path | None = None,
) -> torch.nn.Module:
    model_cfg = cfg["model"]
    dataset = dataset_name(cfg)
    if dataset == "pns":
        if method == "clawno":
            resolution = int(cfg["physics"].get("resolution", model_cfg.get("resolution", 64)))
            model = PeriodicClawFNO2d(
                resolution=resolution,
                length=float(cfg["physics"].get("length", 2 * math.pi)),
                modes=int(model_cfg["modes"]),
                width=int(model_cfg["width"]),
                layers=int(model_cfg["layers"]),
            )
        else:
            model = PNSFNO2d(
                modes=int(model_cfg["modes"]),
                width=int(model_cfg["width"]),
                layers=int(model_cfg["layers"]),
            )
    else:
        if data_path is None:
            data_path = dataset_paths(cfg)["archive"]
        stored_size = int(model_cfg.get("stored_size", infer_lc_stored_size(data_path)))
        model = LCFNO2d(
            stored_size=stored_size,
            modes=int(model_cfg["modes"]),
            width=int(model_cfg["width"]),
            layers=int(model_cfg["layers"]),
        )
    return model.to(device)


def build_dynamics(cfg: dict, device: torch.device, resolution: int | None = None):
    dataset = dataset_name(cfg)
    physics = cfg["physics"]
    if dataset == "pns":
        n = int(resolution or physics.get("resolution", 64))
        operators = PeriodicSpectralOps(
            n=n,
            length=float(physics.get("length", 2 * math.pi)),
            forcing_wavenumber=int(physics.get("forcing_wavenumber", 4)),
            forcing_amplitude=float(physics.get("forcing_amplitude", 0.1)),
            drag=float(physics.get("drag", 0.1)),
            device=device,
        )
        return PNSDynamics(
            operators=operators,
            saved_interval=float(physics.get("saved_interval", 0.05)),
        )
    projection_cfg = cfg["projection"]
    return LCDynamics(
        ProjectionSettings(
            train_iterations=int(projection_cfg.get("train_iterations", 30)),
            eval_iterations=int(projection_cfg.get("eval_iterations", 120)),
            tolerance=float(projection_cfg.get("tolerance", 1e-6)),
            damping=float(projection_cfg.get("damping", 1e-8)),
            train_solver=str(projection_cfg.get("train_solver", "cg")),
            eval_solver=str(projection_cfg.get("eval_solver", "cg")),
        )
    )
