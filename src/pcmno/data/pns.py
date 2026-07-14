from __future__ import annotations

import gc
import math
from pathlib import Path

import torch
from torch.utils.data import Dataset

from pcmno.operators.periodic import PeriodicSpectralOps
from pcmno.utils import safe_torch_load, seed_all


class PNSWindowDataset(Dataset):
    """Autoregressive windows from a released P-NS split."""

    def __init__(self, path: str | Path, horizon: int, stride: int = 1) -> None:
        obj = safe_torch_load(path)
        self.velocity = obj["u"].float()
        self.viscosity = obj["nu"].float()
        self.reynolds = obj["Re"].float()
        self.horizon = int(horizon)
        trajectories, steps = self.velocity.shape[:2]
        if steps < self.horizon + 1:
            raise ValueError(f"Trajectory length {steps} is too short for horizon {horizon}")
        self.index = [
            (trajectory, start)
            for trajectory in range(trajectories)
            for start in range(0, steps - self.horizon, int(stride))
        ]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int):
        trajectory, start = self.index[item]
        return (
            self.velocity[trajectory, start : start + self.horizon + 1],
            self.viscosity[trajectory],
        )


class PNSFullTrajectoryDataset(Dataset):
    def __init__(self, path: str | Path) -> None:
        obj = safe_torch_load(path)
        self.velocity = obj["u"].float()
        self.viscosity = obj["nu"].float()
        self.reynolds = obj["Re"].float()

    def __len__(self) -> int:
        return self.velocity.shape[0]

    def __getitem__(self, item: int):
        return self.velocity[item], self.viscosity[item], self.reynolds[item]


@torch.no_grad()
def generate_pns_batch(
    operators: PeriodicSpectralOps,
    batch_size: int,
    stored_states: int,
    saved_interval: float,
    substeps: int,
    burn_in_intervals: int,
    reynolds_low: float,
    reynolds_high: float,
    initial_velocity_rms: float,
    fixed_reynolds: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if fixed_reynolds is None:
        reynolds = torch.empty(batch_size, device=operators.device).uniform_(
            reynolds_low, reynolds_high
        )
    else:
        reynolds = torch.full(
            (batch_size,), float(fixed_reynolds), device=operators.device
        )
    viscosity = 1.0 / reynolds
    vorticity = operators.random_vorticity(batch_size, initial_velocity_rms)
    dt = float(saved_interval) / int(substeps)
    for _ in range(int(burn_in_intervals) * int(substeps)):
        vorticity = operators.rk4_step(vorticity, viscosity, dt)
    trajectory = []
    for step in range(int(stored_states)):
        velocity = operators.project_velocity(
            operators.velocity_from_vorticity(vorticity)
        )
        trajectory.append(velocity.detach().cpu())
        if step + 1 < stored_states:
            for _ in range(int(substeps)):
                vorticity = operators.rk4_step(vorticity, viscosity, dt)
    return (
        torch.stack(trajectory, dim=1),
        reynolds.detach().cpu(),
        viscosity.detach().cpu(),
    )


def generate_pns_split(
    output: str | Path,
    count: int,
    config: dict,
    device: torch.device,
    seed_offset: int,
    fixed_reynolds: float | None = None,
) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    physics = config["physics"]
    data = config["data_generation"]
    batch_size = int(data.get("batch_size", 8))
    operators = PeriodicSpectralOps(
        n=int(data["resolution"]),
        length=float(physics.get("length", 2 * math.pi)),
        forcing_wavenumber=int(physics.get("forcing_wavenumber", 4)),
        forcing_amplitude=float(physics.get("forcing_amplitude", 0.1)),
        drag=float(physics.get("drag", 0.1)),
        device=device,
    )
    velocity_parts, reynolds_parts, viscosity_parts = [], [], []
    for start in range(0, int(count), batch_size):
        seed_all(int(seed_offset + start // batch_size))
        current = min(batch_size, int(count) - start)
        velocity, reynolds, viscosity = generate_pns_batch(
            operators=operators,
            batch_size=current,
            stored_states=int(data["stored_states"]),
            saved_interval=float(data["saved_interval"]),
            substeps=int(data["substeps"]),
            burn_in_intervals=int(data["burn_in_intervals"]),
            reynolds_low=float(physics["reynolds_range"][0]),
            reynolds_high=float(physics["reynolds_range"][1]),
            initial_velocity_rms=float(physics.get("initial_velocity_rms", 1.0)),
            fixed_reynolds=fixed_reynolds,
        )
        velocity_parts.append(velocity)
        reynolds_parts.append(reynolds)
        viscosity_parts.append(viscosity)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    obj = {
        "u": torch.cat(velocity_parts, dim=0).contiguous(),
        "Re": torch.cat(reynolds_parts, dim=0).contiguous(),
        "nu": torch.cat(viscosity_parts, dim=0).contiguous(),
        "cfg": config,
    }
    torch.save(obj, output)
    return output
