from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from pcmno.operators.mac import apply_mac_boundary, lc_solver_step, mac_project
from pcmno.utils import seed_all


_SPLIT_KEYS = {"train": "idx_train", "val": "idx_val", "test": "idx_test"}


def _load_ids(data: np.lib.npyio.NpzFile, split: str) -> np.ndarray:
    key = _SPLIT_KEYS.get(split)
    if key is None:
        raise ValueError(f"Unknown LC split: {split}")
    if key in data.files:
        return data[key].astype(np.int64)
    # Small public samples intentionally omit paper split indices.
    return np.arange(data["states"].shape[0], dtype=np.int64)


def load_lc_arrays(path: str | Path, split: str):
    with np.load(path, allow_pickle=False) as data:
        ids = _load_ids(data, split)
        states = data["states"][ids].astype(np.float32, copy=True)
        lid = data["lid"][ids].astype(np.float32, copy=True)
        viscosity = data["nu"][ids].astype(np.float32, copy=True)
        reynolds = (
            data["Re"][ids].astype(np.float32, copy=True)
            if "Re" in data.files
            else (1.0 / viscosity).astype(np.float32)
        )
    return states, lid, viscosity, reynolds


class LCWindowDataset(Dataset):
    def __init__(self, path: str | Path, split: str, horizon: int, stride: int = 1) -> None:
        self.states, self.lid, self.viscosity, self.reynolds = load_lc_arrays(path, split)
        self.horizon = int(horizon)
        trajectories, steps = self.states.shape[:2]
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
            torch.from_numpy(
                self.states[trajectory, start : start + self.horizon + 1]
            ).float(),
            torch.tensor(self.lid[trajectory], dtype=torch.float32),
            torch.tensor(self.viscosity[trajectory], dtype=torch.float32),
        )


class LCFullTrajectoryDataset(Dataset):
    def __init__(self, path: str | Path, split: str) -> None:
        self.states, self.lid, self.viscosity, self.reynolds = load_lc_arrays(path, split)

    def __len__(self) -> int:
        return self.states.shape[0]

    def __getitem__(self, item: int):
        return (
            torch.from_numpy(self.states[item]).float(),
            torch.tensor(self.lid[item], dtype=torch.float32),
            torch.tensor(self.viscosity[item], dtype=torch.float32),
            torch.tensor(self.reynolds[item], dtype=torch.float32),
        )


@torch.no_grad()
def generate_lc_dataset(
    output: str | Path,
    config: dict,
    device: torch.device,
    fixed_reynolds: float | None = None,
    fixed_lid: float | None = None,
) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = config["data_generation"]
    physics = config["physics"]
    split = config["splits"]
    total = int(split["train"] + split["val"] + split["test"])
    n = int(data["resolution"])
    size = n + 2
    rng = np.random.default_rng(int(data.get("seed", 0)))
    if fixed_lid is None:
        lids = rng.uniform(*physics["lid_range"], size=total).astype(np.float32)
    else:
        lids = np.full(total, float(fixed_lid), dtype=np.float32)
    if fixed_reynolds is None:
        reynolds = rng.uniform(*physics["reynolds_range"], size=total).astype(np.float32)
    else:
        reynolds = np.full(total, float(fixed_reynolds), dtype=np.float32)
    viscosity = (1.0 / reynolds).astype(np.float32)
    states = np.zeros(
        (total, int(data["stored_states"]), 2, size, size), dtype=np.float32
    )
    batch_size = int(data.get("batch_size", 8))
    for start in range(0, total, batch_size):
        end = min(total, start + batch_size)
        seed_all(int(data.get("seed", 0)) + 1_000_003 * (start // batch_size + 1))
        lid_batch = torch.from_numpy(lids[start:end]).to(device)
        viscosity_batch = torch.from_numpy(viscosity[start:end]).to(device)
        state = float(physics.get("initial_noise", 0.02)) * torch.randn(
            end - start, 2, size, size, device=device
        )
        state = apply_mac_boundary(state, lid_batch)
        state = mac_project(
            state,
            lid=lid_batch,
            iterations=int(data["projection_iterations"]),
            tolerance=float(data["projection_tolerance"]),
            damping=float(data["projection_damping"]),
        )
        for _ in range(int(data["burn_in_intervals"]) * int(data["inner_steps"])):
            state = lc_solver_step(
                state,
                viscosity_batch,
                lid_batch,
                dt=float(data["dt"]),
                drag=float(physics["drag"]),
                projection_iterations=int(data["projection_iterations"]),
                projection_tolerance=float(data["projection_tolerance"]),
                projection_damping=float(data["projection_damping"]),
            )
        for time_index in range(int(data["stored_states"])):
            states[start:end, time_index] = state.detach().cpu().numpy()
            if time_index + 1 < int(data["stored_states"]):
                for _ in range(int(data["inner_steps"])):
                    state = lc_solver_step(
                        state,
                        viscosity_batch,
                        lid_batch,
                        dt=float(data["dt"]),
                        drag=float(physics["drag"]),
                        projection_iterations=int(data["projection_iterations"]),
                        projection_tolerance=float(data["projection_tolerance"]),
                        projection_damping=float(data["projection_damping"]),
                    )
    train_end = int(split["train"])
    val_end = train_end + int(split["val"])
    np.savez_compressed(
        output,
        states=states,
        lid=lids,
        Re=reynolds,
        nu=viscosity,
        idx_train=np.arange(0, train_end, dtype=np.int64),
        idx_val=np.arange(train_end, val_end, dtype=np.int64),
        idx_test=np.arange(val_end, total, dtype=np.int64),
        N=np.asarray(n),
        S=np.asarray(size),
        saved_steps=np.asarray(int(data["stored_states"])),
        inner_steps=np.asarray(int(data["inner_steps"])),
        dt=np.asarray(float(data["dt"])),
        dt_save=np.asarray(float(data["dt"]) * int(data["inner_steps"])),
    )
    return output
