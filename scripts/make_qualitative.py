#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from pcmno.config import load_config
from pcmno.data import LCFullTrajectoryDataset, PNSFullTrajectoryDataset
from pcmno.evaluation import load_checkpoint_model
from pcmno.factory import build_dynamics, dataset_name
from pcmno.utils import choose_device


def parse_args():
    parser = argparse.ArgumentParser(description="Create qualitative stress-field comparisons.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--finalproj", type=Path, required=True)
    parser.add_argument("--pcmno", type=Path, required=True)
    parser.add_argument("--step", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def select_median_energy(trajectories: torch.Tensor, step: int) -> int:
    fields = trajectories[:, step]
    energy = 0.5 * fields.square().sum(dim=(1, 2, 3))
    median = energy.median()
    return int((energy - median).abs().argmin())


@torch.no_grad()
def rollout_pns(model, method, dynamics, initial, viscosity, step):
    state = initial
    for _ in range(step):
        state, _ = dynamics.step(method, model, state, viscosity)
    return state


@torch.no_grad()
def rollout_lc(model, method, dynamics, initial, lid, viscosity, step):
    state = initial
    for _ in range(step):
        state, _ = dynamics.step(method, model, state, lid, viscosity)
    return state


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = choose_device(cfg.get("runtime", {}).get("device", "auto"))
    dataset = dataset_name(cfg)
    final_model, _ = load_checkpoint_model(args.finalproj, cfg, "finalproj", device, args.data)
    pcm_model, _ = load_checkpoint_model(args.pcmno, cfg, "pcmno", device, args.data)

    if dataset == "pns":
        data = PNSFullTrajectoryDataset(args.data)
        index = select_median_energy(data.velocity, args.step)
        trajectory, viscosity, _ = data[index]
        trajectory = trajectory.unsqueeze(0).to(device)
        viscosity = viscosity.view(1).to(device)
        dynamics = build_dynamics(cfg, device, resolution=trajectory.shape[-1])
        target = trajectory[:, args.step]
        final = rollout_pns(final_model, "finalproj", dynamics, trajectory[:, 0], viscosity, args.step)
        pcm = rollout_pns(pcm_model, "pcmno", dynamics, trajectory[:, 0], viscosity, args.step)
        fields = [
            dynamics.operators.vorticity(target)[0].cpu().numpy(),
            dynamics.operators.vorticity(final)[0].cpu().numpy(),
            dynamics.operators.vorticity(pcm)[0].cpu().numpy(),
        ]
        errors = [np.abs(fields[1] - fields[0]), np.abs(fields[2] - fields[0])]
        field_label = "Vorticity"
    else:
        data = LCFullTrajectoryDataset(args.data, "test")
        trajectories = torch.from_numpy(data.states)
        index = select_median_energy(trajectories, args.step)
        trajectory, lid, viscosity, _ = data[index]
        trajectory = trajectory.unsqueeze(0).to(device)
        lid = lid.view(1).to(device)
        viscosity = viscosity.view(1).to(device)
        dynamics = build_dynamics(cfg, device)
        target = trajectory[:, args.step]
        final = rollout_lc(final_model, "finalproj", dynamics, trajectory[:, 0], lid, viscosity, args.step)
        pcm = rollout_lc(pcm_model, "pcmno", dynamics, trajectory[:, 0], lid, viscosity, args.step)
        speed = lambda x: torch.sqrt(x[:, 0].square() + x[:, 1].square())[0].cpu().numpy()
        fields = [speed(target), speed(final), speed(pcm)]
        errors = [
            torch.linalg.vector_norm(final - target, dim=1)[0].cpu().numpy(),
            torch.linalg.vector_norm(pcm - target, dim=1)[0].cpu().numpy(),
        ]
        field_label = "Speed"

    fig, axes = plt.subplots(1, 5, figsize=(12, 2.6), constrained_layout=True)
    field_min = min(float(field.min()) for field in fields)
    field_max = max(float(field.max()) for field in fields)
    error_max = max(float(error.max()) for error in errors)
    titles = ["Ground truth", "FinalProj", "PCM-NO", "FinalProj error", "PCM-NO error"]
    for i, field in enumerate(fields):
        image = axes[i].imshow(field, origin="lower", vmin=field_min, vmax=field_max)
        axes[i].set_title(titles[i])
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    for offset, error in enumerate(errors, start=3):
        axes[offset].imshow(error, origin="lower", vmin=0.0, vmax=error_max)
        axes[offset].set_title(titles[offset])
        axes[offset].set_xticks([])
        axes[offset].set_yticks([])
    fig.colorbar(image, ax=axes[:3], fraction=0.025, label=field_label)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    print(f"selected trajectory={index}")
    print(args.output)


if __name__ == "__main__":
    main()
