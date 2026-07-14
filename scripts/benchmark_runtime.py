#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch

from pcmno.config import load_config
from pcmno.data import LCFullTrajectoryDataset, PNSFullTrajectoryDataset
from pcmno.evaluation import load_checkpoint_model
from pcmno.factory import build_dynamics, dataset_name, dataset_paths
from pcmno.operators.mac import lc_solver_step
from pcmno.utils import choose_device, cuda_sync


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark learned transitions and reference solvers.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, action="append", default=[1, 8])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("outputs/runtime.csv"))
    return parser.parse_args()


def timed(callable_, warmup: int, repeats: int) -> tuple[float, float]:
    for _ in range(warmup):
        callable_()
    cuda_sync()
    samples = []
    for _ in range(repeats):
        cuda_sync()
        start = time.perf_counter()
        callable_()
        cuda_sync()
        samples.append(1000.0 * (time.perf_counter() - start))
    values = torch.tensor(samples)
    return float(values.mean()), float(values.std(unbiased=True))


def main():
    args = parse_args()
    cfg = load_config(args.config)
    checkpoints = json.loads(args.checkpoints.read_text(encoding="utf-8"))
    device = choose_device(cfg.get("runtime", {}).get("device", "auto"))
    dataset = dataset_name(cfg)
    paths = dataset_paths(cfg)
    rows = []
    for batch_size in args.batch_size:
        if dataset == "pns":
            data = PNSFullTrajectoryDataset(paths["test"])
            state = torch.stack([data[i][0][0] for i in range(batch_size)]).to(device)
            viscosity = torch.stack([data[i][1] for i in range(batch_size)]).to(device)
            dynamics = build_dynamics(cfg, device, resolution=state.shape[-1])
            reference_substeps = int(cfg["data_generation"]["substeps"])
            reference_dt = float(cfg["physics"]["saved_interval"]) / reference_substeps
            vorticity = dynamics.operators.vorticity(state)

            def reference_call():
                current = vorticity
                for _ in range(reference_substeps):
                    current = dynamics.operators.rk4_step(current, viscosity, reference_dt)
                return dynamics.operators.velocity_from_vorticity(current)

            mean, std = timed(reference_call, args.warmup, args.repeats)
            rows.append({"dataset": dataset, "batch": batch_size, "method": "Reference solver", "StepMsMean": mean, "StepMsStd": std})
            for method, checkpoint in checkpoints.items():
                model, _ = load_checkpoint_model(checkpoint, cfg, method, device, paths["test"])
                def learned_call(model=model, method=method):
                    return dynamics.step(method, model, state, viscosity)[0]
                mean, std = timed(learned_call, args.warmup, args.repeats)
                rows.append({"dataset": dataset, "batch": batch_size, "method": method, "StepMsMean": mean, "StepMsStd": std})
        else:
            data = LCFullTrajectoryDataset(paths["archive"], "test")
            state = torch.stack([data[i][0][0] for i in range(batch_size)]).to(device)
            lid = torch.stack([data[i][1] for i in range(batch_size)]).to(device)
            viscosity = torch.stack([data[i][2] for i in range(batch_size)]).to(device)
            dynamics = build_dynamics(cfg, device)
            reference_substeps = int(cfg["data_generation"]["inner_steps"])
            def reference_call():
                current = state
                for _ in range(reference_substeps):
                    current = lc_solver_step(
                        current, viscosity, lid,
                        dt=float(cfg["data_generation"]["dt"]),
                        drag=float(cfg["physics"]["drag"]),
                        projection_iterations=int(cfg["data_generation"]["projection_iterations"]),
                        projection_tolerance=float(cfg["data_generation"]["projection_tolerance"]),
                        projection_damping=float(cfg["data_generation"]["projection_damping"]),
                    )
                return current
            mean, std = timed(reference_call, args.warmup, args.repeats)
            rows.append({"dataset": dataset, "batch": batch_size, "method": "Reference solver", "StepMsMean": mean, "StepMsStd": std})
            for method, checkpoint in checkpoints.items():
                model, _ = load_checkpoint_model(checkpoint, cfg, method, device, paths["archive"])
                def learned_call(model=model, method=method):
                    return dynamics.step(method, model, state, lid, viscosity)[0]
                mean, std = timed(learned_call, args.warmup, args.repeats)
                rows.append({"dataset": dataset, "batch": batch_size, "method": method, "StepMsMean": mean, "StepMsStd": std})
    frame = pd.DataFrame(rows)
    frame["Rollout50MsMean"] = 50.0 * frame["StepMsMean"]
    frame["Rollout50MsStd"] = 50.0 * frame["StepMsStd"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(frame)


if __name__ == "__main__":
    main()
