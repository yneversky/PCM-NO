#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from pcmno.config import apply_overrides, load_config
from pcmno.data import LCFullTrajectoryDataset
from pcmno.diagnostics import (
    lc_projection_sensitivity,
    residual_accumulation_reset,
    tangent_filtering_diagnostic,
)
from pcmno.evaluation import load_checkpoint_model
from pcmno.factory import dataset_name, dataset_paths
from pcmno.plotting import plot_residual_reset
from pcmno.utils import choose_device


def parse_args():
    parser = argparse.ArgumentParser(description="Run PCM-NO mechanism diagnostics.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--diagnostic",
        choices=["tangent", "residual-reset", "projection-sensitivity", "all"],
        default="all",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/diagnostics"))
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--max-trajectories", type=int)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.overrides)
    device = choose_device(cfg.get("runtime", {}).get("device", "auto"))
    paths = dataset_paths(cfg)
    data_path = paths.get("archive", paths.get("test"))
    model, _ = load_checkpoint_model(args.checkpoint, cfg, "pcmno", device, data_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.diagnostic in {"tangent", "all"}:
        frame = tangent_filtering_diagnostic(
            cfg, model, "pcmno", device, max_batches=args.max_batches
        )
        frame.to_csv(args.output_dir / f"{dataset_name(cfg)}_tangent_filtering.csv", index=False)
        print(frame.describe())

    if args.diagnostic in {"residual-reset", "all"} and dataset_name(cfg) == "lc":
        frame = residual_accumulation_reset(
            cfg, model, device, max_trajectories=args.max_trajectories
        )
        path = args.output_dir / "lc_residual_accumulation_reset.csv"
        frame.to_csv(path, index=False)
        plot_residual_reset(path, args.output_dir / "lc_residual_accumulation_reset.pdf")

    if args.diagnostic in {"projection-sensitivity", "all"} and dataset_name(cfg) == "lc":
        dataset = LCFullTrajectoryDataset(paths["archive"], "test")
        count = min(32, len(dataset))
        states = torch.stack([dataset[i][0][50] for i in range(count)]).to(device)
        lids = torch.stack([dataset[i][1] for i in range(count)]).to(device)
        frame = lc_projection_sensitivity(states, lids)
        frame.to_csv(args.output_dir / "lc_projection_sensitivity.csv", index=False)
        print(frame)


if __name__ == "__main__":
    main()
