#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pcmno.config import apply_overrides, load_config
from pcmno.evaluation import evaluate_checkpoint
from pcmno.training import train
from pcmno.utils import mean_std

VARIANTS = ("tangent_only", "retraction_only", "div_only", "pcmno")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the LC structural ablation.")
    parser.add_argument("--config", default="configs/lc_ablation.yaml")
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ablation_lc"))
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.overrides)
    seeds = args.seed or [int(x) for x in cfg["training"]["seeds"]]
    rows = []
    for variant in VARIANTS:
        for seed in seeds:
            checkpoint = train(cfg, variant, seed, output_root=args.output_dir / "checkpoints")
            _, summary = evaluate_checkpoint(
                cfg, variant, checkpoint, split="test", horizons=[50]
            )
            summary.update(variant=variant, seed=seed, checkpoint=str(checkpoint))
            rows.append(summary)
            pd.DataFrame(rows).to_csv(args.output_dir / "ablation_seed_level.csv", index=False)
    seed_frame = pd.DataFrame(rows)
    summary_rows = []
    metrics = ["RelL2_50", "LogDiv_50", "LogBC_50", "ProfileErr_50"]
    for variant, group in seed_frame.groupby("variant", sort=False):
        item = {"variant": variant, "n_seeds": group["seed"].nunique()}
        for metric in metrics:
            mean, std = mean_std(group[metric], ddof=1)
            item[f"{metric}_mean"] = mean
            item[f"{metric}_std"] = std
        summary_rows.append(item)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "ablation_summary.csv", index=False)
    (args.output_dir / "manifest.json").write_text(
        json.dumps({"variants": VARIANTS, "seeds": seeds, "config": args.config}, indent=2) + "\n"
    )
    print(summary)


if __name__ == "__main__":
    main()
