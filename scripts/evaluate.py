#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pcmno.config import apply_overrides, load_config
from pcmno.evaluation import evaluate_checkpoint
from pcmno.factory import dataset_name


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an autoregressive checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--data", type=Path, help="Optional ID or stress dataset override.")
    parser.add_argument("--horizons", type=int, nargs="*")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.overrides)
    output_root = Path(cfg["output"]["root"]).expanduser().resolve()
    checkpoint = args.checkpoint or (
        output_root / dataset_name(cfg) / args.method / f"seed_{args.seed}" / "best.pt"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    curve, summary = evaluate_checkpoint(
        cfg,
        method=args.method,
        checkpoint=checkpoint,
        split=args.split,
        data_override=args.data,
        horizons=args.horizons,
    )
    curve.insert(0, "seed", args.seed)
    curve.insert(0, "method", args.method)
    curve.insert(0, "dataset", dataset_name(cfg))
    output_dir = args.output_dir or checkpoint.parent / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    curve_path = output_dir / f"{args.split}_curve.csv"
    summary_path = output_dir / f"{args.split}_summary.json"
    curve.to_csv(curve_path, index=False)
    summary.update(seed=args.seed, checkpoint=str(checkpoint), split=args.split)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(curve)
    print(json.dumps(summary, indent=2))
    print("curve:", curve_path)
    print("summary:", summary_path)


if __name__ == "__main__":
    main()
