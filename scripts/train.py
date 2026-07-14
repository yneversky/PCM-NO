#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from pcmno.config import apply_overrides, load_config
from pcmno.training import train


def parse_args():
    parser = argparse.ArgumentParser(description="Train PCM-NO and paper baselines.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", action="append", help="Method to train; repeat as needed.")
    parser.add_argument("--seed", action="append", type=int, help="Seed; repeat as needed.")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.overrides)
    methods = args.method or list(cfg["methods"])
    seeds = args.seed or [int(x) for x in cfg["training"]["seeds"]]
    for method in methods:
        for seed in seeds:
            checkpoint = train(cfg, method, seed, output_root=args.output_root)
            print(f"[done] method={method} seed={seed} checkpoint={checkpoint}")


if __name__ == "__main__":
    main()
