#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from pcmno.plotting import plot_long_rollout, plot_residual_reset


def parse_args():
    parser = argparse.ArgumentParser(description="Generate paper-style figures from exported CSVs.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    rollout = subparsers.add_parser("long-rollout")
    rollout.add_argument("--curves", type=Path, nargs="+", required=True)
    rollout.add_argument("--output", type=Path, required=True)
    reset = subparsers.add_parser("residual-reset")
    reset.add_argument("--csv", type=Path, required=True)
    reset.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "long-rollout":
        print(plot_long_rollout(args.curves, args.output))
    else:
        print(plot_residual_reset(args.csv, args.output))


if __name__ == "__main__":
    main()
