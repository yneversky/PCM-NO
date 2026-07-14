#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Plot LC projection-solver sensitivity.")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    frame = pd.read_csv(args.csv)
    tolerance = sorted(frame["tolerance"].unique())[len(frame["tolerance"].unique()) // 2]
    subset = frame[frame["tolerance"] == tolerance].sort_values("iterations")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    axes[0].plot(subset["iterations"], subset["LogDiv"], marker="o", label="Div")
    axes[0].plot(subset["iterations"], subset["LogBC"], marker="o", label="BC")
    axes[0].plot(
        subset["iterations"],
        subset["LinearRelativeResidual"].clip(lower=1e-12).map(lambda x: __import__('math').log10(x)),
        marker="o",
        label="CG relative residual",
    )
    axes[0].set_xlabel("CG iterations")
    axes[0].set_ylabel("log10 residual")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(subset["iterations"], subset["RuntimeMsPerBatch"], marker="o")
    axes[1].set_xlabel("CG iterations")
    axes[1].set_ylabel("Runtime (ms/batch)")
    axes[1].grid(alpha=0.25)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
