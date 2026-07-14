from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_long_rollout(curves: list[str | Path], output: str | Path) -> Path:
    frames = [pd.read_csv(path) for path in curves]
    frame = pd.concat(frames, ignore_index=True)
    if "dataset" not in frame or "method" not in frame:
        raise KeyError("Curve CSVs must contain dataset and method columns.")
    metrics_by_dataset = {
        "pns": ["RelL2", "LogDiv", "EnergyRelErr"],
        "lc": ["RelL2", "LogDiv", "ProfileErr"],
    }
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2), constrained_layout=True)
    for row, dataset in enumerate(("pns", "lc")):
        subset = frame[frame["dataset"] == dataset]
        for column, metric in enumerate(metrics_by_dataset[dataset]):
            ax = axes[row, column]
            for method, group in subset.groupby("method"):
                group = group.sort_values("step")
                ax.plot(group["step"], group[metric], label=method)
            ax.set_xlabel("Rollout horizon")
            ax.set_ylabel(metric)
            ax.set_title(f"{dataset.upper()} {metric}")
            ax.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_residual_reset(csv_path: str | Path, output: str | Path) -> Path:
    frame = pd.read_csv(csv_path)
    summary = frame.groupby(["branch", "step"], as_index=False)["LogDiv"].agg(["mean", "std"])
    summary = summary.reset_index()
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for branch, group in summary.groupby("branch"):
        group = group.sort_values("step")
        ax.plot(group["step"], group["mean"], label=branch)
        ax.fill_between(
            group["step"], group["mean"] - group["std"], group["mean"] + group["std"], alpha=0.2
        )
    ax.set_xlabel("Rollout horizon")
    ax.set_ylabel("LogDiv")
    ax.grid(alpha=0.25)
    ax.legend()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output
