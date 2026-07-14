#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pcmno.utils import mean_std


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate per-seed evaluation summaries.")
    parser.add_argument("--summaries", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in args.summaries]
    frame = pd.DataFrame(rows)
    group_columns = [column for column in ("dataset", "method", "split", "setting") if column in frame]
    metric_columns = [
        column for column in frame.select_dtypes(include="number").columns
        if column not in {"seed"}
    ]
    output_rows = []
    for key, group in frame.groupby(group_columns, dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        item = dict(zip(group_columns, key))
        item["n_seeds"] = group["seed"].nunique() if "seed" in group else len(group)
        for column in metric_columns:
            mean, std = mean_std(group[column], ddof=1)
            item[f"{column}_mean"] = mean
            item[f"{column}_std"] = std
        output_rows.append(item)
    output = pd.DataFrame(output_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(output)


if __name__ == "__main__":
    main()
