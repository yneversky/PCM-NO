#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from pcmno.config import apply_overrides, load_config
from pcmno.data.lc import generate_lc_dataset
from pcmno.data.pns import generate_pns_split
from pcmno.factory import dataset_name
from pcmno.utils import choose_device


def parse_args():
    parser = argparse.ArgumentParser(description="Generate PCM-NO datasets from solver code.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.overrides)
    device = choose_device(cfg.get("runtime", {}).get("device", "auto"))
    if dataset_name(cfg) == "pns":
        splits = cfg["splits"]
        root = args.output or Path("generated_data/pns")
        root.mkdir(parents=True, exist_ok=True)
        offsets = {"train": 10_000, "val": 20_000, "test": 30_000}
        for split, count in splits.items():
            path = root / f"{split}.pt"
            generate_pns_split(path, int(count), cfg, device, offsets[split])
            print(path)
    else:
        output = args.output or Path("generated_data/lc/lc_dataset.npz")
        path = generate_lc_dataset(output, cfg, device)
        print(path)


if __name__ == "__main__":
    main()
