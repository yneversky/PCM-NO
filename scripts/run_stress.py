#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pcmno.config import load_config
from pcmno.stress import evaluate_stress_manifest, generate_stress_datasets


def parse_args():
    parser = argparse.ArgumentParser(description="Generate and evaluate PCM-NO stress sets.")
    parser.add_argument("--config", default="configs/stress.yaml")
    parser.add_argument("--pns-config", default="configs/pns_paper.yaml")
    parser.add_argument("--lc-config", default="configs/lc_paper.yaml")
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stress"))
    parser.add_argument("--skip-generation", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    stress_cfg = load_config(args.config)
    pns_cfg = load_config(args.pns_config)
    lc_cfg = load_config(args.lc_config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "stress_manifest.json"
    if args.skip_generation:
        manifest = {key: Path(value) for key, value in json.loads(manifest_path.read_text()).items()}
    else:
        manifest = generate_stress_datasets(stress_cfg, args.output_dir / "data")
        manifest_path.write_text(
            json.dumps({key: str(value) for key, value in manifest.items()}, indent=2) + "\n",
            encoding="utf-8",
        )
    checkpoints = json.loads(args.checkpoints.read_text(encoding="utf-8"))
    frame = evaluate_stress_manifest(manifest, pns_cfg, lc_cfg, checkpoints)
    frame.to_csv(args.output_dir / "stress_summary.csv", index=False)
    print(frame)


if __name__ == "__main__":
    main()
