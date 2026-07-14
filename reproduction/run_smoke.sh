#!/usr/bin/env bash
set -euo pipefail

python scripts/train.py --config configs/pns_smoke.yaml --method pcmno --seed 0
python scripts/evaluate.py \
  --config configs/pns_smoke.yaml \
  --method pcmno \
  --seed 0 \
  --horizons 1 2 3

python scripts/train.py --config configs/lc_smoke.yaml --method pcmno --seed 0
python scripts/evaluate.py \
  --config configs/lc_smoke.yaml \
  --method pcmno \
  --seed 0 \
  --horizons 1 2 3
