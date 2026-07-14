#!/usr/bin/env bash
set -euo pipefail

for seed in 0 1 2; do
  for method in fno divreg pino finalproj pcmno; do
    python scripts/train.py --config configs/pns_paper.yaml --method "$method" --seed "$seed"
    python scripts/evaluate.py --config configs/pns_paper.yaml --method "$method" --seed "$seed"
  done

done

for seed in 0 1 2; do
  for method in fno divreg pino finalproj pcmno; do
    python scripts/train.py --config configs/lc_paper.yaml --method "$method" --seed "$seed"
    python scripts/evaluate.py --config configs/lc_paper.yaml --method "$method" --seed "$seed"
  done

done
