#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh >/dev/null 2>&1
BASE="--config configs/small.yaml --R vector --F add --T step --W every_block --steps 5000"
RUNS=""
for s in 0 1 2 3 4; do
  for w in 01:0.01 1:0.1; do
    tag=${w%%:*}; val=${w##*:}; name="cf_act${tag}_causal_s$s"
    RUNS="$RUNS $name"
    if [ -f "results/raw/$name/metrics.json" ]; then echo "[skip] $name"; continue; fi
    echo "[run ] $name"
    python train.py $BASE --seed $s --run-id "$name" --action-contrastive $val --ac-temperature 0.001 >"results/raw/${name}.log" 2>&1 \
      || echo "[FAIL] $name (see results/raw/${name}.log)"
  done
done
python -m analysis.direction_training --runs $RUNS
echo "[sweep_ac] done"
