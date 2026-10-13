#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh >/dev/null 2>&1
BASE="--config configs/small.yaml --R vector --F add --T step --W every_block --steps 5000"
RUNS=""
for s in 0 1 2 3 4; do
  for w in 001:0.001 01:0.01 1:0.1; do
    tag=${w%%:*}; val=${w##*:}; name="cf_md${tag}_causal_s$s"
    RUNS="$RUNS $name"
    if [ -f "results/raw/$name/metrics.json" ]; then echo "[skip] $name"; continue; fi
    echo "[run ] $name"
    python train.py $BASE --seed $s --run-id "$name" --matched-direction $val >"results/raw/${name}.log" 2>&1 \
      || echo "[FAIL] $name (see results/raw/${name}.log)"
  done
done
BASELINES=""
for s in 0 1 2 3 4; do for o in add delta idm01 agm idm001; do BASELINES="$BASELINES cf_${o}_causal_s$s"; done; done
python -m analysis.direction_training --runs $RUNS $BASELINES
echo "[sweep_md] done"
