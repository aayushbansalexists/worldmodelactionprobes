#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh >/dev/null 2>&1
BASE="--config configs/small.yaml --R vector --F add --T step --W every_block --steps 5000"
RUNS=""
run () { local name=$1; shift; RUNS="$RUNS $name"
  if [ -f "results/raw/$name/metrics.json" ]; then echo "[skip] $name"; return; fi
  echo "[run ] $name"
  python train.py $BASE --run-id "$name" "$@" >"results/raw/${name}.log" 2>&1 || echo "[FAIL] $name"; }
for s in 0 1 2 3 4; do
  run "cf_act001_causal_s$s" --seed $s --action-contrastive 0.001 --ac-temperature 0.001
  run "cf_act003_causal_s$s" --seed $s --action-contrastive 0.003 --ac-temperature 0.001
  run "cf_act01t2_causal_s$s" --seed $s --action-contrastive 0.01 --ac-temperature 0.01
done
python -m analysis.direction_training --runs $RUNS
echo "[sweep_ac_small] done"
