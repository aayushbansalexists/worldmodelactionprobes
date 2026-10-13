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
  run "cf_mdrand01_causal_s$s" --seed $s --matched-direction 0.01 --md-partner random
  run "cf_mdshuf01_causal_s$s" --seed $s --matched-direction 0.01 --md-shuffle
  run "cf_imp01_causal_s$s" --seed $s --md-impute 0.1
  run "cf_imp1_causal_s$s" --seed $s --md-impute 1.0
  run "cf_act01_causal_s$s" --seed $s --action-contrastive 0.01 --ac-temperature 0.001
  run "cf_act1_causal_s$s" --seed $s --action-contrastive 0.1 --ac-temperature 0.001
done
python -m analysis.direction_training --runs $RUNS
echo "[sweep_ablate] done"
