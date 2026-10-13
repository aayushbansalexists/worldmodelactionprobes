#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh >/dev/null 2>&1

STEPS=${STEPS:-5000}
SEEDS=${SEEDS:-"0 1 2 3 4"}
BASE="--config configs/small.yaml --R vector --F add --T step --W every_block --steps $STEPS"

run () {
  local name=$1; shift
  if [ -f "results/raw/$name/metrics.json" ]; then echo "[skip] $name"; return; fi
  echo "[run ] $name"
  python train.py $BASE --run-id "$name" "$@" >"results/raw/${name}.log" 2>&1 \
    || echo "[FAIL] $name (see results/raw/${name}.log)"
}

for s in $SEEDS; do
  for cond in causal shuf; do
    SH=""; [ "$cond" = "shuf" ] && SH="--shuffle-actions"
    run "cf_add_${cond}_s$s"   --seed $s $SH
    run "cf_agm_${cond}_s$s"   --seed $s $SH --agm-weight 0.5
    run "cf_idm_${cond}_s$s"   --seed $s $SH --inverse-dynamics 1.0
    run "cf_delta_${cond}_s$s" --seed $s $SH --delta-prediction
  done
done
echo "[sweep] done"
