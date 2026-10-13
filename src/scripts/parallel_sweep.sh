#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh >/dev/null 2>&1
BASE="--config configs/small.yaml --R vector --F add --T step --W every_block"
LOCKDIR="${LOCKDIR:-/tmp/wmprobe_sweep_locks}"; mkdir -p "$LOCKDIR"
while read -r name args; do
  [ -z "$name" ] && continue
  [ -f "results/raw/$name/metrics.json" ] && continue
  mkdir "$LOCKDIR/$name" 2>/dev/null || continue
  echo "[w$1 run] $name"
  python train.py $BASE --run-id "$name" $args >"results/raw/${name}.log" 2>&1 || echo "[w$1 FAIL] $name"
done < scripts/validity_jobs.txt
echo "[w$1] done"
