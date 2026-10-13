#!/usr/bin/env bash

export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export SRC_ROOT="$PROJECT_ROOT/src"

export VIRTUAL_ENV="$PROJECT_ROOT/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"

export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_HUB_OFFLINE=0

export TORCH_HOME="$PROJECT_ROOT/.cache/torch"

export XDG_CACHE_HOME="$PROJECT_ROOT/.cache/xdg"
export MPLCONFIGDIR="$PROJECT_ROOT/.cache/matplotlib"

export WANDB_DIR="$PROJECT_ROOT/.cache/wandb"
export WANDB_MODE=offline

export DATA_ROOT="$PROJECT_ROOT/data"
export CHECKPOINT_ROOT="$PROJECT_ROOT/checkpoints"
export RESULTS_ROOT="$PROJECT_ROOT/results"

export PYTHONHASHSEED=0

mkdir -p \
  "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" \
  "$TORCH_HOME" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" \
  "$WANDB_DIR" \
  "$DATA_ROOT/raw/libero" "$DATA_ROOT/raw/droid_subset" "$DATA_ROOT/processed/latents" "$DATA_ROOT/splits" \
  "$CHECKPOINT_ROOT" "$RESULTS_ROOT/raw" "$RESULTS_ROOT/tables" "$RESULTS_ROOT/figures"

if [ "${FAC_ENV_QUIET:-0}" != "1" ]; then
  echo "[env.sh] PROJECT_ROOT=$PROJECT_ROOT"
  echo "[env.sh] VIRTUAL_ENV=$VIRTUAL_ENV"
  echo "[env.sh] HF_HOME=$HF_HOME"
  echo "[env.sh] TORCH_HOME=$TORCH_HOME"
  echo "[env.sh] XDG_CACHE_HOME=$XDG_CACHE_HOME"
  echo "[env.sh] DATA_ROOT=$DATA_ROOT"
  echo "[env.sh] CHECKPOINT_ROOT=$CHECKPOINT_ROOT"
  echo "[env.sh] RESULTS_ROOT=$RESULTS_ROOT"
fi
export FAC_ENV_QUIET=1
