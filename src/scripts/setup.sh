#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"

echo "[setup] PROJECT_ROOT=$PROJECT_ROOT"
echo "[setup] creating .venv with python3.12 (explicitly NOT system/global)..."
if [ ! -x "$VIRTUAL_ENV/bin/python" ]; then
  python3.12 -m venv "$VIRTUAL_ENV"
fi

echo "[setup] upgrading pip/setuptools/wheel inside .venv only..."
python -m pip install --upgrade pip setuptools wheel

echo "[setup] installing requirements.txt into .venv only..."
python -m pip install -r "$PROJECT_ROOT/requirements.txt"

echo "[setup] verifying caches are project-local..."
python - "$PROJECT_ROOT" <<'PY'
import os, sys, torch
root = sys.argv[1]
checks = {
    "HF_HOME": os.environ.get("HF_HOME"),
    "TORCH_HOME": os.environ.get("TORCH_HOME"),
    "XDG_CACHE_HOME": os.environ.get("XDG_CACHE_HOME"),
}
for k, v in checks.items():
    inside = v and os.path.realpath(v).startswith(os.path.realpath(root))
    print(f"  {k}={v}  -> {'OK (project-local)' if inside else '!! NOT project-local'}")
print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda device:", torch.cuda.get_device_name(0))
    print("cuda version (torch):", torch.version.cuda)
PY

echo "[setup] done. Activate with: source scripts/env.sh"
