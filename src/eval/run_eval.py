from __future__ import annotations
import torch

from .metrics_prediction import eval_prediction
from .metrics_controllability import eval_controllability


def run_eval(model, ds_val, ds_test, cfg, device="cuda", max_seqs_pred=256, max_seqs_ctrl=64):
    horizons = tuple(cfg.get("eval", {}).get("horizons", [1, 4, 8, 16]))
    pred = eval_prediction(model, ds_test, horizons=horizons, device=device, max_seqs=max_seqs_pred)
    ctrl = eval_controllability(
        model, ds_test, horizons=horizons,
        mask_rates=tuple(cfg.get("eval", {}).get("ctrl", {}).get("mask_rates", [0.0, 0.25, 0.5, 0.75])),
        device=device, max_seqs=max_seqs_ctrl,
    )
    return {"prediction": pred, "controllability": ctrl}


__all__ = ["run_eval"]
