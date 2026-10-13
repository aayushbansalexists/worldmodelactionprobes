from __future__ import annotations
import numpy as np
import torch


@torch.no_grad()
def rollout_horizons(model, obs_seq, act_seq, horizons, device="cuda"):
    L = model.cfg.get("data", {}).get("seq_len", 16)
    T = obs_seq.shape[0]
    Hmax = max(horizons)
    assert T >= L + Hmax, f"trajectory {T} too short for L={L}+H={Hmax}"
    o = obs_seq.to(device).unsqueeze(0)
    a = act_seq.to(device).unsqueeze(0)
    win_o = o[:, :L].clone()
    win_a = a[:, :L].clone()
    sq_err = {h: [] for h in horizons}
    for step in range(1, Hmax + 1):
        out = model(win_o, win_a)
        pred_next = out["pred"][:, -1:]
        true_next = o[:, L + step - 1:L + step]
        err = ((pred_next - true_next) ** 2).mean(dim=-1)
        for h in horizons:
            if step == h:
                sq_err[h].append(err.mean().item())
        win_o = torch.cat([win_o[:, 1:], pred_next], dim=1)
        win_a = torch.cat([win_a[:, 1:], a[:, L + step:L + step + 1]], dim=1)
    return {h: float(np.mean(v)) if v else float("nan") for h, v in sq_err.items()}


@torch.no_grad()
def eval_prediction(model, ds_test, horizons=(1, 4, 8, 16), device="cuda", max_seqs=256):
    model.eval()
    L = model.cfg.get("data", {}).get("seq_len", 16)
    results = {h: [] for h in horizons}
    n = 0
    for i in range(len(ds_test)):
        o, a = ds_test[i]
        T = o.shape[0]
        if T < L + max(horizons):
            continue
        r = rollout_horizons(model, o, a, horizons, device)
        for h in horizons:
            results[h].append(r[h])
        n += 1
        if n >= max_seqs:
            break
    summary = {f"mse_h{h}": float(np.mean(results[h])) for h in horizons}
    summary["n_eval_seqs"] = n
    return summary


__all__ = ["rollout_horizons", "eval_prediction"]
