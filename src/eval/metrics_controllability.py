from __future__ import annotations
import numpy as np
import torch


@torch.no_grad()
def rollout_with_actions(model, ctx_obs, ctx_act, future_act, H, device="cuda"):
    L = model.cfg.get("data", {}).get("seq_len", 16)
    win_o = ctx_obs.to(device).unsqueeze(0).clone()
    win_a = ctx_act.to(device).unsqueeze(0).clone()
    fa = future_act.to(device).unsqueeze(0)
    preds = []
    for step in range(H):
        out = model(win_o, win_a)
        pred = out["pred"][:, -1:]
        preds.append(pred.squeeze(0))
        win_o = torch.cat([win_o[:, 1:], pred], dim=1)
        win_a = torch.cat([win_a[:, 1:], fa[:, step:step + 1]], dim=1)
    return torch.stack(preds, dim=0)


def _norm_clip(x, lo=0.0, hi=1.0):
    return float(np.clip(x, lo, hi))


@torch.no_grad()
def eval_controllability(model, ds_test, horizons=(1, 4, 8, 16),
                         mask_rates=(0.0, 0.25, 0.5, 0.75), device="cuda", max_seqs=64, rng_seed=0):
    model.eval()
    L = model.cfg.get("data", {}).get("seq_len", 16)
    Hmax = max(horizons)
    rng = np.random.default_rng(rng_seed)

    var_proxy, nsamp = 0.0, 0
    cand = []
    for i in range(len(ds_test)):
        o, a = ds_test[i]
        if o.shape[0] < L + Hmax + Hmax:
            continue
        cand.append((o, a))
        var_proxy += o.var(dim=0).mean().item()
        nsamp += 1
        if nsamp >= max_seqs:
            break
    if not cand:
        return {"ctrl_error": "no long-enough test trajectories"}
    var_proxy = max(var_proxy / nsamp, 1e-6)
    n = len(cand)

    C1, C2, C3, C4 = [], [], [], []

    for (o, a) in cand:
        T = o.shape[0]
        ctx_o = o[:L]
        ctx_a = a[:L]
        future_a = a[L:L + Hmax]
        H = future_a.shape[0]
        true_future = o[L:L + Hmax]

        pred_true = rollout_with_actions(model, ctx_o, ctx_a, future_a, H, device)
        zero_act = torch.zeros_like(future_a)
        pred_zero = rollout_with_actions(model, ctx_o, ctx_a, zero_act, H, device)
        d_zero = ((pred_true - pred_zero) ** 2).mean().item()
        C1.append(d_zero / (2.0 * var_proxy))

        perm = torch.from_numpy(rng.permutation(H))
        shuf_act = future_a[perm]
        pred_shuf = rollout_with_actions(model, ctx_o, ctx_a, shuf_act, H, device)
        mse_true = ((pred_true - true_future.to(device)) ** 2).mean().item()
        mse_shuf = ((pred_shuf - true_future.to(device)) ** 2).mean().item()
        C2.append(max(0.0, mse_shuf - mse_true) / (mse_true + 1e-6))

        A = future_a.shape[1]
        sens = []
        for mr in mask_rates:
            if mr == 0.0:
                continue
            mask = torch.from_numpy(rng.random(A) < mr).float()
            masked_act = future_a * (1 - mask)
            pred_masked = rollout_with_actions(model, ctx_o, ctx_a, masked_act, H, device)
            mse_m = ((pred_masked - true_future.to(device)) ** 2).mean().item()
            sens.append(max(0.0, mse_m - mse_true) / (mse_true + 1e-6))
        C3.append(float(np.mean(sens)) if sens else 0.0)

        mags = [0.25, 0.5, 1.0, 2.0, 4.0]
        rhos, gains = [], []
        for dim in range(A):
            changes = []
            for m in mags:
                ca = ctx_a.clone()
                ca[-1, dim] = ca[-1, dim] + m
                pred_p = rollout_with_actions(model, ctx_o, ca, future_a[:1], 1, device)
                ds = ((pred_p[0] - pred_true[0]) ** 2).mean().sqrt().item()
                changes.append(ds)
            if len(changes) >= 2 and np.ptp(changes) > 0:
                r = np.corrcoef(mags, changes)[0, 1]
                rhos.append(max(0.0, 0.0 if np.isnan(r) else r))
                gains.append(float(np.mean(changes)))
        prop = float(np.mean(rhos)) if rhos else 0.0
        sens = float(np.mean(gains) / np.sqrt(var_proxy + 1e-6)) if gains else 0.0
        sens = float(np.clip(sens, 0.0, 1.0))
        C4.append(prop * sens)

    c1 = _norm_clip(np.mean(C1))
    c2 = _norm_clip(np.mean(C2))
    c3 = _norm_clip(np.mean(C3))
    c4 = _norm_clip(np.mean(C4))
    ctrlscore = 0.30 * c1 + 0.20 * c2 + 0.20 * c3 + 0.30 * c4

    return {
        "C1_D_zero": float(np.mean(C1)),
        "C2_Delta_shuf": float(np.mean(C2)),
        "C3_mask_sens": float(np.mean(C3)),
        "C4_cf_separation": float(np.mean(C4)),
        "c1_norm": c1, "c2_norm": c2, "c3_norm": c3, "c4_norm": c4,
        "CtrlScore": float(ctrlscore),
        "n_ctrl_seqs": n,
    }


__all__ = ["eval_controllability", "rollout_with_actions"]
