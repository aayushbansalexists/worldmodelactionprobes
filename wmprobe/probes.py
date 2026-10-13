import numpy as np
import torch


def _prep(model):
    if hasattr(model, "eval"):
        model.eval()
    return model


def _rng(generator):
    if isinstance(generator, np.random.Generator):
        return generator
    return np.random.default_rng(generator)


def _rollout(model, ctx_obs, ctx_act, future_act, horizon):
    if future_act.shape[1] < horizon:
        raise ValueError(
            f"future_act has {future_act.shape[1]} steps but horizon is {horizon}; "
            "the trajectory is too short for this probe")
    win_o = ctx_obs.clone()
    win_a = ctx_act.clone()
    preds = []
    for step in range(horizon):
        pred = model(win_o, win_a)[0, -1]
        preds.append(pred)
        win_o = torch.cat([win_o[:, 1:], pred.unsqueeze(0).unsqueeze(0)], dim=1)
        win_a = torch.cat([win_a[:, 1:], future_act[:, step:step + 1]], dim=1)
    return torch.stack(preds, 0)


def zero_action_divergence(model, obs, act, ctx_len, horizon=8):
    _prep(model)
    ctx_o, ctx_a = obs[:ctx_len][None], act[:ctx_len][None]
    fa = act[ctx_len:ctx_len + horizon][None]
    with torch.no_grad():
        p_true = _rollout(model, ctx_o, ctx_a, fa, horizon)
        p_zero = _rollout(model, ctx_o, ctx_a, torch.zeros_like(fa), horizon)
    return ((p_true - p_zero) ** 2).mean().item()


def shuffled_action_degradation(model, obs, act, ctx_len, horizon=8, generator=None):
    _prep(model)
    rng = _rng(generator)
    ctx_o, ctx_a = obs[:ctx_len][None], act[:ctx_len][None]
    fa = act[ctx_len:ctx_len + horizon][None]
    true_future = obs[ctx_len:ctx_len + horizon]
    perm = torch.from_numpy(rng.permutation(horizon)).to(fa.device)
    with torch.no_grad():
        p_true = _rollout(model, ctx_o, ctx_a, fa, horizon)
        p_shuf = _rollout(model, ctx_o, ctx_a, fa[:, perm], horizon)
    mse_true = ((p_true - true_future) ** 2).mean().item()
    mse_shuf = ((p_shuf - true_future) ** 2).mean().item()
    return max(0.0, mse_shuf - mse_true) / (mse_true + 1e-6)


def mask_sensitivity(model, obs, act, ctx_len, horizon=8, mask_rates=(0.25, 0.5, 0.75),
                     generator=None):
    _prep(model)
    rng = _rng(generator)
    ctx_o, ctx_a = obs[:ctx_len][None], act[:ctx_len][None]
    fa = act[ctx_len:ctx_len + horizon][None]
    true_future = obs[ctx_len:ctx_len + horizon]
    with torch.no_grad():
        p_true = _rollout(model, ctx_o, ctx_a, fa, horizon)
        mse_true = ((p_true - true_future) ** 2).mean().item()
        sens = []
        for mr in mask_rates:
            mask = torch.from_numpy((rng.random(fa.shape[-1]) < mr).astype(np.float32)).to(fa.device)
            p_m = _rollout(model, ctx_o, ctx_a, fa * (1 - mask), horizon)
            mse_m = ((p_m - true_future) ** 2).mean().item()
            sens.append(max(0.0, mse_m - mse_true) / (mse_true + 1e-6))
    return float(np.mean(sens)) if sens else 0.0


def dose_response(model, obs, act, ctx_len, mags=(0.5, 1.0, 2.0)):
    _prep(model)
    ctx_o, ctx_a = obs[:ctx_len][None], act[:ctx_len][None]
    fa = act[ctx_len:ctx_len + 1][None]
    with torch.no_grad():
        p_base = _rollout(model, ctx_o, ctx_a, fa, 1)[0]
        rhos, gains = [], []
        for dim in range(act.shape[-1]):
            changes = []
            for m in mags:
                ca = ctx_a.clone(); ca[0, -1, dim] += m
                p = _rollout(model, ctx_o, ca, fa, 1)[0]
                changes.append(((p - p_base) ** 2).mean().sqrt().item())
            if np.ptp(changes) > 0:
                r = np.corrcoef(mags, changes)[0, 1]
                if not np.isnan(r):
                    rhos.append(max(0.0, r)); gains.append(float(np.mean(changes)))
    prop = float(np.mean(rhos)) if rhos else 0.0
    sens = float(np.mean(gains)) if gains else 0.0
    return prop * min(1.0, sens)


def _rank_of_logged(model, ctx_o, ctx_a, cand, target, rng):
    preds = []
    for c in range(cand.shape[0]):
        ca = ctx_a.clone(); ca[0, -1] = cand[c]
        preds.append(_rollout(model, ctx_o, ca,
                              torch.zeros(1, 1, cand.shape[-1], device=ctx_a.device), 1)[0])
    preds = torch.stack(preds)
    d = (preds - target).pow(2).mean(-1).sqrt().reshape(-1).cpu().numpy()
    d = d + rng.standard_normal(d.shape[0]) * 1e-9 * (np.abs(d).mean() + 1e-12)
    return int(np.argsort(d)[0] == 0), int(np.argsort(np.argsort(d))[0]), preds


def downstream_action_ranking(model, obs, act, ctx_len, n_candidates=16, n_positions=8,
                              generator=None, n_null_draws=8):
    _prep(model)
    rng = _rng(generator)
    hits, ranks, null_hits, pool = 0, [], [], []
    total = 0
    T = obs.shape[0]
    for p in np.linspace(0, T - ctx_len - 2, n_positions).astype(int):
        ctx_o = obs[p:p + ctx_len][None]
        ctx_a = act[p:p + ctx_len][None]
        true_a = act[p + ctx_len - 1]
        true_next = obs[p + ctx_len]
        cand = true_a.repeat(n_candidates, 1)
        cand[1:] = true_a + torch.from_numpy(
            rng.standard_normal((n_candidates - 1, true_a.shape[0])).astype(np.float32) * 0.5
        ).to(true_a.device)
        with torch.no_grad():
            hit, rank, preds = _rank_of_logged(model, ctx_o, ctx_a, cand, true_next, rng)
            hits += hit; ranks.append(rank); total += 1
            for _ in range(n_null_draws):
                if not pool:
                    break
                alt = pool[int(rng.integers(0, len(pool)))]
                d0 = (preds - alt).pow(2).mean(-1).sqrt().reshape(-1).cpu().numpy()
                d0 = d0 + rng.standard_normal(d0.shape[0]) * 1e-9 * (np.abs(d0).mean() + 1e-12)
                null_hits.append(int(np.argsort(d0)[0] == 0))
        pool.append(true_next.clone())
    top1 = hits / max(total, 1)
    null = float(np.mean(null_hits)) if null_hits else float("nan")
    return {"top1": top1, "null": null, "excess": top1 - null,
            "mean_rank": float(np.mean(ranks)) if ranks else float("nan"),
            "n_positions": total, "n_candidates": n_candidates}
