import numpy as np
import torch

TRANSFORMS = ("identity", "reflect", "rotate", "scale_0.25", "scale_4", "blind")


def transform_matrix(name, dim, seed=0):
    if name == "identity":
        return torch.eye(dim)
    if name == "reflect":
        return -torch.eye(dim)
    if name == "rotate":
        g = np.random.default_rng(seed)
        q, r = np.linalg.qr(g.standard_normal((dim, dim)))
        return torch.from_numpy((q * np.sign(np.diag(r))).astype(np.float32))
    if name.startswith("scale_"):
        return float(name.split("_")[1]) * torch.eye(dim)
    if name == "blind":
        return torch.zeros(dim, dim)
    raise ValueError(f"unknown transform {name!r}")


@torch.no_grad()
def rollout(model, ctx_obs, ctx_act, future_act):
    if hasattr(model, "eval"):
        model.eval()
    wo, wa = ctx_obs.clone(), ctx_act.clone()
    wa[:, -1] = future_act[:, 0]
    preds = []
    for k in range(future_act.shape[1]):
        p = model(wo, wa)[:, -1]
        preds.append(p)
        wo = torch.cat([wo[:, 1:], p[:, None]], 1)
        if k + 1 < future_act.shape[1]:
            wa = torch.cat([wa[:, 1:], future_act[:, k + 1:k + 2]], 1)
    return torch.stack(preds, 1)


def transformed_rollouts(model, ctx_obs, ctx_act, ref_act, other_acts, transform, seed=0):
    p_ref = rollout(model, ctx_obs, ctx_act, ref_act)
    T = transform_matrix(transform, p_ref.shape[-1], seed).to(p_ref)
    outs = [p_ref + (rollout(model, ctx_obs, ctx_act, a) - p_ref) @ T.T for a in other_acts]
    return p_ref, outs


@torch.no_grad()
def audit_direction(model, obs, act, ctx_len=16, horizon=8, n_positions=16, n_candidates=16,
                    sigma=0.5, small_ratio=0.25, seed=0):
    g = torch.Generator().manual_seed(seed)
    T_ = obs.shape[0]
    pos = np.linspace(ctx_len, T_ - horizon - 1, n_positions).astype(int)
    if len(pos) == 0 or pos.min() < ctx_len:
        raise ValueError("trajectory too short for this ctx_len and horizon")
    dev = obs.device
    ctx_o = torch.stack([obs[p - ctx_len + 1:p + 1] for p in pos])
    ctx_a = torch.stack([act[p - ctx_len + 1:p + 1] for p in pos])
    ref = torch.stack([act[p:p + horizon] for p in pos])
    zero = torch.zeros_like(ref)
    d = torch.randn(ref.shape[0], 1, ref.shape[-1], generator=g).to(dev)
    d = d / d.norm(dim=-1, keepdim=True)
    large, small = ref + sigma * d, ref + sigma * small_ratio * d
    cand = ref[:, :1].repeat(1, n_candidates, 1)
    cand[:, 1:] += torch.randn(cand[:, 1:].shape, generator=g).to(dev) * sigma
    next_obs = torch.stack([obs[p + 1] for p in pos])

    out = {}
    for t in TRANSFORMS:
        p_ref, (p_zero, p_small, p_large) = transformed_rollouts(
            model, ctx_o, ctx_a, ref, [zero, small, large], t, seed)
        zad = float(((p_zero - p_ref) ** 2).mean())
        ds = ((p_small - p_ref) ** 2).mean((1, 2))
        dl = ((p_large - p_ref) ** 2).mean((1, 2))
        ratio = float((ds / dl.clamp_min(1e-12)).mean()) if float(dl.max()) > 1e-12 else float("nan")
        preds = []
        for c in range(n_candidates):
            preds.append(rollout(model, ctx_o, ctx_a, cand[:, c:c + 1])[:, 0])
        P = torch.stack(preds, 1)
        Tm = transform_matrix(t, P.shape[-1], seed).to(P)
        P = P[:, :1] + (P - P[:, :1]) @ Tm.T
        dist = ((P - next_obs[:, None]) ** 2).mean(-1).sqrt().double()
        dist = dist + torch.randn(dist.shape, generator=g, dtype=torch.float64).to(dev) \
            * 1e-9 * (dist.abs().mean() + 1e-12)
        rank = dist.argsort(1).argsort(1)[:, 0].float().mean().item()
        out[t] = {"zero_action_div": zad, "calib_ratio": ratio, "retrieval_rank": rank}
    return out
