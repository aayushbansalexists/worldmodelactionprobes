import numpy as np
import torch

from .audit import rollout


def find_pairs(start_states, traj_ids, k=1, scale=None):
    S = np.asarray(start_states, dtype=np.float64)
    tid = np.asarray(traj_ids)
    sd = np.asarray(scale, dtype=np.float64) if scale is not None else S.std(0) + 1e-8
    Z = S / sd
    ii, jj, gg = [], [], []
    for i in range(len(Z)):
        d = np.linalg.norm(Z - Z[i], axis=1)
        d[tid == tid[i]] = np.inf
        for j in np.argsort(d)[:k]:
            if np.isfinite(d[j]):
                ii.append(i), jj.append(int(j)), gg.append(float(d[j]))
    return np.array(ii), np.array(jj), np.array(gg)


def _cos_rows(a, b):
    a, b = a.reshape(len(a), -1), b.reshape(len(b), -1)
    na, nb = a.norm(dim=1), b.norm(dim=1)
    c = (a * b).sum(1) / (na * nb).clamp_min(1e-12)
    return torch.where((na > 0) & (nb > 0), c, torch.zeros_like(c))


@torch.no_grad()
def matched_probe(model, ctx_obs, ctx_act, act, future_obs, traj_ids, k=1, transform=None,
                  pairs=None):
    if pairs is None:
        ii, jj, gap = find_pairs(ctx_obs[:, -1].cpu().numpy(), traj_ids, k=k)
    else:
        ii, jj = pairs
        gap = np.full(len(ii), np.nan)
    ii_t, jj_t = torch.as_tensor(ii, device=ctx_obs.device), torch.as_tensor(jj, device=ctx_obs.device)
    s = ctx_obs[:, -1:]
    est = (future_obs[jj_t] - s[jj_t]) - (future_obs[ii_t] - s[ii_t])
    co, ca = ctx_obs[ii_t], ctx_act[ii_t]
    resp = rollout(model, co, ca, act[jj_t]) - rollout(model, co, ca, act[ii_t])
    if transform is not None:
        resp = resp @ torch.as_tensor(transform, dtype=resp.dtype, device=resp.device).T
    c = _cos_rows(resp, est)
    return {"score": float(c.mean()), "per_item": c.cpu().numpy(), "idx_i": ii, "idx_j": jj,
            "gap": gap}
