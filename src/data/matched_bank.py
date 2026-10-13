from __future__ import annotations

import numpy as np
import torch

from data.dataset import load_processed, load_splits


def build_bank(nz, seq_len=16, stride=2, min_action_gap=0.5, device="cpu", partner="nearest", seed=0):
    prng = np.random.default_rng(seed)
    obs, act, ids, _, _, _ = load_processed()
    id2idx = {d: i for i, d in enumerate(ids)}
    train = [d for d in load_splits()["train"] if d in id2idx]
    by_task = {}
    for d in train:
        by_task.setdefault(d.split(":")[0], []).append(d)
    ctx_o, ctx_a, a_sw, e_all, gaps, dq_all = [], [], [], [], [], []
    for task, demos in sorted(by_task.items()):
        O = [nz.norm_obs(obs[id2idx[d]]).astype(np.float32) for d in demos]
        A = [nz.norm_act(act[id2idx[d]]).astype(np.float32) for d in demos]
        cand = [(k, t) for k in range(len(demos)) for t in range(len(A[k]) - 1)]
        cs = torch.from_numpy(np.stack([O[k][t] for k, t in cand])).to(device)
        ca = torch.from_numpy(np.stack([A[k][t] for k, t in cand])).to(device)
        cdemo = torch.tensor([k for k, _ in cand], device=device)
        cdelta = torch.from_numpy(np.stack([O[k][t + 1] - O[k][t] for k, t in cand])).to(device)
        qry = [(k, t) for k in range(len(demos)) for t in range(seq_len - 1, len(A[k]) - 1, stride)]
        if not qry:
            continue
        qs = torch.from_numpy(np.stack([O[k][t] for k, t in qry])).to(device)
        qa = torch.from_numpy(np.stack([A[k][t] for k, t in qry])).to(device)
        qdemo = torch.tensor([k for k, _ in qry], device=device)
        d = torch.cdist(qs, cs)
        d[qdemo[:, None] == cdemo[None, :]] = float("inf")
        d[torch.cdist(qa, ca) < min_action_gap] = float("inf")
        j = d.argmin(1)
        ok = torch.isfinite(d.gather(1, j[:, None]))[:, 0].cpu().numpy()
        j = j.cpu().numpy()
        if partner == "random":
            fin = torch.isfinite(d).cpu().numpy()
            j = np.array([prng.choice(np.flatnonzero(row)) if row.any() else 0 for row in fin])
        for n, (k, t) in enumerate(qry):
            if not ok[n]:
                continue
            ctx_o.append(O[k][t - seq_len + 1:t + 1])
            ctx_a.append(A[k][t - seq_len + 1:t + 1])
            a_sw.append(ca[j[n]].cpu().numpy())
            e_all.append((cdelta[j[n]] - torch.from_numpy(O[k][t + 1] - O[k][t]).to(device)).cpu().numpy())
            dq_all.append(cdelta[j[n]].cpu().numpy())
            gaps.append(float(d[n, j[n]].item()))
    bank = {"ctx_obs": np.stack(ctx_o), "ctx_act": np.stack(ctx_a), "a_swap": np.stack(a_sw),
            "e": np.stack(e_all), "dq": np.stack(dq_all), "gap": np.array(gaps)}
    return bank


def matched_direction_loss(model, bank_t, n, generator=None):
    N = bank_t["e"].shape[0]
    idx = torch.randint(0, N, (n,), generator=generator).to(bank_t["e"].device)
    o = bank_t["ctx_obs"][idx]
    a = bank_t["ctx_act"][idx]
    a2 = a.clone()
    a2[:, -1] = bank_t["a_swap"][idx]
    pred = model(torch.cat([o, o], 0), torch.cat([a, a2], 0))["pred"][:, -1].float()
    r = pred[n:] - pred[:n]
    e = bank_t["e"][idx]
    cos = (r * e).sum(-1) / (r.norm(dim=-1) * e.norm(dim=-1) + 1e-8)
    return (1.0 - cos).mean()


def impute_loss(model, bank_t, n, generator=None):
    N = bank_t["e"].shape[0]
    idx = torch.randint(0, N, (n,), generator=generator).to(bank_t["e"].device)
    a2 = bank_t["ctx_act"][idx].clone()
    a2[:, -1] = bank_t["a_swap"][idx]
    pred = model(bank_t["ctx_obs"][idx], a2)["pred"][:, -1].float()
    target = bank_t["ctx_obs"][idx][:, -1] + bank_t["dq"][idx]
    return ((pred - target) ** 2).mean()
