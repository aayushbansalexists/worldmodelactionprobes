from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from analysis.adversary_study import (ACT_D, DEV, L, OBS_D, Adversary, gen_world, gt_direction,
                                      train_policy, train_wm)


@torch.no_grad()
def matched(model, obs, act, n_pos=6):
    N, T = obs.shape[0], obs.shape[1]
    win = [(i, p) for i in range(N) for p in np.linspace(0, T - L - 2, n_pos).astype(int)]
    S = torch.stack([obs[i][p + L - 1] for i, p in win])
    traj = np.array([i for i, _ in win])
    cs, gaps = [], []
    for w, (i, p) in enumerate(win):
        d = (S - S[w]).norm(dim=1).numpy()
        d[traj == i] = np.inf
        v = int(np.argmin(d))
        j, q = win[v]
        e = (obs[j][q + L] - obs[j][q + L - 1]) - (obs[i][p + L] - obs[i][p + L - 1])
        co, ca = obs[i][p:p + L][None].to(DEV), act[i][p:p + L][None].to(DEV)
        cj = ca.clone()
        cj[0, -1] = act[j][q + L - 1].to(DEV)
        r = (model(co, cj)[0, -1] - model(co, ca)[0, -1]).cpu()
        cs.append(float((r @ e) / (r.norm() * e.norm() + 1e-12)))
        gaps.append(float(d[v]))
    return float(np.mean(cs)), float(np.median(gaps))


def run(seed=0):
    out = {}
    for world in ["G", "N"]:
        obs, act, B_eff = gen_world(world, seed=seed)
        ntr = int(0.8 * obs.shape[0])
        tr_o, tr_a = obs[:ntr], act[:ntr]
        te_o, te_a = obs[ntr:], act[ntr:]
        torch.manual_seed(seed)
        B = (torch.randn(OBS_D, ACT_D) * 0.8).to(DEV)
        abar = tr_a.reshape(-1, act.shape[-1]).mean(0).to(DEV)
        for labels in ["causal", "shuffled"]:
            a_tr = tr_a
            if labels == "shuffled":
                k = int(np.random.default_rng(seed + 1).integers(1, tr_a.shape[0]))
                a_tr = torch.roll(tr_a, shifts=k, dims=0)
            f_base = train_wm(tr_o, a_tr, agm=0.0, seed=seed)
            f_agm = train_wm(tr_o, a_tr, agm=0.5, seed=seed)
            f_blind = train_wm(tr_o, torch.zeros_like(a_tr), agm=0.0, seed=seed)
            pihat = train_policy(tr_o, a_tr, seed=seed)
            models = {
                "mse": f_base, "agm": f_agm, "blind": f_blind,
                "adv_const_c2": Adversary(f_blind, 2.0, B, pi=None, abar=abar).to(DEV).eval(),
                "adv_policy_c2": Adversary(f_blind, 2.0, B, pi=pihat).to(DEV).eval(),
                "adv_policy_c8": Adversary(f_blind, 8.0, B, pi=pihat).to(DEV).eval(),
            }
            for name, m in models.items():
                key = f"{world}/{labels}/{name}"
                mp, gap = matched(m, te_o, te_a)
                out[key] = {"matched": mp, "pair_gap": gap,
                            "direction_gt": gt_direction(m, te_o, te_a, B_eff, seed=seed)
                            if world == "G" else float("nan")}
                print(f"  {key:30s} matched={mp:+.3f} dir={out[key]['direction_gt']:+.3f}",
                      flush=True)
    return out


if __name__ == "__main__":
    p = Path(os.environ.get("RESULTS_ROOT", "results")) / "tables" / "adversary_matched.json"
    allr = json.load(open(p)) if p.exists() else {}
    for s in [0, 1, 2, 3, 4]:
        if str(s) in allr:
            continue
        print(f"=== seed {s} ===", flush=True)
        allr[str(s)] = run(seed=s)
        json.dump(allr, open(p, "w"), indent=2)
    print("wrote", p)
