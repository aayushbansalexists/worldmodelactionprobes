from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from analysis.probe_validity import TRANSFORMS, Data, make_T, rollout
from data.dataset import Normalizer
from eval.counterfactual_eval import load_model

TAG = os.environ.get("GT_TAG", "v3")


class Inverse(nn.Module):
    def __init__(self, obs_dim=21, act_dim=7, width=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * obs_dim, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
                                 nn.Linear(width, width), nn.SiLU(), nn.Linear(width, act_dim))

    def forward(self, x, y):
        return self.net(torch.cat([x, y - x], -1))


def transitions(lo, ids, nz):
    id2 = {str(i): (o, a) for i, o, a in zip(lo["ids"], lo["obs"], lo["act"])}
    X, Y, A = [], [], []
    for i in ids:
        o, a = id2[i]
        o = nz.norm_obs(o).astype(np.float32)
        X.append(o[:-1]); Y.append(o[1:]); A.append(nz.norm_act(a[:-1]).astype(np.float32))
    return [torch.from_numpy(np.concatenate(v)) for v in (X, Y, A)]


def train_inverse(dr, nz, device, seed=0, epochs=30):
    torch.manual_seed(seed)
    lo = np.load(dr / "processed" / "latents" / "libero_lowdim.npz", allow_pickle=True)
    sp = json.load(open(dr / "splits" / "libero_splits.json"))
    Xt, Yt, At = [t.to(device) for t in transitions(lo, sp["train"], nz)]
    Xv, Yv, Av = [t.to(device) for t in transitions(lo, sp["val"], nz)]
    g = Inverse().to(device)
    opt = torch.optim.AdamW(g.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for ep in range(epochs):
        g.train()
        perm = torch.randperm(len(Xt), generator=gen).to(device)
        for i in range(0, len(Xt), 1024):
            b = perm[i:i + 1024]
            loss = ((g(Xt[b], Yt[b]) - At[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    g.eval()
    with torch.no_grad():
        pv = g(Xv, Yv)
        r2 = 1 - ((pv - Av) ** 2).mean(0) / Av.var(0)
    return g, {"n_train": int(len(Xt)), "n_val": int(len(Xv)),
               "val_r2_per_dim": [float(v) for v in r2.cpu()], "val_r2_mean": float(r2.mean())}


@torch.no_grad()
def recover(g, last, P, device, bs=4096):
    prev = np.concatenate([last[:, None], P[:, :-1]], 1)
    x = torch.from_numpy(prev.reshape(-1, P.shape[-1])).float()
    y = torch.from_numpy(P.reshape(-1, P.shape[-1])).float()
    out = [g(x[i:i + bs].to(device), y[i:i + bs].to(device)).cpu() for i in range(0, len(x), bs)]
    return torch.cat(out).numpy().reshape(P.shape[0], P.shape[1], -1)


def cos_rows(a, b):
    a, b = a.reshape(len(a), -1), b.reshape(len(b), -1)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)


def scores(g, last, P_ref, P_q, a_ref, a_q, T, device):
    PT = P_ref + (P_q - P_ref) @ T.T
    g_ref, g_q = recover(g, last, P_ref, device), recover(g, last, PT, device)
    return float(cos_rows(g_q - g_ref, a_q - a_ref).mean()), float(((g_q - a_q) ** 2).sum((1, 2)).mean())


def main():
    dev = "cuda"
    dr = Path(os.environ.get("DATA_ROOT", "data"))
    rr = Path(os.environ.get("RESULTS_ROOT", "results"))
    nz = Normalizer.load(dr / "processed" / "latents" / "normalizer.npz")
    D = Data(TAG, dr, nz)
    mt = np.load(dr / "processed" / "counterfactual" / f"matched_{TAG}.npz", allow_pickle=True)
    K = int(mt["k"])
    row, a_j = mt["row"], mt["a_j"].astype(np.float32)
    rank = np.tile(np.arange(K), len(row) // K)
    r_sim = nz.norm_obs(mt["x_match"]) - nz.norm_obs(mt["x_true"])
    keep = (rank == 0) & (np.linalg.norm(r_sim.reshape(len(row), -1), axis=1) > 1e-6)
    row, a_j = row[keep], a_j[keep]

    g, info = train_inverse(dr, nz, dev)
    print("[inverse] model trained:", json.dumps(info))
    out_path = rr / "tables" / f"inverse_probe_{TAG}.json"
    out = {"inverse_model": info, "n_rand": int(len(D.cf_ref)), "n_swap": int(len(row)), "models": {}}
    base = json.load(open(rr / "tables" / f"probe_validity_{TAG}.json"))
    last_r = D.cf_ctx_x[:, -1]
    cx, ca, ar = D.cf_ctx_x[row], D.cf_ctx_a[row], D.cf_ref[row]
    for rid in sorted(base):
        m, _ = load_model(rid, dev)
        P_ref, P_pert = rollout(m, D.cf_ctx_x, D.cf_ctx_a, D.cf_ref, dev), rollout(m, D.cf_ctx_x, D.cf_ctx_a, D.cf_pert, dev)
        S_ref, S_j = rollout(m, cx, ca, ar, dev), rollout(m, cx, ca, a_j, dev)
        del m
        torch.cuda.empty_cache()
        res = {}
        for t in TRANSFORMS:
            T = make_T(t, P_ref.shape[-1]).astype(np.float32)
            dr_, er_ = scores(g, last_r, P_ref, P_pert, D.cf_ref, D.cf_pert, T, dev)
            ds_, es_ = scores(g, cx[:, -1], S_ref, S_j, ar, a_j, T, dev)
            res[t] = {"inv_dir_rand": dr_, "inv_err_rand": er_, "inv_dir_swap": ds_, "inv_err_swap": es_}
        out["models"][rid] = res
        r = res["identity"]
        print(f"  {rid:24s} dir_rand={r['inv_dir_rand']:+.3f} dir_swap={r['inv_dir_swap']:+.3f} "
              f"err_rand={r['inv_err_rand']:.2f} err_swap={r['inv_err_swap']:.2f} "
              f"reflect_dir_swap={res['reflect']['inv_dir_swap']:+.3f}", flush=True)
    json.dump(out, open(out_path, "w"), indent=1)
    print("[inverse] wrote", out_path)


if __name__ == "__main__":
    main()
