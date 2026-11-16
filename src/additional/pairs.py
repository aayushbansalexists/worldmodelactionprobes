import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from analysis.matched_study import cos_rows
from analysis.probe_validity import TRANSFORMS, Data, make_T, one_step, rollout
from data.dataset import Normalizer
from eval.counterfactual_eval import load_model

OUT = Path(sys.argv[1])
dr = ROOT / "data"
nz = Normalizer.load(dr / "processed" / "latents" / "normalizer.npz")
D = Data("v3", dr, nz)
mt = np.load(dr / "processed" / "counterfactual" / "matched_v3.npz", allow_pickle=True)
K = int(mt["k"])
row, e, a_j = mt["row"], mt["e"].astype(np.float32), mt["a_j"].astype(np.float32)
rank = np.tile(np.arange(K), len(row) // K)
r_sim = (nz.norm_obs(mt["x_match"]) - nz.norm_obs(mt["x_true"])).astype(np.float32)
ok = np.linalg.norm(r_sim.reshape(len(row), -1), axis=1) > 1e-6
base = json.load(open(ROOT / "results" / "tables" / "probe_validity_v3.json"))
runs = sorted(base)

cx, ca, ar = D.cf_ctx_x[row], D.cf_ctx_a[row], D.cf_ref[row]
u, n = D.pos_idx, D.n_cand
N = len(u)
c_est_all, c_sim_all, dirret, plainret = [], [], {}, {}
for rid in runs:
    m, _ = load_model(rid, "cuda")
    R = rollout(m, cx, ca, a_j, "cuda") - rollout(m, cx, ca, ar, "cuda")
    c_est_all.append(cos_rows(R, e))
    c_sim_all.append(cos_rows(R, r_sim))
    flat = one_step(m, np.repeat(D.cf_ctx_x[u], n, 0), np.repeat(D.cf_ctx_a[u], n, 0),
                    D.cand_gauss.reshape(N * n, -1), "cuda")
    Pg = flat.reshape(N, n, -1).astype(np.float64)
    del m
    torch.cuda.empty_cache()
    err = Pg[:, 0] - D.next_obs.astype(np.float64)
    resp = Pg[:, 1:] - Pg[:, :1]
    dirret[rid], plainret[rid] = {}, {}
    for t in TRANSFORMS:
        T = make_T(t, resp.shape[-1])
        rT = resp @ T.T
        s = np.einsum("io,iko->ik", err, rT)
        rk = (s < 0).sum(1) + 0.5 * (s == 0).sum(1)
        dirret[rid][t] = float(rk.mean())
        diff = 2 * s + (rT ** 2).sum(-1)
        plainret[rid][t] = float(((diff < 0).sum(1) + 0.5 * (diff == 0).sum(1)).mean())
    print(rid, f"dirret {dirret[rid]['identity']:.2f} plain {plainret[rid]['identity']:.2f}", flush=True)

np.savez(OUT / "pairs.npz", runs=np.array(runs), c_est=np.array(c_est_all), c_sim=np.array(c_sim_all),
         row=row, rank=rank, ok=ok, gap=mt["gap"], da=mt["da"], est=cos_rows(e, r_sim))
json.dump({"dirret": dirret, "plainret": plainret}, open(OUT / "dirret.json", "w"), indent=1)
print("done")
