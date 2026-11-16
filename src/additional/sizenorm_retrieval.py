import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from analysis.probe_validity import TRANSFORMS, Data, make_T, one_step
from data.dataset import Normalizer
from eval.counterfactual_eval import load_model

OUT = Path(sys.argv[1])
nz = Normalizer.load(ROOT / "data" / "processed" / "latents" / "normalizer.npz")
D = Data("v3", ROOT / "data", nz)
runs = sorted(json.load(open(ROOT / "results" / "tables" / "probe_validity_v3.json")))
u, n = D.pos_idx, D.n_cand
N = len(u)
cache = {}
for rid in runs:
    m, _ = load_model(rid, "cuda")
    flat = one_step(m, np.repeat(D.cf_ctx_x[u], n, 0), np.repeat(D.cf_ctx_a[u], n, 0),
                    D.cand_gauss.reshape(N * n, -1), "cuda")
    del m
    torch.cuda.empty_cache()
    P = flat.reshape(N, n, -1).astype(np.float64)
    cache[rid] = (P[:, 0] - D.next_obs.astype(np.float64), P[:, 1:] - P[:, :1])
s_f = {r: float(np.linalg.norm(cache[r][1], axis=-1).mean()) for r in runs}
c = float(np.median([v for v in s_f.values() if v > 0]))


def rank(err, resp):
    diff = 2 * np.einsum("io,iko->ik", err, resp) + (resp ** 2).sum(-1)
    return float(((diff < 0).sum(1) + 0.5 * (diff == 0).sum(1)).mean())


out = {"c": c, "per_model": {}, "per_item": {}}
for rid in runs:
    err, resp = cache[rid]
    out["per_model"][rid], out["per_item"][rid] = {}, {}
    for t in TRANSFORMS:
        rT = resp @ make_T(t, resp.shape[-1]).T
        sm = float(np.linalg.norm(rT, axis=-1).mean())
        out["per_model"][rid][t] = rank(err, rT * (c / sm)) if sm > 0 else (n - 1) / 2
        nr = np.linalg.norm(rT, axis=-1, keepdims=True)
        out["per_item"][rid][t] = rank(err, np.where(nr > 0, rT * c / np.maximum(nr, 1e-12), 0.0))
    print(rid, {k: round(out[k][rid]["identity"], 2) for k in ("per_model", "per_item")}, flush=True)
json.dump(out, open(OUT / "sizenorm_retrieval.json", "w"), indent=1)
print("done, c =", c)
