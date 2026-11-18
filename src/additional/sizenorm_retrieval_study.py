import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from analysis.probe_validity import Data, one_step
from data.dataset import Normalizer
from eval.counterfactual_eval import load_model

OUT = Path(sys.argv[1])
OBJS = ["md001", "md01", "md1", "mdrand01", "mdshuf01", "imp01", "imp1",
        "act1", "act01", "act003", "act001", "act01t2"]
dt = json.load(open(ROOT / "results" / "tables" / "direction_training_v3.json"))
runs = sorted(r for r in dt if re.fullmatch(r"cf_(" + "|".join(OBJS) + r")_causal_s\d", r))
assert len(runs) == 60, len(runs)
c = json.load(open(OUT / "sizenorm_retrieval.json"))["c"]

nz = Normalizer.load(ROOT / "data" / "processed" / "latents" / "normalizer.npz")
D = Data("v3", ROOT / "data", nz)
u, n = D.pos_idx, D.n_cand
N = len(u)


def rank(err, resp):
    diff = 2 * np.einsum("io,iko->ik", err, resp) + (resp ** 2).sum(-1)
    return float(((diff < 0).sum(1) + 0.5 * (diff == 0).sum(1)).mean())


out = {"c": c, "plain": {}, "per_model": {}, "per_item": {}}
for rid in runs:
    m, _ = load_model(rid, "cuda")
    flat = one_step(m, np.repeat(D.cf_ctx_x[u], n, 0), np.repeat(D.cf_ctx_a[u], n, 0),
                    D.cand_gauss.reshape(N * n, -1), "cuda")
    del m
    torch.cuda.empty_cache()
    P = flat.reshape(N, n, -1).astype(np.float64)
    err, resp = P[:, 0] - D.next_obs.astype(np.float64), P[:, 1:] - P[:, :1]
    sm = float(np.linalg.norm(resp, axis=-1).mean())
    nr = np.linalg.norm(resp, axis=-1, keepdims=True)
    out["plain"][rid] = rank(err, resp)
    out["per_model"][rid] = rank(err, resp * (c / sm)) if sm > 0 else (n - 1) / 2
    out["per_item"][rid] = rank(err, np.where(nr > 0, resp * c / np.maximum(nr, 1e-12), 0.0))
    print(rid, {k: round(out[k][rid], 3) for k in ("plain", "per_model", "per_item")}, flush=True)
json.dump(out, open(OUT / "sizenorm_retrieval_study.json", "w"), indent=1)
print("done, c =", c)
