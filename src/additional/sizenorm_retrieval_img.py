import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from analysis.probe_validity import Data, one_step
from data.dataset import Normalizer
from eval.counterfactual_eval import load_model

out_path = Path(sys.argv[1])
sizes = [a.split("=", 1) for a in sys.argv[2:]]
dr = Path(os.environ["DATA_ROOT"])
nz = Normalizer.load(dr / "processed" / "latents" / "normalizer.npz")
D = Data("v3", dr, nz)
u, n = D.pos_idx, D.n_cand
N = len(u)
STUDY = ("md", "imp", "act")
obj = lambda r: r.split("_")[1]
print(f"{N} positions, {n} candidates", flush=True)


def responses(rid):
    m, _ = load_model(rid, "cuda")
    flat = one_step(m, np.repeat(D.cf_ctx_x[u], n, 0), np.repeat(D.cf_ctx_a[u], n, 0),
                    D.cand_gauss.reshape(N * n, -1), "cuda")
    del m
    torch.cuda.empty_cache()
    P = flat.reshape(N, n, -1).astype(np.float64)
    return P[:, 0] - D.next_obs.astype(np.float64), P[:, 1:] - P[:, :1]


def rank(err, resp):
    diff = 2 * np.einsum("io,iko->ik", err, resp) + (resp ** 2).sum(-1)
    return float(((diff < 0).sum(1) + 0.5 * (diff == 0).sum(1)).mean())


out = {}
for size, roots in sizes:
    res_root, ck_root = roots.split(":")
    os.environ["RESULTS_ROOT"], os.environ["CHECKPOINT_ROOT"] = res_root, ck_root
    runs = sorted(p.name for p in (Path(res_root) / "raw").iterdir() if p.is_dir() and (Path(ck_root) / f"{p.name}.pt").exists())
    s_f = {}
    for rid in runs:
        _, resp = responses(rid)
        s_f[rid] = float(np.linalg.norm(resp, axis=-1).mean())
    ref = [r for r in runs if not obj(r).startswith(STUDY) and s_f[r] > 0]
    c = float(np.median([s_f[r] for r in ref]))
    print(f"[{size}] {len(runs)} runs, c = {c:.6g} from {len(ref)} paper configurations", flush=True)
    res = {"c": c, "n_ref": len(ref), "s_f": s_f, "plain": {}, "per_model": {}, "per_item": {}}
    for rid in runs:
        err, resp = responses(rid)
        nr = np.linalg.norm(resp, axis=-1, keepdims=True)
        res["plain"][rid] = rank(err, resp)
        res["per_model"][rid] = rank(err, resp * (c / s_f[rid])) if s_f[rid] > 0 else (n - 1) / 2
        res["per_item"][rid] = rank(err, np.where(nr > 0, resp * c / np.maximum(nr, 1e-12), 0.0))
        print(size, rid, {k: round(res[k][rid], 3) for k in ("plain", "per_model", "per_item")}, flush=True)
    out[size] = res
    json.dump(out, open(out_path, "w"), indent=1)
print("done", flush=True)
