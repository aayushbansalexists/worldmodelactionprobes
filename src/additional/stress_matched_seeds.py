import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from analysis.matched_study import cos_rows
from analysis.probe_validity import rollout
from data.dataset import Normalizer
from eval.counterfactual_eval import load_model

H = 8
SEEDS = range(5)
dr = ROOT / "data"
nz = Normalizer.load(dr / "processed" / "latents" / "normalizer.npz")
cf = np.load(dr / "processed" / "counterfactual" / "cflow_v3.npz", allow_pickle=True)
lo = np.load(dr / "processed" / "latents" / "libero_lowdim.npz", allow_pickle=True)
sp = json.load(open(dr / "splits" / "libero_splits.json"))
held = sorted(set(sp["val"]) | set(sp["test"]))
id2 = {str(i): (o, a) for i, o, a in zip(lo["ids"], lo["obs"], lo["act"])}
keys = [(str(t), str(d), int(p)) for t, d, p in zip(cf["task"], cf["demo"], cf["pos"])]
first = {}
for i, k in enumerate(keys):
    first.setdefault(k, i)
rows = np.array(sorted(first.values()))
pool = {}
for did in held:
    task, demo = did.split(":")
    pool.setdefault(task, []).append(demo)
pool = {t: sorted(ds) for t, ds in pool.items()}


def build(sigma=0.0, demos=None, seed=0):
    rng = np.random.default_rng(seed)
    obs = {}
    for did in held:
        o = nz.norm_obs(id2[did][0]).astype(np.float32)
        obs[did] = o + rng.standard_normal(o.shape).astype(np.float32) * sigma if sigma > 0 else o
    order = {t: list(np.random.default_rng([seed, j]).permutation(ds)) for j, (t, ds) in enumerate(sorted(pool.items()))}
    e, a_j, n_cand = [], [], []
    for r in rows:
        task, demo, p = keys[r]
        cand = [d for d in order[task] if d != demo]
        if demos is not None:
            cand = cand[:demos]
        n_cand.append(len(cand))
        oi = obs[f"{task}:{demo}"]
        s_i, y_i = oi[p], oi[p + 1:p + H + 1]
        best, bd = None, np.inf
        for dj in cand:
            S = obs[f"{task}:{dj}"][:len(obs[f"{task}:{dj}"]) - H - 1]
            d = np.linalg.norm(S - s_i, axis=1)
            q = int(d.argmin())
            if d[q] < bd:
                bd, best = d[q], (dj, q)
        dj, q = best
        oj = obs[f"{task}:{dj}"]
        e.append((oj[q + 1:q + H + 1] - oj[q]) - (y_i - s_i))
        a_j.append(nz.norm_act(id2[f"{task}:{dj}"][1][q:q + H]).astype(np.float32))
    if demos is not None:
        assert all(n == demos for n in n_cand), (demos, sorted(set(n_cand)))
    return np.stack(e).astype(np.float32), np.stack(a_j)


variants = {"clean": [{}]}
for s in (0.1, 0.3, 1.0):
    variants[f"noise {s}"] = [{"sigma": s, "seed": k} for k in SEEDS]
for n in (8, 4, 2):
    variants[f"{n} other demos per task"] = [{"demos": n, "seed": k} for k in SEEDS]
pairs = {v: [build(**kw) for kw in kws] for v, kws in variants.items()}
print("pairs built", flush=True)
R = ROOT / "results" / "tables"
pv = json.load(open(R / "probe_validity_v3.json"))
ms = json.load(open(R / "matched_v3.json"))["models"]
dt = json.load(open(R / "direction_training_v3.json"))
runs = sorted([r for r in pv if r.split("_")[2] == "causal"] +
              [r for r in dt if r.split("_")[1] not in ("ac001", "ac01", "ac1") and r not in pv])
truth = {r: (ms[r]["identity"]["dir_on"] if r in ms else dt[r]["dir_on"]) for r in runs}
cx, ca, ar = cf["ctx_x"][rows], cf["ctx_a"][rows], cf["fut_a_true"][rows]
cx, ca, ar = nz.norm_obs(cx).astype(np.float32), nz.norm_act(ca).astype(np.float32), nz.norm_act(ar).astype(np.float32)
probe = {v: [{} for _ in kws] for v, kws in variants.items()}
for rid in runs:
    m, _ = load_model(rid, "cuda")
    P0 = rollout(m, cx, ca, ar, "cuda")
    for v in variants:
        for j, (e, a_j) in enumerate(pairs[v]):
            probe[v][j][rid] = float(cos_rows(rollout(m, cx, ca, a_j, "cuda") - P0, e).mean())
    del m
    torch.cuda.empty_cache()
json.dump({"truth": truth, "probe": probe, "variants": variants}, open(ROOT / "results" / "additional" / "stress_matched_seeds.json", "w"), indent=1)
ref = json.load(open(ROOT / "results" / "additional" / "stress_matched.json"))["probe"]["std (paper)"]
print("clean variant against stress_matched.py std (paper): max diff", max(abs(probe["clean"][0][r] - ref[r]) for r in runs))
obj = lambda r: r.split("_")[1]
sets = {"30": [r for r in runs if obj(r) in ("add", "agm", "delta", "idm", "idm01", "idm001")],
        "40": [r for r in runs if not obj(r).startswith(("act", "md"))],
        "55": [r for r in runs if not (obj(r).startswith("act") or obj(r) in ("idm", "mdshuf01"))],
        "65": [r for r in runs if not obj(r).startswith("act")]}
for v in variants:
    out = []
    for s, rs in sets.items():
        vals = [spearmanr([pr[r] for r in rs], [truth[r] for r in rs])[0] for pr in probe[v]]
        out.append(f"{s}: {np.mean(vals):+.2f} [{min(vals):+.2f}, {max(vals):+.2f}]")
    print(f"{v:24s} " + "  ".join(out))
