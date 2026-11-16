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
dr = ROOT / "data"
nz = Normalizer.load(dr / "processed" / "latents" / "normalizer.npz")
cf = np.load(dr / "processed" / "counterfactual" / "cflow_v3.npz", allow_pickle=True)
lo = np.load(dr / "processed" / "latents" / "libero_lowdim.npz", allow_pickle=True)
sp = json.load(open(dr / "splits" / "libero_splits.json"))
held = set(sp["val"]) | set(sp["test"])
id2 = {str(i): (o, a) for i, o, a in zip(lo["ids"], lo["obs"], lo["act"])}
keys = [(str(t), str(d), int(p)) for t, d, p in zip(cf["task"], cf["demo"], cf["pos"])]
first = {}
for i, k in enumerate(keys):
    first.setdefault(k, i)
rows = np.array(sorted(first.values()))
EEF, JOINTS = slice(15, 18), slice(0, 7)


def build(metric="std", sigma=0.0, demos=None, seed=0):
    rng = np.random.default_rng(seed)
    noisy = {}
    for did in held:
        o = nz.norm_obs(id2[did][0]).astype(np.float32)
        noisy[did] = o + rng.standard_normal(o.shape).astype(np.float32) * sigma if sigma > 0 else o
    pool = {}
    for did in held:
        task, demo = did.split(":")
        pool.setdefault(task, []).append(demo)
    keep = {}
    for task, ds in pool.items():
        ds = sorted(ds)
        keep[task] = set(ds if demos is None else list(np.random.default_rng(seed + 1).choice(ds, min(demos, len(ds)), replace=False)))
    e, a_j = [], []
    for r in rows:
        task, demo, p = keys[r]
        oi = noisy[f"{task}:{demo}"]
        s_i, y_i = oi[p], oi[p + 1:p + H + 1]
        best, bd = None, np.inf
        for dj in keep[task]:
            if dj == demo:
                continue
            oj = noisy[f"{task}:{dj}"]
            S = oj[:len(oj) - H - 1]
            if metric == "std":
                d = np.linalg.norm(S - s_i, axis=1)
            elif metric == "raw":
                d = np.linalg.norm(nz.denorm_obs(S) - nz.denorm_obs(s_i), axis=1)
            elif metric == "eef":
                d = np.linalg.norm(S[:, EEF] - s_i[EEF], axis=1)
            elif metric == "joints":
                d = np.linalg.norm(S[:, JOINTS] - s_i[JOINTS], axis=1)
            elif metric == "cosine":
                d = 1 - (S @ s_i) / (np.linalg.norm(S, axis=1) * np.linalg.norm(s_i) + 1e-12)
            q = int(d.argmin())
            if d[q] < bd:
                bd, best = d[q], (dj, q)
        dj, q = best
        oj = noisy[f"{task}:{dj}"]
        e.append((oj[q + 1:q + H + 1] - oj[q]) - (y_i - s_i))
        a_j.append(nz.norm_act(id2[f"{task}:{dj}"][1][q:q + H]).astype(np.float32))
    return np.stack(e).astype(np.float32), np.stack(a_j)


variants = {"std (paper)": {}, "raw": {"metric": "raw"}, "eef only": {"metric": "eef"},
            "joints only": {"metric": "joints"}, "cosine": {"metric": "cosine"},
            "noise 0.1": {"sigma": 0.1}, "noise 0.3": {"sigma": 0.3}, "noise 1.0": {"sigma": 1.0},
            "8 demos per task": {"demos": 8}, "4 demos per task": {"demos": 4}, "2 demos per task": {"demos": 2}}
pairs = {k: build(**v) for k, v in variants.items()}
print("pairs built", flush=True)
R = ROOT / "results" / "tables"
pv = json.load(open(R / "probe_validity_v3.json")); ms = json.load(open(R / "matched_v3.json"))["models"]
dt = json.load(open(R / "direction_training_v3.json"))
runs = sorted([r for r in pv if r.split("_")[2] == "causal"] +
              [r for r in dt if r.split("_")[1] not in ("ac001", "ac01", "ac1") and r not in pv])
truth = {r: (ms[r]["identity"]["dir_on"] if r in ms else dt[r]["dir_on"]) for r in runs}
cx, ca, ar = cf["ctx_x"][rows], cf["ctx_a"][rows], cf["fut_a_true"][rows]
cx, ca, ar = nz.norm_obs(cx).astype(np.float32), nz.norm_act(ca).astype(np.float32), nz.norm_act(ar).astype(np.float32)
probe = {k: {} for k in variants}
for rid in runs:
    m, _ = load_model(rid, "cuda")
    P0 = rollout(m, cx, ca, ar, "cuda")
    for k, (e, a_j) in pairs.items():
        probe[k][rid] = float(cos_rows(rollout(m, cx, ca, a_j, "cuda") - P0, e).mean())
    del m
    torch.cuda.empty_cache()
json.dump({"truth": truth, "probe": probe}, open(ROOT / "results" / "additional" / "stress_matched.json", "w"), indent=1)
obj = lambda r: r.split("_")[1]
sets = {"30 ordinary": [r for r in runs if obj(r) in ("add", "agm", "delta", "idm", "idm01", "idm001")],
        "65 no act*": [r for r in runs if not obj(r).startswith("act")],
        "40 no act*/md*": [r for r in runs if not (obj(r).startswith("act") or obj(r).startswith("md"))],
        "55 working": [r for r in runs if not (obj(r).startswith("act") or obj(r) in ("idm", "mdshuf01"))],
        "90 all": runs}
print(f"{'variant':18s} " + " ".join(f"{s:>15s}" for s in sets))
for k in variants:
    print(f"{k:18s} " + " ".join(f"{spearmanr([probe[k][r] for r in rs], [truth[r] for r in rs])[0]:+15.2f}" for rs in sets.values()))
