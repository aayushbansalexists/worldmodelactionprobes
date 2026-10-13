from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from data.dataset import Normalizer
from eval.counterfactual_eval import load_model, predict_branch

_WSP = Path(__file__).resolve().parents[1] / "third_party" / "WorldSimProbe" / "src"
if str(_WSP) not in sys.path:
    sys.path.insert(0, str(_WSP))
from worldsimprobe.evaluation.task1_action_calibration.oracle_ratio import (
    triangular_ratio_score,
)


@torch.no_grad()
def model_responses(model, d, nz, device="cuda"):
    H = int(d["horizon"])
    resp, true_resp, keys = [], [], []
    for i in range(d["sigma"].shape[0]):
        ctx_x = torch.from_numpy(nz.norm_obs(d["ctx_x"][i])).float().to(device)
        ctx_a = torch.from_numpy(nz.norm_act(d["ctx_a"][i])).float().to(device)
        ft = torch.from_numpy(nz.norm_act(d["fut_a_true"][i])).float().to(device)
        fp = torch.from_numpy(nz.norm_act(d["fut_a_pert"][i])).float().to(device)
        p_true = predict_branch(model, ctx_x, ctx_a, ft, H, device).cpu().numpy()
        p_pert = predict_branch(model, ctx_x, ctx_a, fp, H, device).cpu().numpy()
        resp.append(float(((p_pert - p_true) ** 2).mean()))
        true_resp.append(float(((nz.norm_obs(d["x_pert"][i])
                                 - nz.norm_obs(d["x_true"][i])) ** 2).mean()))
        keys.append((str(d["task"][i]), str(d["demo"][i]), int(d["pos"][i]), float(d["sigma"][i])))
    return np.array(resp), np.array(true_resp), keys


def task1_scores(resp, true_resp, keys):
    by_pos = {}
    for r, tr, (t, dm, p, sg) in zip(resp, true_resp, keys):
        by_pos.setdefault((t, dm, p), {})[sg] = (r, tr)
    scores, pairs = [], []
    for (t, dm, p), bysig in by_pos.items():
        sigmas = sorted(bysig)
        for s_small, s_large in itertools.combinations(sigmas, 2):
            ds, ds_sim = bysig[s_small]
            dl, dl_sim = bysig[s_large]
            if not (dl_sim > 1e-12 and dl_sim > ds_sim and dl > 1e-12):
                continue
            oracle = ds_sim / dl_sim
            if not 0.0 <= oracle < 1.0:
                continue
            sc = triangular_ratio_score(ds / dl, oracle)
            scores.append(sc)
            pairs.append({"task": t, "sigma_small": s_small, "sigma_large": s_large,
                          "model_ratio": ds / dl, "oracle_ratio": oracle, "score": sc})
    return np.array(scores), pairs


def boot_ci(vals, groups, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals, float)
    uniq = sorted(set(groups))
    idx = {g: np.flatnonzero(np.asarray(groups) == g) for g in uniq}
    m = np.empty(n_boot)
    for b in range(n_boot):
        gs = rng.choice(uniq, size=len(uniq), replace=True)
        take = np.concatenate([rng.choice(idx[g], size=len(idx[g]), replace=True) for g in gs])
        m[b] = vals[take].mean()
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dr = Path(os.environ.get("DATA_ROOT", "data"))
    d = np.load(dr / "processed" / "counterfactual" / f"cflow_{args.tag}.npz", allow_pickle=True)
    nz = Normalizer.load(dr / "processed" / "latents" / "normalizer.npz")
    print(f"[external] WorldSimProbe Task 1 scoring, n={d['sigma'].shape[0]} samples, "
          f"{len(set(map(str, d['task'])))} tasks")

    rngv = np.random.default_rng(0)
    oracles = []
    resp0, true0, keys0 = None, None, None
    out = {}
    for rid in args.runs:
        try:
            model, _ = load_model(rid, args.device)
        except Exception as e:
            print(f"  SKIP {rid}: {type(e).__name__}: {e}")
            continue
        resp, true_resp, keys = model_responses(model, d, nz, args.device)
        sc, pairs = task1_scores(resp, true_resp, keys)
        if not len(sc):
            print(f"  {rid}: no scorable pairs")
            continue
        lo, hi = boot_ci(sc, [p["task"] for p in pairs])
        out[rid] = {"task1_score": float(sc.mean() * 100), "ci95": [lo * 100, hi * 100],
                    "n_pairs": int(len(sc))}
        print(f"  {rid:22s} Task1 = {sc.mean()*100:6.2f}  "
              f"[{lo*100:6.2f}, {hi*100:6.2f}]  (n={len(sc)})", flush=True)

    by_pos = {}
    for i in range(d["sigma"].shape[0]):
        k = (str(d["task"][i]), str(d["demo"][i]), int(d["pos"][i]))
        tr = float(((nz.norm_obs(d["x_pert"][i]) - nz.norm_obs(d["x_true"][i])) ** 2).mean())
        by_pos.setdefault(k, {})[float(d["sigma"][i])] = tr
    rand_scores, oracle_vals = [], []
    for k, bysig in by_pos.items():
        sig = sorted(bysig)
        for a, b in itertools.combinations(sig, 2):
            if not (bysig[b] > 1e-12 and bysig[b] > bysig[a]):
                continue
            o = bysig[a] / bysig[b]
            if not 0.0 <= o < 1.0:
                continue
            oracle_vals.append(o)
            rand_scores.append(triangular_ratio_score(float(rngv.random()), o))
    out["_null_random_ratio"] = {"task1_score": float(np.mean(rand_scores) * 100),
                                 "n_pairs": len(rand_scores)}
    out["_null_blind_model"] = {"task1_score": 0.0,
                                "note": "a model with zero response scores 0 by the authors' rule"}
    out["_oracle_ratio_mean"] = float(np.mean(oracle_vals))
    print(f"  {'[null] random ratio':22s} Task1 = {np.mean(rand_scores)*100:6.2f}")
    print(f"  {'[null] blind model':22s} Task1 =   0.00")
    print(f"  mean oracle ratio = {np.mean(oracle_vals):.4f}")

    p = Path(os.environ.get("RESULTS_ROOT", "results")) / "tables" / f"external_probe_{args.tag}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(p, "w"), indent=2)
    print("[external] wrote", p)


if __name__ == "__main__":
    main()
