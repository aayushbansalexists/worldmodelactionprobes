from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from data.dataset import Normalizer
from models.world_model import WorldModel
from eval.metrics_controllability import rollout_with_actions


def load_model(run_id, device="cuda"):
    run_dir = Path(os.environ["RESULTS_ROOT"]) / "raw" / run_id
    cfg = yaml.safe_load(open(run_dir / "config_resolved.yaml"))
    m = WorldModel(cfg, cfg["model"]["obs_dim"], cfg["model"]["action_dim"]).to(device)
    ck = torch.load(Path(os.environ["CHECKPOINT_ROOT"]) / f"{run_id}.pt",
                    map_location=device, weights_only=False)
    m.load_state_dict(ck["state_dict"])
    return m.eval(), cfg


@torch.no_grad()
def predict_branch(model, ctx_x, ctx_a, fut_a, H, device):
    ca = ctx_a.clone()
    ca[-1] = fut_a[0]
    return rollout_with_actions(model, ctx_x, ca, fut_a[1:H], H, device)


def cosine(a, b, eps=1e-12):
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + eps))


def score(model, d, nz, device="cuda"):
    H = int(d["horizon"])
    rows = []
    for i in range(d["sigma"].shape[0]):
        ctx_x = torch.from_numpy(nz.norm_obs(d["ctx_x"][i])).float().to(device)
        ctx_a = torch.from_numpy(nz.norm_act(d["ctx_a"][i])).float().to(device)
        ft = torch.from_numpy(nz.norm_act(d["fut_a_true"][i])).float().to(device)
        fp = torch.from_numpy(nz.norm_act(d["fut_a_pert"][i])).float().to(device)
        p_true = predict_branch(model, ctx_x, ctx_a, ft, H, device).cpu().numpy()
        p_pert = predict_branch(model, ctx_x, ctx_a, fp, H, device).cpu().numpy()
        dx_pred = (p_pert - p_true).ravel()
        dx_true = (nz.norm_obs(d["x_pert"][i]) - nz.norm_obs(d["x_true"][i])).ravel()
        dx_noise = (nz.norm_obs(d["x_noise"][i]) - nz.norm_obs(d["x_true"][i])).ravel()
        nt = np.linalg.norm(dx_true) + 1e-12
        rows.append({
            "sigma": float(d["sigma"][i]),
            "task": str(d["task"][i]),
            "direction": cosine(dx_pred, dx_true),
            "magnitude": float(np.linalg.norm(dx_pred) / nt),
            "cf_error": float(np.linalg.norm(dx_pred - dx_true) / nt),
            "noise_direction": cosine(dx_noise, dx_true),
            "noise_magnitude": float(np.linalg.norm(dx_noise) / nt),
        })
    return rows


def _boot_ci(vals, groups, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals, dtype=float)
    uniq = sorted(set(groups))
    idx = {g: np.flatnonzero(np.asarray(groups) == g) for g in uniq}
    means = np.empty(n_boot)
    for b in range(n_boot):
        gs = rng.choice(uniq, size=len(uniq), replace=True)
        take = np.concatenate([rng.choice(idx[g], size=len(idx[g]), replace=True) for g in gs])
        means[b] = vals[take].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(rows, n_boot=2000):
    g = [r["task"] for r in rows]
    out = {}
    for k in ["direction", "magnitude", "cf_error", "noise_direction", "noise_magnitude"]:
        v = [r[k] for r in rows]
        lo, hi = _boot_ci(v, g, n_boot=n_boot)
        out[k] = {"mean": float(np.mean(v)), "ci95": [lo, hi]}
    out["n"] = len(rows)
    out["n_tasks"] = len(set(g))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dr = Path(os.environ.get("DATA_ROOT", "data"))
    p = dr / "processed" / "counterfactual" / f"cflow_{args.tag}.npz"
    d = np.load(p, allow_pickle=True)
    nz = Normalizer.load(dr / "processed" / "latents" / "normalizer.npz")
    print(f"[cf-eval] {p.name}: n={d['sigma'].shape[0]} horizon={int(d['horizon'])} "
          f"tasks={len(set(map(str, d['task'])))}")

    results = {}
    for rid in args.runs:
        try:
            model, _ = load_model(rid, args.device)
        except Exception as e:
            print(f"[cf-eval] SKIP {rid}: {type(e).__name__}: {e}")
            continue
        rows = score(model, d, nz, args.device)
        results[rid] = {"all": summarize(rows, args.n_boot)}
        for sg in sorted(set(r["sigma"] for r in rows)):
            results[rid][f"sigma_{sg}"] = summarize([r for r in rows if r["sigma"] == sg],
                                                    args.n_boot)
        a = results[rid]["all"]
        print(f"  {rid:24s} dir={a['direction']['mean']:+.3f} "
              f"[{a['direction']['ci95'][0]:+.3f},{a['direction']['ci95'][1]:+.3f}]  "
              f"mag={a['magnitude']['mean']:.3f} "
              f"[{a['magnitude']['ci95'][0]:.3f},{a['magnitude']['ci95'][1]:.3f}]  "
              f"cf_err={a['cf_error']['mean']:.3f}   "
              f"(noise dir={a['noise_direction']['mean']:+.3f} "
              f"mag={a['noise_magnitude']['mean']:.3f})", flush=True)

    out = Path(args.out or (Path(os.environ.get("RESULTS_ROOT", "results")) / "tables" /
                            f"counterfactual_{args.tag}.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out, "w"), indent=2)
    print("[cf-eval] wrote", out)


if __name__ == "__main__":
    main()
