from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from models.world_model import WorldModel
from data.dataset import build_eval_trajectories, build_eval_trajectories_latent
from eval.metrics_controllability import rollout_with_actions


def load(run_id, device="cuda"):
    run_dir = Path(os.environ["RESULTS_ROOT"]) / "raw" / run_id
    cfg = yaml.safe_load(open(run_dir / "config_resolved.yaml"))
    m = WorldModel(cfg, cfg["model"]["obs_dim"], cfg["model"]["action_dim"]).to(device)
    ck = torch.load(Path(os.environ["CHECKPOINT_ROOT"]) / f"{run_id}.pt",
                    map_location=device, weights_only=False)
    m.load_state_dict(ck["state_dict"])
    return m.eval(), cfg


@torch.no_grad()
def retrieval(run_id, n_cand=16, n_pos=8, max_pos=256, seed=0, device="cuda",
              n_null_draws=8):
    model, cfg = load(run_id, device)
    L = cfg["data"]["seq_len"]
    latent = cfg.get("data", {}).get("obs_mode", "lowdim") == "latents"
    trajs, _, _ = (build_eval_trajectories_latent if latent else build_eval_trajectories)("test")
    rng = np.random.default_rng(seed)
    ranks, nulls, null_ranks, pool = [], [], [], []
    for traj in trajs:
        o, a = traj
        T = o.shape[0]
        if T - L < n_pos + 1:
            continue
        for p in np.linspace(0, T - L - 2, n_pos).astype(int):
            ctx_o, ctx_a = o[p:p + L].to(device), a[p:p + L].to(device)
            true_a, true_next = a[p + L - 1].to(device), o[p + L].to(device)
            cand = true_a.repeat(n_cand, 1)
            cand[1:] = true_a + torch.from_numpy(
                rng.standard_normal((n_cand - 1, a.shape[-1])).astype(np.float32) * 0.5).to(device)
            preds = []
            for c in range(n_cand):
                ca = ctx_a.clone(); ca[-1] = cand[c]
                preds.append(rollout_with_actions(
                    model, ctx_o, ca, torch.zeros(1, a.shape[-1], device=device), 1, device)[0])
            preds = torch.stack(preds)
            dist = (preds - true_next).pow(2).mean(-1).sqrt().reshape(-1)
            jit = torch.from_numpy(rng.standard_normal(n_cand).astype(np.float32)
                                   * 1e-6 * (dist.abs().mean().item() + 1e-9)).to(device)
            ranks.append(int(((dist + jit).argsort() == 0).nonzero()[0].item()))
            for _ in range(n_null_draws):
                if not pool:
                    break
                alt = pool[int(rng.integers(0, len(pool)))]
                d0 = (preds - alt).pow(2).mean(-1).sqrt().reshape(-1)
                j0 = torch.from_numpy(rng.standard_normal(n_cand).astype(np.float32)
                                      * 1e-6 * (d0.abs().mean().item() + 1e-9)).to(device)
                nulls.append(int((d0 + j0).argmin().item() == 0))
                null_ranks.append(int(((d0 + j0).argsort() == 0).nonzero()[0].item()))
            pool.append(true_next.clone())
            if len(ranks) >= max_pos:
                break
        if len(ranks) >= max_pos:
            break
    r = np.array(ranks)
    top1 = float((r == 0).mean())
    null = float(np.mean(nulls)) if nulls else float("nan")
    nr = np.array(null_ranks) if null_ranks else np.array([np.nan])
    chance = (n_cand - 1) / 2.0
    return {"top1": top1, "null_empirical": null, "excess": top1 - null,
            "mean_rank": float(r.mean()), "mean_rank_null": float(np.nanmean(nr)),
            "mean_rank_chance": chance,
            "rank_advantage": float(np.nanmean(nr)) - float(r.mean()),
            "rank_advantage_vs_chance": chance - float(r.mean()),
            "n_positions": int(r.size), "n_candidates": n_cand}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None, help="default: every run with a checkpoint")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    res_root = Path(os.environ["RESULTS_ROOT"])
    ck_root = Path(os.environ["CHECKPOINT_ROOT"])
    runs = args.runs or sorted(p.stem for p in ck_root.glob("*.pt") if p.stem != "vae")
    stored_path = res_root / "tables" / "downstream.json"
    stored = json.load(open(stored_path)) if stored_path.exists() else {}

    out = {}
    print(f"{'run':26s} {'stored':>8s} {'top1':>7s} {'null':>7s} {'meanrank':>9s} "
          f"{'rank_null':>10s} {'rank_adv':>9s}")
    for rid in runs:
        if not (res_root / "raw" / rid / "config_resolved.yaml").exists():
            continue
        try:
            out[rid] = retrieval(rid, device=args.device)
        except Exception as e:
            print(f"{rid:26s} ERROR {type(e).__name__}: {e}")
            continue
        s = stored.get(rid, {}).get("top1_agreement")
        r = out[rid]
        print(f"{rid:26s} {('%.3f' % s) if s is not None else '-':>8s} {r['top1']:7.3f} "
              f"{r['null_empirical']:7.3f} {r['mean_rank']:9.2f} {r['mean_rank_null']:10.2f} "
              f"{r['rank_advantage']:+9.2f}", flush=True)

    p = res_root / "tables" / "retrieval_recomputed.json"
    json.dump(out, open(p, "w"), indent=2)
    print("wrote", p)


if __name__ == "__main__":
    main()
