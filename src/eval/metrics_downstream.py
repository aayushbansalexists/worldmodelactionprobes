from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import build_eval_trajectories, build_eval_trajectories_latent
from models.world_model import WorldModel
from eval.metrics_controllability import rollout_with_actions


def _load(run_id, device="cuda"):
    run_dir = Path(os.environ["RESULTS_ROOT"]) / "raw" / run_id
    import yaml
    cfg = yaml.safe_load(open(run_dir / "config_resolved.yaml"))
    model = WorldModel(cfg, cfg["model"]["obs_dim"], cfg["model"]["action_dim"]).to(device)
    ck = torch.load(Path(os.environ["CHECKPOINT_ROOT"]) / f"{run_id}.pt",
                    map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, cfg


@torch.no_grad()
def action_ranking_eval(run_id, n_candidates=16, n_steps_eval=8, max_seqs=32, device="cuda",
                         seed=0):
    model, cfg = _load(run_id, device)
    L = cfg["data"]["seq_len"]
    if cfg.get("data", {}).get("obs_mode", "lowdim") == "latents":
        trajs, _, _ = build_eval_trajectories_latent("test")
    else:
        trajs, _, _ = build_eval_trajectories("test")
    rng = np.random.default_rng(seed)
    top1_hits, topk_hits, n_eval = 0, 0, 0
    null_hits = 0
    null_pool = []
    spearmans = []
    from scipy.stats import spearmanr
    for traj in trajs:
        o, a = traj
        T = o.shape[0]
        H = T - L
        if H < n_steps_eval + 1:
            continue
        positions = np.linspace(0, H - 2, n_steps_eval).astype(int)
        for p in positions:
            ctx_o = o[p:p + L].to(device)
            ctx_a = a[p:p + L].to(device)
            true_a = a[p + L - 1].to(device)
            true_next = o[p + L].to(device)
            cand = torch.zeros(n_candidates, a.shape[-1], device=device)
            cand[0] = true_a
            cand[1:] = true_a + torch.from_numpy(
                rng.standard_normal((n_candidates - 1, a.shape[-1])).astype(np.float32) * 0.5
            ).to(device)
            preds = []
            for c in range(n_candidates):
                ca = ctx_a.clone(); ca[-1] = cand[c]
                pred = rollout_with_actions(model, ctx_o, ca, torch.zeros(1, a.shape[-1], device=device), 1, device)
                preds.append(pred[0])
            preds = torch.stack(preds)
            dist = (preds - true_next.to(device)).pow(2).mean(-1).sqrt()
            dist = dist.reshape(-1)
            jitter = torch.from_numpy(
                rng.standard_normal(dist.shape[0]).astype(np.float32) * 1e-6 * (dist.abs().mean().item() + 1e-9)
            ).to(device)
            model_rank = (dist + jitter).argsort()
            if null_pool:
                alt = null_pool[int(rng.integers(0, len(null_pool)))]
                d0 = (preds - alt).pow(2).mean(-1).sqrt().reshape(-1)
                j0 = torch.from_numpy(rng.standard_normal(d0.shape[0]).astype(np.float32)
                                      * 1e-6 * (d0.abs().mean().item() + 1e-9)).to(device)
                null_hits += int(int((d0 + j0).argmin().item()) == 0)
            null_pool.append(true_next.detach().clone())
            true_idx = 0
            rank_of_true = int((model_rank == true_idx).nonzero(as_tuple=True)[0].item())
            k = max(1, n_candidates // 4)
            top1_hits += int(rank_of_true == 0)
            topk_hits += int(rank_of_true < k)
            true_dist = (cand - true_a).pow(2).mean(-1).sqrt().cpu().numpy()
            if np.ptp(true_dist) > 0:
                sp = spearmanr(dist.cpu().numpy(), true_dist).correlation
                if not np.isnan(sp):
                    spearmans.append(sp)
            n_eval += 1
        if n_eval >= max_seqs * n_steps_eval:
            break
    if n_eval == 0:
        return {"downstream": "no eval positions", "top1": None, "topk": None}
    n_null = max(n_eval - 1, 1)
    top1 = top1_hits / n_eval
    null = null_hits / n_null
    return {
        "downstream": "logged_action_retrieval",
        "top1_agreement": top1,
        "top1_null_empirical": null,
        "top1_excess": top1 - null,
        "topk_agreement": topk_hits / n_eval,
        "ranking_spearman_true": float(np.mean(spearmans)) if spearmans else None,
        "n_candidates": n_candidates, "k": max(1, n_candidates // 4),
        "n_eval_positions": n_eval,
    }


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    r = action_ranking_eval(args.run_id)
    print(json.dumps({args.run_id: r}, indent=2))
    if args.out:
        d = json.load(open(args.out)) if os.path.exists(args.out) else {}
        d[args.run_id] = r
        json.dump(d, open(args.out, "w"), indent=2)
