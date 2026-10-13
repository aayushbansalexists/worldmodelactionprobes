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
from eval.counterfactual_eval import load_model

_WSP = Path(__file__).resolve().parents[1] / "third_party" / "WorldSimProbe" / "src"
sys.path.insert(0, str(_WSP))
from worldsimprobe.evaluation.task1_action_calibration.oracle_ratio import (
    triangular_ratio_score,
)

TRANSFORMS = ["identity", "reflect", "rotate", "scale_0.25", "scale_4", "blind"]


def make_T(name, dim, seed=0):
    if name == "identity":
        return np.eye(dim)
    if name == "reflect":
        return -np.eye(dim)
    if name == "rotate":
        q, r = np.linalg.qr(np.random.default_rng(seed).standard_normal((dim, dim)))
        return q * np.sign(np.diag(r))
    if name.startswith("scale_"):
        return float(name.split("_")[1]) * np.eye(dim)
    if name == "blind":
        return np.zeros((dim, dim))
    raise ValueError(name)


@torch.no_grad()
def rollout(model, ctx_x, ctx_a, fut_a, device, bs=512):
    out = []
    H = fut_a.shape[1]
    for i in range(0, ctx_x.shape[0], bs):
        wo = torch.from_numpy(ctx_x[i:i + bs]).float().to(device)
        wa = torch.from_numpy(ctx_a[i:i + bs]).float().to(device).clone()
        fa = torch.from_numpy(fut_a[i:i + bs]).float().to(device)
        wa[:, -1] = fa[:, 0]
        preds = []
        for k in range(H):
            p = model(wo, wa)["pred"][:, -1]
            preds.append(p)
            wo = torch.cat([wo[:, 1:], p[:, None]], 1)
            if k + 1 < H:
                wa = torch.cat([wa[:, 1:], fa[:, k + 1:k + 2]], 1)
        out.append(torch.stack(preds, 1).cpu().numpy())
    return np.concatenate(out, 0)


@torch.no_grad()
def one_step(model, ctx_x, ctx_a, last_a, device, bs=2048):
    out = []
    for i in range(0, ctx_x.shape[0], bs):
        wo = torch.from_numpy(ctx_x[i:i + bs]).float().to(device)
        wa = torch.from_numpy(ctx_a[i:i + bs]).float().to(device).clone()
        wa[:, -1] = torch.from_numpy(last_a[i:i + bs]).float().to(device)
        out.append(model(wo, wa)["pred"][:, -1].cpu().numpy())
    return np.concatenate(out, 0)


def mean_rank(pred_c, target, rng):
    d = np.sqrt(((pred_c - target[:, None]) ** 2).mean(-1))
    d = d + rng.standard_normal(d.shape) * 1e-9 * (np.abs(d).mean() + 1e-12)
    return float((np.argsort(np.argsort(d, 1), 1)[:, 0]).mean())


class Data:

    def __init__(self, tag, dr, nz, n_cand=16, seed=0):
        rng = np.random.default_rng(seed)
        cf = np.load(dr / "processed" / "counterfactual" / f"cflow_{tag}.npz", allow_pickle=True)
        ws = np.load(dr / "processed" / "counterfactual" / f"wsp_{tag if tag != 'v2' else 'main'}.npz",
                     allow_pickle=True)
        lo = np.load(dr / "processed" / "latents" / "libero_lowdim.npz", allow_pickle=True)
        id2obs = {str(i): o for i, o in zip(lo["ids"], lo["obs"])}
        allact = nz.norm_act(np.concatenate(list(lo["act"]), 0).astype(np.float32))

        self.H = int(cf["horizon"])
        self.cf_ctx_x = nz.norm_obs(cf["ctx_x"]).astype(np.float32)
        self.cf_ctx_a = nz.norm_act(cf["ctx_a"]).astype(np.float32)
        self.cf_ref = nz.norm_act(cf["fut_a_true"]).astype(np.float32)
        self.cf_pert = nz.norm_act(cf["fut_a_pert"]).astype(np.float32)
        self.cf_dtrue = (nz.norm_obs(cf["x_pert"]) - nz.norm_obs(cf["x_true"])).astype(np.float32)
        self.cf_task = np.array([str(t) for t in cf["task"]])
        keys = [(str(t), str(d), int(p)) for t, d, p in zip(cf["task"], cf["demo"], cf["pos"])]
        first = {}
        for i, k in enumerate(keys):
            first.setdefault(k, i)
        u = np.array(sorted(first.values()))
        self.pos_idx = u
        self.pos_task = self.cf_task[u]
        self.next_obs = np.stack([
            nz.norm_obs(id2obs[f"{keys[i][0]}:{keys[i][1]}"][keys[i][2] + 1]) for i in u
        ]).astype(np.float32)
        N = len(u)
        logged = self.cf_ref[u, 0]
        g = logged[:, None].repeat(n_cand, 1)
        g[:, 1:] += rng.standard_normal((N, n_cand - 1, logged.shape[-1])).astype(np.float32) * 0.5
        m = logged[:, None].repeat(n_cand, 1)
        m[:, 1:] = allact[rng.integers(0, allact.shape[0], (N, n_cand - 1))]
        self.cand_gauss, self.cand_marg, self.n_cand = g, m, n_cand
        self.ws_ctx_x = nz.norm_obs(ws["ctx_x"]).astype(np.float32)
        self.ws_ctx_a = nz.norm_act(ws["ctx_a"]).astype(np.float32)
        self.ws = {k: nz.norm_act(ws[f"a_{k}"]).astype(np.float32) for k in ("orig", "small", "large")}
        xo = nz.norm_obs(ws["x_orig"])
        self.ws_ds_sim = ((nz.norm_obs(ws["x_small"]) - xo) ** 2).mean((1, 2))
        self.ws_dl_sim = ((nz.norm_obs(ws["x_large"]) - xo) ** 2).mean((1, 2))


def score(model, D, device, seed=0):
    rng = np.random.default_rng(seed)
    u = D.pos_idx
    P_ref = rollout(model, D.cf_ctx_x, D.cf_ctx_a, D.cf_ref, device)
    P_pert = rollout(model, D.cf_ctx_x, D.cf_ctx_a, D.cf_pert, device)
    P_zero = rollout(model, D.cf_ctx_x[u], D.cf_ctx_a[u], np.zeros_like(D.cf_ref[u]), device)
    N, n = len(u), D.n_cand

    def cands(C):
        flat = one_step(model, np.repeat(D.cf_ctx_x[u], n, 0), np.repeat(D.cf_ctx_a[u], n, 0),
                        C.reshape(N * n, -1), device)
        return flat.reshape(N, n, -1)

    Pg, Pm = cands(D.cand_gauss), cands(D.cand_marg)
    W_o = rollout(model, D.ws_ctx_x, D.ws_ctx_a, D.ws["orig"], device)
    W_s = rollout(model, D.ws_ctx_x, D.ws_ctx_a, D.ws["small"], device)
    W_l = rollout(model, D.ws_ctx_x, D.ws_ctx_a, D.ws["large"], device)

    O = P_ref.shape[-1]
    out = {}
    for tname in TRANSFORMS:
        T = make_T(tname, O).astype(np.float32)
        tr = lambda R: R @ T.T
        dp = tr(P_pert - P_ref).reshape(len(P_ref), -1)
        dt = D.cf_dtrue.reshape(len(P_ref), -1)
        nt = np.linalg.norm(dt, axis=1)
        npd = np.linalg.norm(dp, axis=1)
        cos = (dp * dt).sum(1) / (npd * nt + 1e-12)
        c1 = float((tr(P_zero - P_ref[u]) ** 2).mean())
        tg = Pg[:, :1] + tr(Pg - Pg[:, :1])
        tm = Pm[:, :1] + tr(Pm - Pm[:, :1])
        ds = (tr(W_s - W_o) ** 2).mean((1, 2))
        dl = (tr(W_l - W_o) ** 2).mean((1, 2))
        t1 = []
        for a, b, sa, sb in zip(ds, dl, D.ws_ds_sim, D.ws_dl_sim):
            if not (sb > 1e-12 and sb > sa):
                continue
            o = sa / sb
            if not 0.0 <= o < 1.0:
                continue
            t1.append(0.0 if b <= 1e-12 else triangular_ratio_score(a / b, o))
        out[tname] = {
            "direction": float(cos.mean()),
            "mag_ratio": float(npd.sum() / nt.sum()),
            "cf_error": float(np.median(np.linalg.norm(dp - dt, axis=1) / (nt + 1e-12))),
            "C1": c1,
            "task1": float(np.mean(t1) * 100) if t1 else float("nan"),
            "task1_n": len(t1),
            "ret_gauss_meanrank": mean_rank(tg, D.next_obs, rng),
            "ret_marg_meanrank": mean_rank(tm, D.next_obs, rng),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dr = Path(os.environ.get("DATA_ROOT", "data"))
    nz = Normalizer.load(dr / "processed" / "latents" / "normalizer.npz")
    D = Data(args.tag, dr, nz)
    print(f"[validity] {len(D.pos_idx)} positions over {len(set(D.pos_task))} tasks; "
          f"{len(D.cf_ref)} counterfactual items; {len(D.ws_ds_sim)} WorldSimProbe items")
    res_path = Path(os.environ.get("RESULTS_ROOT", "results")) / "tables" / f"probe_validity_{args.tag}.json"
    results = json.load(open(res_path)) if res_path.exists() else {}
    for rid in args.runs:
        if rid in results:
            continue
        try:
            model, _ = load_model(rid, args.device)
        except Exception as e:
            print(f"  SKIP {rid}: {type(e).__name__}: {e}")
            continue
        results[rid] = score(model, D, args.device)
        r = results[rid]["identity"]
        print(f"  {rid:24s} dir={r['direction']:+.3f} mag={r['mag_ratio']:.2f} "
              f"C1={r['C1']:.4f} task1={r['task1']:.1f} rg={r['ret_gauss_meanrank']:.2f} "
              f"rm={r['ret_marg_meanrank']:.2f}", flush=True)
        json.dump(results, open(res_path, "w"), indent=1)
    print("[validity] wrote", res_path)


if __name__ == "__main__":
    main()
