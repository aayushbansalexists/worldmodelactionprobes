from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import expm

from analysis.probe_validity import Data, mean_rank, one_step, rollout
from data.dataset import Normalizer
from eval.counterfactual_eval import load_model
TAG = os.environ.get("GT_TAG", "v3")

_WSP = Path(__file__).resolve().parents[1] / "third_party" / "WorldSimProbe" / "src"
import sys
sys.path.insert(0, str(_WSP))
from worldsimprobe.evaluation.task1_action_calibration.oracle_ratio import (
    triangular_ratio_score,
)

THETAS = np.linspace(0, np.pi, 13)
LOG2S = np.linspace(-3, 3, 13)


def skew(dim=21, planes=10, seed=0):
    q, r = np.linalg.qr(np.random.default_rng(seed).standard_normal((dim, dim)))
    q = q * np.sign(np.diag(r))
    K = np.zeros((dim, dim))
    for p in range(planes):
        u, v = q[:, 2 * p], q[:, 2 * p + 1]
        K += np.outer(v, u) - np.outer(u, v)
    return K


def cos_rows(a, b):
    a, b = a.reshape(len(a), -1), b.reshape(len(b), -1)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)


def main():
    dr = Path(os.environ.get("DATA_ROOT", "data"))
    rr = Path(os.environ.get("RESULTS_ROOT", "results"))
    nz = Normalizer.load(dr / "processed" / "latents" / "normalizer.npz")
    D = Data(TAG, dr, nz)
    mt = np.load(dr / "processed" / "counterfactual" / f"matched_{TAG}.npz", allow_pickle=True)
    k1 = np.arange(len(mt["row"])) % int(mt["k"]) == 0
    row, e, a_j = mt["row"][k1], mt["e"][k1].astype(np.float32), mt["a_j"][k1].astype(np.float32)
    r_sim = (nz.norm_obs(mt["x_match"][k1]) - nz.norm_obs(mt["x_true"][k1])).astype(np.float32)
    K = skew()
    runs = [f"cf_{m}_causal_s{s}" for m in ("add", "agm", "delta") for s in range(5)]
    u, n = D.pos_idx, D.n_cand
    grid = {q: np.zeros((len(runs), len(THETAS), len(LOG2S))) for q in
            ["dir", "dir_on", "C1", "task1", "ret", "matched", "cf_error"]}
    for mi, rid in enumerate(runs):
        m, _ = load_model(rid, "cuda")
        P_ref = rollout(m, D.cf_ctx_x, D.cf_ctx_a, D.cf_ref, "cuda")
        P_pert = rollout(m, D.cf_ctx_x, D.cf_ctx_a, D.cf_pert, "cuda")
        P_zero = rollout(m, D.cf_ctx_x[u], D.cf_ctx_a[u], np.zeros_like(D.cf_ref[u]), "cuda")
        Pg = one_step(m, np.repeat(D.cf_ctx_x[u], n, 0), np.repeat(D.cf_ctx_a[u], n, 0),
                      D.cand_gauss.reshape(len(u) * n, -1), "cuda").reshape(len(u), n, -1)
        W_o = rollout(m, D.ws_ctx_x, D.ws_ctx_a, D.ws["orig"], "cuda")
        W_s = rollout(m, D.ws_ctx_x, D.ws_ctx_a, D.ws["small"], "cuda")
        W_l = rollout(m, D.ws_ctx_x, D.ws_ctx_a, D.ws["large"], "cuda")
        cx, ca = D.cf_ctx_x[row], D.cf_ctx_a[row]
        R_m = rollout(m, cx, ca, a_j, "cuda") - rollout(m, cx, ca, D.cf_ref[row], "cuda")
        del m
        torch.cuda.empty_cache()
        dt = D.cf_dtrue
        ok_sim = (D.ws_dl_sim > 1e-12) & (D.ws_dl_sim > D.ws_ds_sim)
        o_sim = np.where(ok_sim, D.ws_ds_sim / np.where(ok_sim, D.ws_dl_sim, 1), np.nan)
        for ti, th in enumerate(THETAS):
            Q = expm(th * K).astype(np.float32)
            for si, l2 in enumerate(LOG2S):
                T = (2.0 ** l2) * Q
                tr = lambda X: X @ T.T
                dp = tr(P_pert - P_ref)
                grid["dir"][mi, ti, si] = cos_rows(dp, dt).mean()
                nt = np.linalg.norm(dt.reshape(len(dt), -1), axis=1)
                grid["cf_error"][mi, ti, si] = np.median(
                    np.linalg.norm((dp - dt).reshape(len(dt), -1), axis=1) / (nt + 1e-12))
                grid["C1"][mi, ti, si] = (tr(P_zero - P_ref[u]) ** 2).mean()
                tg = Pg[:, :1] + tr(Pg - Pg[:, :1])
                grid["ret"][mi, ti, si] = mean_rank(tg, D.next_obs, np.random.default_rng(0))
                ds = (tr(W_s - W_o) ** 2).mean((1, 2))
                dl = (tr(W_l - W_o) ** 2).mean((1, 2))
                t1 = [triangular_ratio_score(a / b, o) for a, b, o in zip(ds, dl, o_sim)
                      if np.isfinite(o) and 0 <= o < 1 and b > 1e-12]
                grid["task1"][mi, ti, si] = 100 * np.mean(t1)
                RT = tr(R_m)
                grid["matched"][mi, ti, si] = cos_rows(RT, e).mean()
                grid["dir_on"][mi, ti, si] = cos_rows(RT, r_sim).mean()
        print(f"  {rid}: dir(0,1)={grid['dir'][mi, 0, 6]:+.3f} dir(pi,1)={grid['dir'][mi, -1, 6]:+.3f} "
              f"matched(0,1)={grid['matched'][mi, 0, 6]:+.3f}", flush=True)
    out = {"thetas": THETAS.tolist(), "log2s": LOG2S.tolist(), "runs": runs,
           "mean": {q: g.mean(0).tolist() for q, g in grid.items()},
           "per_model": {q: g.tolist() for q, g in grid.items()}}
    json.dump(out, open(rr / "tables" / f"probe_landscape_{TAG}.json", "w"))
    print("[landscape] wrote", rr / "tables" / f"probe_landscape_{TAG}.json")


if __name__ == "__main__":
    main()
