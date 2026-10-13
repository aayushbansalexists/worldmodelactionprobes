from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from analysis.probe_validity import TRANSFORMS, Data, make_T, rollout
from data.dataset import Normalizer
from eval.counterfactual_eval import load_model
TAG = os.environ.get("GT_TAG", "v3")


def cos_rows(a, b):
    a, b = a.reshape(len(a), -1), b.reshape(len(b), -1)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)


def main():
    dr = Path(os.environ.get("DATA_ROOT", "data"))
    rr = Path(os.environ.get("RESULTS_ROOT", "results"))
    nz = Normalizer.load(dr / "processed" / "latents" / "normalizer.npz")
    D = Data(TAG, dr, nz)
    mt = np.load(dr / "processed" / "counterfactual" / f"matched_{TAG}.npz", allow_pickle=True)
    K = int(mt["k"])
    row, e, a_j = mt["row"], mt["e"].astype(np.float32), mt["a_j"].astype(np.float32)
    rank = np.tile(np.arange(K), len(row) // K)
    r_sim = (nz.norm_obs(mt["x_match"]) - nz.norm_obs(mt["x_true"])).astype(np.float32)
    ok = np.linalg.norm(r_sim.reshape(len(row), -1), axis=1) > 1e-6

    est = cos_rows(e, r_sim)
    summary = {
        "n_positions": int(len(np.unique(row))), "k": K, "n_pairs": int(len(row)),
        "estimate_vs_sim_cos_mean": float(est[ok].mean()),
        "estimate_vs_sim_cos_median": float(np.median(est[ok])),
        "estimate_vs_sim_cos_k1": float(est[ok & (rank == 0)].mean()),
        "state_gap_median": float(np.median(mt["gap"])),
        "action_diff_median": float(np.median(mt["da"])),
        "sim_response_norm_median": float(np.median(np.linalg.norm(r_sim.reshape(len(row), -1), axis=1))),
    }
    print(json.dumps(summary, indent=1))

    base = json.load(open(rr / "tables" / f"probe_validity_{TAG}.json"))
    cx, ca, ar = D.cf_ctx_x[row], D.cf_ctx_a[row], D.cf_ref[row]
    out = {"summary": summary, "models": {}}
    for rid in sorted(base):
        m, _ = load_model(rid, "cuda")
        R = rollout(m, cx, ca, a_j, "cuda") - rollout(m, cx, ca, ar, "cuda")
        del m
        torch.cuda.empty_cache()
        res = {}
        for t in TRANSFORMS:
            T = make_T(t, R.shape[-1]).astype(np.float32)
            RT = R @ T.T
            c_est, c_sim = cos_rows(RT, e), cos_rows(RT, r_sim)
            res[t] = {f"matched_k{k}": float(c_est[ok & (rank < k)].mean()) for k in (1, 3, 5)}
            res[t].update({f"dir_on_k{k}": float(c_sim[ok & (rank < k)].mean()) for k in (1, 3, 5)})
            res[t]["dir_on"] = res[t]["dir_on_k1"]
        out["models"][rid] = res
        print(f"  {rid:24s} matched_k1={res['identity']['matched_k1']:+.3f} "
              f"k5={res['identity']['matched_k5']:+.3f} dir_on={res['identity']['dir_on']:+.3f} "
              f"dir_gauss={base[rid]['identity']['direction']:+.3f}", flush=True)
    json.dump(out, open(rr / "tables" / f"matched_{TAG}.json", "w"), indent=1)
    print("[matched] wrote", rr / "tables" / f"matched_{TAG}.json")


if __name__ == "__main__":
    main()
