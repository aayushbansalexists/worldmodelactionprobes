from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from analysis.matched_study import cos_rows
from analysis.probe_validity import Data, rollout, score
from data.dataset import Normalizer
from eval.counterfactual_eval import load_model

TAG = os.environ.get("GT_TAG", "v3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dr = Path(os.environ.get("DATA_ROOT", "data"))
    rr = Path(os.environ.get("RESULTS_ROOT", "results"))
    nz = Normalizer.load(dr / "processed" / "latents" / "normalizer.npz")
    D = Data(TAG, dr, nz)
    mt = np.load(dr / "processed" / "counterfactual" / f"matched_{TAG}.npz", allow_pickle=True)
    K = int(mt["k"])
    row, e, a_j = mt["row"], mt["e"].astype(np.float32), mt["a_j"].astype(np.float32)
    rank = np.tile(np.arange(K), len(row) // K)
    r_sim = (nz.norm_obs(mt["x_match"]) - nz.norm_obs(mt["x_true"])).astype(np.float32)
    ok = (np.linalg.norm(r_sim.reshape(len(row), -1), axis=1) > 1e-6) & (rank == 0)
    cx, ca, ar = D.cf_ctx_x[row[ok]], D.cf_ctx_a[row[ok]], D.cf_ref[row[ok]]
    out_path = rr / "tables" / f"direction_training_{TAG}.json"
    res = json.load(open(out_path)) if out_path.exists() else {}
    for rid in args.runs:
        if rid in res:
            continue
        m, _ = load_model(rid, args.device)
        s = score(m, D, args.device)["identity"]
        R = rollout(m, cx, ca, a_j[ok], args.device) - rollout(m, cx, ca, ar, args.device)
        del m
        torch.cuda.empty_cache()
        met = json.load(open(rr / "raw" / rid / "metrics.json"))
        res[rid] = {"direction": s["direction"], "dir_on": float(cos_rows(R, r_sim[ok]).mean()),
                    "matched": float(cos_rows(R, e[ok]).mean()), "mag_ratio": s["mag_ratio"],
                    "cf_error": s["cf_error"], "ret_gauss_meanrank": s["ret_gauss_meanrank"],
                    "val_mse": met["best_val_mse"], "matched_direction": met.get("matched_direction", 0.0)}
        r = res[rid]
        print(f"  {rid:24s} D={r['direction']:+.3f} D_log={r['dir_on']:+.3f} matched={r['matched']:+.3f} "
              f"size={r['mag_ratio']:.2f} cf_err={r['cf_error']:.3f} ret={r['ret_gauss_meanrank']:.2f} "
              f"val_mse={r['val_mse']:.6f}", flush=True)
        json.dump(res, open(out_path, "w"), indent=1)
    print("[direction training] wrote", out_path)


if __name__ == "__main__":
    main()
