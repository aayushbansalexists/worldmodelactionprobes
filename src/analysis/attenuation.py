from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.matched_study import cos_rows
from analysis.probe_validity import Data, rollout
from data.dataset import Normalizer
from eval.counterfactual_eval import load_model

TAG = os.environ.get("GT_TAG", "v3")
DATA = Path(os.environ.get("DATA_ROOT", ROOT / "data"))
RES = Path(os.environ.get("RESULTS_ROOT", ROOT / "results")) / "tables"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nz = Normalizer.load(DATA / "processed" / "latents" / "normalizer.npz")
    D = Data(TAG, DATA, nz)
    mt = np.load(DATA / "processed" / "counterfactual" / f"matched_{TAG}.npz", allow_pickle=True)
    k = int(mt["k"])
    row, e, a_j = mt["row"], mt["e"].astype(np.float32), mt["a_j"].astype(np.float32)
    first = np.tile(np.arange(k), len(row) // k) == 0
    r_sim = (nz.norm_obs(mt["x_match"]) - nz.norm_obs(mt["x_true"])).astype(np.float32)
    sel = first & (np.linalg.norm(r_sim.reshape(len(row), -1), axis=1) > 1e-6)
    c = cos_rows(e, r_sim)[sel]
    cx, ca, ar, aj = D.cf_ctx_x[row][sel], D.cf_ctx_a[row][sel], D.cf_ref[row][sel], a_j[sel]
    out = {}
    for rid in [f"cf_{m}_causal_s{s}" for m in ("add", "agm", "delta") for s in range(5)]:
        model, _ = load_model(rid, device)
        with torch.no_grad():
            R = rollout(model, cx, ca, aj, device) - rollout(model, cx, ca, ar, device)
        cs, ce = cos_rows(R, r_sim[sel]), cos_rows(R, e[sel])
        out[rid] = {"matched": float(ce.mean()), "dir_on": float(cs.mean()),
                    "product_of_means": float(c.mean() * cs.mean()), "per_item": float((c * cs).mean())}
    mean = {k_: float(np.mean([v[k_] for v in out.values()])) for k_ in next(iter(out.values()))}
    print({k_: round(v, 3) for k_, v in mean.items()}, "pairs", int(sel.sum()))
    json.dump({"models": out, "mean": mean, "n_pairs": int(sel.sum())}, open(RES / f"attenuation_{TAG}.json", "w"), indent=1)
    print("wrote", RES / f"attenuation_{TAG}.json")


if __name__ == "__main__":
    main()
