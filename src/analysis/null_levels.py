from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.probe_validity import Data, mean_rank, one_step
from data.dataset import Normalizer
from eval.counterfactual_eval import load_model

TAG = os.environ.get("GT_TAG", "v3")
DATA = Path(os.environ.get("DATA_ROOT", ROOT / "data"))
RES = Path(os.environ.get("RESULTS_ROOT", ROOT / "results")) / "tables"
OBJECTIVES = ("add", "delta", "idm01", "idm", "agm")


def derangement(n, rng):
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nz = Normalizer.load(DATA / "processed" / "latents" / "normalizer.npz")
    D = Data(TAG, DATA, nz)
    u, n = D.pos_idx, D.n_cand
    perm = derangement(len(u), np.random.default_rng(1))
    out = {}
    for obj in OBJECTIVES:
        raw, null = [], []
        for s in range(5):
            model, _ = load_model(f"cf_{obj}_causal_s{s}", device)
            flat = one_step(model, np.repeat(D.cf_ctx_x[u], n, 0), np.repeat(D.cf_ctx_a[u], n, 0),
                            D.cand_gauss.reshape(len(u) * n, -1), device)
            P = flat.reshape(len(u), n, -1)
            raw.append(mean_rank(P, D.next_obs, np.random.default_rng(0)))
            null.append(mean_rank(P, D.next_obs[perm], np.random.default_rng(0)))
        out[obj] = {"raw": raw, "null": null}
        print(f"{obj:6s} raw {np.mean(raw):.2f}  null {np.mean(null):.2f}")
    d = lambda o: np.mean(out[o]["raw"]) - np.mean(out["add"]["raw"])
    dn = lambda o: (np.mean(out["add"]["null"]) - np.mean(out["add"]["raw"])) - (np.mean(out[o]["null"]) - np.mean(out[o]["raw"]))
    print(f"delta prediction minus baseline: raw rank difference {d('delta'):.2f}, "
          f"after subtracting each model's null {dn('delta'):.2f}")
    json.dump(out, open(RES / f"null_levels_{TAG}.json", "w"), indent=1)
    print("wrote", RES / f"null_levels_{TAG}.json")


if __name__ == "__main__":
    main()
