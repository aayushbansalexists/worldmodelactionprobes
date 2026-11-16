import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from analysis.inverse_probe import scores, train_inverse
from analysis.probe_validity import Data, make_T, rollout
from data.dataset import Normalizer
from eval.counterfactual_eval import load_model

TAG, dev = "v3", "cuda"
dr, rr = ROOT / "data", ROOT / "results"
nz = Normalizer.load(dr / "processed" / "latents" / "normalizer.npz")
D = Data(TAG, dr, nz)
mt = np.load(dr / "processed" / "counterfactual" / f"matched_{TAG}.npz", allow_pickle=True)
K = int(mt["k"])
row, a_j = mt["row"], mt["a_j"].astype(np.float32)
rank = np.tile(np.arange(K), len(row) // K)
r_sim = nz.norm_obs(mt["x_match"]) - nz.norm_obs(mt["x_true"])
keep = (rank == 0) & (np.linalg.norm(r_sim.reshape(len(row), -1), axis=1) > 1e-6)
row, a_j = row[keep], a_j[keep]
g, info = train_inverse(dr, nz, dev)
saved = json.load(open(rr / "tables" / f"inverse_probe_{TAG}.json"))
print("inverse model:", info["val_r2_mean"], "saved:", saved["inverse_model"]["val_r2_mean"], flush=True)
base = json.load(open(rr / "tables" / f"probe_validity_{TAG}.json"))
dt = json.load(open(rr / "tables" / "direction_training_v3.json"))
extra = sorted(r for r in dt if r.split("_")[1] not in ("ac001", "ac01", "ac1") and r not in base)
check = ["cf_add_causal_s0", "cf_delta_causal_s0"]
last_r = D.cf_ctx_x[:, -1]
cx, ca, ar = D.cf_ctx_x[row], D.cf_ctx_a[row], D.cf_ref[row]
T = make_T("identity", D.cf_ctx_x.shape[-1]).astype(np.float32)
out = {"inverse_model": info, "check": {}, "models": {}}
for rid in check + extra:
    m, _ = load_model(rid, dev)
    P_ref, P_pert = rollout(m, D.cf_ctx_x, D.cf_ctx_a, D.cf_ref, dev), rollout(m, D.cf_ctx_x, D.cf_ctx_a, D.cf_pert, dev)
    S_ref, S_j = rollout(m, cx, ca, ar, dev), rollout(m, cx, ca, a_j, dev)
    del m
    torch.cuda.empty_cache()
    dr_, er_ = scores(g, last_r, P_ref, P_pert, D.cf_ref, D.cf_pert, T, dev)
    ds_, es_ = scores(g, cx[:, -1], S_ref, S_j, ar, a_j, T, dev)
    res = {"inv_dir_rand": dr_, "inv_err_rand": er_, "inv_dir_swap": ds_, "inv_err_swap": es_}
    if rid in check:
        s = saved["models"][rid]["identity"]
        out["check"][rid] = {"new": res, "saved": s}
        print(f"check {rid}: dir_swap new {ds_:+.4f} saved {s['inv_dir_swap']:+.4f}, dir_rand new {dr_:+.4f} saved {s['inv_dir_rand']:+.4f}", flush=True)
    else:
        out["models"][rid] = {"identity": res}
json.dump(out, open(rr / "additional" / f"inverse_extra_{TAG}.json", "w"), indent=1)
print("scored", len(out["models"]), "extra models")
