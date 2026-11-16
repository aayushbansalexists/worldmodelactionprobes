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
from analysis.probe_validity import make_T, rollout
from data.dataset import Normalizer
from eval.counterfactual_eval import load_model

dev = "cuda"
dr = ROOT / "data"
nz = Normalizer.load(dr / "processed" / "latents" / "normalizer.npz")
cf = np.load(dr / "processed" / "counterfactual" / "cflow_v3.npz", allow_pickle=True)
pl = np.load(dr / "processed" / "counterfactual" / "plan_v3.npz")
rows, K = pl["row"], pl["x_q"].shape[1]
cx, ca = nz.norm_obs(cf["ctx_x"][rows]).astype(np.float32), nz.norm_act(cf["ctx_a"][rows]).astype(np.float32)
a_ref = nz.norm_act(pl["a_ref"]).astype(np.float32)
a_q = [nz.norm_act(pl["a_q"][:, k]).astype(np.float32) for k in range(K)]
g, info = train_inverse(dr, nz, dev)
saved = json.load(open(ROOT / "results" / "tables" / "inverse_probe_v3.json"))["inverse_model"]
assert abs(info["val_r2_mean"] - saved["val_r2_mean"]) < 1e-9, (info["val_r2_mean"], saved["val_r2_mean"])
T = make_T("identity", cx.shape[-1]).astype(np.float32)
runs = sorted(json.load(open(ROOT / "results" / "additional" / "dplan.json")))
out = {}
for rid in runs:
    m, _ = load_model(rid, dev)
    P0 = rollout(m, cx, ca, a_ref, dev)
    out[rid] = float(np.mean([scores(g, cx[:, -1], P0, rollout(m, cx, ca, a_q[k], dev), a_ref, a_q[k], T, dev)[0] for k in range(K)]))
    del m
    torch.cuda.empty_cache()
json.dump(out, open(ROOT / "results" / "additional" / "inverse_plan.json", "w"), indent=1)
print("scored", len(out), "models; mean over add/agm/delta",
      np.mean([v for r, v in out.items() if r.split("_")[1] in ("add", "agm", "delta")]))
