import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from analysis.matched_study import cos_rows
from analysis.probe_validity import rollout
from data.dataset import Normalizer
from eval.closed_loop import EEF
from eval.counterfactual_eval import load_model

dr = ROOT / "data"
nz = Normalizer.load(dr / "processed" / "latents" / "normalizer.npz")
cf = np.load(dr / "processed" / "counterfactual" / "cflow_v3.npz", allow_pickle=True)
pl = np.load(dr / "processed" / "counterfactual" / "plan_v3.npz")
rows, K = pl["row"], pl["x_q"].shape[1]
cx, ca = nz.norm_obs(cf["ctx_x"][rows]).astype(np.float32), nz.norm_act(cf["ctx_a"][rows]).astype(np.float32)
a_ref = nz.norm_act(pl["a_ref"]).astype(np.float32)
r_true = nz.norm_obs(pl["x_q"]) - nz.norm_obs(pl["x_ref"])[:, None]
M = json.load(open(ROOT / "results" / "additional" / "dplan.json"))
out = {}
for rid in sorted(M):
    m, _ = load_model(rid, "cuda")
    P0 = rollout(m, cx, ca, a_ref, "cuda")
    R = [rollout(m, cx, ca, nz.norm_act(pl["a_q"][:, k]).astype(np.float32), "cuda") - P0 for k in range(K)]
    out[rid] = dict(Dplan_full=float(np.mean([cos_rows(R[k], r_true[:, k]) for k in range(K)])),
                    Dplan_eef=float(np.mean([cos_rows(R[k][..., EEF], r_true[:, k][..., EEF]) for k in range(K)])))
    del m
    torch.cuda.empty_cache()
json.dump(out, open(ROOT / "results" / "additional" / "dplan_eef.json", "w"), indent=1)
print("max |full - dplan.json|", max(abs(out[r]["Dplan_full"] - M[r]["Dplan"]) for r in M))
print("mean Dplan_eef over add/agm/delta", np.mean([out[r]["Dplan_eef"] for r in M if r.split("_")[1] in ("add", "agm", "delta")]))
