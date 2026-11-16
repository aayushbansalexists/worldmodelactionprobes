import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from analysis.matched_study import cos_rows
from analysis.probe_validity import rollout
from data.dataset import Normalizer
from eval.counterfactual_eval import load_model

dr = ROOT / "data"
nz = Normalizer.load(dr / "processed" / "latents" / "normalizer.npz")
cf = np.load(dr / "processed" / "counterfactual" / "cflow_v3.npz", allow_pickle=True)
pl = np.load(dr / "processed" / "counterfactual" / "plan_v3.npz")
rows, K = pl["row"], pl["x_q"].shape[1]
cx, ca = nz.norm_obs(cf["ctx_x"][rows]).astype(np.float32), nz.norm_act(cf["ctx_a"][rows]).astype(np.float32)
a_ref = nz.norm_act(pl["a_ref"]).astype(np.float32)
r_true = nz.norm_obs(pl["x_q"]) - nz.norm_obs(pl["x_ref"])[:, None]
R = ROOT / "results" / "tables"
pv = json.load(open(R / "probe_validity_v3.json")); ms = json.load(open(R / "matched_v3.json"))["models"]
dt = json.load(open(R / "direction_training_v3.json"))
M = {}
for r in pv:
    if r.split("_")[2] == "causal":
        M[r] = dict(D=pv[r]["identity"]["direction"], Dlog=ms[r]["identity"]["dir_on"], matched=ms[r]["identity"]["matched_k1"],
                    ret=-pv[r]["identity"]["ret_gauss_meanrank"], val=-json.load(open(f"results/raw/{r}/metrics.json"))["best_val_mse"])
for r, v in dt.items():
    if r.split("_")[1] in ("ac001", "ac01", "ac1") or r in M:
        continue
    M[r] = dict(D=v["direction"], Dlog=v["dir_on"], matched=v["matched"], ret=-v["ret_gauss_meanrank"], val=-v["val_mse"])
for rid in sorted(M):
    m, _ = load_model(rid, "cuda")
    P0 = rollout(m, cx, ca, a_ref, "cuda")
    c = [cos_rows(rollout(m, cx, ca, nz.norm_act(pl["a_q"][:, k]).astype(np.float32), "cuda") - P0, r_true[:, k]) for k in range(K)]
    M[rid]["Dplan"] = float(np.mean(c))
    del m
    torch.cuda.empty_cache()
for name, f in (("hard", "closed_loop_hard_g10_e4.json"), ("easy", "closed_loop_hard_g4_e1.json")):
    ep = json.load(open(ROOT / "results" / "additional" / f))
    for r in M:
        M[r][f"miss_{name}"] = -float(np.mean([e["d_final"] for e in ep if e["run"] == r]))
        M[r][f"succ_{name}"] = float(np.mean([e["d_final"] < 0.02 for e in ep if e["run"] == r]))
json.dump(M, open(ROOT / "results" / "additional" / "dplan.json", "w"), indent=1)
obj = lambda r: r.split("_")[1]
sets = {"90 all": list(M), "65 no act*": [r for r in M if not obj(r).startswith("act")],
        "55 working": [r for r in M if not (obj(r).startswith("act") or obj(r) in ("idm", "mdshuf01"))],
        "30 ordinary": [r for r in M if obj(r) in ("add", "agm", "delta", "idm", "idm01", "idm001")]}
print("mean Dplan over the 15 add/agm/delta models:", round(np.mean([M[r]["Dplan"] for r in M if obj(r) in ("add", "agm", "delta") and r.split("_")[2] == "causal"]), 3))
for name, rs in sets.items():
    sp = lambda a, b: spearmanr([M[r][a] for r in rs], [M[r][b] for r in rs])[0]
    print(f"{name} (n={len(rs)})")
    print("   predicts Dplan: " + ", ".join(f"{p} {sp(p, 'Dplan'):+.2f}" for p in ("D", "Dlog", "matched", "ret", "val")))
    print("   Dplan vs control: " + ", ".join(f"{c} {sp('Dplan', c):+.2f}" for c in ("miss_hard", "succ_hard", "miss_easy", "succ_easy")))
    print("   for comparison, control vs val: " + ", ".join(f"{c} {sp('val', c):+.2f}" for c in ("miss_hard", "succ_hard")) +
          " | vs matched: " + ", ".join(f"{c} {sp('matched', c):+.2f}" for c in ("miss_hard", "succ_hard")))
