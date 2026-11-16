import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from analysis.inverse_probe import Inverse, scores, train_inverse, transitions
from analysis.probe_validity import Data, make_T, rollout
from data.dataset import Normalizer
from eval.counterfactual_eval import load_model

TAG, dev = "v3", "cuda"
dr, rt, rr = ROOT / "data", ROOT / "results" / "tables", ROOT / "results" / "additional"
FRACS = (1.0, 0.1, 0.03, 0.01, 0.003)
obj = lambda r: r.split("_")[1]
ORD = ("add", "agm", "delta", "idm", "idm01", "idm001")

pv = json.load(open(rt / "probe_validity_v3.json"))
ms = json.load(open(rt / "matched_v3.json"))["models"]
dt = json.load(open(rt / "direction_training_v3.json"))
ipp = json.load(open(rt / "inverse_probe_v3.json"))["models"]
ipx = json.load(open(rr / "inverse_extra_v3.json"))["models"]
PM = {}
for r in pv:
    if r.split("_")[2] == "causal":
        PM[r] = dict(Dlog=ms[r]["identity"]["dir_on"], matched=ms[r]["identity"]["matched_k1"],
                     ic=ipp[r]["identity"]["inv_dir_swap"])
for r, v in dt.items():
    if obj(r) in ("ac001", "ac01", "ac1") or r in PM:
        continue
    PM[r] = dict(Dlog=v["dir_on"], matched=v["matched"], ic=ipx[r]["identity"]["inv_dir_swap"])
assert len(PM) == 90, len(PM)
runs = sorted(PM)
SETS = {"30": [r for r in runs if obj(r) in ORD],
        "55": [r for r in runs if not (obj(r).startswith("act") or obj(r) in ("idm", "mdshuf01"))],
        "65": [r for r in runs if not obj(r).startswith("act")],
        "90": runs}


def boot(rs, probe, truth, B=4000):
    groups = {}
    for r in rs:
        groups.setdefault(obj(r), []).append(r)
    keys = sorted(groups)
    rng = np.random.default_rng(0)
    sp = lambda pick, p: spearmanr([p[r] for r in pick], [truth[r] for r in pick])[0]
    out = {k: [] for k in probe}
    for _ in range(B):
        pick = [r for k in rng.choice(keys, len(keys)) for r in groups[k]]
        for k in probe:
            out[k].append(sp(pick, probe[k]))
    res = {k: [sp(rs, probe[k]), *np.nanpercentile(out[k], [2.5, 97.5])] for k in probe}
    for k in probe:
        if k != "matched":
            d = np.array(out["matched"]) - np.array(out[k])
            res[f"matched-{k}"] = [res["matched"][0] - res[k][0], *np.nanpercentile(d, [2.5, 97.5])]
    return {k: [round(float(x), 4) for x in v] for k, v in res.items()}


def train_subset(nz, frac, seed=0, base_epochs=30):
    torch.manual_seed(seed)
    lo = np.load(dr / "processed" / "latents" / "libero_lowdim.npz", allow_pickle=True)
    sp = json.load(open(dr / "splits" / "libero_splits.json"))
    ids = sorted(sp["train"])
    keep = sorted(np.random.default_rng(seed).choice(ids, max(1, round(frac * len(ids))), replace=False).tolist())
    Xt, Yt, At = [t.to(dev) for t in transitions(lo, keep, nz)]
    Xv, Yv, Av = [t.to(dev) for t in transitions(lo, sp["val"], nz)]
    n_full = sum(len(o) - 1 for i, o in zip(lo["ids"], lo["obs"]) if str(i) in set(ids))
    steps = base_epochs * math.ceil(n_full / 1024)
    epochs = math.ceil(steps / math.ceil(len(Xt) / 1024))
    g = Inverse(obs_dim=Xt.shape[-1], act_dim=At.shape[-1]).to(dev)
    opt = torch.optim.AdamW(g.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(epochs):
        g.train()
        perm = torch.randperm(len(Xt), generator=gen).to(dev)
        for i in range(0, len(Xt), 1024):
            b = perm[i:i + 1024]
            loss = ((g(Xt[b], Yt[b]) - At[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    g.eval()
    with torch.no_grad():
        r2 = 1 - ((g(Xv, Yv) - Av) ** 2).mean(0) / Av.var(0)
    return g, {"frac": frac, "n_demos": len(keep), "n_train": int(len(Xt)), "epochs": epochs,
               "val_r2_mean": float(r2.mean())}


nz = Normalizer.load(dr / "processed" / "latents" / "normalizer.npz")
D = Data(TAG, dr, nz)
mt = np.load(dr / "processed" / "counterfactual" / f"matched_{TAG}.npz", allow_pickle=True)
K = int(mt["k"])
row, a_j = mt["row"], mt["a_j"].astype(np.float32)
rank = np.tile(np.arange(K), len(row) // K)
r_sim = nz.norm_obs(mt["x_match"]) - nz.norm_obs(mt["x_true"])
keep = (rank == 0) & (np.linalg.norm(r_sim.reshape(len(row), -1), axis=1) > 1e-6)
row, a_j = row[keep], a_j[keep]
cx, ca, ar = D.cf_ctx_x[row], D.cf_ctx_a[row], D.cf_ref[row]
T = make_T("identity", D.cf_ctx_x.shape[-1]).astype(np.float32)

G, info = {}, {}
for f in FRACS:
    if f == 1.0:
        g, inf = train_inverse(dr, nz, dev)
        inf = {"frac": 1.0, "n_demos": len(json.load(open(dr / "splits" / "libero_splits.json"))["train"]),
               "n_train": inf["n_train"], "epochs": 30, "val_r2_mean": inf["val_r2_mean"]}
    else:
        g, inf = train_subset(nz, f)
    G[f], info[str(f)] = g, inf
    print("inverse model", json.dumps(inf), flush=True)

ic = {str(f): {} for f in FRACS}
for rid in runs:
    m, _ = load_model(rid, dev)
    S_ref, S_j = rollout(m, cx, ca, ar, dev), rollout(m, cx, ca, a_j, dev)
    del m
    torch.cuda.empty_cache()
    for f in FRACS:
        ic[str(f)][rid] = scores(G[f], cx[:, -1], S_ref, S_j, ar, a_j, T, dev)[0]
dev_full = max(abs(ic["1.0"][r] - PM[r]["ic"]) for r in runs)
print("full data inverse check vs saved, max abs difference:", dev_full, flush=True)
assert dev_full < 1e-4, dev_full

truth = {r: PM[r]["Dlog"] for r in runs}
res = {}
for s, rs in SETS.items():
    probes = {"matched": {r: PM[r]["matched"] for r in rs}}
    probes.update({f"ic_{f}": {r: ic[str(f)][r] for r in rs} for f in FRACS})
    res[s] = boot(rs, probes, truth)
    print(s, {k: v[0] for k, v in res[s].items() if not k.startswith("matched-")}, flush=True)
json.dump({"inverse_models": info, "max_abs_diff_full_vs_saved": dev_full, "spearman_with_Dlog": res, "ic": ic},
          open(rr / "inverse_strength.json", "w"), indent=1)
print("wrote results/additional/inverse_strength.json")
