import glob
import json
import sys

import numpy as np
from scipy import stats

d, out_path = sys.argv[1], sys.argv[2]
z = np.load(d + "/items.npz", allow_pickle=True)
z2 = np.load(d + "/items_seed2.npz")
z3 = np.load(d + "/items_seeds345.npz")
sel = np.arange(0, 90, 3)
OFF = (1000, 1500, 2000)
assert np.array_equal(z2["sel"], sel)
assert np.array_equal(z3["pos"], np.tile(sel, 3)) and np.array_equal(z3["off"], np.repeat(OFF, 30))
assert np.array_equal(z3["queries"], z["queries"][z3["pos"]][:, :5])
assert np.array_equal(z3["seed"], z["seed"][z3["pos"]] + z3["off"])
assert np.array_equal(z2["queries"], z["queries"][sel][:, :2]) and np.array_equal(z2["seed"], z["seed"][sel] + 500)


def load(pattern, n):
    out, have = None, np.zeros(n, int)
    for f in sorted(glob.glob(pattern)):
        p = np.load(f)
        s, a = int(p["start"]), p["pred"]
        if out is None:
            out = np.zeros((n,) + a.shape[1:], np.uint8)
        out[s:s + len(a)] = a
        have[s:s + len(a)] += 1
    assert out is not None and (have == 1).all(), f"{pattern}: items not covered exactly once {np.flatnonzero(have != 1)}"
    return out


px = lambda a: a[..., ::2, ::2, :].astype(np.float32) / 255.0
view = lambda a: px(a.transpose(0, 1, 3, 2, 4, 5, 6))
P_all = load(d + "/pred_*.npz", 90)
P3 = load(d + "/seeds345/pred_*.npz", 90)
P = {0: view(P_all[sel][:, :5])}
for j, o in enumerate(OFF):
    P[o] = view(P3[30 * j:30 * (j + 1)])
P500 = view(load(d + "/seed2/pred_*.npz", 30))

T = np.zeros((90, 6, 20, 2, 128, 128, 3), np.uint8)
got = np.zeros(90, int)
for f in glob.glob(d + "/truth/truth_*.npz"):
    t = np.load(f)
    for j, i in enumerate(t["idx"]):
        T[i] = t["img"][j]
        got[i] += 1
assert (got == 1).all()
T = px(T[sel])
fut = px(np.stack([z["fut_agent"][sel], z["fut_wrist"][sel]], 2))
start = px(np.stack([z["hist_agent"][sel, -1], z["hist_wrist"][sel, -1]], 1))
p_start = px(np.stack([z["start_agent"][sel], z["start_wrist"][sel]], 1))
p_fut = px(np.stack([z["p_fut_agent"][sel], z["p_fut_wrist"][sel]], 2))
est = (p_fut - p_start[:, None]) - (fut - start[:, None])
n = len(sel)
fl = lambda a: a.reshape(n, -1)
nrm = lambda a: np.linalg.norm(fl(a), axis=1)
cos = lambda a, b: (fl(a) * fl(b)).sum(1) / (nrm(a) * nrm(b) + 1e-12)
r_true = T[:, 1:5] - T[:, :1]


def measures(R):
    return dict(D=np.mean([cos(R[:, k], r_true[:, k]) for k in range(3)], 0), D_log=cos(R[:, 3], r_true[:, 3]),
                matched=cos(R[:, 3], est))


def summ(v):
    v = np.asarray(v)
    return dict(mean=float(v.mean()), se=float(v.std(ddof=1) / np.sqrt(len(v))), p_vs_0=float(stats.ttest_1samp(v, 0).pvalue),
                per_pos=v.tolist())


R = {s: P[s][:, 1:5] - P[s][:, :1] for s in P}
res = {"positions": sel.tolist(), "seeds": ["orig"] + [f"+{o}" for o in OFF], "per_seed": {}, "seed_mean": {}}
for s in P:
    res["per_seed"][str(s)] = {k: summ(v) for k, v in measures(R[s]).items()}
Rbar = np.mean([R[s] for s in P], 0)
res["seed_averaged_response"] = {k: summ(v) for k, v in measures(Rbar).items()}
for k in ("D", "D_log", "matched"):
    res["seed_mean"][k] = float(np.mean([res["per_seed"][str(s)][k]["mean"] for s in P]))
r1 = {s: P[s][:, 1] - P[s][:, 0] for s in P}
r1[500] = P500[:, 1] - P500[:, 0]
res["offset1_5seeds"] = dict(per_seed={str(s): float(cos(r1[s], r_true[:, 0]).mean()) for s in r1},
                             averaged=summ(cos(np.mean(list(r1.values()), 0), r_true[:, 0])))
keys = list(P)
seed_change = [nrm(P[a][:, 0] - P[b][:, 0]).mean() for i, a in enumerate(keys) for b in keys[i + 1:]]
resp = [nrm(R[s][:, k]).mean() for s in P for k in range(3)]
agree = [cos(R[a][:, k], R[b][:, k]).mean() for i, a in enumerate(keys) for b in keys[i + 1:] for k in range(4)]
res["noise"] = dict(seed_change_logged=float(np.mean(seed_change)), offset_response=float(np.mean(resp)),
                    true_offset_response=float(np.mean([nrm(r_true[:, k]).mean() for k in range(3)])),
                    true_repeat=float(nrm(T[:, 5] - T[:, 0]).mean()), cos_between_seeds=float(np.mean(agree)),
                    estimate_cos=float(cos(est, r_true[:, 3]).mean()))
json.dump(res, open(out_path, "w"), indent=1)
for s in P:
    m = res["per_seed"][str(s)]
    print(f"seed {s:>5}: D {m['D']['mean']:+.3f}  D_log {m['D_log']['mean']:+.3f} (p {m['D_log']['p_vs_0']:.3f})  "
          f"matched {m['matched']['mean']:+.3f} (p {m['matched']['p_vs_0']:.3f})")
print("mean of per seed values:", {k: round(v, 3) for k, v in res["seed_mean"].items()})
m = res["seed_averaged_response"]
print(f"seed averaged response: D {m['D']['mean']:+.3f} (se {m['D']['se']:.3f})  D_log {m['D_log']['mean']:+.3f} "
      f"(se {m['D_log']['se']:.3f}, p {m['D_log']['p_vs_0']:.4f})  matched {m['matched']['mean']:+.3f} "
      f"(se {m['matched']['se']:.3f}, p {m['matched']['p_vs_0']:.4f})")
print("first offset, 5 seeds:", {k: round(v, 3) for k, v in res["offset1_5seeds"]["per_seed"].items()},
      "averaged", round(res["offset1_5seeds"]["averaged"]["mean"], 3))
print("noise:", {k: round(v, 3) for k, v in res["noise"].items()})
