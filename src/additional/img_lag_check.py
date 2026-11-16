import json
import sys
from pathlib import Path

import numpy as np

d, out = Path(sys.argv[1]), sys.argv[2]
c = np.load(d / "cflow_v3.npz", allow_pickle=True)
xt, xn, xp = (c[k].astype(np.float64) for k in ("x_true", "x_noise", "x_pert"))
n = len(xt)
flat = lambda a: a.reshape(len(a), -1)
cosr = lambda a, b: (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
lag1 = xn - xt
mot = np.concatenate([xt[:, :1] - c["ctx_x"][:, -1:].astype(np.float64), xt[:, 1:] - xt[:, :-1]], 1)
a, b = mot[:, 1:].ravel(), lag1[:, 1:].ravel()
res = {"n_items": int(n), "sigma_first9": [float(s) for s in c["sigma"][:9]],
       "floor_norm_mean": float(np.linalg.norm(flat(lag1), axis=1).mean()),
       "effect_norm_mean": float(np.linalg.norm(flat(xp - xt), axis=1).mean()),
       "slope_lag_on_motion": float(np.dot(a, b) / np.dot(a, a)), "corr_lag_motion": float(np.corrcoef(a, b)[0, 1]),
       "same_logged_branch_in_all_three_items_of_a_position": bool(np.allclose(xt[0::3], xt[1::3]) and np.allclose(xt[0::3], xt[2::3]))}
r = flat(xp - xt)
res["D_random_truth"] = {}
for s in range(3):
    k = s + 2
    idx = np.arange(s, n, 3)
    art = k * flat(lag1)[idx]
    rc = r[idx] - art
    cs = cosr(r[idx], rc)
    res["D_random_truth"][f"sigma_index_{s}_lag_{k}"] = {
        "sigma": float(c["sigma"][idx[0]]), "response_norm_mean": float(np.linalg.norm(r[idx], axis=1).mean()),
        "lag_over_response_norm_median": float(np.median(np.linalg.norm(art, axis=1) / np.linalg.norm(r[idx], axis=1))),
        "lag_share_of_energy_median": float(np.median((art ** 2).sum(1) / (r[idx] ** 2).sum(1))),
        "cos_measured_vs_corrected_mean": float(cs.mean()), "cos_p10": float(np.quantile(cs, 0.1)), "cos_min": float(cs.min())}
for name in ("matched_v3.npz", "matched_v3pp.npz"):
    if not (d / name).exists():
        continue
    m = np.load(d / name, allow_pickle=True)
    K = int(m["k"])
    xm, xtm, row = m["x_match"].astype(np.float64), m["x_true"].astype(np.float64), m["row"]
    rank = np.arange(len(row)) % K
    ci = row if row.max() >= n // 3 else 3 * row
    l1 = flat(lag1)[ci]
    same = float(np.median(np.linalg.norm(flat(xtm) - flat(xt)[ci], axis=1) / np.linalg.norm(flat(lag1)[ci], axis=1)))
    rr = flat(xm - xtm)
    o = {"k": K, "row_max": int(row.max()), "logged_branch_here_vs_in_cflow_over_one_lag_median": same}
    for j in range(K):
        idx = rank == j
        art = (j + 1) * l1[idx]
        cs = cosr(rr[idx], rr[idx] - art)
        o[f"rank_{j}_lag_{j + 1}"] = {"response_norm_mean": float(np.linalg.norm(rr[idx], axis=1).mean()),
                                      "lag_over_response_norm_median": float(np.median(np.linalg.norm(art, axis=1) / np.linalg.norm(rr[idx], axis=1))),
                                      "lag_share_of_energy_median": float(np.median((art ** 2).sum(1) / (rr[idx] ** 2).sum(1))),
                                      "cos_measured_vs_corrected_mean": float(cs.mean()), "cos_p10": float(np.quantile(cs, 0.1)), "cos_min": float(cs.min())}
    res["D_logged_truth_" + name.split(".")[0]] = o
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res, indent=1))
