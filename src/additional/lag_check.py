import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
c = np.load(ROOT / "data" / "processed" / "counterfactual" / "cflow_v3.npz", allow_pickle=True)
xt, xn, xp = c["x_true"].astype(float), c["x_noise"].astype(float), c["x_pert"].astype(float)
J = slice(0, 7)
mot = np.concatenate([xt[:, :1] - c["ctx_x"][:, -1:].astype(float), xt[:, 1:] - xt[:, :-1]], 1)
a, b = mot[:, 1:, J].ravel(), (xn - xt)[:, 1:, J].ravel()
out = {"slope_repeat_minus_first_on_motion_per_step": float(np.dot(a, b) / np.dot(a, a)), "one_substep": -1 / 25,
       "correlation": float(np.corrcoef(a, b)[0, 1]), "n_items": int(len(xt)), "effect_of_lag_on_D_random_truth": {}}
r = (xp - xt).reshape(len(xt), -1)
for k in (2, 3, 4):
    art = (-(k / 25.0) * mot).reshape(len(xt), -1)
    cl = r - art
    cos = (r * cl).sum(1) / (np.linalg.norm(r, axis=1) * np.linalg.norm(cl, axis=1))
    out["effect_of_lag_on_D_random_truth"][f"{k}_substeps"] = {
        "artifact_over_response_norm_median": float(np.median(np.linalg.norm(art, axis=1) / np.linalg.norm(r, axis=1))),
        "cosine_measured_vs_corrected_mean": float(cos.mean()), "cosine_min": float(cos.min())}
json.dump(out, open(ROOT / "results" / "additional" / "lag_check.json", "w"), indent=1)
print(json.dumps(out, indent=1))
