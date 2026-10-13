from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

H = 8


def build_pairs(tag="v2", k=1, seed=0, dr=None):
    from data.dataset import Normalizer
    dr = Path(dr or os.environ.get("DATA_ROOT", "data"))
    nz = Normalizer.load(dr / "processed" / "latents" / "normalizer.npz")
    cf = np.load(dr / "processed" / "counterfactual" / f"cflow_{tag}.npz", allow_pickle=True)
    lo = np.load(dr / "processed" / "latents" / "libero_lowdim.npz", allow_pickle=True)
    sp = json.load(open(dr / "splits" / "libero_splits.json"))
    held = set(sp["val"]) | set(sp["test"])
    id2 = {str(i): (o, a) for i, o, a in zip(lo["ids"], lo["obs"], lo["act"])}

    keys = [(str(t), str(d), int(p)) for t, d, p in zip(cf["task"], cf["demo"], cf["pos"])]
    first = {}
    for i, key in enumerate(keys):
        first.setdefault(key, i)
    u = np.array(sorted(first.values()))

    pool = {}
    for did in held:
        task, demo = did.split(":")
        o, a = id2[did]
        on, an = nz.norm_obs(o).astype(np.float32), nz.norm_act(a).astype(np.float32)
        for p in range(0, len(o) - H - 1):
            pool.setdefault(task, []).append((demo, on[p], an[p:p + H], on[p + 1:p + H + 1], p))
    rec = {k_: [] for k_ in ["row", "a_j", "e", "gap", "da", "task", "src_demo", "src_pos"]}
    for r in u:
        task, demo, p = keys[r]
        o, a = id2[f"{task}:{demo}"]
        on, an = nz.norm_obs(o).astype(np.float32), nz.norm_act(a).astype(np.float32)
        s_i, a_i, y_i = on[p], an[p:p + H], on[p + 1:p + H + 1]
        cand = [c for c in pool[task] if c[0] != demo]
        S = np.stack([c[1] for c in cand])
        d = np.linalg.norm(S - s_i, axis=1)
        for jj in np.argsort(d)[:k]:
            demo_j, s_j, a_j, y_j, p_j = cand[jj]
            rec["row"].append(r)
            rec["a_j"].append(a_j)
            rec["e"].append((y_j - s_j) - (y_i - s_i))
            rec["gap"].append(d[jj])
            rec["da"].append(np.linalg.norm(a_j - a_i))
            rec["task"].append(task)
            rec["src_demo"].append(demo_j)
            rec["src_pos"].append(p_j)
    return {k_: np.array(v) if k_ not in ("task", "src_demo") else v for k_, v in rec.items()}
