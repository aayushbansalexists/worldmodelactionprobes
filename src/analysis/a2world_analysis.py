from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

TRANSFORMS = ["identity", "reflect", "rotate", "scale_0.25", "scale_4", "blind"]


def cos(a, b):
    a, b = a.reshape(len(a), -1), b.reshape(len(b), -1)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)


def apply(t, R, perm):
    if t == "identity":
        return R
    if t == "reflect":
        return -R
    if t == "rotate":
        return R.reshape(len(R), -1)[:, perm].reshape(R.shape)
    if t.startswith("scale_"):
        return float(t.split("_")[1]) * R
    return np.zeros_like(R)


def measures(P, T, fut, est, rng):
    n = len(P)
    flat = int(np.prod(P.shape[2:]))
    perm = rng.permutation(flat)
    r_true = T[:, 1:5] - T[:, :1]
    out = {}
    for t in TRANSFORMS:
        R = np.stack([apply(t, P[:, k] - P[:, 0], perm) for k in range(1, 6)], 1)
        PT = P[:, :1] + R
        d = np.mean([cos(R[:, k], r_true[:, k]).mean() for k in range(3)])
        dl = cos(R[:, 3], r_true[:, 3]).mean()
        size = sum(np.linalg.norm(R[:, k].reshape(n, -1), axis=1).sum() for k in range(4)) / \
            sum(np.linalg.norm(r_true[:, k].reshape(n, -1), axis=1).sum() for k in range(4))
        c1 = float((R[:, 4] ** 2).reshape(n, -1).mean())
        cand = np.concatenate([P[:, :1], PT[:, :3]], 1)
        dist = np.sqrt(((cand - fut[:, None]) ** 2).reshape(n, 4, -1).mean(-1))
        rk = (dist[:, 1:] < dist[:, :1]).sum(1) + 0.5 * (dist[:, 1:] == dist[:, :1]).sum(1)
        mp = cos(R[:, 3], est).mean()
        out[t] = dict(D=float(d), D_log=float(dl), size=float(size), C1=c1, ret=float(rk.mean()), matched=float(mp),
                      per_pos=dict(D=np.mean([cos(R[:, k], r_true[:, k]) for k in range(3)], 0).tolist(),
                                   D_log=cos(R[:, 3], r_true[:, 3]).tolist(), matched=cos(R[:, 3], est).tolist(),
                                   ret=rk.tolist(), C1=(R[:, 4] ** 2).reshape(n, -1).mean(1).tolist()))
    out["truth"] = dict(noise_floor=float(np.linalg.norm((T[:, 5] - T[:, 0]).reshape(n, -1), axis=1).mean()),
                        effect=float(np.linalg.norm(r_true[:, :3].reshape(n * 3, -1), axis=1).mean()),
                        estimate_cos=cos(est, r_true[:, 3]).tolist())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a2w", required=True, help="directory with items.npz, truth/, pred_*.npz")
    ap.add_argument("--normalizer", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import torch
    from transformers import AutoModel
    from analysis.encode_dino import encode
    d = Path(args.a2w)
    z = np.load(d / "items.npz", allow_pickle=True)
    n = len(z["seed"])
    pred = np.zeros((n, 6, 2, 20, 128, 128, 3), np.uint8)
    have = np.zeros(n, int)
    for f in glob.glob(str(d / "pred_*.npz")):
        p = np.load(f)
        s = int(p["start"])
        pred[s:s + len(p["pred"])] = p["pred"]
        have[s:s + len(p["pred"])] += 1
    assert (have == 1).all(), f"predictions must cover every position exactly once: {np.flatnonzero(have != 1)}"
    T = np.zeros((n, 6, 20, 2, 128, 128, 3), np.uint8)
    got = np.zeros(n, int)
    for f in glob.glob(str(d / "truth" / "truth_*.npz")):
        t = np.load(f)
        for j, i in enumerate(t["idx"]):
            T[i] = t["img"][j]
            got[i] += 1
    assert (got == 1).all(), f"truth must cover every position exactly once: {np.flatnonzero(got != 1)}"
    P = pred.transpose(0, 1, 3, 2, 4, 5, 6)
    fut = np.stack([z["fut_agent"][:n], z["fut_wrist"][:n]], 2)
    start = np.stack([z["hist_agent"][:n, -1], z["hist_wrist"][:n, -1]], 1)
    p_start = np.stack([z["start_agent"][:n], z["start_wrist"][:n]], 1)
    p_fut = np.stack([z["p_fut_agent"][:n], z["p_fut_wrist"][:n]], 2)
    rng = np.random.default_rng(0)
    res = {"n_positions": n, "task": [str(t) for t in z["task"][:n]]}
    px = lambda a: a[..., ::2, ::2, :].astype(np.float32) / 255.0
    est_px = (px(p_fut) - px(p_start)[:, None]) - (px(fut) - px(start)[:, None])
    res["pixels"] = measures(px(P), px(T), px(fut), est_px, rng)
    m = AutoModel.from_pretrained("facebook/dinov2-small").cuda().eval()
    nz = np.load(args.normalizer)
    f = lambda a: (encode(m, np.ascontiguousarray(a), "cuda") - nz["obs_mean"]) / nz["obs_std"]
    FP, FT, Ff = f(P), f(T), f(fut)
    est_f = (f(p_fut) - f(p_start)[:, None]) - (Ff - f(start)[:, None])
    res["dino"] = measures(FP, FT, Ff, est_f, rng)
    tasks = np.array(res["task"])
    for space in ("pixels", "dino"):
        ident = res[space]["identity"]["per_pos"]
        rank = {}
        for probe in ("matched", "ret", "C1"):
            for truth in ("D", "D_log"):
                tm = [np.mean(np.array(ident[truth])[tasks == u]) for u in sorted(set(tasks))]
                sign = -1 if probe == "ret" else 1
                pm = [sign * np.mean(np.array(ident[probe])[tasks == u]) for u in sorted(set(tasks))]
                rank[f"{probe}_vs_{truth}"] = float(spearmanr(pm, tm)[0])
        res[space]["task_ranking"] = rank
    json.dump(res, open(args.out, "w"), indent=1)
    for space in ("pixels", "dino"):
        print(f"== {space}: noise floor {res[space]['truth']['noise_floor']:.3f}, effect {res[space]['truth']['effect']:.3f}, "
              f"estimate cos mean {np.mean(res[space]['truth']['estimate_cos']):.3f}")
        for t in TRANSFORMS:
            r = res[space][t]
            print(f"   {t:11s} D {r['D']:+.3f} D_log {r['D_log']:+.3f} size {r['size']:.2f} C1 {r['C1']:.4f} ret {r['ret']:.2f} matched {r['matched']:+.3f}")
        print("   task ranking (Spearman over tasks):", {k: round(v, 2) for k, v in res[space]["task_ranking"].items()})


if __name__ == "__main__":
    main()
