from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
from pathlib import Path

import numpy as np

H, L = 8, 16


def load_feats(feat, prefix):
    out = {}
    for fp in sorted(glob.glob(str(Path(feat) / f"{prefix}_*.npz"))):
        task = os.path.basename(fp)[len(prefix) + 1:-4]
        out[task] = dict(np.load(fp, allow_pickle=True))
    return out


def estimate(F, nz_std, task, d_i, p_i, d_j, p_j):
    fi, fj = F[task][d_i], F[task][d_j]
    return (((fj[p_j + 1:p_j + 1 + H] - fj[p_j]) - (fi[p_i + 1:p_i + 1 + H] - fi[p_i])) / nz_std).astype(np.float32)


def base(src, feat, dst):
    from data.dataset import Normalizer
    D = load_feats(feat, "demos")
    I = load_feats(feat, "items")
    lo = np.load(src / "processed" / "latents" / "libero_lowdim.npz", allow_pickle=True)
    obs = []
    for did, a in zip(lo["ids"], lo["act"]):
        task, dk = str(did).split(":")
        f = D[task][dk]
        assert len(f) == len(a), (did, len(f), len(a))
        obs.append(f.astype(np.float32))
    (dst / "processed" / "latents").mkdir(parents=True, exist_ok=True)
    (dst / "processed" / "counterfactual").mkdir(parents=True, exist_ok=True)
    (dst / "splits").mkdir(parents=True, exist_ok=True)
    np.savez(dst / "processed" / "latents" / "libero_lowdim.npz", ids=lo["ids"], suites=lo["suites"],
             obs=np.array(obs, dtype=object), act=lo["act"], obs_dim=obs[0].shape[-1], action_dim=lo["action_dim"])
    shutil.copy(src / "splits" / "libero_splits.json", dst / "splits" / "libero_splits.json")
    sp = json.load(open(src / "splits" / "libero_splits.json"))
    tr = set(sp["train"])
    nz = Normalizer.fit_from([o for o, i in zip(obs, lo["ids"]) if str(i) in tr],
                             [a for a, i in zip(lo["act"], lo["ids"]) if str(i) in tr])
    nz.save(dst / "processed" / "latents" / "normalizer.npz")
    F = {t: {k: v for k, v in d.items()} for t, d in D.items()}

    cf = dict(np.load(src / "processed" / "counterfactual" / "cflow_v3.npz", allow_pickle=True))
    n = len(cf["task"])
    fd = obs[0].shape[-1]
    out = {k: np.zeros((n, H, fd), np.float32) for k in ("x_true", "x_pert", "x_noise")}
    ctx = np.zeros((n, L, fd), np.float32)
    replay = []
    for task, it in I.items():
        for j, r in enumerate(it["cf_row"]):
            r = int(r)
            out["x_true"][r], out["x_pert"][r], out["x_noise"][r] = it["cf_img_true"][j], it["cf_img_pert"][j], it["cf_img_noise"][j]
            replay.append(np.abs(it["cf_x_pert"][j] - cf["x_pert"][r]).max())
    for r in range(n):
        p = int(cf["pos"][r])
        ctx[r] = F[str(cf["task"][r])][str(cf["demo"][r])][p - L + 1:p + 1]
    filled = sum(len(it["cf_row"]) for it in I.values())
    assert filled == n, (filled, n)
    cf.update(out, ctx_x=ctx)
    np.savez_compressed(dst / "processed" / "counterfactual" / "cflow_v3.npz", **cf)
    print(f"[base] cflow items {n}; replayed proprioception vs v3, max abs diff: max {max(replay):.4f} "
          f"median {np.median(replay):.4f}")

    ws = dict(np.load(src / "processed" / "counterfactual" / "wsp_v3.npz", allow_pickle=True))
    m = len(ws["task"])
    wo = {k: np.zeros((m, H, fd), np.float32) for k in ("x_orig", "x_small", "x_large")}
    wctx = np.zeros((m, L, fd), np.float32)
    for task, it in I.items():
        for j, r in enumerate(it["ws_row"]):
            r = int(r)
            for k in ("orig", "small", "large"):
                wo[f"x_{k}"][r] = it[f"ws_img_{k}"][j]
    for r in range(m):
        p = int(ws["pos"][r])
        wctx[r] = F[str(ws["task"][r])][str(ws["demo"][r])][p - L + 1:p + 1]
    ws.update(wo, ctx_x=wctx)
    np.savez_compressed(dst / "processed" / "counterfactual" / "wsp_v3.npz", **ws)

    mt = dict(np.load(src / "processed" / "counterfactual" / "matched_v3.npz", allow_pickle=True))
    q_n = len(mt["row"])
    xt, xm = np.zeros((q_n, H, fd), np.float32), np.zeros((q_n, H, fd), np.float32)
    e = np.zeros((q_n, H, fd), np.float32)
    got = 0
    for task, it in I.items():
        for j, q in enumerate(it.get("pr_q", [])):
            q = int(q)
            xt[q], xm[q] = it["pr_img_true"][j], it["pr_img_match"][j]
            got += 1
    for q in range(q_n):
        r = int(mt["row"][q])
        e[q] = estimate(F, nz.obs_std, str(cf["task"][r]), str(cf["demo"][r]), int(cf["pos"][r]),
                        str(mt["src_demo"][q]), int(mt["src_pos"][q]))
    assert got == q_n, (got, q_n)
    mt.update(x_true=xt, x_match=xm, e=e)
    np.savez_compressed(dst / "processed" / "counterfactual" / "matched_v3pp.npz", **mt)
    print(f"[base] wrote {dst}: {len(obs)} demonstrations, feature dim {fd}, wsp items {m}, pairs {q_n}")


def pairs(src, dst, k):
    from analysis.matched_probe import build_pairs
    pr = build_pairs(tag="v3", k=k, dr=dst)
    lo = np.load(src / "processed" / "latents" / "libero_lowdim.npz", allow_pickle=True)
    acts = {str(i): a for i, a in zip(lo["ids"], lo["act"])}
    raw = np.stack([acts[f"{t}:{d}"][p:p + H] for t, d, p in zip(pr["task"], pr["src_demo"], pr["src_pos"])])
    np.savez(dst / "pairs_img.npz", a_match_raw=raw.astype(np.float32), k=k,
             **{kk: (np.array(v, dtype=object) if kk in ("task", "src_demo") else v) for kk, v in pr.items()})
    print(f"[pairs] {len(pr['row'])} pairs, median feature gap {np.median(pr['gap']):.3f}")


def matched(feat, dst):
    pr = dict(np.load(dst / "pairs_img.npz", allow_pickle=True))
    P = load_feats(feat, "pairs")
    n = len(pr["row"])
    fd = next(iter(P.values()))["pr_img_true"].shape[-1]
    xt, xm = np.zeros((n, H, fd), np.float32), np.zeros((n, H, fd), np.float32)
    got = 0
    for task, it in P.items():
        for j, q in enumerate(it["pr_q"]):
            xt[int(q)], xm[int(q)] = it["pr_img_true"][j], it["pr_img_match"][j]
            got += 1
    assert got == n, (got, n)
    np.savez_compressed(dst / "processed" / "counterfactual" / "matched_v3.npz", x_true=xt, x_match=xm,
                        a_match_raw=pr["a_match_raw"], row=pr["row"], src_pos=pr["src_pos"], gap=pr["gap"],
                        da=pr["da"], e=pr["e"], a_j=pr["a_j"], k=pr["k"], task=pr["task"], src_demo=pr["src_demo"])
    print(f"[matched] wrote {n} image matched pairs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["base", "pairs", "matched"])
    ap.add_argument("--src", required=True)
    ap.add_argument("--feat", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    if args.mode == "base":
        base(src, Path(args.feat), dst)
    elif args.mode == "pairs":
        pairs(src, dst, args.k)
    else:
        matched(Path(args.feat), dst)


if __name__ == "__main__":
    main()
