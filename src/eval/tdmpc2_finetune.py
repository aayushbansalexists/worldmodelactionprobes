from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "disabled")

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.tdmpc2_probe import TDMPC2, collect

WEIGHTS = (0.001, 0.01, 0.1)
TRAIN_EPS, TEST_EPS = (0, 1, 2, 3), (4, 5)


def pair_bank(logs, enc, device, min_gap=0.5):
    O = [logs[e][0] for e in TRAIN_EPS]
    A = [logs[e][1] for e in TRAIN_EPS]
    sd = np.concatenate([o for o in O]).std(0) + 1e-6
    X = np.concatenate([o[:-1] / sd for o in O])
    act = np.concatenate(A).astype(np.float32)
    ep = np.concatenate([np.full(len(a), k) for k, a in enumerate(A)])
    t = lambda x: torch.from_numpy(np.asarray(x, np.float32)).to(device)
    z0 = torch.cat([enc(t(o[:-1])) for o in O])
    z1 = torch.cat([enc(t(o[1:])) for o in O])
    d = torch.cdist(t(X), t(X))
    d[torch.from_numpy(ep[:, None] == ep[None, :]).to(device)] = float("inf")
    d[torch.cdist(t(act), t(act)) < min_gap] = float("inf")
    j = d.argmin(1)
    ok = torch.isfinite(d.gather(1, j[:, None]))[:, 0]
    i = torch.arange(len(X), device=device)[ok]
    j = j[ok]
    e = (z1[j] - z0[j]) - (z1[i] - z0[i])
    return {"z0": z0, "z1": z1, "a": t(act), "pi": i, "pj": j, "e": e, "gap": d[i, j]}


def finetune(model, bank, w, steps, seed, device, lr=1e-4, bs=256):
    dyn = copy.deepcopy(model.dyn).to(device)
    for p in dyn.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(dyn.parameters(), lr=lr)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n, m = len(bank["z0"]), len(bank["pi"])
    with torch.enable_grad():
        for _ in range(steps):
            b = torch.randint(0, n, (bs,), generator=gen).to(device)
            c = torch.randint(0, m, (bs,), generator=gen).to(device)
            loss = F.mse_loss(dyn(torch.cat([bank["z0"][b], bank["a"][b]], -1)), bank["z1"][b])
            if w > 0:
                zi, ai, aj = bank["z0"][bank["pi"][c]], bank["a"][bank["pi"][c]], bank["a"][bank["pj"][c]]
                r = dyn(torch.cat([zi, aj], -1)) - dyn(torch.cat([zi, ai], -1))
                e = bank["e"][c]
                cos = (r * e).sum(-1) / (r.norm(dim=-1) * e.norm(dim=-1) + 1e-8)
                loss = loss + w * (1 - cos).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    dyn.eval()
    return dyn


@torch.no_grad()
def evaluate(dyn, model, data, logs, device):
    t = lambda x: torch.from_numpy(np.asarray(x, np.float32)).to(device)
    enc = lambda x: model.encode(t(x).reshape(-1, x.shape[-1])).reshape(*x.shape[:-1], -1)

    def roll(z, A):
        out = []
        for k in range(A.shape[1]):
            z = dyn(torch.cat([z, A[:, k]], -1))
            out.append(z)
        return torch.stack(out, 1)

    keep = np.isin(data["ep"], TEST_EPS)
    z0 = enc(data["o0"][keep])
    dp = (roll(z0, t(data["a_pert"][keep])) - roll(z0, t(data["a"][keep]))).reshape(int(keep.sum()), -1)
    dt = (enc(data["x_pert"][keep]) - enc(data["x_ref"][keep])).reshape(int(keep.sum()), -1)
    ok = dt.norm(dim=1) > 1e-9
    cos = (dp * dt).sum(1)[ok] / (dp.norm(dim=1)[ok] * dt.norm(dim=1)[ok] + 1e-12)
    O = np.concatenate([logs[e][0][:-1] for e in TEST_EPS])
    O1 = np.concatenate([logs[e][0][1:] for e in TEST_EPS])
    A = np.concatenate([logs[e][1] for e in TEST_EPS])
    mse = F.mse_loss(dyn(torch.cat([enc(O), t(A)], -1)), enc(O1)).item()
    return {"direction": float(cos.mean().item()), "consistency_mse": float(mse), "n_items": int(ok.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ck = Path(os.environ.get("DATA_ROOT", ROOT / "data")) / "raw" / "tdmpc2" / "dmcontrol"
    tasks = args.tasks or sorted({p.stem.rsplit("-", 1)[0] for p in ck.glob("*.pt")})
    out_path = Path(args.out or ROOT / "results" / "tables" / "tdmpc2_finetune.json")
    res = json.load(open(out_path)) if out_path.exists() else {}
    for task in tasks:
        if task in res:
            continue
        t0 = time.time()
        model = TDMPC2(ck / f"{task}-1.pt").to(args.device).eval()
        data = collect(task, model, args.device)
        logs = data["logs"]
        bank = pair_bank(logs, model.encode, args.device)
        r = {"released": evaluate(model.dyn, model, data, logs, args.device),
             "consistency": evaluate(finetune(model, bank, 0.0, args.steps, 0, args.device), model, data, logs, args.device)}
        for w in WEIGHTS:
            r[f"matched_{w}"] = evaluate(finetune(model, bank, w, args.steps, 0, args.device), model, data, logs, args.device)
        r["meta"] = {"n_pairs": int(len(bank["pi"])), "n_transitions": int(len(bank["z0"])),
                     "pair_gap_median": float(bank["gap"].median().item())}
        res[task] = r
        print(f"  {task:26s} released D={r['released']['direction']:+.3f} consistency D={r['consistency']['direction']:+.3f} "
              + " ".join(f"w{w}={r[f'matched_{w}']['direction']:+.3f}" for w in WEIGHTS)
              + f" | mse cons={r['consistency']['consistency_mse']:.2e} w0.01={r['matched_0.01']['consistency_mse']:.2e} "
              f"pairs={r['meta']['n_pairs']} ({time.time() - t0:.0f}s)", flush=True)
        json.dump(res, open(out_path, "w"), indent=1)
        del model, data, bank
        torch.cuda.empty_cache()
    print("[finetune] wrote", out_path)


if __name__ == "__main__":
    main()
