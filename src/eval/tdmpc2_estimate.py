from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "disabled")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.tdmpc2_probe import TDMPC2, Sim, make_env, obs_vec


def log_episodes(task, model, device, episodes=4, steps=500, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    env = make_env(task, seed)
    sim = Sim(env)
    adim = env.action_spec().shape[0]
    out = []
    for _ in range(episodes):
        ts = env.reset()
        o = obs_vec(ts.observation)
        O, A, S = [o], [], []
        for _ in range(steps):
            S.append(sim.get())
            z = model.encode(torch.from_numpy(o)[None].to(device))
            a = np.clip(model.act(z)[0].cpu().numpy() + rng.normal(0, noise, adim), -1, 1).astype(np.float32)
            A.append(a)
            o = sim.step(a)
            O.append(o)
        out.append((np.stack(O), np.stack(A), S))
    return sim, out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--pairs", type=int, default=300)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ck = Path(os.environ.get("DATA_ROOT", ROOT / "data")) / "raw" / "tdmpc2" / "dmcontrol"
    tasks = args.tasks or sorted({p.stem.rsplit("-", 1)[0] for p in ck.glob("*.pt")})
    out_path = Path(args.out or ROOT / "results" / "tables" / "tdmpc2_estimate.json")
    res = json.load(open(out_path)) if out_path.exists() else {}
    t = lambda x: torch.from_numpy(np.asarray(x, np.float32)).to(args.device)
    for task in tasks:
        if task in res:
            continue
        t0 = time.time()
        model = TDMPC2(ck / f"{task}-1.pt").to(args.device).eval()
        sim, logs = log_episodes(task, model, args.device)
        train = logs
        sd = np.concatenate([o for o, _, _ in train]).std(0) + 1e-6
        X = np.concatenate([o[:-1] / sd for o, _, _ in train])
        act = np.concatenate([a for _, a, _ in train]).astype(np.float32)
        ep = np.concatenate([np.full(len(a), k) for k, (_, a, _) in enumerate(train)])
        step = np.concatenate([np.arange(len(a)) for _, a, _ in train])
        d = torch.cdist(t(X), t(X))
        d[torch.from_numpy(ep[:, None] == ep[None, :]).to(args.device)] = float("inf")
        d[torch.cdist(t(act), t(act)) < 0.5] = float("inf")
        j = d.argmin(1).cpu().numpy()
        d[torch.from_numpy(ep != len(train) - 1).to(args.device)] = float("inf")
        ok = np.isfinite(d.min(1).values.cpu().numpy())
        rng = np.random.default_rng(1)
        idx = rng.choice(np.flatnonzero(ok), min(args.pairs, int(ok.sum())), replace=False)
        est, mod = [], []
        for i in idx:
            e_i, s_i, k = int(ep[i]), int(step[i]), int(j[i])
            e_j, s_j = int(ep[k]), int(step[k])
            O_i, A_i, S_i = train[e_i]
            O_j, A_j, _ = train[e_j]
            x_swap = sim.branch(S_i[s_i], A_j[s_j:s_j + 1])[0]
            h = lambda o: model.encode(t(o)[None])[0]
            z_p, z_p1, z_q, z_q1, z_sw = h(O_i[s_i]), h(O_i[s_i + 1]), h(O_j[s_j]), h(O_j[s_j + 1]), h(x_swap)
            r_star = z_sw - z_p1
            e = (z_q1 - z_q) - (z_p1 - z_p)
            r = model.next(z_p[None], t(A_j[s_j])[None])[0] - model.next(z_p[None], t(A_i[s_i])[None])[0]
            cs = lambda u, v: float((u @ v) / (u.norm() * v.norm() + 1e-12))
            if r_star.norm() > 1e-9:
                est.append(cs(e, r_star))
                mod.append(cs(r, r_star))
        res[task] = {"estimate_cos_median": float(np.median(est)), "estimate_cos_mean": float(np.mean(est)),
                     "model_cos_median": float(np.median(mod)), "model_cos_mean": float(np.mean(mod)), "n": len(est)}
        r = res[task]
        print(f"  {task:26s} estimate {r['estimate_cos_mean']:+.3f} (median {r['estimate_cos_median']:+.3f})  "
              f"model {r['model_cos_mean']:+.3f} (median {r['model_cos_median']:+.3f})  n={r['n']} ({time.time() - t0:.0f}s)",
              flush=True)
        json.dump(res, open(out_path, "w"), indent=1)
        del model
        torch.cuda.empty_cache()
    print("[estimate] wrote", out_path)


if __name__ == "__main__":
    main()
