from __future__ import annotations

import eval.libero_compat

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

from eval.counterfactual_lowdim import branch, demo_grip_state, make_env, obs_vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.probe_validity import TRANSFORMS, make_T

EEF = slice(15, 18)
GOAL_DIST, STEPS, H, N_SAMP, N_ELITE, ITERS = 0.04, 20, 8, 256, 32, 4
TAG = os.environ.get("GT_TAG", "v3")


@torch.no_grad()
def plan_batch(model, ctx_x, ctx_a, nz, goal, grip, Ts, rngs, device, mus):
    E = ctx_x.shape[0]
    sd = np.full((E, H, 3), 0.5)
    base = np.zeros((E, 1, H, 7), np.float32)
    base[..., 6] = grip
    wo = torch.from_numpy(nz.norm_obs(ctx_x)).float().to(device)
    wa = torch.from_numpy(nz.norm_act(ctx_a)).float().to(device)
    Tt = torch.from_numpy(np.stack(Ts)).float().to(device)

    def rollout(seqs):
        n = seqs.shape[1]
        o = wo[:, None].expand(E, n, *wo.shape[1:]).reshape(E * n, *wo.shape[1:])
        a = wa[:, None].expand(E, n, *wa.shape[1:]).reshape(E * n, *wa.shape[1:]).clone()
        fa = torch.from_numpy(nz.norm_act(seqs.reshape(E * n, H, 7))).float().to(device)
        a[:, -1] = fa[:, 0]
        out = []
        for k in range(H):
            p = model(o, a)["pred"][:, -1]
            out.append(p)
            o = torch.cat([o[:, 1:], p[:, None]], 1)
            if k + 1 < H:
                a = torch.cat([a[:, 1:], fa[:, k + 1:k + 2]], 1)
        return torch.stack(out, 1).reshape(E, n, H, -1)

    P0 = rollout(base)
    mean_o = torch.from_numpy(nz.obs_mean).float().to(device)
    std_o = torch.from_numpy(nz.obs_std).float().to(device)
    g = torch.from_numpy(np.asarray(goal, np.float32)).to(device)
    mu = mus.copy()
    for _ in range(ITERS):
        seqs = np.repeat(base, N_SAMP, 1)
        noise = np.stack([r.standard_normal((N_SAMP, H, 3)) for r in rngs])
        seqs[..., :3] = np.clip(mu[:, None] + sd[:, None] * noise, -1, 1)
        P = rollout(seqs.astype(np.float32))
        P = P0 + torch.einsum("enho,epo->enhp", P - P0, Tt)
        eef = (P * std_o + mean_o)[..., EEF]
        cost = (eef - g).norm(dim=-1).mean(-1).cpu().numpy()
        tie = np.stack([r.standard_normal(N_SAMP) for r in rngs]) * 1e-9
        el = np.argsort(cost + tie, axis=1)[:, :N_ELITE]
        elite = np.take_along_axis(seqs[..., :3], el[:, :, None, None], 1)
        mu, sd = elite.mean(1), elite.std(1) + 0.05
    act = base[:, 0, 0].copy()
    act[:, :3] = mu[:, 0]
    return act, mu


def run_task(fp, rows, cf, runs, nz, device, seed):
    from libero.libero import get_libero_path
    from eval.counterfactual_eval import load_model
    with h5py.File(fp, "r") as h:
        bddl = dict(h["data"].attrs)["bddl_file_name"]
    bddl_path = os.path.join(get_libero_path("bddl_files"), *bddl.split("/")[-2:])
    envs = [make_env(bddl_path) for _ in TRANSFORMS]
    for env in envs:
        env.reset()
    Ts = [make_T(t, 21).astype(np.float32) for t in TRANSFORMS]
    out = []
    with h5py.File(fp, "r") as h:
        for row in rows:
            demo, p = str(cf["demo"][row]), int(cf["pos"][row])
            g = h["data"][demo]
            states, acts = g["states"][:], g["actions"][:].astype(np.float32)
            u = np.random.default_rng(seed * 1000 + row).standard_normal(3)
            goal_rel = GOAL_DIST * u / np.linalg.norm(u)
            gstate = demo_grip_state(acts, p, envs[0].env.robots[0].gripper)
            for env in envs:
                branch(env, env.env.robots[0], states[p], acts[p:p + 1], gstate)
            grip = float(acts[p, 6])
            for rid in runs:
                model = load_model(rid, device)[0]
                rngs = [np.random.default_rng(seed * 7919 + row * 13 + i) for i in range(len(envs))]
                xs = []
                for env in envs:
                    env.sim.reset()
                    env.set_init_state(states[p])
                    env.sim.forward()
                    ctl = env.env.robots[0].controller
                    ctl.update(force=True)
                    ctl.reset_goal()
                    env.env.robots[0].gripper.current_action = gstate.copy()
                    o, _, _, _ = env.step(acts[p])
                    xs.append(obs_vector(o))
                goal = xs[0][EEF] + goal_rel
                ctx_x = np.stack([np.concatenate([cf["ctx_x"][row][1:], x[None]], 0) for x in xs])
                ctx_a = np.stack([np.concatenate([cf["ctx_a"][row][1:], np.zeros((1, 7), np.float32)], 0)
                                  for _ in xs])
                d0 = [float(np.linalg.norm(x[EEF] - goal)) for x in xs]
                mus = np.zeros((len(envs), H, 3))
                for _ in range(STEPS):
                    a, mus = plan_batch(model, ctx_x, ctx_a, nz, goal, grip, Ts, rngs, device, mus)
                    mus = np.concatenate([mus[:, 1:], mus[:, -1:]], 1)
                    for e, env in enumerate(envs):
                        o, _, _, _ = env.step(a[e])
                        xs[e] = obs_vector(o)
                    ctx_a[:, -1] = a
                    ctx_x = np.concatenate([ctx_x[:, 1:], np.stack(xs)[:, None]], 1)
                    ctx_a = np.concatenate([ctx_a[:, 1:], np.zeros((len(envs), 1, 7), np.float32)], 1)
                for e, t in enumerate(TRANSFORMS):
                    out.append({"run": rid, "transform": t, "row": int(row), "d0": d0[e],
                                "d_final": float(np.linalg.norm(xs[e][EEF] - goal))})
                print(f"  row {row} {rid}: " + " ".join(
                    f"{t}={o_['d_final'] * 100:.1f}" for t, o_ in zip(TRANSFORMS, out[-len(envs):])),
                    flush=True)
                del model
            print(f"[closed loop] {os.path.basename(fp)[:36]:36s} row {row} done", flush=True)
    for env in envs:
        env.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=[f"cf_{m}_causal_s{s}" for m in ("add", "agm", "delta")
                                                   for s in range(5)])
    ap.add_argument("--per-task", type=int, default=1)
    ap.add_argument("--tasks", type=int, nargs="*", default=None, help="indices into the 30 tasks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    from data.dataset import Normalizer
    dr = Path(os.environ.get("DATA_ROOT", "data"))
    nz = Normalizer.load(dr / "processed" / "latents" / "normalizer.npz")
    cf = np.load(dr / "processed" / "counterfactual" / f"cflow_{TAG}.npz", allow_pickle=True)
    tasks = sorted({str(t) for t in cf["task"]})
    pick = args.tasks if args.tasks is not None else range(len(tasks))
    out = Path(args.out or Path(os.environ.get("RESULTS_ROOT", "results")) / "tables" / f"closed_loop_{TAG}.json")
    res = json.load(open(out)) if out.exists() else []
    done = {e["row"] for e in res}
    for ti in pick:
        task = tasks[ti]
        rows = sorted({int(i) for i in np.where(cf["task"] == task)[0] if cf["sigma"][i] == 0.25})
        rows = rows[:: max(1, len(rows) // args.per_task)][: args.per_task]
        rows = [r for r in rows if r not in done]
        if not rows:
            continue
        res += run_task(dr / "raw" / "libero" / task, rows, cf, args.runs, nz, "cuda", args.seed)
        json.dump(res, open(out, "w"), indent=1)
    print("[closed loop] wrote", out, len(res), "episodes")


if __name__ == "__main__":
    main()
