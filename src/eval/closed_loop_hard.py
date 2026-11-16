from __future__ import annotations

import eval.libero_compat

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np

from eval.closed_loop import EEF, H, plan_batch
from eval.counterfactual_lowdim import branch, demo_grip_state, make_env, obs_vector


def run_task(fp, rows, cf, runs, nz, device, seed, goal_dist, steps, n_exec):
    from libero.libero import get_libero_path
    from eval.counterfactual_eval import load_model
    with h5py.File(fp, "r") as h:
        bddl = dict(h["data"].attrs)["bddl_file_name"]
    env = make_env(os.path.join(get_libero_path("bddl_files"), *bddl.split("/")[-2:]))
    env.reset()
    T = [np.eye(21, dtype=np.float32)]
    out = []
    with h5py.File(fp, "r") as h:
        for row in rows:
            demo, p = str(cf["demo"][row]), int(cf["pos"][row])
            g = h["data"][demo]
            states, acts = g["states"][:], g["actions"][:].astype(np.float32)
            u = np.random.default_rng(seed * 1000 + row).standard_normal(3)
            goal_rel = goal_dist * u / np.linalg.norm(u)
            gstate = demo_grip_state(acts, p, env.env.robots[0].gripper)
            branch(env, env.env.robots[0], states[p], acts[p:p + 1], gstate)
            grip = float(acts[p, 6])
            for rid in runs:
                model = load_model(rid, device)[0]
                rngs = [np.random.default_rng(seed * 7919 + row * 13)]
                env.sim.reset()
                env.set_init_state(states[p])
                env.sim.forward()
                ctl = env.env.robots[0].controller
                ctl.update(force=True)
                ctl.reset_goal()
                env.env.robots[0].gripper.current_action = gstate.copy()
                env.env.timestep, env.env.done = 0, False
                o, _, _, _ = env.step(acts[p])
                x = obs_vector(o)
                goal = x[EEF] + goal_rel
                ctx_x = np.concatenate([cf["ctx_x"][row][1:], x[None]], 0)[None]
                ctx_a = np.concatenate([cf["ctx_a"][row][1:], np.zeros((1, 7), np.float32)], 0)[None]
                d0 = float(np.linalg.norm(x[EEF] - goal))
                dmin, path, prev = d0, 0.0, x[EEF].copy()
                mus = np.zeros((1, H, 3))
                t = 0
                while t < steps:
                    _, mus = plan_batch(model, ctx_x, ctx_a, nz, goal, grip, T, rngs, device, mus)
                    for k in range(min(n_exec, steps - t)):
                        a = np.zeros(7, np.float32)
                        a[:3] = mus[0, k]
                        a[6] = grip
                        o, _, _, _ = env.step(a)
                        x = obs_vector(o)
                        ctx_a[0, -1] = a
                        ctx_x = np.concatenate([ctx_x[:, 1:], x[None, None]], 1)
                        ctx_a = np.concatenate([ctx_a[:, 1:], np.zeros((1, 1, 7), np.float32)], 1)
                        d = float(np.linalg.norm(x[EEF] - goal))
                        dmin = min(dmin, d)
                        path += float(np.linalg.norm(x[EEF] - prev))
                        prev = x[EEF].copy()
                        t += 1
                    k_done = min(n_exec, H)
                    mus = np.concatenate([mus[:, k_done:], np.repeat(mus[:, -1:], k_done, 1)], 1)
                out.append({"run": rid, "row": int(row), "d0": d0, "d_final": d, "d_min": dmin, "path": path})
                print(f"  row {row} {rid}: final {d * 100:.1f} cm, min {dmin * 100:.1f} cm", flush=True)
                del model
    env.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--per-task", type=int, default=1)
    ap.add_argument("--tasks", type=int, nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--goal-cm", type=float, default=10.0)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--exec", type=int, default=4, dest="n_exec")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert 1 <= args.n_exec <= H, f"--exec must be between 1 and the plan horizon {H}"
    from data.dataset import Normalizer
    dr = Path(os.environ.get("DATA_ROOT", "data"))
    tag = os.environ.get("GT_TAG", "v3")
    nz = Normalizer.load(dr / "processed" / "latents" / "normalizer.npz")
    cf = np.load(dr / "processed" / "counterfactual" / f"cflow_{tag}.npz", allow_pickle=True)
    tasks = sorted({str(t) for t in cf["task"]})
    pick = args.tasks if args.tasks is not None else range(len(tasks))
    out = Path(args.out)
    res = json.load(open(out)) if out.exists() else []
    done = {(e["run"], e["row"]) for e in res}
    for ti in pick:
        task = tasks[ti]
        rows = sorted({int(i) for i in np.where(cf["task"] == task)[0] if cf["sigma"][i] == 0.25})
        rows = rows[:: max(1, len(rows) // args.per_task)][: args.per_task]
        todo = [r for r in args.runs if any((r, row) not in done for row in rows)]
        if not todo:
            continue
        new = run_task(dr / "raw" / "libero" / task, rows, cf, todo, nz, "cuda", args.seed,
                       args.goal_cm / 100, args.steps, args.n_exec)
        res += [e for e in new if (e["run"], e["row"]) not in done]
        json.dump(res, open(out, "w"), indent=1)
        print(f"[closed loop hard] task {ti} done, {len(res)} episodes", flush=True)
    print("[closed loop hard] wrote", out, len(res), "episodes")


if __name__ == "__main__":
    main()
