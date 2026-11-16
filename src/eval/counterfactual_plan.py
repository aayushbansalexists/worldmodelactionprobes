from __future__ import annotations

import eval.libero_compat

import argparse
import os
from pathlib import Path

import h5py
import numpy as np

from eval.counterfactual_lowdim import branch, demo_grip_state, make_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v3")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dr = Path(os.environ.get("DATA_ROOT", "data"))
    from libero.libero import get_libero_path
    cf = np.load(dr / "processed" / "counterfactual" / f"cflow_{args.tag}.npz", allow_pickle=True)
    H = int(cf["horizon"])
    keys = [(str(t), str(d), int(p)) for t, d, p in zip(cf["task"], cf["demo"], cf["pos"])]
    first = {}
    for i, k in enumerate(keys):
        first.setdefault(k, i)
    rows = sorted(first.values())
    rng = np.random.default_rng(args.seed)
    rec = {k: [] for k in ("x_ref", "x_q", "a_ref", "a_q", "row")}
    by_task = {}
    for r in rows:
        by_task.setdefault(keys[r][0], []).append(r)
    for task, rs in sorted(by_task.items()):
        fp = dr / "raw" / "libero" / task
        with h5py.File(fp, "r") as h:
            bddl = dict(h["data"].attrs)["bddl_file_name"]
        env = make_env(os.path.join(get_libero_path("bddl_files"), *bddl.split("/")[-2:]))
        env.reset()
        robot = env.env.robots[0]
        with h5py.File(fp, "r") as h:
            for r in rs:
                _, demo, p = keys[r]
                g = h["data"][demo]
                states, acts = g["states"][:], g["actions"][:].astype(np.float32)
                grip = demo_grip_state(acts, p, robot.gripper)
                a_ref = np.zeros((H, 7), np.float32)
                a_ref[:, 6] = acts[p, 6]
                a_q = np.repeat(a_ref[None], args.k, 0)
                a_q[:, :, :3] = np.clip(rng.standard_normal((args.k, H, 3)) * 0.5, -1, 1)
                env.env.timestep, env.env.done = 0, False
                branch(env, robot, states[p], acts[p:p + H], grip)
                env.env.timestep, env.env.done = 0, False
                rec["x_ref"].append(branch(env, robot, states[p], a_ref, grip))
                xq = []
                for q in range(args.k):
                    env.env.timestep, env.env.done = 0, False
                    xq.append(branch(env, robot, states[p], a_q[q], grip))
                rec["x_q"].append(np.stack(xq))
                rec["a_ref"].append(a_ref); rec["a_q"].append(a_q); rec["row"].append(r)
        env.close()
        print(f"[plan] {task[:48]:48s} positions={len(rs)}", flush=True)
    out = dr / "processed" / "counterfactual" / f"plan_{args.tag}.npz"
    np.savez_compressed(out, **{k: np.stack(v) for k, v in rec.items()})
    print(f"[plan] wrote {out}: {len(rec['row'])} positions x {args.k} queries")


if __name__ == "__main__":
    main()
