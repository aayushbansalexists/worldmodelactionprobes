from __future__ import annotations

import eval.libero_compat

import argparse
import glob
import json
import os
from pathlib import Path

import h5py
import numpy as np

from eval.counterfactual_lowdim import branch, demo_grip_state, make_env

LARGE, SMALL_RATIO = 0.14, 0.25
TRANSLATION_DIMS = (0, 1, 2)


def _data_root():
    return Path(os.environ.get("DATA_ROOT", "data"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--tasks", type=int, default=30)
    ap.add_argument("--demos-per-task", type=int, default=3)
    ap.add_argument("--positions", type=int, default=4)
    ap.add_argument("--large", type=float, default=LARGE)
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()

    from libero.libero import get_libero_path
    bddl_base = get_libero_path("bddl_files")
    test_ids = set(json.load(open(_data_root() / "splits" / "libero_splits.json"))["test"])
    L, H = args.seq_len, args.horizon
    small = args.large * SMALL_RATIO

    rec = {k: [] for k in ["ctx_x", "ctx_a", "a_orig", "a_small", "a_large",
                           "x_orig", "x_small", "x_large", "dim", "task", "demo", "pos"]}
    files = sorted(glob.glob(str(_data_root() / "raw" / "libero" / "*.hdf5")))[: args.tasks]
    for fp in files:
        task = os.path.basename(fp)
        with h5py.File(fp, "r") as h:
            bddl = dict(h["data"].attrs)["bddl_file_name"]
            demo_keys = sorted(h["data"].keys())
        bddl_path = os.path.join(bddl_base, *bddl.split("/")[-2:])
        picked = [d for d in demo_keys if f"{task}:{d}" in test_ids][: args.demos_per_task]
        if not os.path.exists(bddl_path) or not picked:
            continue
        env = make_env(bddl_path)
        env.reset()
        robot = env.env.robots[0]
        for dk in picked:
            with h5py.File(fp, "r") as h:
                g = h["data"][dk]
                states = g["states"][:]
                acts = g["actions"][:].astype(np.float32)
                rec_obs = np.concatenate(
                    [g["obs"][k][:] for k in
                     ["joint_states", "ee_states", "gripper_states", "ee_pos", "ee_ori"]],
                    axis=-1).astype(np.float32)
            if states.shape[0] < L + H + 2:
                continue
            for p in np.linspace(L, states.shape[0] - H - 2, args.positions).astype(int):
                a0 = acts[p:p + H]
                grip = demo_grip_state(acts, p, robot.gripper)
                branch(env, robot, states[p], a0, grip)
                x0 = branch(env, robot, states[p], a0, grip)
                for dim in TRANSLATION_DIMS:
                    a_s, a_l = a0.copy(), a0.copy()
                    a_s[:, dim] = np.clip(a_s[:, dim] + small, -1.0, 1.0)
                    a_l[:, dim] = np.clip(a_l[:, dim] + args.large, -1.0, 1.0)
                    rec["ctx_x"].append(rec_obs[p - L + 1:p + 1])
                    rec["ctx_a"].append(acts[p - L + 1:p + 1])
                    rec["a_orig"].append(a0); rec["a_small"].append(a_s); rec["a_large"].append(a_l)
                    rec["x_orig"].append(x0)
                    rec["x_small"].append(branch(env, robot, states[p], a_s, grip))
                    rec["x_large"].append(branch(env, robot, states[p], a_l, grip))
                    rec["dim"].append(dim); rec["task"].append(task)
                    rec["demo"].append(dk); rec["pos"].append(int(p))
            print(f"[wsp] {task[:36]:36s} {dk:9s} n={len(rec['dim'])}", flush=True)
        env.close()

    out = _data_root() / "processed" / "counterfactual"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"wsp_{args.tag}.npz"
    np.savez_compressed(
        path,
        **{k: np.stack(rec[k]) for k in ["ctx_x", "ctx_a", "a_orig", "a_small", "a_large",
                                         "x_orig", "x_small", "x_large"]},
        dim=np.array(rec["dim"]), pos=np.array(rec["pos"]),
        task=np.array(rec["task"], dtype=object), demo=np.array(rec["demo"], dtype=object),
        horizon=H, seq_len=L, large=args.large, small=small,
    )
    print(f"[wsp] wrote {path}  n={len(rec['dim'])}")


if __name__ == "__main__":
    main()
