from __future__ import annotations

import eval.libero_compat

import argparse
import glob
import os
import sys
from pathlib import Path

import h5py
import numpy as np

from eval.counterfactual_lowdim import branch, demo_grip_state, make_env

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.matched_probe import H, build_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--tag", default="v2")
    args = ap.parse_args()
    dr = Path(os.environ.get("DATA_ROOT", "data"))
    from libero.libero import get_libero_path
    bddl_base = get_libero_path("bddl_files")

    pr = build_pairs(tag=args.tag, k=args.k)
    cf = np.load(dr / "processed" / "counterfactual" / f"cflow_{args.tag}.npz", allow_pickle=True)
    n = len(pr["row"])
    x_true = np.zeros((n, H, 21), np.float32)
    x_match = np.zeros((n, H, 21), np.float32)
    a_match_raw = np.zeros((n, H, 7), np.float32)

    by_task = {}
    for q in range(n):
        by_task.setdefault(pr["task"][q], []).append(q)
    for task, qs in sorted(by_task.items()):
        fp = dr / "raw" / "libero" / task
        with h5py.File(fp, "r") as h:
            bddl = dict(h["data"].attrs)["bddl_file_name"]
        env = make_env(os.path.join(bddl_base, *bddl.split("/")[-2:]))
        env.reset()
        robot = env.env.robots[0]
        done_rows = {}
        with h5py.File(fp, "r") as h:
            for q in qs:
                r = int(pr["row"][q])
                demo_i, p_i = str(cf["demo"][r]), int(cf["pos"][r])
                states = h["data"][demo_i]["states"][:]
                a_i = h["data"][demo_i]["actions"][p_i:p_i + H].astype(np.float32)
                a_j = h["data"][pr["src_demo"][q]]["actions"][
                    int(pr["src_pos"][q]):int(pr["src_pos"][q]) + H].astype(np.float32)
                grip = demo_grip_state(h["data"][demo_i]["actions"][:], p_i, robot.gripper)
                if r not in done_rows:
                    branch(env, robot, states[p_i], a_i, grip)
                    done_rows[r] = branch(env, robot, states[p_i], a_i, grip)
                x_true[q] = done_rows[r]
                x_match[q] = branch(env, robot, states[p_i], a_j, grip)
                a_match_raw[q] = a_j
        env.close()
        print(f"[matched] {task[:40]:40s} pairs={len(qs)}", flush=True)

    out = dr / "processed" / "counterfactual" / f"matched_{args.tag}.npz"
    np.savez_compressed(out, x_true=x_true, x_match=x_match, a_match_raw=a_match_raw,
                        row=pr["row"], src_pos=pr["src_pos"], gap=pr["gap"], da=pr["da"],
                        e=pr["e"], a_j=pr["a_j"], k=args.k,
                        task=np.array(pr["task"], dtype=object),
                        src_demo=np.array(pr["src_demo"], dtype=object))
    print(f"[matched] wrote {out} n={n}")


if __name__ == "__main__":
    main()
