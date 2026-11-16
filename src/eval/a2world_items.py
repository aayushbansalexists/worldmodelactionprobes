from __future__ import annotations

import eval.libero_compat

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np

H, HIST = 20, 20


def prepare(dr, feat, out, per_task, seed):
    cf = np.load(dr / "processed" / "counterfactual" / "cflow_v3.npz", allow_pickle=True)
    lo = np.load(dr / "processed" / "latents" / "libero_lowdim.npz", allow_pickle=True)
    held = set(json.load(open(dr / "splits" / "libero_splits.json"))["val"]) | \
        set(json.load(open(dr / "splits" / "libero_splits.json"))["test"])
    null = np.concatenate(list(lo["act"]), 0).mean(0).astype(np.float32)
    rng = np.random.default_rng(seed)
    rec = {k: [] for k in ("task", "demo", "pos", "p_demo", "p_pos", "queries", "seed",
                           "hist_agent", "hist_wrist", "hist_actions", "fut_agent", "fut_wrist",
                           "start_agent", "start_wrist", "p_fut_agent", "p_fut_wrist")}
    tasks = sorted({str(t) for t in cf["task"]})
    for task in tasks:
        F = dict(np.load(Path(feat) / f"demos_{task}.npz"))
        with h5py.File(dr / "raw" / "libero" / task, "r") as h:
            data = h["data"]
            cand = sorted({(str(d), int(p)) for t, d, p in zip(cf["task"], cf["demo"], cf["pos"])
                           if str(t) == task and int(p) >= HIST - 1 and int(p) + H + 1 <= len(data[str(d)]["actions"])})
            pick = [cand[i] for i in np.linspace(0, len(cand) - 1, per_task).round().astype(int)] if cand else []
            pool = [(d, q) for d in data.keys() if f"{task}:{d}" in held
                    for q in range(0, len(data[d]["actions"]) - H - 1)]
            for d, p in pick:
                g = data[d]
                act = g["actions"][:].astype(np.float32)
                s = F[d][p]
                pc = [(dj, q) for dj, q in pool if dj != d]
                dist = np.array([np.linalg.norm(F[dj][q] - s) for dj, q in pc])
                dj, q = pc[int(dist.argmin())]
                a_log = act[p:p + H]
                offs = [np.clip(a_log + rng.standard_normal(7).astype(np.float32) * 0.5, -1, 1) for _ in range(3)]
                a_j = data[dj]["actions"][q:q + H].astype(np.float32)
                rec["queries"].append(np.stack([a_log, *offs, a_j, np.repeat(null[None], H, 0)]))
                A, Wr = g["obs"]["agentview_rgb"], g["obs"]["eye_in_hand_rgb"]
                rec["hist_agent"].append(A[p - HIST + 1:p + 1]); rec["hist_wrist"].append(Wr[p - HIST + 1:p + 1])
                rec["hist_actions"].append(act[p - HIST + 1:p])
                rec["fut_agent"].append(A[p + 1:p + 1 + H]); rec["fut_wrist"].append(Wr[p + 1:p + 1 + H])
                gj = data[dj]["obs"]
                rec["start_agent"].append(gj["agentview_rgb"][q]); rec["start_wrist"].append(gj["eye_in_hand_rgb"][q])
                rec["p_fut_agent"].append(gj["agentview_rgb"][q + 1:q + 1 + H]); rec["p_fut_wrist"].append(gj["eye_in_hand_rgb"][q + 1:q + 1 + H])
                rec["task"].append(task); rec["demo"].append(d); rec["pos"].append(p)
                rec["p_demo"].append(dj); rec["p_pos"].append(q); rec["seed"].append(1000 + len(rec["seed"]))
        print(f"[a2world items] {task[:48]:48s} {len(pick)} positions", flush=True)
    np.savez(out, **{k: (np.array(v, dtype=object) if k in ("task", "demo", "p_demo") else np.stack(v) if k not in ("pos", "p_pos", "seed") else np.array(v))
                     for k, v in rec.items()})
    print(f"[a2world items] wrote {out}: {len(rec['seed'])} positions, queries {np.stack(rec['queries']).shape}")


def truth(dr, items, tasks, out):
    from eval.counterfactual_lowdim import demo_grip_state
    from eval.render_libero import bddl_of, branch_img, make_cam_env
    z = np.load(items, allow_pickle=True)
    names = sorted({str(t) for t in z["task"]})
    for task in [names[i] for i in tasks] if tasks else names:
        idx = [i for i in range(len(z["task"])) if str(z["task"][i]) == task]
        env = make_cam_env(bddl_of(dr / "raw" / "libero" / task))
        env.reset()
        robot = env.env.robots[0]
        imgs, xs = [], []
        with h5py.File(dr / "raw" / "libero" / task, "r") as h:
            for i in idx:
                g = h["data"][str(z["demo"][i])]
                p = int(z["pos"][i])
                states, acts = g["states"][:], g["actions"][:].astype(np.float32)
                grip = demo_grip_state(acts, p, robot.gripper)
                branch_img(env, robot, states[p], z["queries"][i, 0], grip)
                res = [branch_img(env, robot, states[p], z["queries"][i, k], grip) for k in range(5)]
                res.append(branch_img(env, robot, states[p], z["queries"][i, 0], grip))
                imgs.append(np.stack([r[0] for r in res])); xs.append(np.stack([r[1] for r in res]))
        env.close()
        np.savez(Path(out) / f"truth_{task}.npz", idx=np.array(idx), img=np.stack(imgs), x=np.stack(xs))
        print(f"[a2world truth] {task[:48]:48s} {len(idx)} positions", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prepare", "truth"])
    ap.add_argument("--feat", default=None)
    ap.add_argument("--items", default=None)
    ap.add_argument("--tasks", type=int, nargs="*", default=None)
    ap.add_argument("--per-task", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dr = Path(os.environ.get("DATA_ROOT", "data"))
    if args.mode == "prepare":
        prepare(dr, args.feat, args.out, args.per_task, args.seed)
    else:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        truth(dr, args.items, args.tasks, args.out)


if __name__ == "__main__":
    main()
