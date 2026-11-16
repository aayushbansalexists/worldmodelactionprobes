from __future__ import annotations

import eval.libero_compat

import argparse
import glob
import os
from pathlib import Path

import h5py
import numpy as np

from eval.counterfactual_lowdim import demo_grip_state, obs_vector

VIEWS = ("agentview_image", "robot0_eye_in_hand_image")


def make_cam_env(bddl_path):
    from libero.libero.envs.env_wrapper import ControlEnv
    return ControlEnv(bddl_file_name=bddl_path, has_renderer=False, has_offscreen_renderer=True,
                      use_camera_obs=True, camera_names=["agentview", "robot0_eye_in_hand"],
                      camera_heights=128, camera_widths=128)


def frames(o):
    return np.stack([np.asarray(o[v], dtype=np.uint8) for v in VIEWS])


def render_state(env, state):
    env.sim.reset()
    env.set_init_state(state)
    env.sim.forward()
    return frames(env.env._get_observations(force_update=True))


def branch_img(env, robot, state, actions, grip):
    env.sim.reset()
    env.set_init_state(state)
    env.sim.forward()
    robot.controller.update(force=True)
    robot.controller.reset_goal()
    robot.gripper.current_action = np.array(grip, dtype=np.float64)
    env.env.timestep, env.env.done = 0, False
    imgs, xs = [], []
    for a in actions:
        o, _, _, _ = env.step(a)
        imgs.append(frames(o))
        xs.append(obs_vector(o))
    return np.stack(imgs), np.stack(xs)


def task_files(dr, idx):
    files = sorted(glob.glob(str(dr / "raw" / "libero" / "*.hdf5")))
    return [files[i] for i in idx] if idx else files


def bddl_of(fp):
    from libero.libero import get_libero_path
    with h5py.File(fp, "r") as h:
        bddl = dict(h["data"].attrs)["bddl_file_name"]
    return os.path.join(get_libero_path("bddl_files"), *bddl.split("/")[-2:])


def run_demos(fp, out):
    task = os.path.basename(fp)
    env = make_cam_env(bddl_of(fp))
    env.reset()
    res = {}
    with h5py.File(fp, "r") as h:
        for dk in sorted(h["data"].keys()):
            states = h["data"][dk]["states"][:]
            res[dk] = np.stack([render_state(env, s) for s in states])
    env.close()
    np.savez(out / f"demos_{task}.npz", **res)
    print(f"[render demos] {task[:48]:48s} demos={len(res)} frames={sum(len(v) for v in res.values())}", flush=True)


def run_items(fp, out, dr, tag, pairs, only_pairs=False):
    task = os.path.basename(fp)
    cf = np.load(dr / "processed" / "counterfactual" / f"cflow_{tag}.npz", allow_pickle=True)
    ws = np.load(dr / "processed" / "counterfactual" / f"wsp_{tag}.npz", allow_pickle=True)
    env = make_cam_env(bddl_of(fp))
    env.reset()
    robot = env.env.robots[0]
    rec = {}
    with h5py.File(fp, "r") as h:
        rows = [] if only_pairs else [i for i in range(len(cf["task"])) if str(cf["task"][i]) == task]
        pos_rows = {}
        for i in rows:
            pos_rows.setdefault((str(cf["demo"][i]), int(cf["pos"][i])), []).append(i)
        for (dk, p), its in sorted(pos_rows.items()):
            g = h["data"][dk]
            states, acts = g["states"][:], g["actions"][:].astype(np.float32)
            a_seq = acts[p:p + 8]
            grip = demo_grip_state(acts, p, robot.gripper)
            branch_img(env, robot, states[p], a_seq, grip)
            it, xt = branch_img(env, robot, states[p], a_seq, grip)
            inn, xn = branch_img(env, robot, states[p], a_seq, grip)
            for i in its:
                ip, xp = branch_img(env, robot, states[p], cf["fut_a_pert"][i], grip)
                for k, v in (("img_true", it), ("img_noise", inn), ("img_pert", ip),
                             ("x_true", xt), ("x_noise", xn), ("x_pert", xp)):
                    rec.setdefault(f"cf_{k}", []).append(v)
                rec.setdefault("cf_row", []).append(i)
        for i in ([] if only_pairs else [i for i in range(len(ws["task"])) if str(ws["task"][i]) == task]):
            dk, p = str(ws["demo"][i]), int(ws["pos"][i])
            g = h["data"][dk]
            states, acts = g["states"][:], g["actions"][:].astype(np.float32)
            grip = demo_grip_state(acts, p, robot.gripper)
            branch_img(env, robot, states[p], ws["a_orig"][i], grip)
            for k in ("orig", "small", "large"):
                im, x = branch_img(env, robot, states[p], ws[f"a_{k}"][i], grip)
                rec.setdefault(f"ws_img_{k}", []).append(im)
                rec.setdefault(f"ws_x_{k}", []).append(x)
            rec.setdefault("ws_row", []).append(i)
        if pairs is not None:
            done = {}
            for q in [q for q in range(len(pairs["row"])) if str(cf["task"][int(pairs["row"][q])]) == task]:
                r = int(pairs["row"][q])
                dk, p = str(cf["demo"][r]), int(cf["pos"][r])
                g = h["data"][dk]
                states, acts = g["states"][:], g["actions"][:].astype(np.float32)
                grip = demo_grip_state(acts, p, robot.gripper)
                if r not in done:
                    branch_img(env, robot, states[p], acts[p:p + 8], grip)
                    done[r] = branch_img(env, robot, states[p], acts[p:p + 8], grip)
                im, x = branch_img(env, robot, states[p], pairs["a_match_raw"][q], grip)
                rec.setdefault("pr_img_true", []).append(done[r][0])
                rec.setdefault("pr_x_true", []).append(done[r][1])
                rec.setdefault("pr_img_match", []).append(im)
                rec.setdefault("pr_x_match", []).append(x)
                rec.setdefault("pr_q", []).append(q)
    env.close()
    np.savez(out / f"{'pairs' if only_pairs else 'items'}_{task}.npz", **{k: np.stack(v) if not isinstance(v[0], int) else np.array(v)
                                           for k, v in rec.items()})
    print(f"[render items] {task[:48]:48s} cf={len(rec.get('cf_row', []))} ws={len(rec.get('ws_row', []))} "
          f"pairs={len(rec.get('pr_q', []))}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["demos", "items"])
    ap.add_argument("--tasks", type=int, nargs="*", default=None)
    ap.add_argument("--tag", default="v3")
    ap.add_argument("--pairs", default=None, help="npz with row and a_match_raw, as matched_<tag>.npz")
    ap.add_argument("--only-pairs", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dr = Path(os.environ.get("DATA_ROOT", "data"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs = np.load(args.pairs, allow_pickle=True) if args.pairs else None
    for fp in task_files(dr, args.tasks):
        if args.mode == "demos":
            run_demos(fp, out)
        else:
            run_items(fp, out, dr, args.tag, pairs, args.only_pairs)


if __name__ == "__main__":
    main()
