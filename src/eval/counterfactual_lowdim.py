from __future__ import annotations

import eval.libero_compat

import argparse
import glob
import json
import os
from pathlib import Path

import h5py
import numpy as np
import robosuite.utils.transform_utils as T


def _data_root():
    return Path(os.environ.get("DATA_ROOT", "data"))


def obs_vector(o):
    ee_pos = np.asarray(o["robot0_eef_pos"], dtype=np.float32)
    ee_ori = np.asarray(T.quat2axisangle(o["robot0_eef_quat"]), dtype=np.float32)
    return np.concatenate([
        np.asarray(o["robot0_joint_pos"], dtype=np.float32),
        ee_pos, ee_ori,
        np.asarray(o["robot0_gripper_qpos"], dtype=np.float32),
        ee_pos,
        ee_ori,
    ]).astype(np.float32)


def make_env(bddl_path):
    from libero.libero.envs.env_wrapper import ControlEnv
    return ControlEnv(bddl_file_name=bddl_path, has_renderer=False,
                      has_offscreen_renderer=False, use_camera_obs=False)


def demo_grip_state(acts, p, gripper):
    assert type(gripper).__name__ == "PandaGripper", type(gripper).__name__
    ca = np.zeros(gripper.dof)
    for a in acts[:p]:
        ca = np.clip(ca + np.array([-1.0, 1.0]) * gripper.speed * np.sign(a[-1]), -1.0, 1.0)
    return ca


def branch(env, robot, state, actions, grip=None):
    env.sim.reset()
    env.set_init_state(state)
    env.sim.forward()
    robot.controller.update(force=True)
    robot.controller.reset_goal()
    if grip is not None:
        robot.gripper.current_action = np.array(grip, dtype=np.float64)
    out = []
    for a in actions:
        o, _, _, _ = env.step(a)
        out.append(obs_vector(o))
    return np.stack(out)


def restored_obs(env, robot, state):
    env.sim.reset()
    env.set_init_state(state)
    env.sim.forward()
    return obs_vector(env.env._get_observations())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--tasks", type=int, default=30)
    ap.add_argument("--demos-per-task", type=int, default=3)
    ap.add_argument("--positions", type=int, default=4)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    ap.add_argument("--tag", default="main")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify", action="store_true",
                    help="check the rebuilt observation against the recorded one and exit")
    args = ap.parse_args()

    from libero.libero import get_libero_path
    bddl_base = get_libero_path("bddl_files")
    splits = json.load(open(_data_root() / "splits" / "libero_splits.json"))
    test_ids = set(splits["test"])
    rng = np.random.default_rng(args.seed)
    L, H = args.seq_len, args.horizon

    files = sorted(glob.glob(str(_data_root() / "raw" / "libero" / "*.hdf5")))[: args.tasks]
    rec = {k: [] for k in ["ctx_x", "ctx_a", "fut_a_true", "fut_a_pert",
                           "x_true", "x_pert", "x_noise", "sigma", "task", "demo", "pos"]}
    verify_err = []

    for fp in files:
        task = os.path.basename(fp)
        with h5py.File(fp, "r") as h:
            bddl = dict(h["data"].attrs)["bddl_file_name"]
            demo_keys = sorted(h["data"].keys())
        bddl_path = os.path.join(bddl_base, *bddl.split("/")[-2:])
        if not os.path.exists(bddl_path):
            print(f"[cf] SKIP {task}: bddl missing", flush=True)
            continue
        picked = [d for d in demo_keys if f"{task}:{d}" in test_ids][: args.demos_per_task]
        if not picked:
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
            T_ = states.shape[0]
            if T_ < L + H + 2:
                continue
            if args.verify:
                for t in [5, T_ // 2, T_ - 5]:
                    got = restored_obs(env, robot, states[t])
                    verify_err.append(np.abs(got - rec_obs[t]).max())
                continue
            for p in np.linspace(L, T_ - H - 2, args.positions).astype(int):
                a_seq = acts[p:p + H]
                grip = demo_grip_state(acts, p, robot.gripper)
                branch(env, robot, states[p], a_seq, grip)
                x_true = branch(env, robot, states[p], a_seq, grip)
                x_noise = branch(env, robot, states[p], a_seq, grip)
                for sg in args.sigmas:
                    delta = rng.standard_normal(acts.shape[-1]).astype(np.float32) * sg
                    a_pert = np.clip(a_seq + delta, -1.0, 1.0)
                    x_pert = branch(env, robot, states[p], a_pert, grip)
                    rec["ctx_x"].append(rec_obs[p - L + 1:p + 1])
                    rec["ctx_a"].append(acts[p - L + 1:p + 1])
                    rec["fut_a_true"].append(a_seq)
                    rec["fut_a_pert"].append(a_pert)
                    rec["x_true"].append(x_true)
                    rec["x_pert"].append(x_pert)
                    rec["x_noise"].append(x_noise)
                    rec["sigma"].append(sg)
                    rec["task"].append(task); rec["demo"].append(dk); rec["pos"].append(int(p))
            print(f"[cf] {task[:36]:36s} {dk:9s} n={len(rec['sigma'])}", flush=True)
        env.close()

    if args.verify:
        v = np.array(verify_err)
        print(f"[verify] rebuilt vs recorded observation: n={v.size} "
              f"max={v.max():.3e} mean={v.mean():.3e} median={np.median(v):.3e}")
        return

    out = _data_root() / "processed" / "counterfactual"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"cflow_{args.tag}.npz"
    np.savez_compressed(
        path,
        **{k: np.stack(rec[k]) for k in ["ctx_x", "ctx_a", "fut_a_true", "fut_a_pert",
                                         "x_true", "x_pert", "x_noise"]},
        sigma=np.array(rec["sigma"]), pos=np.array(rec["pos"]),
        task=np.array(rec["task"], dtype=object), demo=np.array(rec["demo"], dtype=object),
        horizon=H, seq_len=L,
    )
    xt, xn, xp = np.stack(rec["x_true"]), np.stack(rec["x_noise"]), np.stack(rec["x_pert"])
    floor = np.linalg.norm(xn - xt, axis=(1, 2)).mean()
    signal = np.linalg.norm(xp - xt, axis=(1, 2)).mean()
    print(f"[cf] wrote {path}  n={len(rec['sigma'])}")
    print(f"[cf] noise floor ||x_noise - x_true|| = {floor:.5f}")
    print(f"[cf] signal      ||x_pert  - x_true|| = {signal:.5f}   SNR = {signal/max(floor,1e-12):.1f}")


if __name__ == "__main__":
    main()
