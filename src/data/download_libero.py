from __future__ import annotations
import argparse
import glob
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def _project_root():
    return Path(__file__).resolve().parents[1]


def _data_root():
    return Path(os.environ.get("DATA_ROOT", str(_project_root() / "data")))


def _lowdim_obs_keys(h5_demo, want_keys):
    obs = h5_demo["obs"]
    have = set(obs.keys())
    chosen = []
    for k in want_keys:
        if k in have and obs[k].ndim == 2:
            chosen.append(k)
    if not chosen:
        for k in sorted(have):
            ds = obs[k]
            if "image" in k:
                continue
            try:
                if ds.ndim == 2 and ds.shape[-1] <= 64:
                    chosen.append(k)
            except Exception:
                continue
    return chosen


def extract_real(processed_path):
    raw = _data_root() / "raw" / "libero"
    files = sorted(glob.glob(str(raw / "*.hdf5")))
    if not files:
        return None, "no .hdf5 files under data/raw/libero/"
    try:
        import h5py
    except ImportError:
        return None, "h5py not installed"
    want_keys = ["joint_states", "ee_states", "gripper_states", "ee_pos", "ee_ori"]
    demos = []
    used_files = []
    for fp in files:
        suite = os.path.basename(fp).split("_")[0]
        task = os.path.basename(fp).replace("_demo.hdf5", "")
        try:
            with h5py.File(fp, "r") as h5:
                data_grp = h5["data"]
                for did in sorted(data_grp.keys()):
                    grp = data_grp[did]
                    if "actions" not in grp:
                        continue
                    act = grp["actions"][:]
                    keys = _lowdim_obs_keys(grp, want_keys)
                    if not keys:
                        continue
                    parts = [grp["obs"][k][:] for k in keys]
                    T = act.shape[0]
                    obs = np.concatenate([p[:T] for p in parts], axis=-1).astype(np.float32)
                    act = act.astype(np.float32)
                    demos.append((f"{task}:{did}", suite, obs, act))
            used_files.append(fp)
        except Exception as e:
            print(f"[extract] WARNING failed to read {fp}: {e}")
    if not demos:
        return None, "extraction yielded 0 demos (keys may not match)"
    return demos, {"files": used_files, "keys": want_keys}


def gen_synthetic(n_demos=200, min_len=60, max_len=110, obs_dim=8, action_dim=7, seed=1234):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((obs_dim, obs_dim)) * 0.15
    A *= 0.6 / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-6)
    c = rng.standard_normal(obs_dim) * 0.1
    B0 = rng.standard_normal((obs_dim, action_dim)) * 0.5
    Bk = rng.standard_normal((obs_dim, action_dim, obs_dim)) * 0.2
    demos = []
    for i in range(n_demos):
        L = int(rng.integers(min_len, max_len + 1))
        s = rng.standard_normal(obs_dim) * 0.3
        obs = np.zeros((L, obs_dim), np.float32)
        act = np.zeros((L, action_dim), np.float32)
        phase = rng.standard_normal(action_dim) * 0.3
        for t in range(L):
            a = np.sin(2 * np.pi * (0.02 * t + phase)) + 0.5 * rng.standard_normal(action_dim)
            a = a.astype(np.float32)
            Bs = B0 + np.einsum("oak,k->oa", Bk, s)
            s = np.tanh(A @ s + c) + (Bs @ a) * 0.2 + rng.standard_normal(obs_dim) * 0.02
            obs[t] = s
            act[t] = a
        demos.append((f"synthetic:demo_{i:04d}", "synthetic", obs, act))
    return demos, {"generator": "nonlinear_controlled_synth", "seed": seed,
                   "obs_dim": obs_dim, "action_dim": action_dim}


def build_processed(demos, info, out_path):
    ids = np.array([d[0] for d in demos], dtype=object)
    suites = np.array([d[1] for d in demos], dtype=object)
    obs = np.array([d[2] for d in demos], dtype=object)
    act = np.array([d[3] for d in demos], dtype=object)
    obs_dim = demos[0][2].shape[-1]
    action_dim = demos[0][3].shape[-1]
    np.savez_compressed(
        out_path, ids=ids, suites=suites, obs=obs, act=act,
        obs_dim=obs_dim, action_dim=action_dim,
    )
    return obs_dim, action_dim, len(demos)


def write_manifest(source, real_info, obs_dim, action_dim, n_demos, demos):
    from .splits import build_and_save_splits
    splits_dir = _data_root() / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    demo_meta = {d[0]: {"suite": d[1], "length": int(d[2].shape[0])} for d in demos}
    splits = build_and_save_splits(_data_root(), demo_meta)
    manifest = {
        "source": source,
        "real_info": real_info if source == "real" else None,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "n_demos": n_demos,
        "n_train": len(splits["train"]), "n_val": len(splits["val"]), "n_test": len(splits["test"]),
        "processed_file": "data/processed/latents/libero_lowdim.npz",
        "splits_file": "data/splits/libero_splits.json",
        "note": ("Real LIBERO lowdim proprioception+actions extracted from official HF mirror "
                 "yifengzhu-hf/LIBERO-datasets (no images, no simulator).")
                 if source == "real" else
                ("SYNTHETIC FALLBACK in use: real LIBERO extraction unavailable. Nonlinear "
                 "controlled world used to keep the pipeline reproducible. Replace with real data."),
    }
    with open(splits_dir / "libero_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def prepare(force_synth=False, synth_args=None):
    out_dir = _data_root() / "processed" / "latents"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "libero_lowdim.npz"
    demos, info = (None, None)
    source = None
    if not force_synth:
        demos, info = extract_real(out_path)
    if demos:
        source = "real"
    else:
        print(f"[prepare] real extraction unavailable ({info}); using synthetic fallback.")
        sa = synth_args or {}
        demos, info = gen_synthetic(**sa)
        source = "synthetic_fallback"
    obs_dim, action_dim, n = build_processed(demos, info, out_path)
    manifest = write_manifest(source, info, obs_dim, action_dim, n, demos)
    print(f"[prepare] source={source} demos={n} obs_dim={obs_dim} action_dim={action_dim}")
    print(f"[prepare] processed -> {out_path}")
    print(f"[prepare] manifest -> {_data_root()/'splits'/'libero_manifest.json'}")
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force-synth", action="store_true")
    p.add_argument("--n-demos", type=int, default=200)
    args = p.parse_args()
    sa = {} if not args.force_synth else {"n_demos": args.n_demos}
    m = prepare(force_synth=args.force_synth, synth_args=sa)
    print(json.dumps(m, indent=2))
