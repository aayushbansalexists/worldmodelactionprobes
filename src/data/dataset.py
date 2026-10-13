from __future__ import annotations
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _data_root():
    return Path(os.environ.get("DATA_ROOT", str(Path(__file__).resolve().parents[1] / "data")))


def load_processed():
    p = _data_root() / "processed" / "latents" / "libero_lowdim.npz"
    d = np.load(p, allow_pickle=True)
    obs = d["obs"]
    act = d["act"]
    ids = [str(x) for x in d["ids"]]
    suites = [str(x) for x in d["suites"]]
    obs_dim = int(d["obs_dim"])
    action_dim = int(d["action_dim"])
    return obs, act, ids, suites, obs_dim, action_dim


def load_splits():
    p = _data_root() / "splits" / "libero_splits.json"
    with open(p) as f:
        return json.load(f)


class Normalizer:

    def __init__(self, obs_mean, obs_std, act_mean, act_std):
        self.obs_mean = obs_mean.astype(np.float32)
        self.obs_std = np.clip(obs_std.astype(np.float32), 1e-6, None)
        self.act_mean = act_mean.astype(np.float32)
        self.act_std = np.clip(act_std.astype(np.float32), 1e-6, None)

    def norm_obs(self, x):
        return (x - self.obs_mean) / self.obs_std

    def norm_act(self, x):
        return (x - self.act_mean) / self.act_std

    def denorm_obs(self, x):
        return x * self.obs_std + self.obs_mean

    def save(self, path):
        np.savez(path, obs_mean=self.obs_mean, obs_std=self.obs_std,
                 act_mean=self.act_mean, act_std=self.act_std)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(d["obs_mean"], d["obs_std"], d["act_mean"], d["act_std"])

    @classmethod
    def fit_from(cls, obs_list, act_list):
        all_obs = np.concatenate([o.reshape(-1, o.shape[-1]) for o in obs_list], axis=0)
        all_act = np.concatenate([a.reshape(-1, a.shape[-1]) for a in act_list], axis=0)
        return cls(all_obs.mean(0), all_obs.std(0), all_act.mean(0), all_act.std(0))


def get_or_fit_normalizer(train_obs, train_act):
    path = _data_root() / "processed" / "latents" / "normalizer.npz"
    if path.exists():
        return Normalizer.load(path)
    nz = Normalizer.fit_from(train_obs, train_act)
    nz.save(path)
    return nz


class LiberoWindowDataset(Dataset):

    def __init__(self, obs, act, ids, normalizer, seq_len=16, overfit=0, split_name="train"):
        self.seq_len = int(seq_len)
        self.nz = normalizer
        self.split_name = split_name
        self.index = []
        self.obs_list = []
        self.act_list = []
        n_trajs = len(obs) if overfit == 0 else min(overfit, len(obs))
        for i in range(n_trajs):
            o = normalizer.norm_obs(obs[i])
            a = normalizer.norm_act(act[i])
            self.obs_list.append(o)
            self.act_list.append(a)
            L = self.seq_len + 1
            if o.shape[0] < L:
                continue
            if overfit > 0:
                self.index.append((i, 0))
            else:
                step = max(1, (self.seq_len + 1) // 2)
                for start in range(0, o.shape[0] - L + 1, step):
                    self.index.append((i, start))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        ti, start = self.index[i]
        L = self.seq_len + 1
        o = self.obs_list[ti][start:start + L]
        a = self.act_list[ti][start:start + L - 1]
        return torch.from_numpy(np.ascontiguousarray(o)), torch.from_numpy(np.ascontiguousarray(a))


def build_datasets(seq_len=16, overfit=0):
    obs, act, ids, suites, obs_dim, action_dim = load_processed()
    splits = load_splits()
    id2idx = {did: i for i, did in enumerate(ids)}
    train_ids = splits["train"]; val_ids = splits["val"]; test_ids = splits["test"]
    train_obs = [obs[id2idx[d]] for d in train_ids if d in id2idx]
    train_act = [act[id2idx[d]] for d in train_ids if d in id2idx]
    nz = get_or_fit_normalizer(train_obs, train_act)

    def make(split_ids, name):
        oi = [obs[id2idx[d]] for d in split_ids if d in id2idx]
        ai = [act[id2idx[d]] for d in split_ids if d in id2idx]
        return LiberoWindowDataset(oi, ai, split_ids, nz, seq_len=seq_len,
                                   overfit=(overfit if name == "train" else 0), split_name=name)

    return (make(train_ids, "train"), make(val_ids, "val"), make(test_ids, "test"),
            nz, obs_dim, action_dim)


class TrajectoryList:

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        o, a = self.items[i]
        return torch.as_tensor(o, dtype=torch.float32), torch.as_tensor(a, dtype=torch.float32)


def build_eval_trajectories(split="test"):
    obs, act, ids, suites, obs_dim, action_dim = load_processed()
    splits = load_splits()
    id2idx = {did: i for i, did in enumerate(ids)}
    train_ids = splits["train"]
    train_obs = [obs[id2idx[d]] for d in train_ids if d in id2idx]
    train_act = [act[id2idx[d]] for d in train_ids if d in id2idx]
    nz = get_or_fit_normalizer(train_obs, train_act)
    ids_for = splits[split]
    items = []
    for d in ids_for:
        if d not in id2idx:
            continue
        o = nz.norm_obs(obs[id2idx[d]])
        a = nz.norm_act(act[id2idx[d]])
        items.append((np.ascontiguousarray(o), np.ascontiguousarray(a)))
    return TrajectoryList(items), obs_dim, action_dim


def get_or_fit_normalizer_latent(train_lat, train_act):
    path = _data_root() / "processed" / "latents" / "normalizer_latent.npz"
    if path.exists():
        return Normalizer.load(path)
    nz = Normalizer.fit_from(train_lat, train_act)
    nz.save(path)
    return nz


def load_latents():
    p = _data_root() / "processed" / "latents" / "libero_imglatents.npz"
    d = np.load(p, allow_pickle=True)
    lat = d["latents"]
    ids = [str(x) for x in d["ids"]]
    suites = [i.split(":")[0] for i in ids]
    flat = [z.astype(np.float32).reshape(z.shape[0], -1) for z in lat]
    ld = np.load(_data_root() / "processed" / "latents" / "libero_lowdim.npz", allow_pickle=True)
    id2act = {str(i): a for i, a in zip([str(x) for x in ld["ids"]], ld["act"])}
    act = [id2act[i] for i in ids]
    latent_dim = int(flat[0].shape[-1])
    return flat, act, ids, suites, latent_dim, int(act[0].shape[-1])


def build_datasets_latent(seq_len=16, overfit=0):
    obs, act, ids, suites, obs_dim, action_dim = load_latents()
    splits = load_splits()
    id2idx = {did: i for i, did in enumerate(ids)}
    train_ids = splits["train"]
    train_obs = [obs[id2idx[d]] for d in train_ids if d in id2idx]
    train_act = [act[id2idx[d]] for d in train_ids if d in id2idx]
    nz = get_or_fit_normalizer_latent(train_obs, train_act)

    def make(split_ids, name):
        oi = [obs[id2idx[d]] for d in split_ids if d in id2idx]
        ai = [act[id2idx[d]] for d in split_ids if d in id2idx]
        return LiberoWindowDataset(oi, ai, split_ids, nz, seq_len=seq_len,
                                   overfit=(overfit if name == "train" else 0), split_name=name)

    return (make(train_ids, "train"), make(splits["val"], "val"), make(splits["test"], "test"),
            nz, obs_dim, action_dim)


def build_eval_trajectories_latent(split="test"):
    obs, act, ids, suites, obs_dim, action_dim = load_latents()
    splits = load_splits()
    id2idx = {did: i for i, did in enumerate(ids)}
    train_ids = splits["train"]
    train_obs = [obs[id2idx[d]] for d in train_ids if d in id2idx]
    train_act = [act[id2idx[d]] for d in train_ids if d in id2idx]
    nz = get_or_fit_normalizer_latent(train_obs, train_act)
    ids_for = splits[split]
    items = []
    for d in ids_for:
        if d not in id2idx:
            continue
        o = nz.norm_obs(obs[id2idx[d]])
        a = nz.norm_act(act[id2idx[d]])
        items.append((np.ascontiguousarray(o), np.ascontiguousarray(a)))
    return TrajectoryList(items), obs_dim, action_dim


__all__ = ["build_datasets", "LiberoWindowDataset", "Normalizer",
           "TrajectoryList", "build_eval_trajectories",
           "build_datasets_latent", "build_eval_trajectories_latent", "load_latents"]
