from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

for _name, _attrs in (("data", {"Batch": object}),
                      ("utils", {"LossAndLogs": object, "StateDictMixin": object})):
    _m = types.ModuleType(_name)
    for _k, _v in _attrs.items():
        setattr(_m, _k, _v)
    sys.modules.setdefault(_name, _m)
sys.path.insert(0, str(ROOT / "third_party" / "diamond" / "src"))

from models.diffusion.denoiser import Denoiser, DenoiserConfig
from models.diffusion.diffusion_sampler import (
    DiffusionSampler, DiffusionSamplerConfig)
from models.diffusion.inner_model import InnerModelConfig

H = 4
N_COND = 4
DENOISE_STEPS = 3
TRANSFORMS = ["identity", "reflect", "rotate", "scale_0.25", "scale_4", "blind"]


def load_denoiser(path, device):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    n_act = sd["denoiser.inner_model.act_emb.0.weight"].shape[0]
    cfg = DenoiserConfig(
        inner_model=InnerModelConfig(img_channels=3, num_steps_conditioning=N_COND,
                                     cond_channels=256, depths=[2, 2, 2, 2],
                                     channels=[64, 64, 64, 64], attn_depths=[0, 0, 0, 0],
                                     num_actions=n_act),
        sigma_data=0.5, sigma_offset_noise=0.3)
    den = Denoiser(cfg)
    den.load_state_dict({k[len("denoiser."):]: v for k, v in sd.items()
                         if k.startswith("denoiser.")}, strict=True)
    den = den.to(device).eval()
    return den, DiffusionSampler(den, DiffusionSamplerConfig(num_steps_denoising=DENOISE_STEPS)), n_act


@torch.no_grad()
def _denoise_from(sampler, prev_obs, prev_act, x0):
    b, t, c, h, w = prev_obs.shape
    obs = prev_obs.reshape(b, t * c, h, w)
    x = x0
    for sigma, next_sigma in zip(sampler.sigmas[:-1], sampler.sigmas[1:]):
        denoised = sampler.denoiser.denoise(x, sigma, obs, prev_act)
        x = x + ((x - denoised) / sigma) * (next_sigma - sigma)
    return x


@torch.no_grad()
def rollout(sampler, ctx_obs, ctx_act, fut_act, noise, chunk=16):
    B = ctx_obs.shape[0]
    outs = []
    for s in range(0, B, chunk):
        e = min(s + chunk, B)
        pad = chunk - (e - s)
        sl = slice(s, e)
        rep = lambda x: torch.cat([x[sl], x[e - 1:e].expand(pad, *x.shape[1:])], 0) if pad else x[sl]
        obs, act = rep(ctx_obs).clone(), rep(ctx_act).clone()
        fa, nz = rep(fut_act), rep(noise)
        got = []
        for k in range(fut_act.shape[1]):
            act = torch.cat([act[:, 1:], fa[:, k:k + 1]], 1)
            nxt = _denoise_from(sampler, obs, act, nz[:, k])
            got.append(nxt)
            obs = torch.cat([obs[:, 1:], nxt[:, None]], 1)
        outs.append(torch.stack(got, 1)[:e - s])
        del obs, act, got
    return torch.cat(outs, 0)


class Sim:

    def __init__(self, game, seed=0):
        import importlib.util
        import gymnasium
        import ale_py
        _p = ROOT / "third_party" / "diamond" / "src" / "envs" / "atari_preprocessing.py"
        _spec = importlib.util.spec_from_file_location("diamond_atari_preprocessing", _p)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        AtariPreprocessing = _mod.AtariPreprocessing
        gid = f"{game}NoFrameskip-v4"
        env = gymnasium.make(gid, full_action_space=False, frameskip=1, render_mode="rgb_array")
        self.env = AtariPreprocessing(env=env, noop_max=30, frame_skip=4, screen_size=64)
        self.ale = self.env.env.unwrapped.ale
        self.ale.setFloat("repeat_action_probability", 0.0)
        self.n_act = self.env.action_space.n
        self.seed = seed
        self.n_resets = 0

    def reset(self):
        obs, _ = self.env.reset(seed=self.seed * 1000 + self.n_resets)
        self.n_resets += 1
        return obs

    def step(self, a):
        obs, _, term, trunc, _ = self.env.step(int(a))
        return obs, bool(term or trunc)

    def clone(self):
        return self.ale.cloneState()

    def restore(self, st):
        self.ale.restoreState(st)

    def branch(self, st, actions):
        self.restore(st)
        frames, ended = [], False
        for a in actions:
            if ended:
                frames.append(frames[-1])
                continue
            obs, done = self.step(a)
            frames.append(obs)
            ended = done
        return np.stack(frames), ended


def to_tensor(x, device):
    t = torch.as_tensor(np.asarray(x), device=device).float().div(255).mul(2).sub(1)
    return t.permute(*range(t.ndim - 3), t.ndim - 1, t.ndim - 3, t.ndim - 2).contiguous()


def collect(game, n_positions, device, seed=0, warmup=120):
    rng = np.random.default_rng(seed)
    sim = Sim(game, seed)
    obs = sim.reset()
    frames = [obs] * N_COND
    acts = [0] * N_COND
    rec = {k: [] for k in ["ctx_obs", "ctx_act", "a_log", "x_log", "x_rep", "state", "ep"]}
    played = {"x0": [], "a": [], "x1": [], "ep": []}
    t, ep, skipped = 0, 0, 0
    per_ep = max(1, n_positions // 6)
    while len(rec["a_log"]) < n_positions:
        a = int(rng.integers(sim.n_act))
        prev = frames[-1]
        obs, done = sim.step(a)
        if not done:
            played["x0"].append(prev); played["a"].append(a); played["x1"].append(obs); played["ep"].append(ep)
        frames = frames[1:] + [obs]
        acts = acts[1:] + [a]
        t += 1
        if done or (t > warmup and len(rec["a_log"]) and len(rec["a_log"]) % per_ep == 0
                    and rec["ep"][-1] == ep and t % 7 == 0):
            obs = sim.reset()
            frames, acts = [obs] * N_COND, [0] * N_COND
            t, ep = 0, ep + 1
            continue
        if t < warmup or t % 7:
            continue
        st = sim.clone()
        a_log = rng.integers(sim.n_act, size=H)
        _, e0 = sim.branch(st, a_log)
        x_log, e1 = sim.branch(st, a_log)
        x_rep, e2 = sim.branch(st, a_log)
        sim.restore(st)
        if e0 or e1 or e2:
            skipped += 1
            continue
        rec["ctx_obs"].append(np.stack(frames)); rec["ctx_act"].append(np.array(acts))
        rec["a_log"].append(a_log); rec["x_log"].append(x_log); rec["x_rep"].append(x_rep)
        rec["state"].append(st); rec["ep"].append(ep)
    data = {k: (np.stack(v) if k != "state" else v) for k, v in rec.items()}
    data["n_episodes"] = int(len(set(rec["ep"])))
    data["n_skipped_ended"] = skipped
    data["sim"], data["n_act"] = sim, sim.n_act
    data["played"] = {k: np.stack(v) if k != "a" and k != "ep" else np.array(v) for k, v in played.items()}
    return data


def matched_pairs(data, rng):
    C = data["ctx_obs"].reshape(len(data["ctx_obs"]), -1).astype(np.float32) / 255.0
    A, ep = data["a_log"], data["ep"]
    pairs = []
    for i in range(len(C)):
        d = np.linalg.norm(C - C[i], axis=1)
        d[i] = np.inf
        d[ep == ep[i]] = np.inf
        d[(A == A[i]).all(1)] = np.inf
        j = int(np.argmin(d))
        if np.isfinite(d[j]):
            pairs.append((i, j, float(d[j])))
    return pairs


class InverseCNN(torch.nn.Module):

    def __init__(self, n_act):
        super().__init__()
        nn = torch.nn
        self.net = nn.Sequential(nn.Conv2d(6, 32, 4, 2, 1), nn.SiLU(), nn.Conv2d(32, 64, 4, 2, 1), nn.SiLU(),
                                 nn.Conv2d(64, 64, 4, 2, 1), nn.SiLU(), nn.Flatten(),
                                 nn.Linear(64 * 8 * 8, 256), nn.SiLU(), nn.Linear(256, n_act))

    def forward(self, x0, x1):
        return self.net(torch.cat([x0, x1 - x0], 1))


def fit_inverse(played, n_act, device, steps=3000, seed=0):
    last = played["ep"].max()
    tr, va = played["ep"] < last, played["ep"] == last
    X0, X1 = to_tensor(played["x0"], device), to_tensor(played["x1"], device)
    A = torch.as_tensor(played["a"], device=device).long()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        g = InverseCNN(n_act).to(device)
    opt = torch.optim.AdamW(g.parameters(), 1e-3, weight_decay=1e-4)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.as_tensor(np.flatnonzero(tr))
    for _ in range(steps):
        b = idx[torch.randint(0, len(idx), (128,), generator=gen)].to(device)
        loss = torch.nn.functional.cross_entropy(g(X0[b], X1[b]), A[b])
        opt.zero_grad(); loss.backward(); opt.step()
    g.eval()
    with torch.no_grad():
        v = torch.as_tensor(np.flatnonzero(va), device=device)
        acc = (g(X0[v], X1[v]).argmax(1) == A[v]).float().mean().item() if len(v) else float("nan")
    return g, {"inv_val_acc": acc, "inv_chance": 1.0 / n_act, "inv_n_train": int(tr.sum())}


@torch.no_grad()
def inv_probs(g, first, P):
    prev = torch.cat([first[:, None], P[:, :-1]], 1)
    n, h = P.shape[:2]
    return torch.softmax(g(prev.reshape(n * h, *P.shape[2:]), P.reshape(n * h, *P.shape[2:])), -1).reshape(n, h, -1)


_PERMS = {}


def make_perm(width, device, seed=0):
    key = (width, str(device), seed)
    if key not in _PERMS:
        g = torch.Generator(device="cpu").manual_seed(seed * 100003 + width)
        _PERMS[key] = torch.randperm(width, generator=g).to(device)
    return _PERMS[key]


def apply_T(R, name, seed=0):
    if name == "identity":
        return R
    if name == "reflect":
        return -R
    if name == "rotate":
        return R[..., make_perm(R.shape[-1], R.device, seed)]
    if name.startswith("scale_"):
        return float(name.split("_")[1]) * R
    if name == "blind":
        return torch.zeros_like(R)
    raise ValueError(name)


def cos_rows(a, b):
    na, nb = a.norm(dim=1), b.norm(dim=1)
    return ((a * b).sum(1) / (na * nb).clamp_min(1e-12)).cpu().numpy()


def score(sampler, data, pairs, device, seed=0):
    N, n_act = len(data["a_log"]), data["n_act"]
    flat = lambda x: x.reshape(len(x), -1)
    sim = data["sim"]

    swaps, keep = [], []
    for i, j, g in pairs:
        x, ended = sim.branch(data["state"][i], data["a_log"][j])
        if not ended:
            swaps.append(x)
            keep.append((i, j, g))
    n_dropped, pairs = len(pairs) - len(keep), keep

    ctx_o = to_tensor(data["ctx_obs"], device)
    ctx_a = torch.as_tensor(data["ctx_act"], device=device).long()
    a_log = torch.as_tensor(data["a_log"], device=device).long()
    true_log = to_tensor(data["x_log"], device)

    noise = torch.randn(N, H, 3, 64, 64, generator=torch.Generator(device="cpu").manual_seed(seed)).to(device)
    noise_b = torch.randn(N, H, 3, 64, 64,
                          generator=torch.Generator(device="cpu").manual_seed(seed + 7919)).to(device)

    P_log = rollout(sampler, ctx_o, ctx_a, a_log, noise)
    P_zero = rollout(sampler, ctx_o, ctx_a, torch.zeros_like(a_log), noise)
    i_idx = torch.as_tensor([p[0] for p in pairs], device=device)
    j_idx = torch.as_tensor([p[1] for p in pairs], device=device)
    P_swap = rollout(sampler, ctx_o[i_idx], ctx_a[i_idx], a_log[j_idx], noise[i_idx])
    P_swap_b = rollout(sampler, ctx_o[i_idx], ctx_a[i_idx], a_log[j_idx], noise_b[i_idx])

    rep = lambda x, n: x.repeat_interleave(n, 0)
    cand_a = torch.arange(n_act, device=device).repeat(N)[:, None]
    P_cand = rollout(sampler, rep(ctx_o, n_act), rep(ctx_a, n_act), cand_a,
                     rep(noise[:, :1], n_act))[:, 0].reshape(N, n_act, -1)
    target1 = flat(true_log[:, 0])

    r_star = flat(to_tensor(np.stack(swaps), device) - true_log[i_idx])
    s_i = ctx_o[i_idx, -1][:, None].expand(-1, H, -1, -1, -1)
    s_j = ctx_o[j_idx, -1][:, None].expand(-1, H, -1, -1, -1)
    e_hat = flat((true_log[j_idx] - s_j) - (true_log[i_idx] - s_i))

    R_model = flat(P_swap - P_log[i_idx])
    R_indep = flat(P_swap_b - P_log[i_idx])
    D_zero = flat(P_zero - P_log)

    g, inv_meta = fit_inverse(data["played"], n_act, device)
    first = ctx_o[i_idx, -1]
    p_ref = inv_probs(g, first, P_log[i_idx])
    oh = lambda a: torch.nn.functional.one_hot(a, n_act).float()
    da = flat(oh(a_log[j_idx]) - oh(a_log[i_idx]))

    out = {}
    for t in TRANSFORMS:
        Rt = apply_T(R_model, t, seed)
        p_q = inv_probs(g, first, P_log[i_idx] + Rt.reshape(P_swap.shape))
        inv = {"inv_dir_swap": float(np.mean(cos_rows(flat(p_q - p_ref), da))),
               "inv_err_swap": float(-(p_q.gather(-1, a_log[j_idx][..., None]).clamp_min(1e-12).log()).mean().item())}
        out[t] = {
            "direction": float(np.mean(cos_rows(Rt, r_star))),
            "mag_ratio": float((Rt.norm(dim=1).sum() / r_star.norm(dim=1).sum()).item()),
            "C1": float((apply_T(D_zero, t, seed) ** 2).mean().item()),
            "matched": float(np.mean(cos_rows(Rt, e_hat))),
            "ret_meanrank": retrieval_rank(P_cand, a_log[:, 0], target1, t, seed),
            **inv,
        }
    extra = {"n_pairs": len(pairs), "n_pairs_dropped_ended": n_dropped, **inv_meta,
             "direction_no_crn": float(np.mean(cos_rows(R_indep, r_star))),
             "matched_no_crn": float(np.mean(cos_rows(R_indep, e_hat))),
             "sampler_noise_max": float((P_swap_b - P_swap).abs().max().item()),
             "action_effect_max": float(R_model.abs().max().item())}
    return out, extra


def retrieval_rank(P_cand, idx, target, tname, seed):
    rows = torch.arange(len(P_cand), device=P_cand.device)
    base = P_cand[rows, idx][:, None]
    Pt = base + apply_T(P_cand - base, tname, seed)
    d = (Pt - target[:, None]).double().pow(2).mean(-1)
    d_log = d[rows, idx][:, None]
    below = (d < d_log).sum(1).double()
    ties = (d == d_log).sum(1).double() - 1.0
    return float((below + 0.5 * ties).mean().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="+", default=["Pong", "Breakout", "Freeway"])
    ap.add_argument("--positions", type=int, default=60)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ck = ROOT / "data" / "raw" / "diamond" / "models"
    out_path = Path(args.out or ROOT / "results" / "tables" / "diamond_validity.json")
    res = json.load(open(out_path)) if out_path.exists() else {}

    for game in args.games:
        if game in res:
            print(f"[diamond] {game}: already done")
            continue
        t0 = time.time()
        den, sampler, n_act = load_denoiser(ck / f"{game}.pt", args.device)
        data = collect(game, args.positions, args.device, seed=args.seed)
        assert data["n_act"] == n_act, f"{game}: env has {data['n_act']} actions, model has {n_act}"
        floor = np.abs(data["x_rep"].astype(int) - data["x_log"].astype(int)).max()
        rng = np.random.default_rng(0)
        pairs = matched_pairs(data, rng)
        r, extra = score(sampler, data, pairs, args.device, seed=args.seed)
        r["meta"] = {"n_positions": int(len(data["a_log"])), "n_episodes": data["n_episodes"],
                     "n_positions_skipped_ended": data["n_skipped_ended"],
                     "n_actions": int(n_act), "emulator_noise_floor": int(floor),
                     "chance_rank": (n_act - 1) / 2, **extra}
        res[game] = r
        i = r["identity"]
        print(f"  {game:14s} dir={i['direction']:+.3f} mag={i['mag_ratio']:.2f} C1={i['C1']:.4f} "
              f"ret={i['ret_meanrank']:.2f}/{(n_act-1)/2:.1f} matched={i['matched']:+.3f} "
              f"floor={floor} eps={data['n_episodes']} dropped={extra['n_pairs_dropped_ended']} "
              f"dir_noCRN={extra['direction_no_crn']:+.3f} inv={i['inv_dir_swap']:+.3f} "
              f"inv_acc={extra['inv_val_acc']:.2f}/{extra['inv_chance']:.2f} ({time.time()-t0:.0f}s)", flush=True)
        json.dump(res, open(out_path, "w"), indent=1)
        del den, sampler, data
        torch.cuda.empty_cache()
    print("[diamond] wrote", out_path)


if __name__ == "__main__":
    main()
