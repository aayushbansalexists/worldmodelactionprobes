from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "disabled")

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.probe_validity import TRANSFORMS, make_T

_WSP = ROOT / "third_party" / "WorldSimProbe" / "src"
sys.path.insert(0, str(_WSP))
from worldsimprobe.evaluation.task1_action_calibration.oracle_ratio import (
    triangular_ratio_score,
)

H = 3
LARGE, SMALL = 0.14, 0.035
TD_TASKS = ROOT / "third_party" / "tdmpc2" / "tdmpc2" / "envs" / "tasks"


def _register_custom_tasks():
    from dm_control import suite
    for name in ["cheetah", "walker", "hopper", "reacher", "ball_in_cup", "pendulum", "fish"]:
        spec = importlib.util.spec_from_file_location(f"tdmpc2_tasks_{name}", TD_TASKS / f"{name}.py")
        spec.loader.exec_module(importlib.util.module_from_spec(spec))
    suite.ALL_TASKS = suite.ALL_TASKS + suite._get_tasks("custom")
    suite.TASKS_BY_DOMAIN = suite._get_tasks_by_domain(suite.ALL_TASKS)
    return suite


_SUITE = None


def make_env(task, seed):
    global _SUITE
    if _SUITE is None:
        _SUITE = _register_custom_tasks()
    domain, t = task.replace("-", "_").split("_", 1)
    domain = dict(cup="ball_in_cup", pointmass="point_mass").get(domain, domain)
    return _SUITE.load(domain, t, task_kwargs={"random": seed}, visualize_reward=False)


def obs_vec(obs):
    return np.concatenate([np.asarray(v, dtype=np.float32).ravel() for v in obs.values()])


class Sim:

    def __init__(self, env):
        self.env, self.phys, self.task = env, env.physics, env._task
        spec = env.action_spec()
        self.lo, self.hi = spec.minimum.astype(np.float64), spec.maximum.astype(np.float64)

    def step(self, a):
        a = self.lo + (np.clip(a, -1, 1) + 1.0) * (self.hi - self.lo) / 2.0
        self.reward = 0.0
        for _ in range(2):
            self.task.before_step(a, self.phys)
            self.phys.step(self.env._n_sub_steps)
            self.task.after_step(self.phys)
            self.reward += float(self.task.get_reward(self.phys))
        return obs_vec(self.task.get_observation(self.phys))

    def get(self):
        return self.phys.get_state().copy()

    def set(self, s):
        with self.phys.reset_context():
            self.phys.set_state(s)

    def branch(self, s, actions):
        self.set(s)
        return np.stack([self.step(a) for a in actions])


class NormedLinear(nn.Linear):
    def __init__(self, i, o, act):
        super().__init__(i, o)
        self.ln = nn.LayerNorm(o)
        self.act = act

    def forward(self, x):
        return self.act(self.ln(super().forward(x)))


class SymLog(nn.Module):
    def forward(self, x):
        return torch.sign(x) * torch.log1p(x.abs())


class SimNorm(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        s = x.shape
        return F.softmax(x.view(*s[:-1], -1, self.dim), -1).view(*s)


def _flat(sd, prefix, final_simnorm):
    sub = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    if any(k.startswith("0.0.") for k in sub):
        inner = {k[2:]: v for k, v in sub.items() if k.startswith("0.")}
        seq = _flat_seq(inner, None)
        ln = nn.LayerNorm(sub["1.weight"].shape[0])
        ln.load_state_dict({"weight": sub["1.weight"], "bias": sub["1.bias"]})
        return nn.Sequential(seq, ln, SimNorm())
    return _flat_seq(sub, SimNorm() if final_simnorm else None)


def _flat_seq(sub, final_act):
    idx = {int(k.split(".")[0]) for k in sub}
    top = max(idx)
    mods = []
    for j in range(top + 1):
        if j not in idx:
            if j == 0:
                mods.append(SymLog())
            continue
        w, b = sub[f"{j}.weight"], sub[f"{j}.bias"]
        if w.ndim == 2:
            m = nn.Linear(w.shape[1], w.shape[0])
        else:
            m = nn.LayerNorm(w.shape[0])
        m.load_state_dict({"weight": w, "bias": b})
        mods.append(m)
        if w.ndim == 1:
            mods.append(final_act if (j == top and final_act is not None) else nn.Mish())
    return nn.Sequential(*mods)


def _mlp(sd, prefix, final_simnorm):
    if not any(k.startswith(prefix + "0.ln.") for k in sd) and any(k.startswith(prefix) for k in sd):
        return _flat(sd, prefix, final_simnorm)
    idx = sorted({int(k[len(prefix):].split(".")[0]) for k in sd if k.startswith(prefix)})
    layers = []
    for n, i in enumerate(idx):
        w = sd[f"{prefix}{i}.weight"]
        last = n == len(idx) - 1
        if last and not final_simnorm:
            layers.append(nn.Linear(w.shape[1], w.shape[0]))
        else:
            layers.append(NormedLinear(w.shape[1], w.shape[0], SimNorm() if last else nn.Mish()))
    m = nn.Sequential(*layers)
    m.load_state_dict({k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)})
    return m


class TDMPC2(nn.Module):

    def __init__(self, path):
        super().__init__()
        sd = torch.load(path, map_location="cpu", weights_only=False)["model"]
        self.enc = _mlp(sd, "_encoder.state.", True)
        self.dyn = _mlp(sd, "_dynamics.", True)
        self.pi = _mlp(sd, "_pi.", False)
        last_pi = max((k for k in sd if k.startswith("_pi.") and k.endswith(".weight")
                       and sd[k].ndim == 2), key=lambda k: int(k.split(".")[1]))
        self.adim = sd[last_pi].shape[0] // 2
        self.latent = sd["_pi.0.weight"].shape[1]

    @torch.no_grad()
    def encode(self, o):
        return self.enc(o)

    @torch.no_grad()
    def next(self, z, a):
        return self.dyn(torch.cat([z, a], -1))

    @torch.no_grad()
    def act(self, z):
        return torch.tanh(self.pi(z).chunk(2, -1)[0])

    @torch.no_grad()
    def rollout(self, z, A):
        out = []
        for k in range(A.shape[1]):
            z = self.next(z, A[:, k])
            out.append(z)
        return torch.stack(out, 1)


def collect(task, behavior, device, episodes=6, per_ep=25, steps=500, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    env = make_env(task, seed)
    sim = Sim(env)
    adim = env.action_spec().shape[0]
    R = {k: [] for k in ["o0", "a", "x_ref", "x_rep", "a_pert", "x_pert", "sigma", "o1",
                         "w_dim", "w_orig", "w_small", "w_large", "w_ctx", "ep"]}
    logs = []
    returns = []
    for ep in range(episodes):
        ts = env.reset()
        o = obs_vec(ts.observation)
        O, A, S = [o], [], []
        ret = 0.0
        for t in range(steps):
            S.append(sim.get())
            z = behavior.encode(torch.from_numpy(o)[None].to(device))
            a = behavior.act(z)[0].cpu().numpy()
            a = np.clip(a + rng.normal(0, noise, adim), -1, 1).astype(np.float32)
            A.append(a)
            o = sim.step(a)
            ret += sim.reward
            O.append(o)
        returns.append(ret)
        O, A = np.stack(O), np.stack(A)
        logs.append((O, A))
        for t in np.sort(rng.choice(np.arange(5, steps - H - 1), per_ep, replace=False)):
            a_log = A[t:t + H]
            sim.branch(S[t], a_log)
            x_ref = sim.branch(S[t], a_log)
            x_rep = sim.branch(S[t], a_log)
            for sg in (0.2, 0.5):
                ap = np.clip(a_log + rng.normal(0, sg, a_log.shape), -1, 1).astype(np.float32)
                R["o0"].append(O[t]); R["a"].append(a_log); R["x_ref"].append(x_ref)
                R["x_rep"].append(x_rep); R["a_pert"].append(ap)
                R["x_pert"].append(sim.branch(S[t], ap)); R["sigma"].append(sg)
                R["o1"].append(O[t + 1]); R["ep"].append(ep)
            for dim in rng.permutation(adim)[:min(3, adim)]:
                a_s, a_l = a_log.copy(), a_log.copy()
                a_s[:, dim] = np.clip(a_s[:, dim] + SMALL, -1, 1)
                a_l[:, dim] = np.clip(a_l[:, dim] + LARGE, -1, 1)
                R["w_ctx"].append(O[t]); R["w_dim"].append(int(dim)); R["w_orig"].append(x_ref)
                R["w_small"].append(sim.branch(S[t], a_s)); R["w_large"].append(sim.branch(S[t], a_l))
                R.setdefault("w_a_orig", []).append(a_log)
                R.setdefault("w_a_small", []).append(a_s)
                R.setdefault("w_a_large", []).append(a_l)
    data = {k: np.stack(v) if len(v) and not isinstance(v[0], (int, float)) else np.array(v)
            for k, v in R.items()}
    data["returns"] = np.array(returns)
    data["logs"] = logs
    return data


def matched_pairs(logs, n_pos=150, seed=0):
    rng = np.random.default_rng(seed)
    win = [(e, t) for e, (O, A) in enumerate(logs) for t in range(len(A) - H)]
    allO = np.concatenate([O for O, _ in logs])
    sd = allO.std(0) + 1e-6
    S = np.stack([logs[e][0][t] / sd for e, t in win])
    ep = np.array([e for e, _ in win])
    pick = rng.choice(len(win), n_pos, replace=False)
    out = []
    for i in pick:
        d = np.linalg.norm(S - S[i], axis=1)
        d[ep == ep[i]] = np.inf
        j = int(np.argmin(d))
        out.append((win[i], win[j], float(d[j])))
    return out


def fit_inverse(enc, logs, device, steps=3000, seed=0):
    Z = [enc(O) for O, _ in logs]
    X = np.concatenate([z[:-1] for z in Z[:-1]]); Y = np.concatenate([z[1:] for z in Z[:-1]])
    A = np.concatenate([a for _, a in logs[:-1]])
    t = lambda x: torch.from_numpy(np.asarray(x, np.float32)).to(device)
    X, Y, A = t(X), t(Y), t(A)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    L, adim = X.shape[1], A.shape[1]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        g = nn.Sequential(nn.Linear(2 * L, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(),
                          nn.Linear(256, adim)).to(device)
    opt = torch.optim.AdamW(g.parameters(), 1e-3, weight_decay=1e-4)
    with torch.enable_grad():
        for _ in range(steps):
            b = torch.randint(0, len(X), (256,), generator=gen).to(device)
            loss = ((g(torch.cat([X[b], Y[b] - X[b]], -1)) - A[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    g.eval()
    zv, av = Z[-1], t(logs[-1][1])
    with torch.no_grad():
        pv = g(torch.cat([t(zv[:-1]), t(zv[1:] - zv[:-1])], -1))
        r2 = 1 - ((pv - av) ** 2).mean(0) / (av.var(0) + 1e-12)
    return g, float(r2.mean())


@torch.no_grad()
def inv_read(g, z0, P, device):
    prev = np.concatenate([z0[:, None], P[:, :-1]], 1)
    x = torch.from_numpy(prev.reshape(-1, P.shape[-1])).float().to(device)
    y = torch.from_numpy(P.reshape(-1, P.shape[-1])).float().to(device)
    return g(torch.cat([x, y - x], -1)).cpu().numpy().reshape(P.shape[0], P.shape[1], -1)


def _cos(a, b):
    a, b = a.reshape(len(a), -1), b.reshape(len(b), -1)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)


def mean_rank(pred_c, target, rng):
    d = np.sqrt(((pred_c - target[:, None]) ** 2).mean(-1))
    d = d + rng.standard_normal(d.shape) * 1e-9 * (np.abs(d).mean() + 1e-12)
    return float((np.argsort(np.argsort(d, 1), 1)[:, 0]).mean())


def score(model, data, pairs, device, n_cand=16, seed=0):
    rng = np.random.default_rng(seed)
    t = lambda x: torch.from_numpy(np.asarray(x, np.float32)).to(device)
    enc = lambda x: model.encode(t(x).reshape(-1, x.shape[-1])).reshape(*x.shape[:-1], -1).cpu().numpy()
    roll = lambda z, A: model.rollout(t(z), t(A)).cpu().numpy()

    z0 = enc(data["o0"])
    P_ref, P_pert = roll(z0, data["a"]), roll(z0, data["a_pert"])
    P_zero = roll(z0, np.zeros_like(data["a"]))
    dtrue = enc(data["x_pert"]) - enc(data["x_ref"])
    N = len(z0)
    cand = np.repeat(data["a"][:, :1], n_cand, 1)
    cand[:, 1:] = np.clip(cand[:, 1:] + rng.normal(0, 0.5, cand[:, 1:].shape), -1, 1)
    logged = cand[:, :1]
    for _ in range(50):
        same = (cand[:, 1:] == logged).all(-1)
        if not same.any():
            break
        base = np.broadcast_to(logged, cand[:, 1:].shape)[same]
        cand[:, 1:][same] = np.clip(base + rng.normal(0, 0.5, base.shape), -1, 1)
    assert not (cand[:, 1:] == logged).all(-1).any(), "an alternative still equals the logged action"
    Pc = model.next(t(np.repeat(z0, n_cand, 0)), t(cand.reshape(N * n_cand, -1))).cpu().numpy()
    Pc = Pc.reshape(N, n_cand, -1)
    tgt = enc(data["o1"])
    wz = enc(data["w_ctx"])
    W_o, W_s, W_l = roll(wz, data["w_a_orig"]), roll(wz, data["w_a_small"]), roll(wz, data["w_a_large"])
    ho = enc(data["w_orig"])
    ds_sim = ((enc(data["w_small"]) - ho) ** 2).mean((1, 2))
    dl_sim = ((enc(data["w_large"]) - ho) ** 2).mean((1, 2))
    logs = data["logs"]
    Oi = np.stack([logs[e][0][s:s + H + 1] for (e, s), _, _ in pairs])
    Oj = np.stack([logs[e][0][s:s + H + 1] for _, (e, s), _ in pairs])
    Ai = np.stack([logs[e][1][s:s + H] for (e, s), _, _ in pairs])
    Aj = np.stack([logs[e][1][s:s + H] for _, (e, s), _ in pairs])
    hi, hj = enc(Oi), enc(Oj)
    e_ij = (hj[:, 1:] - hj[:, :1]) - (hi[:, 1:] - hi[:, :1])
    zi = hi[:, 0]
    M_i, M_j = roll(zi, Ai), roll(zi, Aj)
    g, inv_r2 = fit_inverse(enc, logs, device)
    gP_ref, gM_i = inv_read(g, z0, P_ref, device), inv_read(g, zi, M_i, device)

    L = P_ref.shape[-1]
    out = {}
    for name in TRANSFORMS:
        T = make_T(name, L).astype(np.float32)
        tr = lambda R: R @ T.T
        dp = tr(P_pert - P_ref).reshape(N, -1)
        dt = dtrue.reshape(N, -1)
        nt, npd = np.linalg.norm(dt, axis=1), np.linalg.norm(dp, axis=1)
        ok = nt > 1e-9
        cos = (dp * dt).sum(1)[ok] / (npd[ok] * nt[ok] + 1e-12)
        tc = Pc[:, :1] + tr(Pc - Pc[:, :1])
        ds = (tr(W_s - W_o) ** 2).mean((1, 2))
        dl = (tr(W_l - W_o) ** 2).mean((1, 2))
        t1 = []
        for a, b, sa, sb in zip(ds, dl, ds_sim, dl_sim):
            if not (sb > 1e-12 and sb > sa):
                continue
            o = sa / sb
            if not 0.0 <= o < 1.0:
                continue
            t1.append(0.0 if b <= 1e-12 else triangular_ratio_score(a / b, o))
        R = tr(M_j - M_i).reshape(len(pairs), -1)
        E = e_ij.reshape(len(pairs), -1)
        mc = (R * E).sum(1) / (np.linalg.norm(R, axis=1) * np.linalg.norm(E, axis=1) + 1e-12)
        out[name] = {
            "direction": float(cos.mean()),
            "mag_ratio": float(npd[ok].sum() / nt[ok].sum()),
            "cf_error": float(np.median(np.linalg.norm(dp - dt, axis=1)[ok] / nt[ok])),
            "C1": float((tr(P_zero - P_ref) ** 2).mean()),
            "task1": float(np.mean(t1) * 100) if t1 else float("nan"),
            "task1_n": len(t1),
            "ret_gauss_meanrank": mean_rank(tc, tgt, rng),
            "matched": float(mc.mean()),
        }
        gq = inv_read(g, z0, P_ref + tr(P_pert - P_ref), device)
        gs = inv_read(g, zi, M_i + tr(M_j - M_i), device)
        out[name].update({
            "inv_dir_rand": float(_cos(gq - gP_ref, data["a_pert"] - data["a"]).mean()),
            "inv_err_rand": float(((gq - data["a_pert"]) ** 2).sum((1, 2)).mean()),
            "inv_dir_swap": float(_cos(gs - gM_i, Aj - Ai).mean()),
            "inv_err_swap": float(((gs - Aj) ** 2).sum((1, 2)).mean()),
            "inv_val_r2": inv_r2,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ck = Path(os.environ.get("DATA_ROOT", ROOT / "data")) / "raw" / "tdmpc2" / "dmcontrol"
    tasks = args.tasks or sorted({p.stem.rsplit("-", 1)[0] for p in ck.glob("*.pt")})
    out_path = Path(args.out or Path(os.environ.get("RESULTS_ROOT", ROOT / "results")) / "tables"
                    / "tdmpc2_validity.json")
    res = json.load(open(out_path)) if out_path.exists() else {}
    for task in tasks:
        if all(f"{task}-{s}" in res for s in args.seeds):
            continue
        t0 = time.time()
        beh = TDMPC2(ck / f"{task}-1.pt").to(args.device).eval()
        data = collect(task, beh, args.device)
        floor = np.linalg.norm(data["x_rep"] - data["x_ref"], axis=(1, 2)).mean()
        signal = np.linalg.norm(data["x_pert"] - data["x_ref"], axis=(1, 2)).mean()
        pairs = matched_pairs(data["logs"])
        meta = {"noise_floor": float(floor), "signal": float(signal),
                "behavior_return": float(np.mean(data["returns"])),
                "n_items": int(len(data["o0"])), "n_wsp": int(len(data["w_ctx"])),
                "pair_gap_median": float(np.median([g for _, _, g in pairs]))}
        for s in args.seeds:
            m = TDMPC2(ck / f"{task}-{s}.pt").to(args.device).eval()
            r = score(m, data, pairs, args.device)
            r["meta"] = meta
            res[f"{task}-{s}"] = r
            i = r["identity"]
            print(f"  {task}-{s:<3d} dir={i['direction']:+.3f} mag={i['mag_ratio']:.2f} "
                  f"C1={i['C1']:.4f} t1={i['task1']:.1f} ret={i['ret_gauss_meanrank']:.2f} "
                  f"matched={i['matched']:+.3f} inv={i['inv_dir_rand']:+.3f} "
                  f"inv_r2={i['inv_val_r2']:.2f}", flush=True)
        print(f"[tdmpc2] {task}: floor={floor:.2e} signal={signal:.3f} "
              f"return={meta['behavior_return']:.0f} ({time.time() - t0:.0f}s)", flush=True)
        json.dump(res, open(out_path, "w"), indent=1)
    print("[tdmpc2] wrote", out_path)


if __name__ == "__main__":
    main()
