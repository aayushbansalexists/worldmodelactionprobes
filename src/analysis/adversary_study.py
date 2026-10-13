from __future__ import annotations
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OBS_D, ACT_D, HID, L = 8, 4, 128, 12


def gen_world(kind, n_traj=400, T=40, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((OBS_D, OBS_D)) * 0.3
    Bt = rng.standard_normal((OBS_D, ACT_D)) * 0.8
    C = rng.standard_normal((ACT_D, OBS_D)) * 0.5
    obs = np.zeros((n_traj, T, OBS_D), np.float32)
    act = np.zeros((n_traj, T, ACT_D), np.float32)
    for i in range(n_traj):
        s = rng.standard_normal(OBS_D) * 0.5
        for t in range(T):
            a = np.tanh(C @ s) + rng.standard_normal(ACT_D) * 0.15
            nxt = np.tanh(W @ s) + rng.standard_normal(OBS_D) * 0.05
            if kind == "G":
                nxt = nxt + Bt @ a
            obs[i, t] = s
            act[i, t] = a
            s = nxt
    B_eff = Bt if kind == "G" else np.zeros_like(Bt)
    return torch.from_numpy(obs), torch.from_numpy(act), torch.from_numpy(B_eff.astype(np.float32))


class WM(nn.Module):

    def __init__(self):
        super().__init__()
        self.eo = nn.Linear(OBS_D, HID)
        self.ea = nn.Linear(ACT_D, HID)
        self.pos = nn.Parameter(torch.zeros(1, 64, HID))
        self.tr = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(HID, 4, HID * 4, dropout=0.0, batch_first=True), 2)
        self.dec = nn.Linear(HID, OBS_D)

    def forward(self, obs, act):
        T = obs.shape[1]
        h = self.eo(obs) + self.ea(act) + self.pos[:, :T]
        m = torch.triu(torch.ones(T, T, device=obs.device, dtype=torch.bool), 1)
        return self.dec(self.tr(h, mask=m))


class Policy(nn.Module):

    def __init__(self):
        super().__init__()
        self.eo = nn.Linear(OBS_D, HID)
        self.ea = nn.Linear(ACT_D, HID)
        self.pos = nn.Parameter(torch.zeros(1, 64, HID))
        self.tr = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(HID, 4, HID * 4, dropout=0.0, batch_first=True), 2)
        self.dec = nn.Linear(HID, ACT_D)

    def forward(self, obs, act):
        T = obs.shape[1]
        a_prev = torch.zeros_like(act)
        a_prev[:, 1:] = act[:, :-1]
        h = self.eo(obs) + self.ea(a_prev) + self.pos[:, :T]
        m = torch.triu(torch.ones(T, T, device=obs.device, dtype=torch.bool), 1)
        return self.dec(self.tr(h, mask=m))


class Adversary(nn.Module):

    def __init__(self, base, c, B, pi=None, abar=None):
        super().__init__()
        self.base, self.c, self.pi = base, c, pi
        self.register_buffer("B", B)
        self.register_buffer("abar", abar if abar is not None else torch.zeros(ACT_D))

    def forward(self, obs, act):
        out = self.base(obs, act)
        ref = self.pi(obs, act) if self.pi is not None else self.abar.expand_as(act)
        return out + self.c * (act - ref) @ self.B.T


def train_wm(obs, act, agm=0.0, steps=1500, seed=0):
    torch.manual_seed(seed)
    m = WM().to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(steps):
        i = torch.randint(0, obs.shape[0], (64,), generator=g)
        o, a = obs[i].to(DEV), act[i].to(DEV)
        pred = m(o[:, :-1], a[:, :-1])
        loss = ((pred - o[:, 1:]) ** 2).mean()
        if agm > 0:
            pp = m(o[:, :-1], a[:, :-1] + torch.randn_like(a[:, :-1]) * 0.1)
            loss = loss - agm * ((pp - pred) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()


def train_policy(obs, act, steps=1500, seed=0):
    torch.manual_seed(seed + 777)
    m = Policy().to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(steps):
        i = torch.randint(0, obs.shape[0], (64,), generator=g)
        o, a = obs[i].to(DEV), act[i].to(DEV)
        loss = ((m(o, a) - a) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()


@torch.no_grad()
def retrieval(model, obs, act, n_cand=16, n_pos=6, seed=0, null=False):
    rng = np.random.default_rng(seed)
    hits = tot = 0
    for i in range(obs.shape[0]):
        o, a = obs[i].to(DEV), act[i].to(DEV)
        T = o.shape[0]
        for p in np.linspace(0, T - L - 2, n_pos).astype(int):
            co, ca = o[p:p + L][None], a[p:p + L][None]
            true_a = a[p + L - 1]
            if null:
                q = int(rng.integers(0, obs.shape[0]))
                r = int(rng.integers(0, T - L - 1))
                target = obs[q][r + L].to(DEV)
            else:
                target = o[p + L]
            cand = true_a.repeat(n_cand, 1)
            cand[1:] = true_a + torch.from_numpy(
                rng.standard_normal((n_cand - 1, ACT_D)).astype(np.float32) * 0.5).to(DEV)
            d = []
            for c in range(n_cand):
                cc = ca.clone(); cc[0, -1] = cand[c]
                d.append(((model(co, cc)[0, -1] - target) ** 2).mean().item())
            d = np.array(d)
            d = d + rng.standard_normal(n_cand) * 1e-9 * (np.abs(d).mean() + 1e-12)
            hits += int(d.argmin() == 0); tot += 1
    return hits / max(tot, 1)


@torch.no_grad()
def c1_zero_action(model, obs, act, n=32):
    v = []
    for i in range(min(n, obs.shape[0])):
        o, a = obs[i][:L][None].to(DEV), act[i][:L][None].to(DEV)
        v.append(((model(o, a)[0, -1] - model(o, torch.zeros_like(a))[0, -1]) ** 2).mean().item())
    return float(np.mean(v))


@torch.no_grad()
def gt_direction(model, obs, act, B_eff, n=64, n_dirs=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    cs = []
    for i in range(min(n, obs.shape[0])):
        o, a = obs[i][:L][None].to(DEV), act[i][:L][None].to(DEV)
        base = model(o, a)[0, -1]
        for _ in range(n_dirs):
            d = torch.randn(ACT_D, generator=g).to(DEV) * 0.5
            ap = a.clone(); ap[0, -1] += d
            r = model(o, ap)[0, -1] - base
            t = B_eff.to(DEV) @ d
            cs.append(float((r @ t) / (r.norm() * t.norm() + 1e-12)))
    return float(np.mean(cs))


@torch.no_grad()
def response_size(model, obs, act, n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    v = []
    for i in range(min(n, obs.shape[0])):
        o, a = obs[i][:L][None].to(DEV), act[i][:L][None].to(DEV)
        ap = a.clone(); ap[0, -1] += torch.randn(ACT_D, generator=g).to(DEV) * 0.5
        v.append(float((model(o, ap)[0, -1] - model(o, a)[0, -1]).norm()))
    return float(np.mean(v))


@torch.no_grad()
def one_step_mse(model, obs, act, n=64):
    v = []
    for i in range(min(n, obs.shape[0])):
        o, a = obs[i][None].to(DEV), act[i][None].to(DEV)
        v.append(((model(o[:, :-1], a[:, :-1]) - o[:, 1:]) ** 2).mean().item())
    return float(np.mean(v))


def run(seed=0):
    out = {}
    for world in ["G", "N"]:
        obs, act, B_eff = gen_world(world, seed=seed)
        ntr = int(0.8 * obs.shape[0])
        tr_o, tr_a = obs[:ntr], act[:ntr]
        te_o, te_a = obs[ntr:], act[ntr:]
        torch.manual_seed(seed)
        B = (torch.randn(OBS_D, ACT_D) * 0.8).to(DEV)
        abar = tr_a.reshape(-1, ACT_D).mean(0).to(DEV)

        for labels in ["causal", "shuffled"]:
            a_tr = tr_a
            if labels == "shuffled":
                k = int(np.random.default_rng(seed + 1).integers(1, tr_a.shape[0]))
                a_tr = torch.roll(tr_a, shifts=k, dims=0)

            f_base = train_wm(tr_o, a_tr, agm=0.0, seed=seed)
            f_agm = train_wm(tr_o, a_tr, agm=0.5, seed=seed)
            f_blind = train_wm(tr_o, torch.zeros_like(a_tr), agm=0.0, seed=seed)
            pihat = train_policy(tr_o, a_tr, seed=seed)

            models = {
                "mse": f_base,
                "agm": f_agm,
                "blind": f_blind,
                "adv_const_c2": Adversary(f_blind, 2.0, B, pi=None, abar=abar).to(DEV).eval(),
                "adv_policy_c2": Adversary(f_blind, 2.0, B, pi=pihat).to(DEV).eval(),
                "adv_policy_c8": Adversary(f_blind, 8.0, B, pi=pihat).to(DEV).eval(),
                "adv_policy_c2_on_trained": Adversary(f_base, 2.0, B, pi=pihat).to(DEV).eval(),
            }
            for name, m in models.items():
                key = f"{world}/{labels}/{name}"
                out[key] = {
                    "retrieval": retrieval(m, te_o, te_a, seed=seed),
                    "retrieval_null": retrieval(m, te_o, te_a, seed=seed, null=True),
                    "C1": c1_zero_action(m, te_o, te_a),
                    "mse": one_step_mse(m, te_o, te_a),
                    "response": response_size(m, te_o, te_a, seed=seed),
                    "direction_gt": gt_direction(m, te_o, te_a, B_eff, seed=seed)
                    if world == "G" else float("nan"),
                }
                r = out[key]
                print(f"  {key:38s} ret={r['retrieval']:.3f} null={r['retrieval_null']:.3f} "
                      f"C1={r['C1']:.4f} mse={r['mse']:.4f} resp={r['response']:.3f} "
                      f"dir={r['direction_gt']:+.3f}", flush=True)
        p = train_policy(tr_o, tr_a, seed=seed)
        with torch.no_grad():
            pa = p(te_o.to(DEV), te_a.to(DEV))
            ss_res = ((pa - te_a.to(DEV)) ** 2).sum().item()
            ss_tot = ((te_a.to(DEV) - te_a.to(DEV).mean()) ** 2).sum().item()
        out[f"{world}/policy_r2"] = 1 - ss_res / ss_tot
    return out


if __name__ == "__main__":
    allr = {}
    for s in [0, 1, 2, 3, 4]:
        print(f"=== seed {s} ===", flush=True)
        allr[s] = run(seed=s)
    p = Path(os.environ.get("RESULTS_ROOT", "results")) / "tables" / "adversary_study_v2.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(allr, open(p, "w"), indent=2)
    print("wrote", p)
