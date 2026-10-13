from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "disabled")

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.tdmpc2_probe import TDMPC2, Sim, SymLog, _mlp, make_env, obs_vec


def reward(head, z, a):
    x = (torch.softmax(head(torch.cat([z, a], -1)), -1)
         * torch.linspace(-10, 10, 101, device=z.device)).sum(-1)
    return torch.sign(x) * (torch.exp(x.abs()) - 1)


@torch.no_grad()
def check(m, head, task, device, steps=300, seed=0):
    env = make_env(task, seed)
    sim = Sim(env)
    rng = np.random.default_rng(seed)
    o = obs_vec(env.reset().observation)
    O, A, Rw = [o], [], []
    for _ in range(steps):
        z = m.encode(torch.from_numpy(o)[None].to(device))
        a = np.clip(m.act(z)[0].cpu().numpy() + rng.normal(0, 0.3, m.adim), -1, 1).astype(np.float32)
        o = sim.step(a)
        O.append(o); A.append(a); Rw.append(sim.reward)
    O = torch.tensor(np.stack(O), device=device)
    A = torch.tensor(np.stack(A), device=device)
    Rw = np.array(Rw)
    z = m.encode(O)
    r1 = reward(head, z[:-1], A).cpu().numpy()
    z2 = m.next(m.next(z[:-3], A[:-2]), A[1:-1])
    r3 = reward(head, z2, A[2:]).cpu().numpy()
    err = (m.next(z[:-1], A) - z[1:]).norm(dim=-1).mean().item()
    move = (z[1:] - z[:-1]).norm(dim=-1).mean().item()
    c = lambda x, y: float(np.corrcoef(x, y)[0, 1]) if np.std(y) > 1e-9 and np.std(x) > 1e-9 else float("nan")
    return {"reward_corr": c(r1, Rw), "reward_corr_2step": c(r3, Rw[2:]),
            "consistency": err / max(move, 1e-9), "return": float(Rw.sum())}


def main():
    torch.set_num_threads(2)
    device = "cuda"
    ck = Path(os.environ.get("DATA_ROOT", ROOT / "data")) / "raw" / "tdmpc2" / "dmcontrol"
    out_path = Path(os.environ.get("RESULTS_ROOT", ROOT / "results")) / "tables" / "tdmpc2_verify.json"
    res = json.load(open(out_path)) if out_path.exists() else {}
    for p in sorted(ck.glob("*.pt")):
        name = p.stem
        if name in res:
            continue
        task = name.rsplit("-", 1)[0]
        sd = torch.load(p, map_location="cpu", weights_only=False)["model"]
        head = _mlp(sd, "_reward.", False).to(device).eval()
        m = TDMPC2(p).to(device).eval()
        old = isinstance(m.enc[0], SymLog)
        r = {"old_layout": old, "loaded": check(m, head, task, device)}
        if old:
            m.enc[0] = nn.Identity()
            r["without_symlog"] = check(m, head, task, device)
        res[name] = r
        L = r["loaded"]
        msg = (f"  {name:28s} reward {L['reward_corr']:+.3f} via 2 steps {L['reward_corr_2step']:+.3f} "
               f"consistency {L['consistency']:.3f}")
        if old:
            W = r["without_symlog"]
            msg += f" | without symlog: reward {W['reward_corr']:+.3f} consistency {W['consistency']:.3f}"
        print(msg, flush=True)
        json.dump(res, open(out_path, "w"), indent=1)
    print("[verify] wrote", out_path)


if __name__ == "__main__":
    main()
