import numpy as np
import torch
import torch.nn as nn

from wmprobe import (UnconditionalInflation, PolicyAlignedInflation, fit_policy, policy_r2,
                     zero_action_divergence, downstream_action_ranking)

OBS, ACT, HID, L = 8, 4, 128, 12


def make_world(n_traj=300, T=40, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((OBS, OBS)) * 0.3
    B = rng.standard_normal((OBS, ACT)) * 0.8
    C = rng.standard_normal((ACT, OBS)) * 0.5
    obs = np.zeros((n_traj, T, OBS), np.float32)
    act = np.zeros((n_traj, T, ACT), np.float32)
    for i in range(n_traj):
        s = rng.standard_normal(OBS) * 0.5
        for t in range(T):
            a = np.tanh(C @ s) + rng.standard_normal(ACT) * 0.15
            obs[i, t], act[i, t] = s, a
            s = np.tanh(W @ s) + B @ a + rng.standard_normal(OBS) * 0.05
    return torch.from_numpy(obs), torch.from_numpy(act)


class WM(nn.Module):
    def __init__(self):
        super().__init__()
        self.eo, self.ea = nn.Linear(OBS, HID), nn.Linear(ACT, HID)
        self.pos = nn.Parameter(torch.zeros(1, 64, HID))
        self.tr = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(HID, 4, HID * 4, dropout=0.0, batch_first=True), 2)
        self.dec = nn.Linear(HID, OBS)

    def forward(self, obs, act):
        T = obs.shape[1]
        h = self.eo(obs) + self.ea(act) + self.pos[:, :T]
        m = torch.triu(torch.ones(T, T, device=obs.device, dtype=torch.bool), 1)
        return self.dec(self.tr(h, mask=m))


def train(obs, act, steps=1200, seed=0):
    torch.manual_seed(seed)
    m = WM()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        i = torch.randint(0, obs.shape[0], (64,), generator=g)
        o, a = obs[i], act[i]
        loss = ((m(o[:, :-1], a[:, :-1]) - o[:, 1:]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()


def main():
    obs, act = make_world()
    tr_o, tr_a = obs[:240], act[:240]
    te_o, te_a = obs[240:], act[240:]

    print("training the real model ...")
    base = train(tr_o, tr_a)
    print("training a behavior policy predictor (the adversary's fuel) ...")
    pi = fit_policy(tr_o, tr_a, device="cpu")
    print(f"policy predictability R2 = {policy_r2(pi, te_o, te_a):.3f}\n")

    models = {
        "real model (MSE)": base,
        "unconditional inflation": UnconditionalInflation(base, OBS, ACT, c=2.0).eval(),
        "policy aligned inflation": PolicyAlignedInflation(base, pi, OBS, ACT, c=2.0).eval(),
    }
    print(f"{'model':26s} {'c1':>8s} {'top1':>7s} {'null':>7s} {'excess':>8s} {'mean rank':>10s}")
    for name, m in models.items():
        c1 = np.mean([zero_action_divergence(m, te_o[i], te_a[i], ctx_len=L) for i in range(8)])
        rs = [downstream_action_ranking(m, te_o[i], te_a[i], ctx_len=L, n_positions=6)
              for i in range(8)]
        f = lambda k: float(np.nanmean([r[k] for r in rs]))
        print(f"{name:26s} {c1:8.3f} {f('top1'):7.3f} {f('null'):7.3f} "
              f"{f('excess'):+8.3f} {f('mean_rank'):10.2f}")

    print()
    print("read it this way. both adversaries contain zero learned dynamics beyond the real")
    print("model they wrap, so every point they score is a point the probe gives away. c1 goes")
    print("up for both, because it only measures response size. the ranking probe moves only")
    print("for the policy aligned one, because what it really rewards is knowing what the")
    print("demonstrator did. and note the gap between raw top1 and excess: against a nominal")
    print("chance of 1/16 = 0.0625 an adversary can look excellent while its excess is nil.")


if __name__ == "__main__":
    main()
