import numpy as np
import torch
import torch.nn as nn

from wmprobe import (agm_loss, shuffle_actions, zero_action_divergence,
                     downstream_action_ranking)


def make_data(n_traj=200, length=60, obs_dim=8, act_dim=4, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((obs_dim, obs_dim)) * 0.2
    B = rng.standard_normal((obs_dim, act_dim)) * 0.5
    obs = np.zeros((n_traj, length, obs_dim), np.float32)
    act = np.zeros((n_traj, length, act_dim), np.float32)
    for i in range(n_traj):
        s = rng.standard_normal(obs_dim) * 0.3
        for t in range(length):
            a = np.sin(0.1 * t + rng.standard_normal(act_dim)).astype(np.float32)
            s = np.tanh(A @ s + B @ a) + rng.standard_normal(obs_dim) * 0.05
            obs[i, t] = s; act[i, t] = a
    return torch.from_numpy(obs), torch.from_numpy(act)


class WorldModel(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.enc = nn.Linear(obs_dim, hidden)
        self.act = nn.Linear(act_dim, hidden)
        self.tr = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden, 4, batch_first=True), num_layers=2)
        self.dec = nn.Linear(hidden, obs_dim)

    def forward(self, obs, act):
        L = obs.shape[1]
        h = self.enc(obs) + self.act(act)
        mask = torch.triu(torch.ones(L, L, device=obs.device, dtype=torch.bool), 1)
        return self.dec(self.tr(h, mask=mask))


def train(model, data, use_agm, shuffle, steps=600, lr=3e-4):
    obs, act = data
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for step in range(steps):
        i = torch.randint(0, obs.shape[0], (32,))
        o, a = obs[i], act[i]
        if shuffle:
            a = shuffle_actions(a)
        pred = model(o[:, :-1], a[:, :-1])
        loss = ((pred - o[:, 1:]) ** 2).mean()
        if use_agm:
            loss = loss + 0.5 * agm_loss(model, o[:, :-1], a[:, :-1], eps=0.1)
        opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()


def probe_all(model, obs, act, tag):
    c1 = zero_action_divergence(model, obs, act, ctx_len=8)
    r = downstream_action_ranking(model, obs, act, ctx_len=8)
    print(f"{tag:30s} c1={c1:.3f}  retrieval excess={r['excess']:+.3f} "
          f"(top1={r['top1']:.3f}, null={r['null']:.3f}, mean rank={r['mean_rank']:.1f})")
    return c1, r


def main():
    torch.manual_seed(0)
    data = make_data()
    obs, act = data[0][0], data[1][0]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    obs, act = obs.to(device), act.to(device)
    data = (data[0].to(device), data[1].to(device))

    m_base = train(WorldModel(8, 4).to(device), data, use_agm=False, shuffle=False)
    m_agm = train(WorldModel(8, 4).to(device), data, use_agm=True, shuffle=False)

    print("causal actions:")
    probe_all(m_base, obs, act, "  baseline (mse)")
    probe_all(m_agm, obs, act, "  baseline + agm")

    print("shuffled actions (genuineness test):")
    m_base_s = train(WorldModel(8, 4).to(device), data, use_agm=False, shuffle=True)
    m_agm_s = train(WorldModel(8, 4).to(device), data, use_agm=True, shuffle=True)
    probe_all(m_base_s, obs, act, "  baseline (mse), shuffled")
    probe_all(m_agm_s, obs, act, "  baseline + agm, shuffled")

    print()
    print("how to read this. compare EXCESS over the null, not top1 against 1/16. a gain that")
    print("survives the shuffled condition is an artifact. a gain that collapses has passed one")
    print("negative control, which is necessary and not sufficient. run examples/adversary.py")
    print("to see what these probes hand out to a model with no dynamics at all.")


if __name__ == "__main__":
    main()
