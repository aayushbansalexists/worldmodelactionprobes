import numpy as np
import torch
import torch.nn as nn

from wmprobe import matched_probe, transform_matrix

L, H = 8, 3


def make_data(n_traj=300, length=40, obs_dim=6, act_dim=3, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((obs_dim, obs_dim)) * 0.3
    B = rng.standard_normal((obs_dim, act_dim)) * 0.6
    C = rng.standard_normal((act_dim, obs_dim)) * 0.5
    obs = np.zeros((n_traj, length, obs_dim), np.float32)
    act = np.zeros((n_traj, length, act_dim), np.float32)
    for i in range(n_traj):
        s = rng.standard_normal(obs_dim) * 0.5
        for t in range(length):
            a = np.tanh(C @ s) + rng.standard_normal(act_dim) * 0.5
            obs[i, t], act[i, t] = s, a
            s = np.tanh(A @ s) + B @ a + rng.standard_normal(obs_dim) * 0.05
    return torch.from_numpy(obs), torch.from_numpy(act)


class WorldModel(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=64, blind=False):
        super().__init__()
        self.blind = blind
        self.net = nn.GRU(obs_dim + act_dim, hidden, batch_first=True)
        self.out = nn.Linear(hidden, obs_dim)

    def forward(self, obs, act):
        if self.blind:
            act = torch.zeros_like(act)
        h, _ = self.net(torch.cat([obs, act], -1))
        return obs + self.out(h)


def train(model, obs, act, steps=600):
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(steps):
        i = torch.randint(0, obs.shape[0], (64,))
        pred = model(obs[i, :-1], act[i, :-1])
        loss = ((pred - obs[i, 1:]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()


def main():
    torch.manual_seed(0)
    obs, act = make_data()
    tr, te = slice(0, 240), slice(240, 300)
    model = train(WorldModel(obs.shape[-1], act.shape[-1]), obs[tr], act[tr])
    blind = train(WorldModel(obs.shape[-1], act.shape[-1], blind=True), obs[tr], act[tr])

    O, A = obs[te], act[te]
    seg = [(i, t) for i in range(O.shape[0]) for t in range(L, O.shape[1] - H, 4)]
    ctx_obs = torch.stack([O[i, t - L + 1:t + 1] for i, t in seg])
    ctx_act = torch.stack([A[i, t - L + 1:t + 1] for i, t in seg])
    fut_act = torch.stack([A[i, t:t + H] for i, t in seg])
    fut_obs = torch.stack([O[i, t + 1:t + H + 1] for i, t in seg])
    traj = np.array([i for i, _ in seg])

    base = matched_probe(model, ctx_obs, ctx_act, fut_act, fut_obs, traj)
    pairs = (base["idx_i"], base["idx_j"])
    print(f"{len(pairs[0])} matched pairs, median standardized start state gap {np.median(base['gap']):.2f}")
    print(f"  trained model      {base['score']:+.3f}")
    for name in ["reflect", "rotate", "scale_0.25", "scale_4"]:
        T = transform_matrix(name, obs.shape[-1])
        r = matched_probe(model, ctx_obs, ctx_act, fut_act, fut_obs, traj, transform=T, pairs=pairs)
        print(f"  {name:18s} {r['score']:+.3f}")
    r = matched_probe(blind, ctx_obs, ctx_act, fut_act, fut_obs, traj, pairs=pairs)
    print(f"  action blind model {r['score']:+.3f}")


if __name__ == "__main__":
    main()
