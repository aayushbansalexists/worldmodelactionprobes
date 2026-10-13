import torch
import torch.nn as nn


class UnconditionalInflation(nn.Module):

    def __init__(self, base, obs_dim, act_dim, c=2.0, abar=None, generator=None):
        super().__init__()
        self.base, self.c = base, c
        g = generator or torch.Generator().manual_seed(0)
        self.register_buffer("B", torch.randn(obs_dim, act_dim, generator=g) * 0.8)
        self.register_buffer("abar", torch.zeros(act_dim) if abar is None else abar)

    def forward(self, obs, act):
        return self.base(obs, act) + self.c * (act - self.abar) @ self.B.T


class PolicyAlignedInflation(nn.Module):

    def __init__(self, base, policy, obs_dim, act_dim, c=2.0, generator=None):
        super().__init__()
        self.base, self.policy, self.c = base, policy, c
        g = generator or torch.Generator().manual_seed(0)
        self.register_buffer("B", torch.randn(obs_dim, act_dim, generator=g) * 0.8)

    def forward(self, obs, act):
        return self.base(obs, act) + self.c * (act - self.policy(obs, act)) @ self.B.T


class CausalPolicy(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden=128, layers=2, heads=4, max_len=256):
        super().__init__()
        self.eo = nn.Linear(obs_dim, hidden)
        self.ea = nn.Linear(act_dim, hidden)
        self.pos = nn.Parameter(torch.zeros(1, max_len, hidden))
        self.tr = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden, heads, hidden * 4, dropout=0.0,
                                       batch_first=True), layers)
        self.dec = nn.Linear(hidden, act_dim)

    def forward(self, obs, act):
        T = obs.shape[1]
        a_prev = torch.zeros_like(act)
        a_prev[:, 1:] = act[:, :-1]
        h = self.eo(obs) + self.ea(a_prev) + self.pos[:, :T]
        m = torch.triu(torch.ones(T, T, device=obs.device, dtype=torch.bool), 1)
        return self.dec(self.tr(h, mask=m))


def fit_policy(obs, act, steps=1500, batch=64, lr=3e-4, hidden=128, seed=0, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    m = CausalPolicy(obs.shape[-1], act.shape[-1], hidden=hidden).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        i = torch.randint(0, obs.shape[0], (batch,), generator=g)
        o, a = obs[i].to(device), act[i].to(device)
        loss = ((m(o, a) - a) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()


@torch.no_grad()
def policy_r2(policy, obs, act, device=None):
    device = device or next(policy.parameters()).device
    o, a = obs.to(device), act.to(device)
    pred = policy(o, a)
    ss_res = ((pred - a) ** 2).sum().item()
    ss_tot = ((a - a.mean()) ** 2).sum().item()
    return 1.0 - ss_res / max(ss_tot, 1e-12)
