from __future__ import annotations
import torch
import torch.nn as nn


class Representation(nn.Module):
    def forward(self, action: torch.Tensor, action_mask=None) -> torch.Tensor:
        raise NotImplementedError


class VectorRep(Representation):

    def __init__(self, action_dim, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(action_dim, hidden_dim)
        self.act = nn.SiLU()

    def forward(self, action, action_mask=None):
        return self.act(self.proj(action))


class PerDimRep(Representation):

    def __init__(self, action_dim, hidden_dim, d_per=None):
        super().__init__()
        d_per = d_per or max(8, hidden_dim // 4)
        self.d_per = d_per
        self.per_dim = nn.Linear(1, d_per)
        self.out = nn.Linear(action_dim * d_per, hidden_dim)
        self.act = nn.SiLU()
        self.action_dim = action_dim

    def forward(self, action, action_mask=None):
        B, T, A = action.shape
        x = action.unsqueeze(-1)
        x = self.act(self.per_dim(x))
        x = x.reshape(B, T, A * self.d_per)
        return self.out(x)


class GroupedRep(Representation):

    def __init__(self, action_dim, hidden_dim, grouped_size=4):
        super().__init__()
        self.grouped_size = grouped_size
        self.padded_dim = ((action_dim + grouped_size - 1) // grouped_size) * grouped_size
        self.n_groups = self.padded_dim // grouped_size
        self.action_dim = action_dim
        d_g = max(16, hidden_dim // self.n_groups)
        self.d_g = d_g
        self.in_proj = nn.Linear(grouped_size, d_g)
        self.out = nn.Linear(self.n_groups * d_g, hidden_dim)
        self.act = nn.SiLU()

    def forward(self, action, action_mask=None):
        B, T, A = action.shape
        if A < self.padded_dim:
            pad = torch.zeros(B, T, self.padded_dim - A, device=action.device, dtype=action.dtype)
            action = torch.cat([action, pad], dim=-1)
        x = action.view(B, T, self.n_groups, self.grouped_size)
        x = self.act(self.in_proj(x))
        x = x.reshape(B, T, self.n_groups * self.d_g)
        return self.out(x)


class RelativeDeltaRep(Representation):

    def __init__(self, action_dim, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(action_dim, hidden_dim)
        self.act = nn.SiLU()

    def forward(self, action, action_mask=None):
        delta = action.clone()
        delta[:, 1:] = action[:, 1:] - action[:, :-1]
        return self.act(self.proj(delta))


_REP = {
    "vector": VectorRep,
    "per_dim": PerDimRep,
    "grouped": GroupedRep,
    "relative_delta": RelativeDeltaRep,
}


def build_representation(cfg) -> Representation:
    name = cfg.get("R", "vector")
    a = _REP[name]
    if name == "grouped":
        return a(cfg["_action_dim"], cfg["_hidden_dim"], cfg.get("grouped_size", 4))
    return a(cfg["_action_dim"], cfg["_hidden_dim"])


__all__ = ["Representation", "build_representation", *_REP.keys()]
