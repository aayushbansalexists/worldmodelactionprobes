from __future__ import annotations
import torch
import torch.nn as nn


class LinearEncoder(nn.Module):
    def __init__(self, obs_dim, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(obs_dim, hidden_dim)

    def forward(self, obs):
        return self.proj(obs)


class MLPEncoder(nn.Module):
    def __init__(self, obs_dim, hidden_dim, depth=2):
        super().__init__()
        layers = []
        d = obs_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(d, hidden_dim), nn.SiLU()]
            d = hidden_dim
        layers += [nn.Linear(d, hidden_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        return self.net(obs)


def build_encoder(cfg, obs_dim, hidden_dim):
    t = cfg.get("type", "linear")
    if t == "linear":
        return LinearEncoder(obs_dim, hidden_dim)
    if t == "mlp":
        return MLPEncoder(obs_dim, hidden_dim)
    raise ValueError(f"unknown encoder type {t!r}")


__all__ = ["build_encoder", "LinearEncoder", "MLPEncoder"]
