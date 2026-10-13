from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class Fusion(nn.Module):
    def forward(self, h: torch.Tensor, af: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class AddFusion(Fusion):

    def __init__(self, hidden_dim):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(1))

    def forward(self, h, af):
        return h + self.gain * af


class FilmFusion(Fusion):

    def __init__(self, hidden_dim):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, 2 * hidden_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h, af):
        s, b = self.proj(af).chunk(2, dim=-1)
        return (1 + s) * self.ln(h) + b


class AdaLNZeroFusion(Fusion):

    def __init__(self, hidden_dim):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, 3 * hidden_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h, af):
        shift, scale, gate = self.proj(af).chunk(3, dim=-1)
        m = self.ln(h) * (1 + scale) + shift
        return h + gate * m


class CrossAttentionFusion(Fusion):

    def __init__(self, hidden_dim, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.ln_q = nn.LayerNorm(hidden_dim)

    def forward(self, h, af):
        B, T, C = h.shape
        q = self.q(self.ln_q(h))
        k = self.k(af)
        v = self.v(af)
        hd = C // self.n_heads
        q = q.view(B, T, self.n_heads, hd).transpose(1, 2)
        k = k.view(B, T, self.n_heads, hd).transpose(1, 2)
        v = v.view(B, T, self.n_heads, hd).transpose(1, 2)
        causal = torch.triu(torch.ones(T, T, device=h.device, dtype=torch.bool), diagonal=1)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=causal)
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        return h + self.out(attn)


class GatedFusion(Fusion):

    def __init__(self, hidden_dim):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        self.gate_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h, af):
        g = torch.sigmoid(self.gate_proj(af))
        return g * self.ln(h) + (1 - g) * af


_FUS = {
    "add": AddFusion,
    "film": FilmFusion,
    "adaln_zero": AdaLNZeroFusion,
    "cross_attention": CrossAttentionFusion,
    "gated": GatedFusion,
}


def build_fusion(cfg) -> Fusion:
    name = cfg.get("F", "add")
    a = _FUS[name]
    return a(cfg["_hidden_dim"])


__all__ = ["Fusion", "build_fusion", *_FUS.keys()]
