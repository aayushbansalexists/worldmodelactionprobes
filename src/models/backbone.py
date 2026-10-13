from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, h, attn_mask=None):
        a, _ = self.attn(self.ln1(h), self.ln1(h), self.ln1(h), attn_mask=attn_mask, need_weights=False)
        h = h + a
        h = h + self.mlp(self.ln2(h))
        return h


class TransformerBackbone(nn.Module):
    def __init__(self, hidden_dim, n_layers, n_heads, dropout=0.0, max_len=64):
        super().__init__()
        self.blocks = nn.ModuleList(
            [Block(hidden_dim, n_heads, dropout) for _ in range(n_layers)]
        )
        self.pos = nn.Parameter(torch.zeros(1, max_len, hidden_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.max_len = max_len

    def forward(self, h, causal=True):
        T = h.shape[1]
        assert T <= self.max_len, f"seq len {T} > max_len {self.max_len}"
        h = h + self.pos[:, :T]
        mask = None
        if causal:
            mask = torch.triu(torch.ones(T, T, device=h.device, dtype=torch.bool), diagonal=1)
            mask = mask
        for blk in self.blocks:
            h = blk(h, attn_mask=mask)
        return h


def build_backbone(cfg):
    b = cfg.get("backbone", {})
    return TransformerBackbone(
        hidden_dim=cfg.get("hidden_dim", 128),
        n_layers=b.get("n_layers", 4),
        n_heads=b.get("n_heads", 4),
        dropout=b.get("dropout", 0.0),
        max_len=b.get("max_len", 64),
    )


__all__ = ["TransformerBackbone", "build_backbone"]
