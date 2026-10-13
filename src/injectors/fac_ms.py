from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ActionInjector
from .representation import VectorRep, PerDimRep, GroupedRep
from .fusion import AdaLNZeroFusion, CrossAttentionFusion, GatedFusion
from .timescale import ChunkTimescale


class FACMS(ActionInjector):
    def __init__(self, cfg):
        super().__init__()
        H = cfg["_hidden_dim"]
        A = cfg["_action_dim"]
        self.cfg = cfg
        self.R = cfg.get("R", "vector")
        self.F = "fac_ms"
        self.T = "multiscale"
        self.chunk_size = cfg.get("chunk_size", 4)
        self.shared_encoder = cfg.get("fac_ms_shared_encoder", False)

        rep_kind = cfg.get("fac_ms_rep", "vector")
        self.rep_kind = rep_kind
        def _make():
            if rep_kind == "per_dim":
                return PerDimRep(A, H)
            if rep_kind == "grouped":
                return GroupedRep(A, H, cfg.get("grouped_size", 4))
            return VectorRep(A, H)
        self.step_rep = _make()
        self.chunk_rep = self.step_rep if self.shared_encoder else _make()

        self.chunk_fusion = AdaLNZeroFusion(H)
        step_fuse = cfg.get("fac_ms_step_fusion", "cross_attention")
        self.step_fusion = CrossAttentionFusion(H, n_heads=4) if step_fuse == "cross_attention" else GatedFusion(H)

        self.chunk_ts = ChunkTimescale(self.chunk_size)

        self.gate_proj = nn.Linear(2 * H, H)
        self.gate_bias_init = 0.0
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

        self.use_chunk = cfg.get("fac_ms_use_chunk", True)
        self.use_step = cfg.get("fac_ms_use_step", True)
        self.use_gate = cfg.get("fac_ms_use_gate", True)

    def forward(self, h, action, action_mask=None):
        h_chunk = h
        if self.use_chunk:
            af = self.chunk_rep(action, action_mask=action_mask)
            af_global = self.chunk_ts(af)
            h_chunk = self.chunk_fusion(h, af_global)
        if not self.use_step:
            return h_chunk

        af_step = self.step_rep(action, action_mask=action_mask)
        h_step = self.step_fusion(h, af_step)

        if not self.use_chunk:
            return h_step

        if not self.use_gate:
            return h_chunk + h_step

        g = torch.sigmoid(self.gate_proj(torch.cat([h_chunk, h_step], dim=-1)))
        return (1 - g) * h_chunk + g * h_step


class FACMSLite(ActionInjector):

    def __init__(self, cfg):
        super().__init__()
        H = cfg["_hidden_dim"]
        A = cfg["_action_dim"]
        self.R = cfg.get("R", "vector")
        self.F = "fac_ms_lite"
        self.T = "multiscale"
        self.chunk_size = cfg.get("chunk_size", 4)
        self.warm = cfg.get("fac_ms_warm", False)
        self.bypass = cfg.get("fac_ms_bypass", False)
        self.sg = cfg.get("fac_ms_sg", False)
        if self.sg:
            self.bypass = True
        self.rep = VectorRep(A, H)
        self.chunk_ts = ChunkTimescale(self.chunk_size)
        self.add_chunk = nn.Linear(H, H)
        self.add_step = nn.Linear(H, H)
        if self.sg:
            self.gate_proj = nn.Linear(H, 1)
            self.gate = None
            nn.init.zeros_(self.gate_proj.weight)
            nn.init.zeros_(self.gate_proj.bias)
            nn.init.normal_(self.add_chunk.weight, std=0.02)
            nn.init.zeros_(self.add_chunk.bias)
        else:
            self.gate = nn.Parameter(torch.tensor(0.0))
            if self.bypass:
                nn.init.normal_(self.add_chunk.weight, std=0.02)
                nn.init.zeros_(self.add_chunk.bias)
            elif self.warm:
                nn.init.normal_(self.add_chunk.weight, std=0.02)
                nn.init.zeros_(self.add_chunk.bias)
                nn.init.normal_(self.add_step.weight, std=0.02)
                nn.init.zeros_(self.add_step.bias)
            else:
                nn.init.zeros_(self.add_chunk.weight); nn.init.zeros_(self.add_chunk.bias)
                nn.init.zeros_(self.add_step.weight); nn.init.zeros_(self.add_step.bias)

    def forward(self, h, action, action_mask=None):
        af = self.rep(action, action_mask=action_mask)
        chunk_af = self.chunk_ts(af)
        h_chunk = h + self.add_chunk(chunk_af)
        if self.bypass:
            h_step = h + af
        else:
            h_step = h + self.add_step(af)
        if self.sg:
            hf = af - chunk_af
            g = torch.sigmoid(self.gate_proj(hf))
        else:
            g = torch.sigmoid(self.gate)
        return (1 - g) * h_chunk + g * h_step


def build_fac_ms(cfg) -> ActionInjector:
    if cfg.get("fac_ms_lite", False) or cfg.get("fac_ms_warm", False) \
            or cfg.get("fac_ms_bypass", False) or cfg.get("fac_ms_sg", False):
        return FACMSLite(cfg)
    return FACMS(cfg)


__all__ = ["FACMS", "FACMSLite", "build_fac_ms"]
