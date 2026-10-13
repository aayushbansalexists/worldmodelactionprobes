from __future__ import annotations
import torch
import torch.nn as nn

from .encoder import build_encoder
from .backbone import build_backbone
from injectors.base import build_injector
from injectors.placement import resolve_placement
from injectors.fac_ms import build_fac_ms


class WorldModel(nn.Module):
    def __init__(self, cfg, obs_dim, action_dim):
        super().__init__()
        self.cfg = cfg
        H = cfg.get("model", {}).get("hidden_dim", 128)
        self.H = H
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        inj_cfg = dict(cfg.get("injector", {}))
        inj_cfg["_action_dim"] = action_dim
        inj_cfg["_hidden_dim"] = H
        self.W = resolve_placement(inj_cfg)
        self.every_k = inj_cfg.get("W_every_k", 2)
        inj_cfg["W"] = self.W
        self.inj_cfg = inj_cfg

        if inj_cfg.get("F") in ("fac_ms", "fac_ms_lite", "fac_ms_warm", "fac_ms_bypass", "fac_ms_sg"):
            if inj_cfg.get("F") == "fac_ms_warm":
                inj_cfg["fac_ms_warm"] = True
            elif inj_cfg.get("F") == "fac_ms_bypass":
                inj_cfg["fac_ms_bypass"] = True
            elif inj_cfg.get("F") == "fac_ms_sg":
                inj_cfg["fac_ms_sg"] = True
            self.injector = build_fac_ms(inj_cfg)
        else:
            self.injector = build_injector(inj_cfg)

        self.encoder = build_encoder(cfg.get("model", {}).get("encoder", {}), obs_dim, H)
        self.backbone = build_backbone(cfg.get("model", {}))
        self.decoder = nn.Linear(H, obs_dim)
        self.out_ln = nn.LayerNorm(H)
        self.dab = cfg.get("model", {}).get("decoder_action_bypass", False)
        if self.dab:
            self.action_bypass = nn.Linear(action_dim, obs_dim)
            nn.init.normal_(self.action_bypass.weight, std=0.02)
            nn.init.zeros_(self.action_bypass.bias)

        self.delta_prediction = cfg.get("model", {}).get("delta_prediction", False)
        self.action_blind = cfg.get("model", {}).get("action_blind", False)

        self.idm = float(cfg.get("train", {}).get("inverse_dynamics", 0.0) or 0.0) > 0
        if self.idm:
            self.idm_head = nn.Sequential(
                nn.Linear(2 * obs_dim, H), nn.SiLU(), nn.Linear(H, action_dim))

    def forward(self, obs, action, action_mask=None):
        L = obs.shape[1]
        assert action.shape[1] == L, f"obs/action length mismatch: {L} vs {action.shape[1]}"
        if self.action_blind:
            action = torch.zeros_like(action)
        h = self.encoder(obs)
        if self.W == "early":
            h = self.injector(h, action, action_mask=action_mask)
        T = h.shape[1]
        h = h + self.backbone.pos[:, :T]
        device = h.device
        causal = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)
        for i, blk in enumerate(self.backbone.blocks):
            h = blk(h, attn_mask=causal)
            if self.W == "every_block":
                h = self.injector(h, action, action_mask=action_mask)
            elif self.W == "every_k" and (i + 1) % self.every_k == 0:
                h = self.injector(h, action, action_mask=action_mask)
        h = self.out_ln(h)
        if self.W == "late":
            h = self.injector(h, action, action_mask=action_mask)
        pred = self.decoder(h)
        if self.dab:
            pred = pred + self.action_bypass(action)
        if self.delta_prediction:
            pred = obs + pred
        out = {"pred": pred, "h": h}
        if self.idm:
            out["a_hat"] = self.idm_head(torch.cat([obs, pred], dim=-1))
        return out

    def next_step_loss(self, pred, obs_full):
        y = obs_full[:, 1:]
        return ((pred - y) ** 2).mean()

    def factor_tuple(self):
        inj = self.inj_cfg
        return {
            "R": inj.get("R"), "F": inj.get("F"), "T": inj.get("T"), "W": self.W,
            "W_every_k": self.every_k,
        }

    def param_counts(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        injector = sum(p.numel() for p in self.injector.parameters() if p.requires_grad)
        enc = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        bb = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        dec = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        return {"total": total, "injector": injector, "encoder": enc, "backbone": bb, "decoder": dec}

    def estimate_flops(self, B=1, L=16):
        H = self.H
        f = 2 * self.obs_dim * H
        f += self.cfg["model"]["backbone"]["n_layers"] * (L * L * H + L * H * H * 4 + L * H * 4 * H * 2)
        f += 2 * self.action_dim * H * 2
        f += 2 * H * self.obs_dim
        return f * B


def F_mse(p, y):
    return ((p - y) ** 2)


__all__ = ["WorldModel"]
