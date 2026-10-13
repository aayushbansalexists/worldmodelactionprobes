from __future__ import annotations
import torch
import torch.nn as nn


class ActionInjector(nn.Module):

    def forward(self, h: torch.Tensor, action: torch.Tensor, action_mask=None) -> torch.Tensor:
        raise NotImplementedError

    def config_tuple(self):
        return (self.R, self.F, self.T, getattr(self, "W", "?"))

    def injector_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_injector(cfg) -> ActionInjector:
    from . import representation as R_mod
    from . import fusion as F_mod
    from . import timescale as T_mod

    rep = R_mod.build_representation(cfg)
    ts = T_mod.build_timescale(cfg)
    fus = F_mod.build_fusion(cfg)
    return ComposedInjector(rep, ts, fus, cfg)


class ComposedInjector(ActionInjector):

    def __init__(self, rep, ts, fus, cfg):
        super().__init__()
        self.rep = rep
        self.ts = ts
        self.fus = fus
        self.R = cfg.get("R", "vector")
        self.F = cfg.get("F", "add")
        self.T = cfg.get("T", "step")

    def forward(self, h, action, action_mask=None):
        af = self.rep(action, action_mask=action_mask)
        af = self.ts(af)
        return self.fus(h, af)


__all__ = ["ActionInjector", "ComposedInjector", "build_injector"]
