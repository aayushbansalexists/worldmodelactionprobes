from __future__ import annotations
import torch
import torch.nn as nn


class Timescale(nn.Module):
    def forward(self, af: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class StepTimescale(Timescale):
    def forward(self, af):
        return af


class ChunkTimescale(Timescale):
    def __init__(self, chunk_size=4):
        super().__init__()
        self.chunk_size = chunk_size

    def forward(self, af):
        B, T, C = af.shape
        k = self.chunk_size
        pad = (-T) % k
        if pad:
            af = F_pad(af, pad)
        Tp = af.shape[1]
        x = af.view(B, Tp // k, k, C).mean(dim=2)
        x = x.repeat_interleave(k, dim=1)
        return x[:, :T]


class MultiscaleTimescale(Timescale):
    def __init__(self, chunk_size=4):
        super().__init__()
        self.chunk = ChunkTimescale(chunk_size)
        self.lam = nn.Parameter(torch.tensor(0.0))

    def forward(self, af):
        return af + self.lam * self.chunk(af)


def F_pad(af, pad):
    pad_t = torch.zeros(af.shape[0], pad, af.shape[2], device=af.device, dtype=af.dtype)
    return torch.cat([af, pad_t], dim=1)


_TS = {
    "step": StepTimescale,
    "chunk": ChunkTimescale,
    "multiscale": MultiscaleTimescale,
}


def build_timescale(cfg) -> Timescale:
    name = cfg.get("T", "step")
    a = _TS[name]
    if name == "step":
        return a()
    return a(cfg.get("chunk_size", 4))


__all__ = ["Timescale", "build_timescale", *_TS.keys()]
