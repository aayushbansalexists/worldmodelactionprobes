from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvVAE(nn.Module):
    def __init__(self, in_ch=3, latent_ch=4, base=32):
        super().__init__()
        b = base
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, b, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(b, b*2, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(b*2, b*4, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(b*4, b*4, 3, 1, 1), nn.SiLU(),
        )
        self.fc_mu = nn.Conv2d(b*4, latent_ch, 1, 1, 0)
        self.fc_lv = nn.Conv2d(b*4, latent_ch, 1, 1, 0)
        self.dec_in = nn.Conv2d(latent_ch, b*4, 1, 1, 0)
        self.dec = nn.Sequential(
            nn.SiLU(),
            nn.ConvTranspose2d(b*4, b*4, 3, 1, 1), nn.SiLU(),
            nn.ConvTranspose2d(b*4, b*2, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(b*2, b, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(b, in_ch, 4, 2, 1),
        )

    def encode(self, x):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_lv(h)

    def reparam(self, mu, lv):
        if self.training:
            return mu + torch.randn_like(mu) * (0.5 * lv).exp()
        return mu

    def decode(self, z):
        return self.dec(self.dec_in(z))

    def forward(self, x):
        mu, lv = self.encode(x)
        z = self.reparam(mu, lv)
        rec = self.decode(z)
        return rec, mu, lv

    @torch.no_grad()
    def encode_mu(self, x):
        return self.encode(x)[0]


def vae_loss(rec, x, mu, lv, beta=1e-4):
    rec_l = F.mse_loss(rec, x, reduction="mean")
    kl = (-0.5 * (1 + lv - mu.pow(2) - lv.exp())).mean()
    return rec_l + beta * kl, {"rec": rec_l.item(), "kl": kl.item()}


__all__ = ["ConvVAE", "vae_loss"]
