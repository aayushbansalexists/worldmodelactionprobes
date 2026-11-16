from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@torch.no_grad()
def encode(model, frames, device, bs=1024):
    flat = frames.reshape(-1, 128, 128, 3)
    out = []
    for i in range(0, len(flat), bs):
        x = torch.from_numpy(flat[i:i + bs]).to(device).permute(0, 3, 1, 2).float() / 255.0
        x = F.interpolate(x, size=224, mode="bilinear", antialias=True, align_corners=False)
        x = (x - MEAN.to(device)) / STD.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model(pixel_values=x).last_hidden_state.float()
        out.append(torch.cat([h[:, 0], h[:, 1:].mean(1)], -1).cpu().numpy())
    f = np.concatenate(out, 0)
    return f.reshape(*frames.shape[:-4], 2 * f.shape[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pattern", default="*.npz")
    ap.add_argument("--model", default="facebook/dinov2-small")
    args = ap.parse_args()
    from transformers import AutoModel
    device = "cuda"
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for fp in sorted(glob.glob(os.path.join(args.inp, args.pattern))):
        dst = out / os.path.basename(fp)
        if dst.exists():
            continue
        z = np.load(fp, allow_pickle=True)
        res = {}
        for k in z.files:
            a = z[k]
            if a.dtype == np.uint8 and a.shape[-4:] == (2, 128, 128, 3):
                res[k] = encode(model, a, device).astype(np.float32)
            else:
                res[k] = a
        np.savez(dst, **res)
        print(f"[encode] {os.path.basename(fp)[:60]:60s} keys={len(res)}", flush=True)


if __name__ == "__main__":
    main()
