from __future__ import annotations

import argparse
import os
import sys
import time
from argparse import Namespace

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--a2world", required=True)
    ap.add_argument("--steps", type=int, default=35)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=None)
    args = ap.parse_args()
    sys.path.insert(0, args.a2world)
    import cv2
    import torch
    from einops import rearrange
    from a2world.actions import libero_servo_actions
    from a2world.history import action_path_features, sample_history_indices
    from a2world.rollout import LATENT_FRAMES, pack_conditioning_frames, setup_pipeline

    z = np.load(args.items)
    n = len(z["seed"])
    stop = n if args.stop is None else min(n, args.stop)
    pipe = setup_pipeline(Namespace(checkpoint=args.checkpoint, variant="libero", history=True,
                                    guardrail=False, seed=0, num_gpus=1))
    prep = lambda f: cv2.resize(np.ascontiguousarray(f[:, ::-1]), (256, 256), interpolation=cv2.INTER_AREA)
    back = lambda f: cv2.resize(np.ascontiguousarray(f[:, ::-1]), (128, 128), interpolation=cv2.INTER_AREA)
    views = torch.tensor([2, 0], dtype=torch.int64).repeat_interleave(LATENT_FRAMES)
    Q = z["queries"].shape[1]
    pred = np.zeros((stop - args.start, Q, 2, 20, 128, 128, 3), np.uint8)
    t0 = time.time()
    for i in range(args.start, stop):
        hist = [[prep(f) for f in z["hist_agent"][i]], [prep(f) for f in z["hist_wrist"][i]]]
        current = [hist[0][-1], hist[1][-1]]
        h_act = libero_servo_actions(z["hist_actions"][i])
        idx = sample_history_indices(h_act, 20)
        h_vid = np.concatenate([np.stack([v[k] for k in idx]) for v in hist], 0)
        h_ten = torch.from_numpy(h_vid).permute(3, 0, 1, 2).unsqueeze(0)
        for q in range(Q):
            act = libero_servo_actions(z["queries"][i, q])
            state = action_path_features(h_act, act, idx)[None]
            video = pipe(state=state, video_tensor=pack_conditioning_frames(current), input_path=None, actions=act,
                         blacks=None, history=h_ten, latent_view_indices=views, num_conditional_frames=1,
                         guidance=0.0, n_views=2, num_sampling_step=args.steps, seed=int(z["seed"][i]))
            v = rearrange(video, "b c (v t) h w -> b v t h w c", v=2)[0]
            v = ((v.float().cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
            for k in range(2):
                pred[i - args.start, q, k] = np.stack([back(f) for f in v[k, 1:]])
        print(f"[a2world] item {i + 1}/{stop}  {(time.time() - t0) / (i - args.start + 1):.1f} s per item", flush=True)
        if (i - args.start) % 10 == 9 or i == stop - 1:
            np.savez(args.out, pred=pred[:i - args.start + 1], start=args.start)


if __name__ == "__main__":
    main()
