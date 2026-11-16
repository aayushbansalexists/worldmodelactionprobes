import os
import sys

import numpy as np

src, out = sys.argv[1], sys.argv[2]
assert not os.path.exists(out), f"{out} exists"
z = np.load(src, allow_pickle=True)
sel = np.arange(0, 90, 3)
offsets = (1000, 1500, 2000)
pos = np.concatenate([sel for _ in offsets])
off = np.concatenate([np.full(len(sel), o) for o in offsets])
keys = ["hist_agent", "hist_wrist", "hist_actions"]
np.savez(out, queries=z["queries"][pos][:, :5], seed=z["seed"][pos] + off, pos=pos, off=off,
         **{k: z[k][pos] for k in keys})
print(f"saved {len(pos)} items: positions {len(sel)} x offsets {offsets}, queries {z['queries'][pos][:, :5].shape}")
print("seeds unique:", len(np.unique(z["seed"][pos] + off)) == len(pos), "| overlap with original or +500 seeds:",
      len(set((z["seed"][pos] + off).tolist()) & (set(z["seed"].tolist()) | set((z["seed"] + 500).tolist()))))
