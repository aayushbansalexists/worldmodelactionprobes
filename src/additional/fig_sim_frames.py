import json
import os
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
out = Path(sys.argv[1])
FLIP = os.environ.get("FRAME_FLIP", "v")
BLUE, INK, INK2 = "#2a78d6", "#0b0b0b", "#52514e"
BLUE_LIGHT, BLUE_DARK = "#dbe8f8", "#123e75"
if os.environ.get("FIG_PALETTE"):
    _p = json.load(open(os.environ["FIG_PALETTE"]))
    BLUE, BLUE_LIGHT, BLUE_DARK = _p["colors"]["BLUE"], _p["derived"]["BLUE_LIGHT"], _p["derived"]["BLUE_DARK"]
SEQ = LinearSegmentedColormap.from_list("seq", ["#ffffff", BLUE_LIGHT, BLUE, BLUE_DARK])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 8.5, "text.color": INK,
                     "figure.dpi": 220, "savefig.bbox": "tight", "savefig.facecolor": "white"})

z = np.load(ROOT / "results" / "additional" / "a2world_frames.npz", allow_pickle=True)
up = (lambda a: a[::-1]) if FLIP == "v" else (lambda a: a[::-1, ::-1])
f = lambda a: a.astype(np.float32) / 255.0
K = 19
sim_l, sim_o = f(z["sim_logged"]), f(z["sim_offset"])
pr_l, pr_o = f(z["pred_logged"]).transpose(1, 0, 2, 3, 4), f(z["pred_offset"]).transpose(1, 0, 2, 3, 4)
r_true, r_mod = sim_o - sim_l, pr_o - pr_l
small = lambda a: np.stack([[cv2.resize(a[t, v], (64, 64), interpolation=cv2.INTER_AREA) for v in range(2)] for t in range(a.shape[0])])
a, b = small(r_true).ravel(), small(r_mod).ravel()
num = {"position": int(z["pos"]), "task": str(z["task"]), "D_position_mean_over_3_offsets": float(z["D"]),
       "cosine_this_offset": float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))),
       "size_ratio_this_offset": float(np.linalg.norm(b) / np.linalg.norm(a)), "step_shown": K + 1}
json.dump(num, open(ROOT / "results" / "additional" / "fig_sim_frames.json", "w"), indent=1)
print(num)

WITH_MODEL = os.environ.get("WITH_MODEL", "0") == "1"
NC = 5 if WITH_MODEL else 4
fig = plt.figure(figsize=(7.2 if WITH_MODEL else 6.2, 2.95 if WITH_MODEL else 3.15))
gs = fig.add_gridspec(2, NC + 1, width_ratios=[1] * NC + [0.06], wspace=0.05, hspace=0.05)
cols = ["recorded frame", "simulator,\nlogged actions", "simulator,\nchanged actions", "true response\n(simulator)", "response\nof A2World"]
start = [f(z["cur_agent"]), f(z["cur_wrist"])]
for v, cam in enumerate(("front camera", "wrist camera")):
    mt, mm = np.abs(r_true[K, v]).mean(-1), np.abs(r_mod[K, v]).mean(-1)
    vmax = float(max(mt.max(), mm.max())) if WITH_MODEL else float(mt.max())
    for c, im in enumerate([start[v], sim_l[K, v], sim_o[K, v], mt, mm][:NC]):
        ax = fig.add_subplot(gs[v, c])
        if im.ndim == 3:
            ax.imshow(up(im), interpolation="nearest", aspect="equal")
        else:
            h = ax.imshow(up(im), cmap=SEQ, vmin=0, vmax=vmax, interpolation="nearest", aspect="equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(INK2); s.set_linewidth(0.5)
        if v == 0:
            ax.set_title(cols[c], pad=4, fontsize=7.8)
        if c == 0:
            ax.set_ylabel(cam, fontsize=8, color=INK)
    cax = fig.add_subplot(gs[v, NC])
    cb = fig.colorbar(h, cax=cax)
    cb.ax.tick_params(labelsize=6.5, colors=INK2, length=2)
    cb.outline.set_edgecolor(INK2); cb.outline.set_linewidth(0.5)
    cb.set_label("absolute pixel\ndifference", fontsize=6.8, color=INK2)
fig.savefig(out / "fig_sim_frames.png", pad_inches=0.05)
print("wrote", out / "fig_sim_frames.png")
