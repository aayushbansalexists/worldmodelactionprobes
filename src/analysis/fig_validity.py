from __future__ import annotations

import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = Path(os.environ.get("RESULTS_ROOT", ROOT / "results"))
FIG = Path(os.environ.get("FIG_OUT", ROOT / "figures"))
FIG.mkdir(parents=True, exist_ok=True)
TAG = os.environ.get("GT_TAG", "v3")

BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
MID, INK, INK2, GRID = "#f0efec", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9,
    "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 220, "savefig.bbox": "tight", "savefig.facecolor": "white",
})
TRANSFORMS = ["identity", "reflect", "rotate", "scale_0.25", "scale_4", "blind"]
TLABEL = {"identity": "trained\nmodel", "reflect": "reflected\nresponse", "rotate": "rotated\nresponse",
          "scale_0.25": "response\n×0.25", "scale_4": "response\n×4", "blind": "no\nresponse"}


def _v():
    sys.path.insert(0, str(ROOT))
    from analysis.make_tables import load_libero
    return load_libero()


def _trained(V):
    return [f"cf_{m}_causal_s{s}" for m in ["add", "agm", "delta"] for s in range(5)
            if f"cf_{m}_causal_s{s}" in V]


def fig_verdicts():
    V = _v()
    runs = _trained(V)
    mean = {t: {k: np.mean([V[r][t][k] for r in runs]) for k in V[runs[0]][t]} for t in TRANSFORMS}
    size_err = {t: abs(np.log2(max(mean[t]["mag_ratio"], 1e-3))) for t in TRANSFORMS}
    def rel(val, ref, higher_better):
        d = (val - ref) / (abs(ref) + 1e-12)
        return float(np.clip(d if higher_better else -d, -1, 1))
    rows = [
        ("direction, random", "truth", lambda t: mean[t]["direction"], True, "{:+.2f}"),
        ("direction, logged swaps", "truth", lambda t: mean[t]["dir_on"], True, "{:+.2f}"),
        ("size error", "truth", lambda t: size_err[t], False, "{:.2f}"),
        ("$C_1$ zero action", "probe", lambda t: mean[t]["C1"], True, "{:.3f}"),
        ("WSP Task 1", "probe", lambda t: mean[t]["task1"], True, "{:.1f}"),
        ("retrieval rank", "probe", lambda t: mean[t]["ret_gauss_meanrank"], False, "{:.2f}"),
        ("retrieval rank, real neg.", "probe", lambda t: mean[t]["ret_marg_meanrank"], False, "{:.2f}"),
        ("matched probe", "probe", lambda t: mean[t]["matched"], True, "{:+.2f}"),
    ]
    M = np.zeros((len(rows), len(TRANSFORMS)))
    txt = [[""] * len(TRANSFORMS) for _ in rows]
    for i, (_, _, get, hb, fmt) in enumerate(rows):
        ref = get("identity")
        for j, t in enumerate(TRANSFORMS):
            v = get(t)
            M[i, j] = 0.0 if t == "identity" else rel(v, ref, hb)
            txt[i][j] = "∞" if (t == "blind" and rows[i][0] == "size error") else fmt.format(v)
    cmap = LinearSegmentedColormap.from_list("div", [RED, MID, BLUE])
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    ax.imshow(M, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    ax.grid(False)
    for i in range(len(rows)):
        for j in range(len(TRANSFORMS)):
            ax.text(j, i, txt[i][j], ha="center", va="center", fontsize=7.8,
                    color="white" if abs(M[i, j]) > 0.6 else INK)
    ax.set_xticks(range(len(TRANSFORMS)))
    ax.set_xticklabels([TLABEL[t] for t in TRANSFORMS])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.axhline(2.5, color=INK, lw=1.4)
    ax.axvline(0.5, color="white", lw=3)
    n = len(TRANSFORMS)
    for y, lab in [(1.0, "simulator\nground truth"), (5.0, "probes")]:
        ax.text(n - 0.35, y, lab, rotation=270, ha="center", va="center", fontsize=7.5,
                color=INK2, clip_on=False)
    ax.set_xlim(-0.5, n - 0.5)
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=BLUE, edgecolor=INK2, lw=0.4, label="metric says: more faithful"),
               Patch(facecolor=MID, edgecolor=INK2, lw=0.4, label="unchanged"),
               Patch(facecolor=RED, edgecolor=INK2, lw=0.4, label="metric says: less faithful")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.07), handlelength=1.2, columnspacing=1.8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_verdicts.png")
    plt.close(fig)
    print("[fig] fig_verdicts.png")


def fig_scatter():
    V = _v()
    models = sorted(V)
    groups = [("trained models", ["identity"], BLUE, "o"),
              ("direction destroyed\n(reflected, rotated)", ["reflect", "rotate"], ORANGE, "^"),
              ("size changed\n(×0.25, ×4)", ["scale_0.25", "scale_4"], AQUA, "s"),
              ("no response", ["blind"], "#8a8984", "x")]
    probes = [("C1", "$C_1$ zero action div.", 1, True),
              ("task1", "WSP Task 1", 1, False),
              ("ret_gauss_meanrank", "retrieval rank", -1, False),
              ("ret_marg_meanrank", "retrieval rank, real neg.", -1, False),
              ("matched", "matched probe", 1, False)]
    GT = "dir_on"
    fig, axes = plt.subplots(1, 5, figsize=(8.6, 2.6))
    for ax, (key, lab, sgn, logy) in zip(axes, probes):
        for name, ts, col, mk in groups:
            x = [V[m][t][GT] for m in models for t in ts]
            y = [V[m][t][key] for m in models for t in ts]
            if key == "C1":
                y = [max(v, 1e-5) for v in y]
            kw = dict(edgecolors="white", linewidths=0.4) if mk != "x" else dict(linewidths=0.9)
            ax.scatter(x, y, s=16, c=col, marker=mk, label=name,
                       zorder=3 if col == BLUE else 2, **kw)
        xi = [V[m]["identity"][GT] for m in models]
        yi = [sgn * V[m]["identity"][key] for m in models]
        xa = [V[m][t][GT] for m in models for t in TRANSFORMS]
        ya = [sgn * V[m][t][key] for m in models for t in TRANSFORMS]
        ri = stats.spearmanr(xi, yi).correlation
        ra = stats.spearmanr(xa, ya).correlation
        ax.set_title(f"{lab}\nρ trained {ri:+.2f}   ρ all {ra:+.2f}", fontsize=8, color=INK)
        if logy:
            ax.set_yscale("log")
        if key == "task1":
            ax.set_ylim(-3, 82)
            _, _, best, lin = task1_levels()
            for yv, ls, t, dy in [(lin, "-", "linear model", -8.5), (best, "--", "constant", 1.5)]:
                ax.axhline(yv, color=INK2, lw=0.9, ls=ls, zorder=1)
                ax.text(0.98, yv + dy, t, transform=ax.get_yaxis_transform(), ha="right",
                        fontsize=6.5, color=INK2)
        if sgn < 0:
            ax.set_ylim(15.5, -0.8)
        ax.axvline(0, color=INK2, lw=0.6, ls=":")
    axes[0].set_ylabel("probe score", fontsize=8)
    fig.supxlabel("true direction on logged action swaps, from the simulator", fontsize=8.5, y=0.08)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.1))
    axes[-1].set_ylim(-1.05, 1.05)
    fig.tight_layout(w_pad=0.9, rect=[0, 0.06, 1, 1])
    fig.savefig(FIG / "fig_scatter.png")
    plt.close(fig)
    print("[fig] fig_scatter.png")


def task1_levels():
    sys.path.insert(0, str(ROOT / "third_party" / "WorldSimProbe" / "src"))
    sys.path.insert(0, str(ROOT))
    from worldsimprobe.evaluation.task1_action_calibration.oracle_ratio import triangular_ratio_score
    from data.dataset import Normalizer
    data = Path(os.environ.get("DATA_ROOT", ROOT / "data"))
    nz = Normalizer.load(data / "processed" / "latents" / "normalizer.npz")
    w = np.load(data / "processed" / "counterfactual" / (f"wsp_{TAG}.npz" if TAG != "v2" else "wsp_main.npz"),
                allow_pickle=True)
    xo = nz.norm_obs(w["x_orig"])
    ds = ((nz.norm_obs(w["x_small"]) - xo) ** 2).mean((1, 2))
    dl = ((nz.norm_obs(w["x_large"]) - xo) ** 2).mean((1, 2))
    ok = (dl > 1e-12) & (dl > ds)
    o = ds[ok] / dl[ok]
    o = o[(o >= 0) & (o < 1)]
    grid = np.linspace(0.005, 0.6, 120)
    sc = [np.mean([triangular_ratio_score(r, x) for x in o]) for r in grid]
    rstar, best = grid[int(np.argmax(sc))], max(sc) * 100
    lin = np.mean([triangular_ratio_score(0.0625, x) for x in o]) * 100
    return o, rstar, best, lin


def fig_task1():
    o, rstar, best, lin = task1_levels()
    V = _v()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.4), gridspec_kw={"width_ratios": [1.25, 1]})
    a1.hist(o, bins=np.linspace(0, 1, 41), color=BLUE, edgecolor="white", linewidth=0.6)
    a1.axvline(0.0625, color=INK, lw=1.3)
    a1.axvline(rstar, color=INK2, lw=1.3, ls="--")
    ymax = a1.get_ylim()[1]
    a1.text(rstar + 0.04, ymax * 0.93, "any nonzero linear\nresponse gives 0.0625", fontsize=7,
            color=INK, va="top")
    a1.annotate("", xy=(0.0625, ymax * 0.84), xytext=(rstar + 0.035, ymax * 0.86),
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.7))
    a1.text(rstar + 0.012, ymax * 0.55, f"best constant\n{rstar:.2f}", fontsize=7, color=INK, va="top")
    a1.set_xlabel("simulator response ratio, small / large perturbation")
    a1.set_ylabel("items")
    a1.set_xlim(0, 1)

    groups = [("trained on\nlogged actions", ["causal"], BLUE, "o"),
              ("trained under a\nnegative control", ["shuf", "derange", "tstep"], BLUE, "^")]
    ypos = {g[0]: i for i, g in enumerate(groups)}
    rng = np.random.default_rng(0)
    for name, sufs, col, mk in groups:
        vals = [V[m]["identity"]["task1"] for m in V
                if any(f"_{s}_s" in m for s in sufs)]
        y = ypos[name] + rng.uniform(-0.18, 0.18, len(vals))
        a2.scatter(vals, y, s=16, c=col, marker=mk, edgecolors="white", linewidths=0.4, zorder=3)
    a2.axvline(lin, color=INK, lw=1.3)
    a2.axvline(best, color=INK2, lw=1.3, ls="--")
    a2.text(lin - 1.8, 1.55, f"linear\nmodel {lin:.1f}", fontsize=7, ha="right", va="center")
    a2.text(best - 1.8, 1.55, f"stateless\nconstant {best:.1f}", fontsize=7, ha="right", va="center")
    a2.set_yticks(range(len(groups)))
    a2.set_yticklabels([g[0] for g in groups])
    a2.set_ylim(-0.6, 2.0)
    a2.set_xlim(40, 80)
    a2.set_xlabel("WSP Task 1 score")
    a2.grid(axis="y", visible=False)
    fig.tight_layout(w_pad=1.6)
    fig.savefig(FIG / "fig_task1.png")
    plt.close(fig)
    print(f"[fig] fig_task1.png  (constant {rstar:.3f} -> {best:.2f}, linear -> {lin:.2f})")


if __name__ == "__main__":
    fig_verdicts()
    fig_scatter()
    fig_task1()
