from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = Path(os.environ.get("RESULTS_ROOT", ROOT / "results"))
DATA = Path(os.environ.get("DATA_ROOT", ROOT / "data"))
FIG = Path(os.environ.get("FIG_OUT", ROOT / "figures"))
TAG = os.environ.get("GT_TAG", "v3")

BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
MID, INK, INK2, GRID, GRAY = "#f0efec", "#0b0b0b", "#52514e", "#e4e3df", "#9a988f"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9,
    "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 220, "savefig.bbox": "tight", "savefig.facecolor": "white",
})
TRANSFORMS = ["identity", "reflect", "rotate", "scale_0.25", "scale_4", "blind"]
EEF = slice(15, 18)


def cos_rows(a, b):
    a, b = a.reshape(len(a), -1), b.reshape(len(b), -1)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)


def fig_eef3d(run="cf_add_causal_s0", n_items=3):
    import torch
    from analysis.probe_validity import Data, rollout
    from data.dataset import Normalizer
    from eval.counterfactual_eval import load_model
    nz = Normalizer.load(DATA / "processed" / "latents" / "normalizer.npz")
    D = Data(TAG, DATA, nz)
    mt = np.load(DATA / "processed" / "counterfactual" / f"matched_{TAG}.npz", allow_pickle=True)
    k1 = np.arange(len(mt["row"])) % int(mt["k"]) == 0
    row, a_j = mt["row"][k1], mt["a_j"][k1].astype(np.float32)
    xt, xm = mt["x_true"][k1], mt["x_match"][k1]
    m, _ = load_model(run, "cuda")
    cx, ca = D.cf_ctx_x[row], D.cf_ctx_a[row]
    Pi = nz.denorm_obs(rollout(m, cx, ca, D.cf_ref[row], "cuda"))
    Pj = nz.denorm_obs(rollout(m, cx, ca, a_j, "cuda"))
    del m
    torch.cuda.empty_cache()
    c = cos_rows((Pj - Pi)[:, :, EEF], (xm - xt)[:, :, EEF])
    tasks = np.array([str(t) for t in mt["task"][k1]])
    big = np.linalg.norm((xm - xt)[:, -1, EEF], axis=1) > 0.03
    order = np.argsort(np.abs(c - np.median(c)))
    pick, seen = [], set()
    for i in order:
        if big[i] and tasks[i] not in seen:
            pick.append(i)
            seen.add(tasks[i])
        if len(pick) == n_items:
            break
    fig = plt.figure(figsize=(7.2, 3.1))
    for p, i in enumerate(pick):
        ax = fig.add_subplot(1, n_items, p + 1, projection="3d")
        eff = lambda X: np.vstack([np.zeros(3), X[:, EEF]]) * 100.0
        true_e = eff(xm[i] - xt[i])
        model_e = eff(Pj[i] - Pi[i])
        refl_e = -model_e
        ax.plot(*true_e.T, color=INK, lw=2.0, label="simulator")
        ax.plot(*model_e.T, color=BLUE, lw=2.0, label="trained model")
        ax.plot(*refl_e.T, color=ORANGE, lw=2.0, ls=(0, (3, 1.6)),
                label="trained model, reflected")
        for X, col in [(true_e, INK), (model_e, BLUE), (refl_e, ORANGE)]:
            ax.scatter(*X[-1], color=col, s=12, depthshade=False)
        ax.scatter(0, 0, 0, color=GRAY, s=16, depthshade=False)
        allp = np.vstack([true_e, model_e, refl_e])
        ctr, half = allp.mean(0), (allp.max(0) - allp.min(0)).max() / 2 + 0.2
        ax.set_xlim(ctr[0] - half, ctr[0] + half)
        ax.set_ylim(ctr[1] - half, ctr[1] + half)
        ax.set_zlim(ctr[2] - half, ctr[2] + half)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("$\\Delta x$ (cm)", labelpad=-4, fontsize=9.5)
        ax.set_ylabel("$\\Delta y$ (cm)", labelpad=-4, fontsize=9.5)
        ax.set_zlabel("$\\Delta z$ (cm)", labelpad=-1, fontsize=9.5)
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(4)); ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(axis="both", pad=-2, labelsize=8)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor((1, 1, 1, 0))
            axis.pane.set_edgecolor(GRID)
        ax.view_init(elev=22, azim=-55)
        ax.set_title(f"model {c[i]:+.2f},  reflected {-c[i]:+.2f}", fontsize=9.5, pad=-2)
    h, lab = fig.axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, 0.005))
    fig.subplots_adjust(left=0.0, right=0.955, top=0.93, bottom=0.14, wspace=0.3)
    fig.savefig(FIG / "fig_eef3d.png", pad_inches=0.28)
    plt.close(fig)
    print("[fig] eef3d items", [(tasks[i][:30], round(float(c[i]), 3)) for i in pick],
          "median gripper cos", round(float(np.median(c)), 3), "n", len(c))


def fig_landscape():
    L = json.load(open(RES / "tables" / f"probe_landscape_{TAG}.json"))
    th, ls = np.array(L["thetas"]), np.array(L["log2s"])
    TH, LS = np.meshgrid(th, ls, indexing="ij")
    g = {k: np.array(v) for k, v in L["mean"].items()}
    div = LinearSegmentedColormap.from_list("div", [RED, MID, BLUE])
    seq = LinearSegmentedColormap.from_list("seq", ["#dbe8f8", BLUE, "#123e75"])
    panels = [
        ("dir", "truth: direction under\nrandom perturbations", div, (-1, 1)),
        ("dir_on", "truth: direction under\nlogged action swaps", div, (-1, 1)),
        ("matched", "probe: matched\n(logged data only)", div, (-1, 1)),
        ("C1", "probe: zero action\ndivergence, $\\log_{10} C_1$", seq, None),
        ("task1", "probe: WSP Task 1\n(0 to 100)", seq, (0, 100)),
        ("ret", "probe: retrieval,\nminus mean rank", seq, (-7.5, 0)),
    ]
    fig = plt.figure(figsize=(7.2, 4.9))
    for p, (k, title, cmap, zl) in enumerate(panels):
        ax = fig.add_subplot(2, 3, p + 1, projection="3d")
        Z = g[k].copy()
        if k == "C1":
            Z = np.log10(Z)
        if k == "ret":
            Z = -Z
        lo, hi = zl if zl else (Z.min(), Z.max() + 1e-9)
        if k == "task1":
            lo, hi = 0, 100
        norm = plt.Normalize(lo, hi)
        ax.plot_surface(TH, LS, Z, facecolors=cmap(norm(Z)), rstride=1, cstride=1,
                        linewidth=0.25, edgecolor=(1, 1, 1, 0.5), antialiased=True, shade=False)
        if zl:
            ax.set_zlim(*zl)
        ax.set_xticks([0, np.pi / 2, np.pi])
        ax.set_xticklabels(["0", "$\\pi/2$", "$\\pi$"])
        ax.set_yticks([-3, 0, 3])
        ax.set_yticklabels(["1/8", "1", "8"])
        ax.set_xlabel("rotation $\\theta$", labelpad=-2, fontsize=11.5)
        ax.set_ylabel("scale $s$", labelpad=-2, fontsize=11.5)
        ax.tick_params(axis="both", pad=-1, labelsize=10)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor((1, 1, 1, 0))
            axis.pane.set_edgecolor(GRID)
        ax.view_init(elev=24, azim=-128)
        ax.set_title(title, fontsize=12.5, pad=0)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.94, bottom=0.01, wspace=0.05, hspace=0.34)
    fig.savefig(FIG / "fig_landscape.png")
    plt.close(fig)
    for k in g:
        Z = g[k] if k != "C1" else np.log10(g[k])
        print(f"[landscape] {k:8s} range over theta (s=1) {Z[:, 6].min():+.3f}..{Z[:, 6].max():+.3f}"
              f"  range over s (theta=0) {Z[0].min():+.3f}..{Z[0].max():+.3f}")


def fig_validity_bars():
    from analysis.make_tables import PROBES, TRANSFORMS as TR, load_libero, validity
    V = load_libero()
    R = json.load(open(RES / "tables" / "tdmpc2_validity.json"))
    lib, td = sorted(V), sorted(k for k in R if "identity" in R[k])
    panels = [("LIBERO, 92 trained models\ntruth: random perturbations", V, lib, "direction"),
              ("LIBERO, 92 trained models\ntruth: logged action swaps", V, lib, "dir_on"),
              (f"TD-MPC2, {len(td)} released models\ntruth: random perturbations", R, td, "direction")]
    probes = [p for p in PROBES if p[0] != "ret_marg_meanrank"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), sharey=True)
    y = np.arange(len(probes))[::-1]
    for ax, (title, D_, runs, gt) in zip(axes, panels):
        for off, trs, col, lab in [(0.18, ["identity"], BLUE, "models as trained"),
                                   (-0.18, TR, ORANGE, "with the 6 response transforms")]:
            v = validity(D_, runs, trs, gt, probes)
            m = np.array([v[k][0] for k, _, _ in probes])
            lo = np.array([v[k][1] for k, _, _ in probes])
            hi = np.array([v[k][2] for k, _, _ in probes])
            ax.barh(y + off, m, height=0.34, color=col, label=lab, zorder=2)
            ax.errorbar(m, y + off, xerr=[m - lo, hi - m], fmt="none", ecolor=INK2, elinewidth=0.8,
                        capsize=1.5, zorder=3)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_xlim(-0.8, 1.05)
        ax.set_xticks([-0.5, 0, 0.5, 1])
        ax.set_title(title, fontsize=7.5)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([lab for _, lab, _ in probes], fontsize=7.5)
    for t in axes[0].get_yticklabels():
        if "matched" in t.get_text():
            t.set_fontweight("bold")
    fig.supxlabel("Spearman correlation with true direction across models", fontsize=7.5,
                  x=0.58, y=0.02)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.58, -0.12))
    fig.subplots_adjust(wspace=0.14, bottom=0.2)
    fig.savefig(FIG / "fig_validity_bars.png")
    plt.close(fig)


def fig_tdmpc2():
    R = json.load(open(RES / "tables" / "tdmpc2_validity.json"))
    runs = sorted(k for k in R if "identity" in R[k])
    tasks = sorted({r.rsplit("-", 1)[0] for r in runs})
    d = {t: [R[r]["identity"]["direction"] for r in runs if r.rsplit("-", 1)[0] == t] for t in tasks}
    mp = {t: [R[r]["identity"]["matched"] for r in runs if r.rsplit("-", 1)[0] == t] for t in tasks}
    order = sorted(tasks, key=lambda t: np.mean(d[t]))
    fig, ax = plt.subplots(figsize=(7.2, 0.155 * len(tasks) + 0.7))
    for i, t in enumerate(order):
        ax.plot([min(d[t] + mp[t]), max(d[t] + mp[t])], [i, i], color=GRID, lw=0.8, zorder=1)
        ax.scatter(d[t], [i] * len(d[t]), s=12, color=BLUE, zorder=3,
                   label="direction, simulator" if i == 0 else None)
        ax.scatter(mp[t], [i] * len(mp[t]), s=12, facecolors="white", edgecolors=ORANGE, lw=1.0,
                   zorder=3, label="matched probe, logged data only" if i == 0 else None)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=6.5)
    ax.set_xlabel("cosine between the model's response and the true one (3 seeds per task)")
    ax.set_xlim(-0.05, 1.02)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", frameon=False)
    fig.savefig(FIG / "fig_tdmpc2.png")
    plt.close(fig)
    print("[fig] tdmpc2 tasks", len(tasks), "models", len(runs))


def fig_closed_loop():
    E = json.load(open(RES / "tables" / f"closed_loop_{TAG}.json"))
    order = [("identity", "trained model", BLUE, "o"), ("scale_0.25", "response ×0.25", AQUA, "s"),
             ("scale_4", "response ×4", AQUA, "s"), ("rotate", "rotated response", ORANGE, "^"),
             ("reflect", "reflected response", ORANGE, "^"), ("blind", "no response", GRAY, "x")]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7.0, 2.5))
    for i, (t, lab, col, mk) in enumerate(order):
        d = np.array([e["d_final"] for e in E if e["transform"] == t]) * 100
        y = len(order) - 1 - i + rng.uniform(-0.22, 0.22, len(d))
        kw = dict(edgecolors="white", linewidths=0.3) if mk != "x" else dict(linewidths=0.8)
        ax.scatter(np.maximum(d, 0.03), y, s=9, c=col, marker=mk, alpha=0.7, zorder=2, **kw)
        ax.plot([np.median(d)] * 2, [len(order) - 1 - i - 0.34, len(order) - 1 - i + 0.34],
                color=INK, lw=1.6, zorder=3)
        ax.text(80, len(order) - 1 - i, f"median {np.median(d):.1f} cm", va="center", fontsize=7,
                color=INK2)
    ax.axvline(4.0, color=INK2, lw=0.9, ls="--", zorder=1)
    ax.text(4.4, len(order) - 0.45, "start, 4 cm", fontsize=7, color=INK2, ha="left")
    ax.axvline(1.0, color=INK2, lw=0.6, ls=":", zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(0.03, 300)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([o[1] for o in order][::-1])
    ax.set_xlabel("gripper distance to the goal after 20 steps of model predictive control (cm)")
    ax.grid(axis="y", visible=False)
    fig.savefig(FIG / "fig_closed_loop.png")
    plt.close(fig)
    print("[fig] closed loop", len(E), "episodes")


if __name__ == "__main__":
    import sys
    for name in sys.argv[1:] or ["eef3d"]:
        globals()[f"fig_{name}"]()
