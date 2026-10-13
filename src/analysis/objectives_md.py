from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy import stats

TAG = os.environ.get("GT_TAG", "v3")
ROOT = Path(__file__).resolve().parents[1]
RES = Path(os.environ.get("RESULTS_ROOT", ROOT / "results")) / "tables"
ROWS = [("delta", "delta prediction"), ("idm01", r"+ inv.\ dyn.\ ($w=0.01$)"), ("agm", "+ AGM (size)"),
        ("act01", r"+ contrastive ($w=0.01$)"), ("act1", r"+ contrastive ($w=0.1$)"),
        ("md001", r"+ matched swap alignment ($w=0.001$)"), ("md01", r"+ matched swap alignment ($w=0.01$)"),
        ("md1", r"+ matched swap alignment ($w=0.1$)"),
        ("mdrand01", r"\quad random partner ($w=0.01$)"), ("mdshuf01", r"\quad shuffled targets ($w=0.01$)"),
        ("imp01", r"+ squared error imputation ($w=0.1$)"), ("imp1", r"+ squared error imputation ($w=1$)")]
MAIN = ("delta", "agm", "act01", "act1", "md01", "mdrand01", "mdshuf01", "imp1")
COLS = [("direction", r"D random $\uparrow$", 3), ("dir_on", r"D logged $\uparrow$", 3),
        ("cf_error", r"cf.\ error $\downarrow$", 3), ("mag_ratio", "size", 3), ("ret_gauss_meanrank", r"ret.\ rank", 2),
        ("val_mse", r"val.\ MSE ($\times10^{-4}$) $\downarrow$", 2)]


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "tables"))
    args = ap.parse_args()
    R = json.load(open(RES / f"direction_training_{TAG}.json"))
    seeds = range(5)
    get = lambda obj, k: np.array([R[f"cf_{obj}_causal_s{s}"][k] for s in seeds])
    stats_ = {}
    for k, _, _ in COLS:
        base = get("add", k)
        diffs = {o: get(o, k) - base for o, _ in ROWS}
        pv = [stats.ttest_rel(get(o, k), base).pvalue for o, _ in ROWS]
        ph = holm(pv)
        for (o, _), p_raw, p_adj in zip(ROWS, pv, ph):
            stats_[(o, k)] = (float(diffs[o].mean()), float(p_raw), float(p_adj),
                              float((diffs[o] > 0).sum()))
    scale = {"val_mse": 1e4}
    head = [r"\begin{tabular}{l" + "r" * len(COLS) + "}", r"\toprule",
            "objective & " + " & ".join(r"$\Delta$ " + lab for _, lab, _ in COLS) + r" \\", r"\midrule"]
    lines, main_lines = list(head), list(head)
    print("[objectives] paired by seed against MSE, mean difference (Holm adjusted p, seeds improved)")
    for o, lab in ROWS:
        cells = []
        for k, _, nd in COLS:
            m, p, pa, npos = stats_[(o, k)]
            v = m * scale.get(k, 1.0)
            star = "$^{*}$" if pa < 0.05 else ""
            txt = f"{v:+.{nd}f}"
            if float(txt) == 0:
                txt = f"{0:.{nd}f}"
            cells.append(txt + star)
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
        if o in MAIN:
            main_lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
        print(f"  {o:7s} " + "  ".join(f"{k}={stats_[(o, k)][0] * scale.get(k, 1.0):+.4f} "
                                       f"(p={stats_[(o, k)][1]:.4f}, holm {stats_[(o, k)][2]:.4f}, "
                                       f"{int(stats_[(o, k)][3])}/5 up)" for k, _, _ in COLS))
    lines += [r"\bottomrule", r"\end{tabular}"]
    main_lines += [r"\bottomrule", r"\end{tabular}"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "objectives_md.tex").write_text("\n".join(main_lines) + "\n")
    (out / "objectives_md_full.tex").write_text("\n".join(lines) + "\n")
    base = {k: float(get("add", k).mean()) for k, _, _ in COLS}
    print("[objectives] MSE baseline means:", {k: round(v, 6) for k, v in base.items()})
    for o in ("md001", "md01", "md1", "delta", "act01", "act1", "mdrand01", "mdshuf01", "imp01", "imp1"):
        rel = 100 * stats_[(o, "direction")][0] / base["direction"]
        rel_on = 100 * stats_[(o, "dir_on")][0] / base["dir_on"]
        print(f"[objectives] {o}: D {base['direction']:.3f} -> {get(o, 'direction').mean():.3f} ({rel:+.0f}%), "
              f"D_log {base['dir_on']:.3f} -> {get(o, 'dir_on').mean():.3f} ({rel_on:+.1f}%)")
    print("[objectives] wrote", out / "objectives_md.tex")


if __name__ == "__main__":
    main()
