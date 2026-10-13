from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analysis import make_tables as mk

LABEL = mk.TRANSFORM_LABEL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "tables"))
    args = ap.parse_args()
    out = Path(args.out)
    V = mk.load_libero()
    R = mk._load("tdmpc2_validity.json")
    A = mk._load("diamond_validity.json")
    I = mk._load(f"inverse_probe_{mk.TAG}.json")
    trained = [f"cf_{m}_causal_s{s}" for m in ["add", "agm", "delta"] for s in mk.S5]
    td = sorted(k for k in R if "identity" in R[k])
    games = sorted(A)
    mean = lambda D, runs, t, k: float(np.mean([D[r][t][k] for r in runs]))

    lines = [r"\begin{tabular}{l|rrr|rrr|rrr|rrr}", r"\toprule",
             r" & \multicolumn{6}{c|}{LIBERO, 15 trained models} & \multicolumn{3}{c|}{TD-MPC2, "
             + str(len(td)) + r"} & \multicolumn{3}{c}{Atari, " + str(len(games)) + r"} \\",
             r" & \multicolumn{3}{c|}{random perturbations} & \multicolumn{3}{c|}{logged swaps} & "
             r"\multicolumn{3}{c|}{random perturbations} & \multicolumn{3}{c}{swaps} \\",
             r"model & truth & $\IC$ & error & truth & $\IC$ & error & truth & $\IC$ & error & truth & $\IC$ & error \\",
             r"\midrule"]
    print("[inverse] mean per transform: LIBERO rand | LIBERO swap | TD-MPC2 | Atari (truth, IC, error)")
    for t in mk.TRANSFORMS:
        c = [mean(V, trained, t, "direction"), mean(V, trained, t, "inv_dir_rand"), mean(V, trained, t, "inv_err_rand"),
             mean(V, trained, t, "dir_on"), mean(V, trained, t, "inv_dir_swap"), mean(V, trained, t, "inv_err_swap"),
             mean(R, td, t, "direction"), mean(R, td, t, "inv_dir_rand"), mean(R, td, t, "inv_err_rand"),
             mean(A, games, t, "direction"), mean(A, games, t, "inv_dir_swap"), mean(A, games, t, "inv_err_swap")]
        fmt = ["{:+.3f}", "{:+.3f}", "{:.0f}", "{:+.3f}", "{:+.3f}", "{:.1f}",
               "{:+.3f}", "{:+.3f}", "{:.2f}", "{:+.3f}", "{:+.3f}", "{:.2f}"]
        cells = [f.format(v) for f, v in zip(fmt, c)]
        lines.append(f"{LABEL[t]} & " + " & ".join(cells) + r" \\")
        print(f"  {t:11s} " + " ".join(cells))
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "inverse_transforms.tex").write_text("\n".join(lines) + "\n")

    lib_all = sorted(V)
    ordinary = sorted(r for r in lib_all if "_causal_" in r)
    groups = {}
    for r in td:
        groups.setdefault(r.rsplit("-", 1)[0], []).append(r)
    rows = [("inv_dir_swap", r"$\IC$, logged swaps", 1), ("inv_dir_rand", r"$\IC$, random queries", 1),
            ("inv_err_swap", "cycle error, logged swaps", -1), ("inv_err_rand", "cycle error, random queries", -1),
            ("matched", "matched probe", 1)]
    lines = [r"\begin{tabular}{l|rrrr|rr|rrr}", r"\toprule",
             r" & \multicolumn{4}{c|}{LIBERO, 92 models} & \multicolumn{2}{c|}{LIBERO, 30 ordinary}"
             r" & \multicolumn{3}{c}{TD-MPC2, " + str(len(td)) + r"} \\",
             r" & \multicolumn{2}{c}{D random} & \multicolumn{2}{c|}{D logged} & D random & D logged"
             r" & \multicolumn{3}{c}{direction} \\",
             r"query set & trained & transf. & trained & transf. & trained & trained & trained & transf. & within task \\",
             r"\midrule"]
    print("\n[inverse] Spearman with truth: LIBERO D (trained, transf), D_log (trained, transf), "
          "30 ordinary (D, D_log), TD-MPC2 (trained, transf, within task)")
    for key, lab, sgn in rows:
        pr = [(key, lab, sgn)]
        v = [mk.validity(V, lib_all, trs, gt, pr)[key][0] for gt in ("direction", "dir_on")
             for trs in (["identity"], mk.TRANSFORMS)]
        v += [mk.validity(V, ordinary, ["identity"], gt, pr)[key][0] for gt in ("direction", "dir_on")]
        if key in R[td[0]]["identity"]:
            v += [mk.validity(R, td, trs, "direction", pr)[key][0] for trs in (["identity"], mk.TRANSFORMS)]
            v += [mk.demeaned_validity(R, groups, key, sgn)[0]]
            cells = [f"{x:+.2f}" for x in v]
        else:
            cells = [f"{x:+.2f}" for x in v] + ["", "", ""]
        lines.append(lab + " & " + " & ".join(cells) + r" \\")
        print(f"  {key:14s} " + " ".join(cells))
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "inverse_validity.tex").write_text("\n".join(lines) + "\n")

    info = I["inverse_model"]
    r2 = [R[k]["identity"]["inv_val_r2"] for k in td]
    acc = [A[g]["meta"]["inv_val_acc"] / A[g]["meta"]["inv_chance"] for g in games]
    print(f"\n[inverse] LIBERO: {info['n_train']} training transitions, {info['n_val']} held out, "
          f"R2 per dim {min(info['val_r2_per_dim']):.2f} to {max(info['val_r2_per_dim']):.2f}, mean {info['val_r2_mean']:.2f}")
    print(f"[inverse] TD-MPC2: held out R2 median {np.median(r2):.2f}, range {min(r2):.2f} to {max(r2):.2f}, "
          f"above 0.5 in {np.mean(np.array(r2) > 0.5) * 100:.0f} percent of models")
    print(f"[inverse] Atari: held out accuracy / chance median {np.median(acc):.1f}, range {min(acc):.1f} to {max(acc):.1f}; "
          f"training transitions {min(A[g]['meta']['inv_n_train'] for g in games)} to "
          f"{max(A[g]['meta']['inv_n_train'] for g in games)}")
    print("[inverse] wrote", out / "inverse_transforms.tex", "and", out / "inverse_validity.tex")


if __name__ == "__main__":
    main()
