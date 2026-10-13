from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = Path(os.environ.get("RESULTS_ROOT", ROOT / "results")) / "tables"
WEIGHTS = ("0.001", "0.01", "0.1")


def main():
    R = json.load(open(RES / "tdmpc2_finetune.json"))
    tasks = sorted(R)
    get = lambda arm, k: np.array([R[t][arm][k] for t in tasks])
    base = get("consistency", "direction")
    print(f"[finetune] {len(tasks)} tasks, seed 1 checkpoints, direction on held out episodes 4 and 5")
    print(f"  released        D {get('released', 'direction').mean():.3f}   consistency fine tune D {base.mean():.3f}")
    for w in WEIGHTS:
        d = get(f"matched_{w}", "direction") - base
        m = get(f"matched_{w}", "consistency_mse") / get("consistency", "consistency_mse")
        print(f"  matched w={w:5s}  D {get(f'matched_{w}', 'direction').mean():.3f}  change {d.mean():+.3f} "
              f"(median {np.median(d):+.3f}), tasks improved {int((d > 0).sum())}/{len(tasks)}, "
              f"paired t p={stats.ttest_rel(get(f'matched_{w}', 'direction'), base).pvalue:.2g}, "
              f"held out consistency error x{np.median(m):.2f} (median ratio)")
    out = Path(os.environ.get("TABLES_OUT", ROOT / "tables"))
    out.mkdir(parents=True, exist_ok=True)
    rows = [("released", "released model"), ("consistency", "consistency loss only")] + \
           [(f"matched_{w}", f"+ matched swap alignment ($w={w}$)") for w in WEIGHTS]
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"fine tuning & direction $\uparrow$ & $\Delta$ direction $\uparrow$ & tasks improved & held out error $\downarrow$ \\", r"\midrule"]
    for arm, lab in rows:
        dd = get(arm, "direction") - base
        er = np.median(get(arm, "consistency_mse") / get("consistency", "consistency_mse"))
        imp = "" if arm in ("consistency", "released") else f"{int((dd > 0).sum())} of {len(tasks)}"
        dtxt = "" if arm == "consistency" else f"{dd.mean():+.3f}"
        lines.append(f"{lab} & {get(arm, 'direction').mean():.3f} & {dtxt} & {imp} & $\\times${er:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "finetune.tex").write_text("\n".join(lines) + "\n")
    E = RES / "tdmpc2_estimate.json"
    if E.exists():
        E = json.load(open(E))
        te = sorted(set(E) & set(tasks))
        est = np.array([E[t]["estimate_cos_mean"] for t in te]); mod = np.array([E[t]["model_cos_mean"] for t in te])
        d01 = np.array([R[t]["matched_0.01"]["direction"] - R[t]["consistency"]["direction"] for t in te])
        print(f"  estimate accuracy {est.mean():.2f}, model accuracy {mod.mean():.2f}, model better in "
              f"{int((mod > est).sum())} of {len(te)}; change where the estimate is better {d01[est > mod].mean():+.3f} "
              f"({int((est > mod).sum())} tasks), elsewhere {d01[est <= mod].mean():+.3f}")
    for w in WEIGHTS:
        dD = get(f"matched_{w}", "direction") - base
        dM = get(f"matched_{w}", "consistency_mse") / get("consistency", "consistency_mse") - 1
        r = stats.spearmanr(dM, dD)
        print(f"  w={w}: Spearman(change in held out error, change in direction) {r.correlation:+.2f} (p={r.pvalue:.2g})")
    print("[finetune] wrote", out / "finetune.tex")
    d01 = get("matched_0.01", "direction") - base
    r = stats.spearmanr(get("released", "direction"), d01)
    print(f"  change at w=0.01 against released direction: Spearman {r.correlation:+.2f} (p={r.pvalue:.2g})")
    low = get("released", "direction") < np.median(get("released", "direction"))
    print(f"  less faithful half: change {d01[low].mean():+.3f}; more faithful half: {d01[~low].mean():+.3f}")


if __name__ == "__main__":
    main()
