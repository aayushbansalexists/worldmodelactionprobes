from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy import stats

RES = Path(os.environ.get("RESULTS_ROOT", "results")) / "tables"
OUT = Path(os.environ.get("TABLES_OUT",
                          Path(__file__).resolve().parents[1] / "tables"))
OUT.mkdir(parents=True, exist_ok=True)
TAG = os.environ.get("GT_TAG", "v3")

TRANSFORMS = ["identity", "reflect", "rotate", "scale_0.25", "scale_4", "blind"]
TRANSFORM_LABEL = {"identity": "trained model", "reflect": "reflected response",
                   "rotate": "rotated response", "scale_0.25": "response $\\times0.25$",
                   "scale_4": "response $\\times4$", "blind": "no response"}
PROBES = [("C1", "zero action divergence", 1), ("task1", "WSP Task 1", 1),
          ("ret_gauss_meanrank", "retrieval", -1),
          ("ret_marg_meanrank", "retrieval, real negatives", -1),
          ("inv_err@", "inverse model, cycle error", -1),
          ("inv_dir@", "inverse model, direction", 1),
          ("matched", "matched probe (ours)", 1)]


def resolve(key, gt):
    if key.endswith("@"):
        return key[:-1] + ("_swap" if gt.startswith("dir_on") else "_rand")
    return key


S5 = range(5)


def _load(name):
    p = RES / name
    return json.load(open(p)) if p.exists() else {}


def load_libero():
    V = _load(f"probe_validity_{TAG}.json")
    M = _load(f"matched_{TAG}.json").get("models", {})
    for r in V:
        for t in TRANSFORMS:
            if r in M:
                V[r][t]["matched"] = M[r][t]["matched_k1"]
                V[r][t]["matched_k3"] = M[r][t]["matched_k3"]
                V[r][t]["matched_k5"] = M[r][t]["matched_k5"]
                V[r][t]["dir_on"] = M[r][t]["dir_on"]
                for k in (1, 3, 5):
                    if f"dir_on_k{k}" in M[r][t]:
                        V[r][t][f"dir_on_k{k}"] = M[r][t][f"dir_on_k{k}"]
    I = _load(f"inverse_probe_{TAG}.json").get("models", {})
    for r in V:
        if r in I:
            for t in TRANSFORMS:
                V[r][t].update(I[r][t])
    return V


def spearman_ci(x, y, n_boot=5000, seed=0, groups=None):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    g = np.arange(len(x)) if groups is None else np.asarray(groups)
    x, y, g = x[ok], y[ok], g[ok]
    rho = stats.spearmanr(x, y).correlation
    rng = np.random.default_rng(seed)
    uniq = list(dict.fromkeys(g.tolist()))
    members = [np.flatnonzero(g == u) for u in uniq]
    bs = []
    for _ in range(n_boot):
        i = np.concatenate([members[k] for k in rng.integers(0, len(uniq), len(uniq))])
        if np.ptp(x[i]) > 0 and np.ptp(y[i]) > 0:
            bs.append(stats.spearmanr(x[i], y[i]).correlation)
    return float(rho), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), int(len(x))


def validity(V, runs, transforms, gt="direction", probes=PROBES, group_of=None):
    pts = [(r, t) for r in runs for t in transforms if gt in V[r][t]]
    y = [V[r][t][gt] for r, t in pts]
    unit = group_of or (lambda r: r)
    out = {}
    for key, _, sgn in probes:
        k = resolve(key, gt)
        if not all(k in V[r][t] for r, t in pts):
            continue
        out[key] = spearman_ci([sgn * V[r][t][k] for r, t in pts], y, groups=[unit(r) for r, _ in pts])
    return out


def demeaned_validity(V, groups, key, sgn, gt="direction"):
    x, y, g = [], [], []
    key = resolve(key, gt)
    for name, runs in groups.items():
        px = np.array([sgn * V[r]["identity"][key] for r in runs])
        py = np.array([V[r]["identity"][gt] for r in runs])
        x += list(px - px.mean())
        y += list(py - py.mean())
        g += [name] * len(runs)
    return spearman_ci(x, y, groups=g)


def fmt_rho(v):
    return "" if v is None else f"{v[0]:+.2f}"


def main():
    V = load_libero()
    R = _load("tdmpc2_validity.json")
    lib_all = sorted(V)
    trained = [f"cf_{m}_causal_s{s}" for m in ["add", "agm", "delta"] for s in S5 if f"cf_{m}_causal_s{s}" in V]
    td_all = sorted(k for k in R if isinstance(R[k], dict) and "identity" in R[k])
    print(f"[tables] LIBERO {len(lib_all)} models, {len(trained)} trained causal; TD-MPC2 {len(td_all)} models")

    keys = ["direction", "dir_on", "mag_ratio", "cf_error", "C1", "task1", "ret_gauss_meanrank", "matched"]
    mean = {t: {k: float(np.mean([V[r][t][k] for r in trained])) for k in keys} for t in TRANSFORMS}
    CL = _load(f"closed_loop_{TAG}.json")
    cl = {}
    if CL:
        for t in TRANSFORMS:
            d = np.array([e["d_final"] for e in CL if e["transform"] == t]) * 100
            cl[t] = (float(d.mean()), float(np.median(d)), float((d < 1.0).mean()), len(d))
    lines = [r"\begin{tabular}{lrrrr|r|rrrr}" if cl else r"\begin{tabular}{lrrrr|rrrr}", r"\toprule",
             r" & \multicolumn{4}{c|}{simulator ground truth} & "
             + (r"\multicolumn{1}{c|}{control} & " if cl else "")
             + r"\multicolumn{4}{c}{probe} \\",
             r"model & D random $\uparrow$ & D logged $\uparrow$ & size & cf.\ error $\downarrow$ & "
             + (r"miss (cm) $\downarrow$ & " if cl else "")
             + r"$C_1$ $\uparrow$ & Task 1 $\uparrow$ & ret.\ rank $\downarrow$ & matched $\uparrow$ \\",
             r"\midrule"]
    print("\n[transforms] LIBERO, mean over 15 trained models")
    for t in TRANSFORMS:
        v = mean[t]
        lines.append(f"{TRANSFORM_LABEL[t]} & {v['direction']:+.3f} & {v['dir_on']:+.3f} & "
                     f"{v['mag_ratio']:.2f} & {v['cf_error']:.2f} & "
                     + (f"{cl[t][0]:.1f} & " if cl else "")
                     + f"{v['C1']:.4f} & {v['task1']:.1f} & "
                     f"{v['ret_gauss_meanrank']:.2f} & {v['matched']:+.3f} \\\\")
        print(f"  {t:11s} " + " ".join(f"{k}={v[k]:.4g}" for k in keys)
              + (f" closed_loop_mean_cm={cl[t][0]:.2f} median={cl[t][1]:.2f} within1cm={cl[t][2]:.2f} n={cl[t][3]}" if cl else ""))
    if CL:
        cells = {}
        for e in CL:
            cells.setdefault((e["run"], e["transform"]), []).append(e["d_final"])
        pts = sorted(cells)
        miss = [-np.mean(cells[k]) for k in pts]
        print(f"\n[closed loop] Spearman with minus final distance over {len(pts)} model x transform cells")
        for key, lab, sgn in [("direction", "D random", 1), ("dir_on", "D logged", 1)] + PROBES:
            x = [sgn * V[r][t][resolve(key, "direction")] for r, t in pts]
            print(f"  {lab:28s} {stats.spearmanr(x, miss).correlation:+.2f}")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "transforms.tex").write_text("\n".join(lines) + "\n")

    if td_all:
        tm = {t: {k: float(np.mean([R[r][t][k] for r in td_all])) for k in
                  ["direction", "mag_ratio", "C1", "task1", "ret_gauss_meanrank", "matched"]} for t in TRANSFORMS}
        lines = [r"\begin{tabular}{lrr|rrrr}", r"\toprule",
                 r" & \multicolumn{2}{c|}{simulator} & \multicolumn{4}{c}{probe} \\",
                 r"model & direction $\uparrow$ & size & $C_1$ $\uparrow$ & Task 1 $\uparrow$ & ret.\ rank $\downarrow$ & matched $\uparrow$ \\", r"\midrule"]
        print(f"\n[transforms] TD-MPC2, mean over {len(td_all)} released models")
        for t in TRANSFORMS:
            v = tm[t]
            lines.append(f"{TRANSFORM_LABEL[t]} & {v['direction']:+.3f} & {v['mag_ratio']:.2f} & "
                         f"{v['C1']:.4f} & {v['task1']:.1f} & {v['ret_gauss_meanrank']:.2f} & "
                         f"{v['matched']:+.3f} \\\\")
            print(f"  {t:11s} " + " ".join(f"{k}={v[k]:.4g}" for k in v))
        lines += [r"\bottomrule", r"\end{tabular}"]
        (OUT / "transforms_tdmpc2.tex").write_text("\n".join(lines) + "\n")

    cols = [("LIBERO", V, lib_all, ["identity"], "direction"),
            ("LIBERO", V, lib_all, TRANSFORMS, "direction"),
            ("LIBERO", V, lib_all, ["identity"], "dir_on"),
            ("LIBERO", V, lib_all, TRANSFORMS, "dir_on")]
    if td_all:
        cols += [("TD-MPC2", R, td_all, ["identity"], "direction"),
                 ("TD-MPC2", R, td_all, TRANSFORMS, "direction")]
    DG = _load("diamond_validity.json")
    games = sorted(DG)
    if games:
        DV = {g: {t: {"dir_on": DG[g][t]["direction"], "C1": DG[g][t]["C1"], "matched": DG[g][t]["matched"],
                      "ret_gauss_meanrank": DG[g][t]["ret_meanrank"] / DG[g]["meta"]["chance_rank"],
                      **{k: DG[g][t][k] for k in ("inv_err_swap", "inv_dir_swap") if k in DG[g][t]}}
                  for t in TRANSFORMS} for g in games}
        cols += [("Atari", DV, games, ["identity"], "dir_on"),
                 ("Atari", DV, games, TRANSFORMS, "dir_on")]
    task_of = lambda r: r.rsplit("-", 1)[0]
    res = [validity(D_, runs, trs, gt, group_of=task_of if name == "TD-MPC2" else None)
           for name, D_, runs, trs, gt in cols]
    head = (r" & \multicolumn{4}{c}{LIBERO, " + str(len(lib_all)) + r" models we trained}"
            + (r" & \multicolumn{2}{c}{TD-MPC2, " + str(len(td_all)) + r" released}" if td_all else "")
            + (r" & \multicolumn{2}{c}{Atari, " + str(len(games)) + r" released}" if games else "") + r" \\")
    sub = (r" & \multicolumn{2}{c}{random perturbations} & \multicolumn{2}{c}{logged action swaps}"
           + (r" & \multicolumn{2}{c}{random perturbations}" if td_all else "")
           + (r" & \multicolumn{2}{c}{logged action swaps}" if games else "") + r" \\")
    rule = (r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}" + (r"\cmidrule(lr){6-7}" if td_all else "")
            + (r"\cmidrule(lr){8-9}" if games else ""))
    lines = [r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule", head, sub, rule,
             "probe" + " & as trained & transformed" * (len(cols) // 2) + r" \\", r"\midrule"]
    ci_lines = [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule", head, sub, rule,
                "probe" + " & as trained & transformed" * (len(cols) // 2) + r" \\", r"\midrule"]
    print("\n[validity] Spearman rho with ground truth direction")
    for key, lab, _ in PROBES:
        cells = [fmt_rho(r_.get(key)) for r_ in res]
        if all(c == "" for c in cells):
            continue
        lines.append(lab + " & " + " & ".join(cells) + r" \\")
        ci_lines.append(lab + " & " + " & ".join(
            "" if r_.get(key) is None else f"[{r_[key][1]:+.2f}, {r_[key][2]:+.2f}]" for r_ in res) + r" \\")
        print(f"  {lab:28s} " + "  ".join(
            "  n/a " if r_.get(key) is None else f"{r_[key][0]:+.2f}[{r_[key][1]:+.2f},{r_[key][2]:+.2f}]"
            for r_ in res))
    REFV = {"identity": 1.0, "scale_0.25": 1.0, "scale_4": 1.0, "reflect": -1.0, "rotate": 0.0, "blind": 0.0}
    ref_cells = []
    for name, D_, runs, trs, gt in cols:
        if len(trs) == 1:
            ref_cells.append("")
            continue
        pts = [(r, t) for r in runs for t in trs if gt in D_[r][t]]
        ref_cells.append(fmt_rho((stats.spearmanr([REFV[t] for _, t in pts], [D_[r][t][gt] for r, t in pts]).correlation,)))
    lines.append(r"\midrule")
    lines.append(r"reference, no data or model & " + " & ".join(ref_cells) + r" \\")
    print("  reference (no data, no model), transformed columns: " + " ".join(c for c in ref_cells if c))
    lines += [r"\bottomrule", r"\end{tabular}"]
    ci_lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "validity.tex").write_text("\n".join(lines) + "\n")
    (OUT / "validity_ci.tex").write_text("\n".join(ci_lines) + "\n")
    for f in ["validity_models.tex", "validity_transforms.tex"]:
        if (OUT / f).exists():
            (OUT / f).unlink()

    ordinary = sorted(r for r in lib_all if "_causal_" in r)
    no_outlier = [r for r in ordinary if not r.startswith("cf_idm_causal")]
    objective = lambda r: r.split("_")[1]
    objs = sorted({objective(r) for r in ordinary})
    print(f"\n[validity] within {len(ordinary)} models differing only in objective and seed "
          f"({len(objs)} objectives); intervals resample objectives, since seeds of one objective are not "
          f"independent draws")
    for gt in ("dir_on", "direction"):
        sd = [np.std([V[r]["identity"][gt] for r in ordinary if objective(r) == o], ddof=1) for o in objs]
        print(f"  seed standard deviation of {gt} per objective: median {np.median(sd):.3f}")
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r" & \multicolumn{3}{c}{D logged} & D random & D logged \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-5}\cmidrule(lr){6-6}",
             r"probe & all 30 & objective means & seed level & seed level & without inv.\ dyn.\ $w=1$ \\",
             r"\midrule"]

    def levels(key, sgn, gt, runs):
        k = resolve(key, gt)
        gm = [np.mean([sgn * V[r]["identity"][k] for r in runs if objective(r) == o]) for o in objs]
        gt_m = [np.mean([V[r]["identity"][gt] for r in runs if objective(r) == o]) for o in objs]
        x, y = [], []
        for o in objs:
            px = np.array([sgn * V[r]["identity"][k] for r in runs if objective(r) == o])
            py = np.array([V[r]["identity"][gt] for r in runs if objective(r) == o])
            x += list(px - px.mean()); y += list(py - py.mean())
        return stats.spearmanr(gm, gt_m).correlation, stats.spearmanr(x, y).correlation

    for key, lab, sgn in PROBES:
        a = validity(V, ordinary, ["identity"], "dir_on", [(key, lab, sgn)], group_of=objective)[key]
        c = validity(V, no_outlier, ["identity"], "dir_on", [(key, lab, sgn)], group_of=objective)[key]
        om, sl = levels(key, sgn, "dir_on", ordinary)
        _, sr = levels(key, sgn, "direction", ordinary)
        lines.append(f"{lab} & {a[0]:+.2f} & {om:+.2f} & {sl:+.2f} & {sr:+.2f} & {c[0]:+.2f} \\\\")
        print(f"  {lab:28s} all30 D_log {a[0]:+.2f} [{a[1]:+.2f},{a[2]:+.2f}]  objective means {om:+.2f}  "
              f"seed level D_log {sl:+.2f}  seed level D_rand {sr:+.2f}  no-outlier D_log {c[0]:+.2f} "
              f"[{c[1]:+.2f},{c[2]:+.2f}]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "validity_within.tex").write_text("\n".join(lines) + "\n")

    print("\n[matched k] LIBERO rho with on support / random perturbation ground truth")
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r" & \multicolumn{2}{c}{logged action swaps} & \multicolumn{2}{c}{random perturbations} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"pairs per position & as trained & transformed & as trained & transformed \\", r"\midrule"]
    for k in (1, 3, 5):
        key = "matched" if k == 1 else f"matched_k{k}"
        pr = [(key, "", 1)]
        on = f"dir_on_k{k}"
        vals = [validity(V, lib_all, trs, gt, pr)[key][0] for gt in (on, "direction")
                for trs in (["identity"], TRANSFORMS)]
        lines.append(f"{k} & " + " & ".join(f"{v:+.3f}" for v in vals) + r" \\")
        print(f"  k={k} " + " ".join(f"{v:+.3f}" for v in vals))
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "matched_k.tex").write_text("\n".join(lines) + "\n")

    if td_all:
        groups = {}
        for r in td_all:
            groups.setdefault(r.rsplit("-", 1)[0], []).append(r)
        print(f"\n[validity] TD-MPC2 within task (task means removed, {len(groups)} tasks)")
        for key, lab, sgn in PROBES:
            if resolve(key, "direction") not in R[td_all[0]]["identity"]:
                continue
            r_ = demeaned_validity(R, groups, key, sgn)
            print(f"  {lab:28s} {r_[0]:+.2f} [{r_[1]:+.2f},{r_[2]:+.2f}] n={r_[3]}")

    groups = {"logged": "causal", "sequence perm.": "shuf", "derangement": "derange",
              "per timestep": "tstep", "action blind": "blind"}
    ck = [("direction", r"D random $\uparrow$", "{:.3f}"), ("dir_on", r"D logged $\uparrow$", "{:.3f}"),
          ("C1", r"$C_1$ $\uparrow$", "{:.3f}"), ("task1", r"Task 1 $\uparrow$", "{:.1f}"),
          ("ret_gauss_meanrank", r"ret.\ rank $\downarrow$", "{:.2f}"), ("matched", r"matched $\uparrow$", "{:.3f}")]
    lines = [r"\begin{tabular}{l" + "r" * len(ck) + "}", r"\toprule",
             "training & " + " & ".join(k[1] for k in ck) + r" \\", r"\midrule"]
    print("\n[controls] additive objective, mean over seeds")
    for lab, suf in groups.items():
        runs = [f"cf_add_{suf}_s{s}" for s in S5 if f"cf_add_{suf}_s{s}" in V]
        cells = [f.format(np.mean([V[r]["identity"][k] for r in runs])) for k, _, f in ck]
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
        print(f"  {lab:16s} n={len(runs)} " + " ".join(f"{k}={c}" for (k, _, _), c in zip(ck, cells)))
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "controls.tex").write_text("\n".join(lines) + "\n")

    print("\n[objectives] paired by seed against additive")
    base = [f"cf_add_causal_s{s}" for s in S5]
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r" & \multicolumn{2}{c}{simulator: $\Delta$ direction} & \multicolumn{3}{c}{probe: $\Delta$ score} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}",
             r"objective vs.\ additive & random $\uparrow$ & logged $\uparrow$ & ret.\ rank & matched $\uparrow$ & $p$ (matched) \\",
             r"\midrule"]
    objs = [("agm", "+ AGM"), ("idm", "+ inv.\\ dyn.\\ ($w=1$)"), ("idm01", "+ inv.\\ dyn.\\ ($w=0.01$)"),
            ("idm001", "+ inv.\\ dyn.\\ ($w=0.001$)"), ("delta", "delta prediction")]
    for m, lab in objs:
        runs = [f"cf_{m}_causal_s{s}" for s in S5]
        d = {k: np.array([V[r]["identity"][k] for r in runs]) - np.array([V[r]["identity"][k] for r in base])
             for k in ["direction", "dir_on", "ret_gauss_meanrank", "matched"]}
        p = {k: stats.ttest_rel(np.array([V[r]["identity"][k] for r in runs]),
                                np.array([V[r]["identity"][k] for r in base])).pvalue for k in d}
        ps = f"{p['matched']:.3f}" if p["matched"] >= 0.001 else r"$<$0.001"
        lines.append(f"{lab} & {d['direction'].mean():+.3f} & {d['dir_on'].mean():+.3f} & "
                     f"{d['ret_gauss_meanrank'].mean():+.2f} & {d['matched'].mean():+.3f} & {ps} \\\\")
        print(f"  {m:7s} " + " ".join(f"d_{k}={d[k].mean():+.4f}(p={p[k]:.4f})" for k in d))
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "objectives.tex").write_text("\n".join(lines) + "\n")

    A, AM = _load("adversary_study_v2.json"), _load("adversary_matched.json")
    if A:
        seeds = sorted(A)
        names = [("mse", "trained, MSE"), ("agm", "trained, + AGM"), ("blind", "action blind base"),
                 ("adv_const_c2", "blind + unconditional inflation"),
                 ("adv_policy_c2", "blind + policy aligned inflation"),
                 ("adv_policy_c8", "blind + policy aligned, $c=8$")]
        has_m = bool(AM) and all(s in AM for s in seeds)
        lines = [r"\begin{tabular}{lrrrr" + ("r" if has_m else "") + "}", r"\toprule",
                 r"model (world G) & true direction & $C_1$ & response & ret.\ excess"
                 + (r" & matched" if has_m else "") + r" \\", r"\midrule"]
        print("\n[synthetic] world G, logged actions, mean over seeds")
        for k, lab in names:
            key = f"G/causal/{k}"
            ex = [A[s][key]["retrieval"] - A[s][key]["retrieval_null"] for s in seeds]
            c1 = [A[s][key]["C1"] for s in seeds]
            rs = [A[s][key]["response"] for s in seeds]
            dr = [A[s][key]["direction_gt"] for s in seeds]
            row = (f"{lab} & {np.nanmean(dr):+.3f} & {np.mean(c1):.2f} & {np.mean(rs):.2f} & "
                   f"{np.mean(ex):+.3f}")
            msg = (f"  {lab:34s} dir={np.nanmean(dr):+.3f} C1={np.mean(c1):.3f} resp={np.mean(rs):.3f} "
                   f"excess={np.mean(ex):+.3f}+-{np.std(ex, ddof=1):.3f}")
            if has_m:
                mv = [AM[s][key]["matched"] for s in seeds]
                d2 = [AM[s][key]["direction_gt"] for s in seeds]
                row += f" & {np.mean(mv):+.3f}"
                msg += (f" matched={np.mean(mv):+.3f}+-{np.std(mv, ddof=1):.3f} "
                        f"(model check: dir rerun {np.nanmean(d2):+.3f})")
            lines.append(row + r" \\")
            print(msg)
        lines += [r"\bottomrule", r"\end{tabular}"]
        (OUT / "synthetic.tex").write_text("\n".join(lines) + "\n")
        if has_m:
            print("[synthetic] world N matched (should be near 0): " + ", ".join(
                f"{k}={np.mean([AM[s][f'N/causal/{k}']['matched'] for s in seeds]):+.3f}" for k, _ in names))
            print("[synthetic] shuffled labels, world G matched: " + ", ".join(
                f"{k}={np.mean([AM[s][f'G/shuffled/{k}']['matched'] for s in seeds]):+.3f}" for k, _ in names))

    if td_all:
        tasks = sorted({r.rsplit("-", 1)[0] for r in td_all})
        lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
                 r"task & direction & size & $C_1$ & Task 1 & ret.\ rank & matched \\", r"\midrule"]
        for task in tasks:
            runs = [r for r in td_all if r.rsplit("-", 1)[0] == task]
            g = lambda k: np.array([R[r]["identity"][k] for r in runs])
            d = g("direction")
            lines.append(f"{task} & {d.mean():+.3f}$\\pm${d.std(ddof=1) if len(d) > 1 else 0:.3f} & "
                         f"{g('mag_ratio').mean():.2f} & {g('C1').mean():.4f} & {g('task1').mean():.1f} & "
                         f"{g('ret_gauss_meanrank').mean():.2f} & {g('matched').mean():+.3f} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (OUT / "tdmpc2_tasks.tex").write_text("\n".join(lines) + "\n")
        dirs = np.array([R[r]["identity"]["direction"] for r in td_all])
        mags = np.array([R[r]["identity"]["mag_ratio"] for r in td_all])
        print(f"\n[tdmpc2] direction over {len(td_all)} released models: mean {dirs.mean():.3f} "
              f"min {dirs.min():.3f} max {dirs.max():.3f}; size mean {mags.mean():.2f} "
              f"[{mags.min():.2f}, {mags.max():.2f}]")
        worst = sorted(td_all, key=lambda r: R[r]["identity"]["direction"])[:5]
        print("[tdmpc2] least faithful:", [(r, round(R[r]["identity"]["direction"], 3)) for r in worst])
    print(f"\n[tables] wrote {sorted(p.name for p in OUT.glob('*.tex'))}")


if __name__ == "__main__":
    main()
