from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "tables"
TRANSFORMS = ["identity", "reflect", "rotate", "scale_0.25", "scale_4", "blind"]
LABEL = {"identity": "trained model", "reflect": "reflected response", "rotate": "rotated response",
         "scale_0.25": r"response $\times0.25$", "scale_4": r"response $\times4$",
         "blind": "no response"}


def load(tag=""):
    return json.load(open(RES / f"diamond_validity{tag}.json"))


def _norm_ret(R, g, t):
    return R[g][t]["ret_meanrank"] / R[g]["meta"]["chance_rank"]


def transforms_table(R, games):
    rows = []
    for t in TRANSFORMS:
        rows.append("{} & {:+.3f} & {:.2f} & {:.4f} & {:.2f} & {:+.3f} \\\\".format(
            LABEL[t],
            np.mean([R[g][t]["direction"] for g in games]),
            np.mean([R[g][t]["mag_ratio"] for g in games]),
            np.mean([R[g][t]["C1"] for g in games]),
            np.mean([_norm_ret(R, g, t) for g in games]),
            np.mean([R[g][t]["matched"] for g in games])))
    return ("\\begin{tabular}{lrr|rrr}\n\\toprule\n"
            " & \\multicolumn{2}{c|}{simulator} & \\multicolumn{3}{c}{probe (no simulator)} \\\\\n"
            "model & direction $\\uparrow$ & size & $C_1$ $\\uparrow$ & ret.\\ rank $\\downarrow$ & matched $\\uparrow$ \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def validity(R, games):
    P1 = [(g, "identity") for g in games]
    P6 = [(g, t) for g in games for t in TRANSFORMS]
    truth = {(g, t): R[g][t]["direction"] for g, t in P6}

    def demean(d):
        out = {}
        for g in games:
            mu = np.mean([d[(g, t)] for t in TRANSFORMS])
            for t in TRANSFORMS:
                out[(g, t)] = d[(g, t)] - mu
        return out

    cols = {}
    probes = [("C1", lambda g, t: R[g][t]["C1"]),
              ("retrieval", lambda g, t: -_norm_ret(R, g, t)),
              ("matched", lambda g, t: R[g][t]["matched"])]
    if "inv_dir_swap" in R[games[0]]["identity"]:
        probes += [("inverse err", lambda g, t: -R[g][t]["inv_err_swap"]),
                   ("inverse dir", lambda g, t: R[g][t]["inv_dir_swap"])]
    for name, get in probes:
        d = {(g, t): get(g, t) for g, t in P6}
        dd, tt = demean(d), demean(truth)
        cols[name] = (
            stats.spearmanr([d[k] for k in P1], [truth[k] for k in P1]).correlation,
            stats.spearmanr([d[k] for k in P6], [truth[k] for k in P6]).correlation,
            stats.spearmanr([dd[k] for k in P6], [tt[k] for k in P6]).correlation)
    return cols


REFERENCE = {"identity": 1.0, "scale_0.25": 1.0, "scale_4": 1.0, "reflect": -1.0, "rotate": 0.0,
             "blind": 0.0}


def invariance_reference(R, games):
    P6 = [(g, t) for g in games for t in TRANSFORMS]
    return stats.spearmanr([REFERENCE[t] for g, t in P6], [R[g][t]["direction"] for g, t in P6]).correlation


def as_trained_intervals(R, games, n_boot=5000, seed=0):
    d = np.array([R[g]["identity"]["direction"] for g in games])
    vals = {"matched": np.array([R[g]["identity"]["matched"] for g in games]),
            "retrieval": np.array([-_norm_ret(R, g, "identity") for g in games]),
            "zero action divergence": np.array([R[g]["identity"]["C1"] for g in games])}
    diffs = [("matched minus retrieval", "matched", "retrieval")]
    if "inv_dir_swap" in R[games[0]]["identity"]:
        vals["inverse direction"] = np.array([R[g]["identity"]["inv_dir_swap"] for g in games])
        vals["inverse cycle error"] = np.array([-R[g]["identity"]["inv_err_swap"] for g in games])
        diffs.append(("matched minus inverse direction", "matched", "inverse direction"))
    rng = np.random.default_rng(seed)
    boot = {k: [] for k in list(vals) + [d[0] for d in diffs]}
    for _ in range(n_boot):
        i = rng.integers(0, len(games), len(games))
        r = {k: stats.spearmanr(v[i], d[i]).correlation for k, v in vals.items()}
        for k in vals:
            boot[k].append(r[k])
        for name, a, b in diffs:
            boot[name].append(r[a] - r[b])
    point = {k: stats.spearmanr(v, d).correlation for k, v in vals.items()}
    for name, a, b in diffs:
        point[name] = point[a] - point[b]
    return {k: (point[k], *np.nanpercentile(boot[k], [2.5, 97.5])) for k in boot}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "tables"))
    args = ap.parse_args()
    R = load()
    games = sorted(R)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "transforms_diamond.tex").write_text(transforms_table(R, games))
    print("[diamond] wrote", out / "transforms_diamond.tex")

    meta = R[games[0]]["meta"]
    floor = max(R[g]["meta"]["emulator_noise_floor"] for g in games)
    d = np.array([R[g]["identity"]["direction"] for g in games])
    print(f"\ngames {len(games)}, positions {meta['n_positions']}, episodes {meta['n_episodes']}, "
          f"worst emulator repeat difference {floor}")
    print(f"direction: mean {d.mean():.2f}, min {d.min():.2f} ({games[int(np.argmin(d))]}), "
          f"max {d.max():.2f} ({games[int(np.argmax(d))]})")
    print(f"size: mean {np.mean([R[g]['identity']['mag_ratio'] for g in games]):.2f}")
    c = validity(R, games)
    print("\nSpearman with direction   as trained | transformed | within game")
    for k, v in c.items():
        print(f"  {k:12s} {v[0]:+11.2f} | {v[1]:+11.2f} | {v[2]:+11.2f}")
    print(f"\ninvariance reference (no data, no model), transformed population: "
          f"{invariance_reference(R, games):+.2f}")
    eps = [R[g]["meta"]["n_episodes"] for g in games]
    print(f"episodes per game {min(eps)} to {max(eps)}; positions skipped because a branch ended "
          f"{sum(R[g]['meta'].get('n_positions_skipped_ended', 0) for g in games)}; swaps dropped "
          f"{sum(R[g]['meta'].get('n_pairs_dropped_ended', 0) for g in games)}")
    crn = np.array([R[g]["identity"]["direction"] - R[g]["meta"]["direction_no_crn"] for g in games])
    print(f"common random numbers: direction minus direction with independent noise, mean "
          f"{crn.mean():+.3f}, largest {np.abs(crn).max():.3f} ({games[int(np.argmax(np.abs(crn)))]})")
    if "inv_val_acc" in meta:
        acc = np.array([R[g]["meta"]["inv_val_acc"] / R[g]["meta"]["inv_chance"] for g in games])
        print(f"inverse model held out accuracy, as a multiple of chance: median {np.nanmedian(acc):.1f}, "
              f"range {np.nanmin(acc):.1f} to {np.nanmax(acc):.1f}")
        for t in TRANSFORMS:
            print(f"  inverse direction, {t:11s} mean {np.mean([R[g][t]['inv_dir_swap'] for g in games]):+.3f}")
    ci = as_trained_intervals(R, games)
    print("\nas trained, 95% bootstrap over games (resampling games):")
    for k, (r, lo, hi) in ci.items():
        print(f"  {k:24s} {r:+.2f} [{lo:+.2f}, {hi:+.2f}]")
    best = max(abs(c['C1'][1]), abs(c['retrieval'][1]))
    print(f"\nbest existing probe with transforms (absolute): {best:.2f}; "
          f"matched {c['matched'][1]:+.2f}")

    p1 = RES / "diamond_validity_seed1.json"
    if p1.exists():
        S = json.load(open(p1))
        gs = sorted(S)
        a = np.array([R[g]["identity"]["direction"] for g in gs])
        b = np.array([S[g]["identity"]["direction"] for g in gs])
        ma = np.array([R[g]["identity"]["matched"] for g in gs])
        mb = np.array([S[g]["identity"]["matched"] for g in gs])
        print(f"\nseed replication on {len(gs)} games: direction mean |diff| {np.abs(a-b).mean():.3f} "
              f"(rank corr {stats.spearmanr(a, b).correlation:+.2f}), "
              f"matched mean |diff| {np.abs(ma-mb).mean():.3f} "
              f"(rank corr {stats.spearmanr(ma, mb).correlation:+.2f})")


if __name__ == "__main__":
    main()
