import json
import os
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
RR, RT = ROOT / "results" / "additional", ROOT / "results" / "tables"
OUT = Path(os.environ.get("TABLES_OUT", ROOT / "tables"))
OUT.mkdir(parents=True, exist_ok=True)
N = {}

obj = lambda r: r.split("_")[1]
ORD = ("add", "agm", "delta", "idm", "idm01", "idm001")
TR = ("identity", "reflect", "rotate", "scale_0.25", "scale_4", "blind")


def subsets(runs):
    runs = sorted(runs)
    return {"30": [r for r in runs if obj(r) in ORD],
            "40": [r for r in runs if not obj(r).startswith(("act", "md"))],
            "55": [r for r in runs if not (obj(r).startswith("act") or obj(r) in ("idm", "mdshuf01"))],
            "65": [r for r in runs if not obj(r).startswith("act")],
            "90": runs}


def boot(rs, probes, truth, leads=(), B=4000):
    groups = {}
    for r in rs:
        groups.setdefault(obj(r), []).append(r)
    keys = sorted(groups)
    rng = np.random.default_rng(0)
    t = {r: truth(r) for r in rs}
    pv_ = {p: {r: probes[p](r) for r in rs} for p in probes}
    sp = lambda pick, p: spearmanr([pv_[p][r] for r in pick], [t[r] for r in pick])[0]
    bs = {p: [] for p in probes}
    for _ in range(B):
        pick = [r for k in rng.choice(keys, len(keys)) for r in groups[k]]
        for p in probes:
            bs[p].append(sp(pick, p))
    bs = {p: np.array(v) for p, v in bs.items()}
    res = {p: (sp(rs, p), *np.nanpercentile(bs[p], [2.5, 97.5])) for p in probes}
    for a, b in leads:
        res[f"{a}-{b}"] = (res[a][0] - res[b][0], *np.nanpercentile(bs[a] - bs[b], [2.5, 97.5]))
    return res


f2 = lambda x: f"{x:+.2f}".replace("+-", "-")
ci = lambda v: f"{f2(v[0])} [{f2(v[1])}, {f2(v[2])}]"
num = lambda v: [round(float(x), 4) for x in v]


def write(name, lines):
    (OUT / name).write_text("\n".join(lines) + "\n")
    print(f"wrote {name}")


pv = json.load(open(RT / "probe_validity_v3.json"))
ms = json.load(open(RT / "matched_v3.json"))["models"]
dt = json.load(open(RT / "direction_training_v3.json"))
ipp = json.load(open(RT / "inverse_probe_v3.json"))["models"]
PM = {}
for r in pv:
    if r.split("_")[2] == "causal":
        PM[r] = dict(D=pv[r]["identity"]["direction"], Dlog=ms[r]["identity"]["dir_on"],
                     matched=ms[r]["identity"]["matched_k1"], ret=-pv[r]["identity"]["ret_gauss_meanrank"],
                     val=-json.load(open(ROOT / f"results/raw/{r}/metrics.json"))["best_val_mse"],
                     size=pv[r]["identity"]["mag_ratio"], ic=ipp[r]["identity"]["inv_dir_swap"])
for r, v in dt.items():
    if obj(r) in ("ac001", "ac01", "ac1") or r in PM:
        continue
    PM[r] = dict(D=v["direction"], Dlog=v["dir_on"], matched=v["matched"], ret=-v["ret_gauss_meanrank"],
                 val=-v["val_mse"], size=v["mag_ratio"], ic=None)
assert len(PM) == 90, len(PM)
ipx = json.load(open(RR / "inverse_extra_v3.json"))
assert all(abs(c["new"]["inv_dir_swap"] - c["saved"]["inv_dir_swap"]) < 1e-6 for c in ipx["check"].values())
for r, v in ipx["models"].items():
    PM[r]["ic"] = v["identity"]["inv_dir_swap"]
assert all(PM[r]["ic"] is not None for r in PM)
assert max(abs(ms[r]["identity"]["dir_on"] - ms[r]["identity"]["dir_on_k1"]) for r in ms) == 0
SN92 = json.load(open(RR / "sizenorm_retrieval.json"))
SN60 = json.load(open(RR / "sizenorm_retrieval_study.json"))
assert SN60["c"] == SN92["c"]
for r in PM:
    PM[r]["sn"] = -(SN92["per_model"][r]["identity"] if r in SN92["per_model"] else SN60["per_model"][r])
PS = subsets(PM)
assert [len(PS[k]) for k in ("30", "40", "55", "65", "90")] == [30, 40, 55, 65, 90]
g = lambda k: (lambda r: PM[r][k])
prop = {}
for s in ("30", "55", "65", "90"):
    probes = {"matched": g("matched"), "ret": g("ret"), "val": g("val"), "ic": g("ic"), "sn": g("sn")}
    prop[s] = boot(PS[s], probes, g("Dlog"), leads=[("matched", "ret"), ("matched", "val"), ("matched", "ic"), ("matched", "sn")])
N["proprio_validity"] = {s: {k: num(v) for k, v in prop[s].items()} for s in prop}
N["proprio_n_objectives"] = {s: len({obj(r) for r in PS[s]}) for s in PS}
fam = lambda r: next((f for f in ("act", "md", "imp", "idm") if obj(r).startswith(f)), obj(r))
N["proprio_leave_family_out_matched"] = {f: round(float(spearmanr([PM[r]["matched"] for r in PS["90"] if fam(r) != f],
                                                                    [PM[r]["Dlog"] for r in PS["90"] if fam(r) != f])[0]), 4)
                                         for f in sorted({fam(r) for r in PS["90"]})}

IMG = {}
for size, folder in (("1.2M", "img"), ("5.5M", "img_medium"), ("26.8M", "img_large")):
    d = RR / folder
    ipv = json.load(open(d / "probe_validity_v3.json"))
    imi = json.load(open(d / "matched_v3.json"))["models"]
    ival = json.load(open(d / "val_mse.json"))
    iip = json.load(open(d / "inverse_probe_v3.json"))["models"] if (d / "inverse_probe_v3.json").exists() else None
    IMG[size] = dict(pv=ipv, mi=imi, val=ival, ip=iip)
    causal = sorted(r for r in ipv if r.split("_")[2] == "causal")
    IMG[size]["sets"] = subsets(causal) if len(causal) == 90 else {"30": [r for r in causal if obj(r) in ORD]}
SNI = json.load(open(RR / "server" / "sizenorm_img_20261118.json"))
assert len(IMG["1.2M"]["sets"]["90"]) == 90 and len(IMG["5.5M"]["sets"]["30"]) == 30 and len(IMG["26.8M"]["sets"]["30"]) == 30


def ig(size, key, t="identity"):
    I = IMG[size]
    return {"matched": lambda r: I["mi"][r][t]["matched_k1"], "Dlog": lambda r: I["mi"][r][t]["dir_on"],
            "D": lambda r: I["pv"][r][t]["direction"], "ret": lambda r: -I["pv"][r][t]["ret_gauss_meanrank"],
            "ret_real": lambda r: -I["pv"][r][t]["ret_marg_meanrank"], "C1": lambda r: I["pv"][r][t]["C1"],
            "task1": lambda r: I["pv"][r][t]["task1"], "val": lambda r: -I["val"][r],
            "ic": lambda r: I["ip"][r][t]["inv_dir_swap"], "sn": lambda r: -SNI[size]["per_model"][r]}[key]


img = {}
for s in ("30", "55", "65", "90"):
    probes = {k: ig("1.2M", k) for k in ("matched", "ret", "ret_real", "C1", "task1", "ic", "val", "sn")}
    img[s] = boot(IMG["1.2M"]["sets"][s], probes, ig("1.2M", "Dlog"), leads=[("matched", "ret"), ("matched", "ic"), ("matched", "val"), ("matched", "sn")])
for size in ("5.5M", "26.8M"):
    keys = ("matched", "ret", "C1", "val", "sn") + (("ic",) if IMG[size]["ip"] is not None else ())
    probes = {k: ig(size, k) for k in keys}
    img[f"30_{size}"] = boot(IMG[size]["sets"]["30"], probes, ig(size, "Dlog"),
                             leads=[("matched", "ret"), ("matched", "val"), ("matched", "sn")] + ([("matched", "ic")] if "ic" in keys else []))
N["image_validity"] = {s: {k: num(v) for k, v in img[s].items()} for s in img}
N["image_n_objectives"] = {s: len({obj(r) for r in IMG["1.2M"]["sets"][s]}) for s in ("30", "55", "65", "90")}

rows = [("LIBERO, proprioception", "0.8M", "30 ordinary", prop["30"], True),
        ("", "", "65 without contrastive", prop["65"], True),
        ("", "", "all 90 trained", prop["90"], True),
        ("LIBERO, image features", "1.2M", "30 ordinary", img["30"], True),
        ("", "", "all 90 trained", img["90"], True),
        ("", "5.5M", "30 ordinary", img["30_5.5M"], True),
        ("", "26.8M", "30 ordinary", img["30_26.8M"], True)]
L = [r"\begin{tabular}{lllrrrrrc}", r"\toprule",
     r"inputs & params & models & matched probe & retrieval & size norm.\ retrieval & inverse check & val.\ error & matched minus retrieval \\",
     r"\midrule"]
for i, (a, b, c, v, has_ic) in enumerate(rows):
    if i == 3:
        L.append(r"\midrule")
    icv = f2(v["ic"][0]) if has_ic and "ic" in v else ""
    L.append(f"{a} & {b} & {c} & {f2(v['matched'][0])} & {f2(v['ret'][0])} & {f2(v['sn'][0])} & {icv} & {f2(v['val'][0])} & "
             f"{ci(v['matched-ret'])} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
write("extra_main.tex", L)

names = [("matched", "matched probe (ours)"), ("ic", "inverse model, direction"), ("ret", "retrieval"),
         ("ret_real", "retrieval, real negatives"), ("sn", "size normalized retrieval"), ("C1", "zero action divergence"), ("task1", "WSP Task 1"),
         ("val", "validation error"), ("matched-ret", "matched minus retrieval"), ("matched-sn", "matched minus size normalized retrieval"), ("matched-ic", "matched minus inverse"),
         ("matched-val", "matched minus validation error")]
L = [r"\begin{tabular}{lcccc}", r"\toprule",
     r"probe & 30 ordinary & 55 working & 65 without contrastive & all 90 \\", r"\midrule"]
for k, lab in names:
    if k == "matched-ret":
        L.append(r"\midrule")
    L.append(lab + " & " + " & ".join(ci(img[s][k]) for s in ("30", "55", "65", "90")) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
write("extra_img_ci.tex", L)

L = [r"\begin{tabular}{lcccc}", r"\toprule",
     r"probe & 30 ordinary & 55 working & 65 without contrastive & all 90 \\", r"\midrule"]
for k, lab in (("matched", "matched probe (ours)"), ("ic", "inverse model, direction"), ("ret", "retrieval"), ("sn", "size normalized retrieval"), ("val", "validation error"),
               ("matched-ret", "matched minus retrieval"), ("matched-sn", "matched minus size normalized retrieval"), ("matched-ic", "matched minus inverse"), ("matched-val", "matched minus validation error")):
    if k == "matched-ret":
        L.append(r"\midrule")
    L.append(lab + " & " + " & ".join(ci(prop[s][k]) for s in ("30", "55", "65", "90")) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
write("extra_prop_ci.tex", L)

tr15 = lambda size: [r for r in IMG[size]["sets"]["30"] if obj(r) in ("add", "agm", "delta")]
L = [r"\begin{tabular}{llrrrrrrr}", r"\toprule",
     r"params & transform & D random & D logged & size & zero action div. & WSP Task 1 & ret.\ rank $\downarrow$ & matched probe \\",
     r"\midrule"]
scale = {}
for size in ("1.2M", "5.5M", "26.8M"):
    I = IMG[size]
    for t in TR:
        rs = tr15(size)
        v = dict(D=np.mean([I["pv"][r][t]["direction"] for r in rs]), Dlog=np.mean([I["mi"][r][t]["dir_on"] for r in rs]),
                 size=np.mean([I["pv"][r][t]["mag_ratio"] for r in rs]), C1=np.mean([I["pv"][r][t]["C1"] for r in rs]),
                 task1=np.mean([I["pv"][r][t]["task1"] for r in rs]), ret=np.mean([I["pv"][r][t]["ret_gauss_meanrank"] for r in rs]),
                 matched=np.mean([I["mi"][r][t]["matched_k1"] for r in rs]))
        scale[(size, t)] = v
        lab = {"identity": "as trained", "reflect": "reflected", "rotate": "rotated", "scale_0.25": r"$\times 0.25$",
               "scale_4": r"$\times 4$", "blind": "no response"}[t]
        z3 = lambda x: "0.000" if abs(x) < 5e-4 else f"{x:+.3f}"
        L.append(f"{size if t == 'identity' else ''} & {lab} & {z3(v['D'])} & {z3(v['Dlog'])} & {v['size']:.2f} & "
                 f"{v['C1']:.3f} & {v['task1']:.1f} & {v['ret']:.2f} & {z3(v['matched'])} \\\\")
    if size != "26.8M":
        L.append(r"\midrule")
L += [r"\bottomrule", r"\end{tabular}"]
write("extra_img_transforms.tex", L)
N["image_transforms_15"] = {f"{s}|{t}": {k: round(float(x), 4) for k, x in v.items()} for (s, t), v in scale.items()}

for size in ("1.2M", "5.5M", "26.8M"):
    rs = IMG[size]["sets"]["30"]
    for p in ("matched", "ret", "C1"):
        a = [ig(size, p, t)(r) for r in rs for t in TR]
        b = [ig(size, "Dlog", t)(r) for r in rs for t in TR]
        N.setdefault("image_transformed_30", {})[f"{size}|{p}"] = round(float(spearmanr(a, b)[0]), 4)
    N.setdefault("image_ratio_matched_over_Dlog", {})[size] = {
        o: round(float(np.mean([ig(size, "matched")(r) for r in rs if obj(r) == o]) /
                       np.mean([ig(size, "Dlog")(r) for r in rs if obj(r) == o])), 3) for o in ORD}
I1 = IMG["1.2M"]
N["image_estimate_cos_k1"] = round(json.load(open(RR / "img" / "matched_v3.json"))["summary"]["estimate_vs_sim_cos_k1"], 4)
N["image_estimate_cos_k1_proprio_pairs"] = round(json.load(open(RR / "img" / "matched_v3pp.json"))["summary"]["estimate_vs_sim_cos_k1"], 4)
N["image_inverse_r2"] = round(json.load(open(RR / "img" / "inverse_probe_v3.json"))["inverse_model"]["val_r2_mean"], 4)
N["image_val_blind_vs_trained"] = {o: dict(trained=round(float(np.mean([I1["val"][f"cf_{o}_causal_s{s}"] for s in range(5)])), 4),
                                           blind=round(float(np.mean([I1["val"][f"cf_{o}_blind_s{s}"] for s in range(5)])), 4))
                                   for o in ("add", "delta")}
pp = json.load(open(RR / "img" / "matched_v3pp.json"))["models"]
paper92 = sorted(I1["pv"])[:0] + [r for r in sorted(I1["pv"]) if r in pv]
assert len(paper92) == 92
N["image_paper92_as_trained"] = dict(
    matched_img_pairs=round(float(spearmanr([I1["mi"][r]["identity"]["matched_k1"] for r in paper92],
                                            [I1["mi"][r]["identity"]["dir_on"] for r in paper92])[0]), 4),
    matched_proprio_pairs=round(float(spearmanr([pp[r]["identity"]["matched_k1"] for r in paper92],
                                                [I1["mi"][r]["identity"]["dir_on"] for r in paper92])[0]), 4),
    ic=round(float(spearmanr([I1["ip"][r]["identity"]["inv_dir_swap"] for r in paper92],
                             [I1["mi"][r]["identity"]["dir_on"] for r in paper92])[0]), 4))

L = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
     r" & & & \multicolumn{4}{c}{30 ordinary models, as trained} & \multicolumn{2}{c}{with transformed copies} \\",
     r"\cmidrule(lr){4-7}\cmidrule(lr){8-9}",
     r"params & D random & D logged & matched probe & retrieval & zero action div. & val.\ error & matched probe & retrieval \\",
     r"\midrule"]
for size, key in (("1.2M", "30"), ("5.5M", "30_5.5M"), ("26.8M", "30_26.8M")):
    v = img[key]
    t = scale[(size, "identity")]
    L.append(f"{size} & {t['D']:.3f} & {t['Dlog']:.3f} & {f2(v['matched'][0])} & {f2(v['ret'][0])} & {f2(v['C1'][0])} & "
             f"{f2(v['val'][0])} & {f2(N['image_transformed_30'][size + '|matched'])} & {f2(N['image_transformed_30'][size + '|ret'])} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
write("extra_scale.tex", L)

def paired(get, base, run):
    a = [get(f"cf_{run}_causal_s{s}") for s in range(5)]
    b = [get(f"cf_{base}_causal_s{s}") for s in range(5)]
    out = {}
    for k in ("D", "Dlog"):
        x, y = np.array([u[k] for u in a]), np.array([u[k] for u in b])
        out[k] = (float((x - y).mean()), float(stats.ttest_rel(x, y).pvalue), int((x < y).sum()))
    vr = np.array([u["val"] for u in a]) / np.array([u["val"] for u in b])
    out["val_ratio"] = (float(vr.mean()), float(vr.min()), float(vr.max()))
    out["ret"] = (float(np.mean([u["ret"] for u in b])), float(np.mean([u["ret"] for u in a])),
                  float(stats.ttest_rel([u["ret"] for u in a], [u["ret"] for u in b]).pvalue))
    out["size"] = float(np.mean([u["size"] for u in a]) / np.mean([u["size"] for u in b]))
    return out


def pget(r):
    if r in PM:
        m = PM[r]
        return dict(D=m["D"], Dlog=m["Dlog"], ret=-m["ret"], val=-m["val"], size=m["size"])
    v = dt[r]
    return dict(D=v["direction"], Dlog=v["dir_on"], ret=v["ret_gauss_meanrank"], val=v["val_mse"], size=v["mag_ratio"])


def iget(size):
    I = IMG[size]
    return lambda r: dict(D=I["pv"][r]["identity"]["direction"], Dlog=I["mi"][r]["identity"]["dir_on"],
                          ret=I["pv"][r]["identity"]["ret_gauss_meanrank"], val=I["val"][r], size=I["pv"][r]["identity"]["mag_ratio"])


G = [("proprioception", "0.8M", r"contrastive, $w=0.1$", pget, "act1"),
     ("", "", r"contrastive, $w=0.01$", pget, "act01"),
     ("", "", r"contrastive, $w=0.003$", pget, "act003"),
     ("", "", r"contrastive, $w=0.001$", pget, "act001"),
     ("", "", r"contrastive, $w=0.01$, temperature $10^{-2}$", pget, "act01t2"),
     ("", "", r"contrastive, $w=0.1$, no temperature", pget, "ac1"),
     ("", "", r"contrastive, $w=0.01$, no temperature", pget, "ac01"),
     ("", "", r"contrastive, $w=0.001$, no temperature", pget, "ac001"),
     ("image features", "1.2M", r"contrastive, $w=0.1$", iget("1.2M"), "act1"),
     ("", "5.5M", r"contrastive, $w=0.1$", iget("5.5M"), "act1"),
     ("", "26.8M", r"contrastive, $w=0.1$", iget("26.8M"), "act1"),
     ("proprioception", "0.8M", r"matched swap alignment, $w=0.01$", pget, "md01"),
     ("image features", "1.2M", r"matched swap alignment, $w=0.01$", iget("1.2M"), "md01"),
     ("", "5.5M", r"matched swap alignment, $w=0.01$", iget("5.5M"), "md01"),
     ("", "26.8M", r"matched swap alignment, $w=0.01$", iget("26.8M"), "md01")]
pfmt = lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"
zz = lambda x: "0.000" if abs(x) < 5e-4 else f"{x:+.3f}"
L = [r"\begin{tabular}{lllcccc}", r"\toprule",
     r"inputs & params & objective added to MSE & val.\ error ratio & ret.\ rank $\downarrow$ & $\Delta$ D random ($p$) & $\Delta$ D logged ($p$, seeds lower) \\",
     r"\midrule"]
N["paired"] = {}
for i, (a, b, c, get, run) in enumerate(G):
    o = paired(get, "add", run)
    N["paired"][f"{a or G[i - 1][0]}|{b}|{run}"] = o
    if i in (8, 11):
        L.append(r"\midrule")
    L.append(f"{a} & {b} & {c} & {o['val_ratio'][0]:.2f} [{o['val_ratio'][1]:.2f}, {o['val_ratio'][2]:.2f}] & "
             f"{o['ret'][0]:.2f} $\\to$ {o['ret'][1]:.2f} & {zz(o['D'][0])} ({pfmt(o['D'][1])}) & "
             f"{zz(o['Dlog'][0])} ({pfmt(o['Dlog'][1])}, {o['Dlog'][2]}/5) \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
write("extra_goodhart.tex", L)
o = paired(pget, "mdrand01", "md01")
N["md01_vs_mdrand01"] = o
gain = paired(pget, "add", "md01"), paired(pget, "add", "mdrand01")
N["mdrand_keeps"] = dict(D=gain[1]["D"][0] / gain[0]["D"][0], Dlog=gain[1]["Dlog"][0] / gain[0]["Dlog"][0])

a2 = json.load(open(RR / "server" / "a2world_results.json"))
L = [r"\begin{tabular}{lrrrrrrrrrr}", r"\toprule",
     r" & \multicolumn{6}{c}{pixels} & \multicolumn{4}{c}{DINOv2 features} \\", r"\cmidrule(lr){2-7}\cmidrule(lr){8-11}",
     r"transform & D random & D logged & size & zero action div. & ret.\ rank $\downarrow$ & matched probe & D random & D logged & ret.\ rank $\downarrow$ & matched probe \\",
     r"\midrule"]
for t in TR:
    p, q = a2["pixels"][t], a2["dino"][t]
    lab = {"identity": "as released", "reflect": "reflected", "rotate": "rotated", "scale_0.25": r"$\times 0.25$",
           "scale_4": r"$\times 4$", "blind": "no response"}[t]
    L.append(f"{lab} & {p['D']:+.3f} & {p['D_log']:+.3f} & {p['size']:.2f} & {p['C1']:.3f} & {p['ret']:.2f} & {p['matched']:+.3f} & "
             f"{q['D']:+.3f} & {q['D_log']:+.3f} & {q['ret']:.2f} & {q['matched']:+.3f} \\\\".replace("+-", "-").replace("-0.000", "0.000").replace("+0.000", "0.000"))
L += [r"\bottomrule", r"\end{tabular}"]
write("extra_a2world.tex", L)
N["a2world"] = {}
for sp_ in ("pixels", "dino"):
    pp_ = a2[sp_]["identity"]["per_pos"]
    for k in ("D", "D_log", "matched"):
        v = np.array(pp_[k])
        N["a2world"][f"{sp_}|{k}"] = dict(mean=float(v.mean()), se=float(v.std(ddof=1) / np.sqrt(len(v))),
                                          p=float(stats.ttest_1samp(v, 0).pvalue))
    ret = np.array(pp_["ret"])
    N["a2world"][f"{sp_}|logged_first"] = int((ret == 0).sum())
    N["a2world"][f"{sp_}|estimate_cos"] = float(np.mean(a2[sp_]["truth"]["estimate_cos"]))
    N["a2world"][f"{sp_}|noise_floor_effect"] = (a2[sp_]["truth"]["noise_floor"], a2[sp_]["truth"]["effect"])
    for t in ("reflect", "rotate", "scale_4"):
        N["a2world"][f"{sp_}|{t}|logged_first"] = int((np.array(a2[sp_][t]["per_pos"]["ret"]) == 0).sum())
seeds_f = RR / "server" / "a2world_seeds.json"
if seeds_f.exists():
    sd = json.load(open(seeds_f))
    L = [r"\begin{tabular}{lccc}", r"\toprule", r"seed & D random & D logged & matched probe \\", r"\midrule"]
    lab = {"0": "original", "1000": "original $+1000$", "1500": "original $+1500$", "2000": "original $+2000$"}
    fm = lambda m: f"{m['mean']:+.3f} ({m['se']:.3f})".replace("+-", "-")
    for s, m in sd["per_seed"].items():
        L.append(f"{lab[s]} & {fm(m['D'])} & {fm(m['D_log'])} & {fm(m['matched'])} \\\\")
    L.append(r"\midrule")
    m = sd["seed_averaged_response"]
    L.append(f"response averaged over the 4 seeds & {fm(m['D'])} & {fm(m['D_log'])} & {fm(m['matched'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("extra_a2world_seeds.tex", L)
    N["a2world_seeds"] = {k: v for k, v in sd.items() if k != "positions"}

z = np.load(RR / "pairs.npz", allow_pickle=True)
runs = [str(r) for r in z["runs"]]
ce, cs, ok, gap, da, est, rank = z["c_est"], z["c_sim"], z["ok"], z["gap"], z["da"], z["est"], z["rank"]
ordi = [i for i, r in enumerate(runs) if r.split("_")[2] == "causal"]
assert len(ordi) == 30 and ok.all()


def bin_stats(mask):
    p, t = ce[:, mask].mean(1), cs[:, mask].mean(1)
    pr = {runs[i]: p[i] for i in ordi}
    tt = {runs[i]: t[i] for i in ordi}
    b = boot([runs[i] for i in ordi], {"m": lambda r: pr[r]}, lambda r: tt[r])["m"]
    return dict(n=int(mask.sum()), med=float(np.median(gap[mask])), med_da=float(np.median(da[mask])),
                est=float(est[mask].mean()), rho92=float(spearmanr(p, t)[0]), rho30=b)


bins = {}
L = [r"\begin{tabular}{lccccc}", r"\toprule",
     r" & \multicolumn{5}{c}{fifths of the 1800 pairs} \\", r"\cmidrule(lr){2-6}",
     r" & 1 & 2 & 3 & 4 & 5 \\", r"\midrule"]
for name, x in (("state gap", gap), ("action difference", da)):
    q = np.quantile(x, [0, .2, .4, .6, .8, 1])
    res = [bin_stats((x >= a) & (x <= b)) for a, b in zip(q[:-1], q[1:])]
    bins[name] = res
    key = "med" if name == "state gap" else "med_da"
    L.append(f"{name}, median & " + " & ".join(f"{r[key]:.2f}" for r in res) + r" \\")
    L.append(r"\quad estimate vs simulator, mean cosine & " + " & ".join(f"{r['est']:.2f}" for r in res) + r" \\")
    L.append(r"\quad probe vs truth, 92 models & " + " & ".join(f"{r['rho92']:.2f}" for r in res) + r" \\")
    L.append(r"\quad probe vs truth, 30 ordinary & " + " & ".join(f"{r['rho30'][0]:.2f}" for r in res) + r" \\")
    L.append(r"\quad \quad 95\% interval & " + " & ".join(f"[{r['rho30'][1]:.2f}, {r['rho30'][2]:.2f}]".replace("[-", "[$-$").replace(", -", ", $-$") for r in res) + r" \\")
    if name == "state gap":
        L.append(r"\midrule")
L += [r"\bottomrule", r"\end{tabular}"]
write("extra_bins.tex", L)
N["bins"] = {k: [{kk: (num(vv) if isinstance(vv, tuple) else vv) for kk, vv in r.items()} for r in v] for k, v in bins.items()}
no_idm1 = [i for i in ordi if obj(runs[i]) != "idm"]
q = np.quantile(gap, [0, .2, .4, .6, .8, 1])
N["bins_gap_truth_sd25"] = [float(cs[:, (gap >= a) & (gap <= b)].mean(1)[no_idm1].std()) for a, b in zip(q[:-1], q[1:])]
N["bins_gap_rho25"] = [float(spearmanr(ce[:, (gap >= a) & (gap <= b)].mean(1)[no_idm1], cs[:, (gap >= a) & (gap <= b)].mean(1)[no_idm1])[0])
                       for a, b in zip(q[:-1], q[1:])]
k1 = rank == 0
N["pairs_k1_da_quantiles"] = num(np.quantile(da[k1], [.1, .5, .9]))
N["pairs_k1_gap_quantiles"] = num(np.quantile(gap[k1], [.1, .5, .9]))
p1 = ce[:, k1].mean(1)
t25 = cs[:, (rank > 0)].mean(1)
no_idm1 = [i for i in ordi if obj(runs[i]) != "idm"]
N["probe_k1_vs_truth_rank2to5"] = dict(r30=float(spearmanr(p1[ordi], t25[ordi])[0]), r25=float(spearmanr(p1[no_idm1], t25[no_idm1])[0]),
                                       r92=float(spearmanr(p1, t25)[0]))
mv = np.load(ROOT / "data" / "processed" / "counterfactual" / "matched_v3.npz", allow_pickle=True)
assert (mv["row"] == z["row"]).all() and int(mv["k"]) == 5
sd, sp_ = mv["src_demo"], mv["src_pos"]
d0, p0 = sd.reshape(-1, 5)[:, :1].repeat(5, 1).reshape(-1), sp_.reshape(-1, 5)[:, :1].repeat(5, 1).reshape(-1)
same_d, other_d = (rank > 0) & (sd == d0), (rank > 0) & (sd != d0)
tod = cs[:, other_d].mean(1)
N["probe_k1_vs_truth_other_demo"] = dict(
    n_same_demo=int(same_d.sum()), n_same_demo_within_4_steps=int((same_d & (np.abs(sp_ - p0) <= 4)).sum()),
    n_other_demo=int(other_d.sum()), n_positions=int(len(np.unique(mv["row"][other_d]))),
    r30=float(spearmanr(p1[ordi], tod[ordi])[0]), r25=float(spearmanr(p1[no_idm1], tod[no_idm1])[0]), r92=float(spearmanr(p1, tod)[0]))

st = json.load(open(RR / "stress_matched.json"))
ss = json.load(open(RR / "stress_matched_seeds.json"))
assert all(abs(st["truth"][r] - PM[r]["Dlog"]) < 1e-6 for r in PM)
assert all(abs(ss["truth"][r] - PM[r]["Dlog"]) < 1e-6 for r in PM)
SETS4 = ("30", "55", "65")
rho_s = lambda pr, s: spearmanr([pr[r] for r in PS[s]], [PM[r]["Dlog"] for r in PS[s]])[0]
L = [r"\begin{tabular}{lccc}", r"\toprule",
     r"variant & 30 ordinary & 55 working & 65 without contrastive \\", r"\midrule"]
N["stress"] = {}
for v, lab in (("std (paper)", "standardized distance (paper)"), ("raw", "raw distance"), ("eef only", "end effector only"),
               ("joints only", "joints only"), ("cosine", "cosine distance")):
    vals = [rho_s(st["probe"][v], s) for s in SETS4]
    N["stress"][v] = num(vals)
    L.append(f"{lab} & " + " & ".join(f2(x) for x in vals) + r" \\")
L.append(r"\midrule")
for v, lab in (("noise 0.1", "noise 0.1 sd"), ("noise 0.3", "noise 0.3 sd"), ("noise 1.0", "noise 1.0 sd"),
               ("8 other demos per task", "8 other demonstrations"), ("4 other demos per task", "4 other demonstrations"),
               ("2 other demos per task", "2 other demonstrations")):
    if v == "8 other demos per task":
        L.append(r"\midrule")
    cells = []
    for s in SETS4:
        vals = [rho_s(pr, s) for pr in ss["probe"][v]]
        N["stress"][f"{v}|{s}"] = dict(mean=float(np.mean(vals)), min=float(min(vals)), max=float(max(vals)), draws=num(vals))
        cells.append(f"{f2(np.mean(vals))} [{f2(min(vals))}, {f2(max(vals))}]")
    L.append(f"{lab} & " + " & ".join(cells) + r" \\")
assert abs(rho_s(ss["probe"]["clean"][0], "30") - rho_s(st["probe"]["std (paper)"], "30")) < 1e-9
vals = [spearmanr([PM[r]["ret"] for r in PS[s]], [PM[r]["Dlog"] for r in PS[s]])[0] for s in SETS4]
N["stress"]["retrieval"] = num(vals)
L += [r"\midrule", r"retrieval, for comparison & " + " & ".join(f2(x) for x in vals) + r" \\", r"\bottomrule", r"\end{tabular}"]
write("extra_stress.tex", L)

DP = json.load(open(RR / "dplan.json"))
assert sorted(DP) == sorted(PM)
dg = lambda k: (lambda r: DP[r][k])
IP = json.load(open(RR / "inverse_plan.json"))
assert sorted(IP) == sorted(DP)
for r in DP:
    DP[r]["ic_plan"] = IP[r]
    DP[r]["sn"] = PM[r]["sn"]
PSETS = ("30", "55", "65")
PRED = ("D", "Dlog", "matched", "ic_plan", "ret", "sn", "val")
dpl = {s: boot(PS[s], {k: dg(k) for k in PRED}, dg("Dplan"),
               leads=[("Dlog", "D"), ("matched", "ret"), ("ic_plan", "matched"), ("ic_plan", "ret"), ("matched", "sn")]) for s in PSETS}
N["dplan"] = {s: {k: num(v) for k, v in dpl[s].items()} for s in dpl}
N["dplan_mean_15"] = float(np.mean([DP[r]["Dplan"] for r in PS["30"] if obj(r) in ("add", "agm", "delta")]))
DE = json.load(open(RR / "dplan_eef.json"))
assert max(abs(DE[r]["Dplan_full"] - DP[r]["Dplan"]) for r in DP) < 1e-9
for r in DP:
    DP[r]["Dplan_eef"] = DE[r]["Dplan_eef"]
dpe = {s: boot(PS[s], {k: dg(k) for k in PRED}, dg("Dplan_eef"),
               leads=[("Dlog", "D"), ("matched", "ret"), ("ic_plan", "matched"), ("ic_plan", "ret"), ("matched", "sn")]) for s in PSETS}
N["dplan_eef"] = {s: {k: num(v) for k, v in dpe[s].items()} for s in dpe}
N["dplan_eef_mean_15"] = float(np.mean([DP[r]["Dplan_eef"] for r in PS["30"] if obj(r) in ("add", "agm", "delta")]))
N["dplan_full_vs_eef"] = {s: float(spearmanr([DP[r]["Dplan"] for r in PS[s]], [DP[r]["Dplan_eef"] for r in PS[s]])[0]) for s in ("30", "65", "90")}
N["ic_plan_mean_15"] = float(np.mean([DP[r]["ic_plan"] for r in PS["30"] if obj(r) in ("add", "agm", "delta")]))
LAB = (("D", "D random (simulator)"), ("Dlog", "D logged (simulator)"), ("matched", "matched probe"),
       ("ic_plan", "inverse check on the planner changes"), ("ret", "retrieval"), ("sn", "size normalized retrieval"), ("val", "validation error"),
       ("Dlog-D", "D logged minus D random"), ("matched-ret", "matched minus retrieval"), ("matched-sn", "matched minus size normalized retrieval"),
       ("ic_plan-matched", "inverse check minus matched"), ("ic_plan-ret", "inverse check minus retrieval"))
L = [r"\begin{tabular}{lccc}", r"\toprule",
     r"predictor of D planner & 30 ordinary & 55 working & 65 without contrastive \\", r"\midrule",
     r"\multicolumn{4}{l}{\textit{D planner on the full state}} \\"]
for k, lab in LAB:
    L.append(lab + " & " + " & ".join(ci(dpl[s][k]) for s in PSETS) + r" \\")
L += [r"\midrule", r"\multicolumn{4}{l}{\textit{D planner on the end effector position only}} \\"]
for k, lab in LAB:
    L.append(lab + " & " + " & ".join(ci(dpe[s][k]) for s in PSETS) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
write("extra_dplan.tex", L)

ctl = {}
for s in ("90", "65", "55"):
    ctl[s] = {}
    for out in ("miss_hard", "succ_hard", "miss_easy", "succ_easy"):
        ctl[s][out] = boot(PS[s], {k: dg(k) for k in ("D", "Dlog", "matched", "ret", "val", "Dplan")}, dg(out))
N["control"] = {s: {o: {k: num(v) for k, v in ctl[s][o].items()} for o in ctl[s]} for s in ctl}
L = [r"\begin{tabular}{lcccccc}", r"\toprule",
     r" & \multicolumn{2}{c}{all 90} & \multicolumn{2}{c}{65 without contrastive} & \multicolumn{2}{c}{55 working} \\",
     r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
     r"predictor & original task & harder task & original task & harder task & original task & harder task \\", r"\midrule"]
for k, lab in (("D", "D random (simulator)"), ("Dlog", "D logged (simulator)"), ("Dplan", "D planner (simulator)"),
               ("matched", "matched probe"), ("ret", "retrieval"), ("val", "validation error")):
    L.append(lab + " & " + " & ".join(ci(ctl[s][o][k]) for s in ("90", "65", "55") for o in ("miss_easy", "miss_hard")) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
write("extra_control.tex", L)
L = L[:6]
for k, lab in (("D", "D random (simulator)"), ("Dlog", "D logged (simulator)"), ("Dplan", "D planner (simulator)"),
               ("matched", "matched probe"), ("ret", "retrieval"), ("val", "validation error")):
    L.append(lab + " & " + " & ".join(ci(ctl[s][o][k]) for s in ("90", "65", "55") for o in ("succ_easy", "succ_hard")) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
write("extra_control_succ.tex", L)
from scipy.stats import pearsonr, rankdata


def partial(rs, a, b, c):
    x, y, w = [rankdata([DP[r][k] for r in rs]) for k in (a, b, c)]
    rx = x - np.polyval(np.polyfit(w, x, 1), w)
    ry = y - np.polyval(np.polyfit(w, y, 1), w)
    return pearsonr(rx, ry)[0]


N["control_partial_given_val_55"] = {}
groups = {}
for r in PS["55"]:
    groups.setdefault(obj(r), []).append(r)
gk = sorted(groups)
for out in ("miss_easy", "miss_hard", "succ_easy", "succ_hard"):
    for p in ("Dlog", "matched", "Dplan"):
        rng = np.random.default_rng(0)
        bs = [partial([r for k in rng.choice(gk, len(gk)) for r in groups[k]], p, out, "val") for _ in range(4000)]
        N["control_partial_given_val_55"][f"{p}|{out}"] = [float(partial(PS["55"], p, out, "val")), *map(float, np.nanpercentile(bs, [2.5, 97.5]))]
N["control_paired"] = {}
for run in ("md001", "md01", "md1", "delta"):
    for out in ("succ_easy", "succ_hard", "miss_easy", "miss_hard"):
        a = np.array([DP[f"cf_{run}_causal_s{s}"][out] for s in range(5)])
        b = np.array([DP[f"cf_add_causal_s{s}"][out] for s in range(5)])
        N["control_paired"][f"{run}|{out}"] = dict(model=float(a.mean()), mse=float(b.mean()), p=float(stats.ttest_rel(a, b).pvalue))
ep = json.load(open(RR / "closed_loop_hard_g10_e4.json"))
ep2 = json.load(open(RR / "closed_loop_hard_g4_e1.json"))
N["control_episodes"] = dict(hard=len(ep), easy=len(ep2), hard_models=len({e["run"] for e in ep}), easy_models=len({e["run"] for e in ep2}))

sn = json.load(open(RR / "sizenorm_retrieval.json"))
L = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
     r"normalization & as trained & reflected & rotated & $\times 0.25$ & $\times 4$ & no response \\", r"\midrule"]
tr15p = [r for r in PS["30"] if obj(r) in ("add", "agm", "delta")]
N["sizenorm"] = {}
for k, lab in (("per_model", "per model"), ("per_item", "per item")):
    vals = [np.mean([sn[k][r][t] for r in tr15p]) for t in TR]
    N["sizenorm"][k] = num(vals)
    L.append(f"{lab} & " + " & ".join(f"{x:.2f}" for x in vals) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
write("extra_sizenorm.tex", L)
snb = boot(PS["30"], {"sn": lambda r: -sn["per_model"][r]["identity"], "matched": g("matched")}, g("Dlog"), leads=[("matched", "sn")])
N["sizenorm_30"] = {k: num(v) for k, v in snb.items()}
paper_runs = sorted(pv)
for truth_key, tf in (("D", lambda r, t: pv[r][t]["direction"]), ("Dlog", lambda r, t: ms[r][t]["dir_on"])):
    a = [-sn["per_model"][r][t] for r in paper_runs for t in TR]
    b = [tf(r, t) for r in paper_runs for t in TR]
    N["sizenorm"][f"transformed_vs_{truth_key}"] = round(float(spearmanr(a, b)[0]), 4)

ft = json.load(open(RT / "tdmpc2_finetune.json"))
fell = rose = fell_err = rose_err = 0
dd, de = [], []
for task, v in ft.items():
    for w in ("0.001", "0.01", "0.1"):
        ddir = v[f"matched_{w}"]["direction"] - v["consistency"]["direction"]
        derr = v[f"matched_{w}"]["consistency_mse"] - v["consistency"]["consistency_mse"]
        dd.append(ddir)
        de.append(derr)
        if ddir < 0:
            fell += 1
            fell_err += derr > 0
        else:
            rose += 1
            rose_err += derr > 0
N["heldout_warning"] = dict(fell=fell, fell_err_rose=fell_err, rose=rose, rose_err_rose=rose_err, tasks=len(ft),
                            spearman_all=float(spearmanr(dd, de)[0]))

rs = IMG["1.2M"]["sets"]["30"]
nd = [r for r in rs if obj(r) != "delta"]
N["image_val_delta"] = dict(
    delta_Dlog=float(np.mean([ig("1.2M", "Dlog")(r) for r in rs if obj(r) == "delta"])),
    others_Dlog=float(np.mean([ig("1.2M", "Dlog")(r) for r in nd])),
    delta_val=float(np.mean([I1["val"][r] for r in rs if obj(r) == "delta"])),
    others_val=float(np.mean([I1["val"][r] for r in nd])),
    val_rho_without_delta=float(spearmanr([ig("1.2M", "val")(r) for r in nd], [ig("1.2M", "Dlog")(r) for r in nd])[0]))
N["ac1_ret_seeds_improved"] = int(sum(dt[f"cf_ac1_causal_s{s}"]["ret_gauss_meanrank"] < -PM[f"cf_add_causal_s{s}"]["ret"] for s in range(5)))
N["ac1_abs_changes"] = dict(D=float(np.mean([dt[f"cf_ac1_causal_s{s}"]["direction"] - PM[f"cf_add_causal_s{s}"]["D"] for s in range(5)])),
                            Dlog=float(np.mean([dt[f"cf_ac1_causal_s{s}"]["dir_on"] - PM[f"cf_add_causal_s{s}"]["Dlog"] for s in range(5)])),
                            matched=float(np.mean([dt[f"cf_ac1_causal_s{s}"]["matched"] - PM[f"cf_add_causal_s{s}"]["matched"] for s in range(5)])))
N["control_by_objective"] = {o: {out: float(np.mean([DP[r][out] for r in PS["90"] if obj(r) == o]))
                                 for out in ("succ_easy", "succ_hard", "miss_easy", "miss_hard")} for o in sorted({obj(r) for r in PS["90"]})}

N["image_pair_overlap"] = json.load(open(RR / "server" / "img_pair_overlap.json"))
L = [r"\begin{tabular}{lcccc}", r"\toprule",
     r" & \multicolumn{2}{c}{final distance} & \multicolumn{2}{c}{success within 2 cm} \\", r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
     r"measure, given validation error & original task & harder task & original task & harder task \\", r"\midrule"]
for p_, lab in (("Dlog", "D logged (simulator)"), ("Dplan", "D planner (simulator)"), ("matched", "matched probe")):
    L.append(lab + " & " + " & ".join(ci(N["control_partial_given_val_55"][f"{p_}|{o}"]) for o in ("miss_easy", "miss_hard", "succ_easy", "succ_hard")) + r" \\")
L.append(r"\midrule")
for p_, lab in (("Dlog", "D logged, alone"), ("val", "validation error, alone")):
    L.append(lab + " & " + " & ".join(ci(ctl["55"][o][p_]) for o in ("miss_easy", "miss_hard", "succ_easy", "succ_hard")) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
write("extra_partial.tex", L)
for size in ("1.2M", "5.5M", "26.8M"):
    I = IMG[size]
    rs_all = sorted(r for r in I["pv"] if r.split("_")[2] == "causal")
    N.setdefault("image_ratio_all_objectives", {})[size] = {
        o: round(float(np.mean([I["mi"][r]["identity"]["matched_k1"] for r in rs_all if obj(r) == o]) /
                       np.mean([I["mi"][r]["identity"]["dir_on"] for r in rs_all if obj(r) == o])), 3)
        for o in sorted({obj(r) for r in rs_all})}
def exact_lead(rs, pa, pb, truth):
    import itertools
    import math
    groups = {}
    for r in rs:
        groups.setdefault(obj(r), []).append(r)
    keys = sorted(groups)
    n = len(keys)
    assert n <= 6
    tt, A, Bv = {r: truth(r) for r in rs}, {r: pa(r) for r in rs}, {r: pb(r) for r in rs}
    f = lambda pick: (spearmanr([A[r] for r in pick], [tt[r] for r in pick])[0] -
                      spearmanr([Bv[r] for r in pick], [tt[r] for r in pick])[0])
    vals, w = [], []
    for c in itertools.combinations_with_replacement(range(n), n):
        v = f([r for i in c for r in groups[keys[i]]])
        if v == v:
            vals.append(v)
            w.append(math.factorial(n) / np.prod([math.factorial(int(k)) for k in np.bincount(c, minlength=n)]) / n ** n)
    vals, w = np.array(vals), np.array(w)
    w = w / w.sum()
    o = np.argsort(vals)
    cw = np.cumsum(w[o])
    q = lambda pr: float(vals[o][min(int(np.searchsorted(cw, pr)), len(vals) - 1)])
    return dict(point=round(float(f(rs)), 4), lo=round(q(0.025), 4), hi=round(q(0.975), 4), p_le_0=round(float(w[vals <= 0].sum()), 4), n=len(vals))


N["exact_30"] = {
    "lead_over_retrieval": {"proprio 0.8M": exact_lead(PS["30"], g("matched"), g("ret"), g("Dlog")),
                            **{f"image {s}": exact_lead(IMG[s]["sets"]["30"], ig(s, "matched"), ig(s, "ret"), ig(s, "Dlog"))
                               for s in ("1.2M", "5.5M", "26.8M")}},
    "dplan_Dlog_minus_D": {"full": exact_lead(PS["30"], dg("Dlog"), dg("D"), dg("Dplan")),
                           "eef": exact_lead(PS["30"], dg("Dlog"), dg("D"), dg("Dplan_eef"))}}
print("exact 30:", json.dumps(N["exact_30"]))

NOI, IDMF = ("add", "agm", "delta"), ("idm", "idm01", "idm001")
def by_family(rs, get, truth):
    out = {}
    for name, fam in (("all30", ORD), ("without_idm", NOI), ("with_idm", IDMF)):
        pick = [r for r in rs if obj(r) in fam]
        out[name] = {k: round(float(spearmanr([get(k)(r) for r in pick], [truth(r) for r in pick])[0]), 4) for k in ("ic", "matched", "ret")}
    return out
N["probes_by_objective_family"] = {"proprio 0.8M": by_family(PS["30"], g, g("Dlog")),
                                   **{f"image {s}": by_family(IMG[s]["sets"]["30"], lambda k, s=s: ig(s, k), ig(s, "Dlog"))
                                      for s in ("1.2M", "5.5M", "26.8M")}}
def clone_gap(rs, truth):
    dl = {r: truth(r) for r in rs}
    v = lambda o, s: dl[f"cf_{o}_causal_s{s}"]
    return {**{f"{o}_vs_add": round(float(np.mean([abs(v(o, s) - v("add", s)) for s in range(5)])), 5) for o in ("agm", "idm001", "idm01", "delta", "idm")},
            "sd_over_seeds": round(float(np.mean([np.std([v(o, s) for s in range(5)], ddof=1) for o in ORD])), 5),
            "sd_over_seeds_add": round(float(np.std([v("add", s) for s in range(5)], ddof=1)), 5)}
N["clone_gap_Dlog"] = {"proprio 0.8M": clone_gap(PS["30"], g("Dlog")),
                       **{f"image {s}": clone_gap(IMG[s]["sets"]["30"], ig(s, "Dlog")) for s in ("1.2M", "5.5M", "26.8M")}}
_obj = obj
obj = lambda r: "add" if _obj(r) in ("agm", "idm001") else _obj(r)
N["exact_30_four_units"] = {"proprio 0.8M": exact_lead(PS["30"], g("matched"), g("ret"), g("Dlog")),
                            **{f"image {s}": exact_lead(IMG[s]["sets"]["30"], ig(s, "matched"), ig(s, "ret"), ig(s, "Dlog"))
                               for s in ("1.2M", "5.5M", "26.8M")}}
obj = _obj
tk = np.array(a2["task"]) if "task" in a2 else None
if tk is not None:
    for sp_name in ("pixels", "dino"):
        v = np.array(a2[sp_name]["identity"]["per_pos"]["D_log"])
        m = np.array([v[tk == u].mean() for u in np.unique(tk)])
        N["a2world"][f"{sp_name}|D_log_by_task"] = dict(n_tasks=int(len(m)), mean=float(m.mean()), p=float(stats.ttest_1samp(m, 0).pvalue))
print("family numbers:", json.dumps({k: N[k] for k in ("probes_by_objective_family", "clone_gap_Dlog", "exact_30_four_units")}))

S = json.load(open(RR / "inverse_strength.json"))
assert S["max_abs_diff_full_vs_saved"] == 0.0
fr = ("1.0", "0.1", "0.03", "0.01", "0.003")
inv = {k: {s: boot(PS[s], {"matched": g("matched"), "ic": (lambda r, k=k: S["ic"][k][r])}, g("Dlog"), leads=[("matched", "ic")])
           for s in ("30", "55", "65", "90")} for k in fr}
assert all(np.allclose(inv["1.0"][s]["ic"], prop[s]["ic"]) and np.allclose(inv["1.0"][s]["matched"], prop[s]["matched"]) for s in inv["1.0"])
assert all(abs(inv[k][s]["ic"][0] - S["spearman_with_Dlog"][s][f"ic_{k}"][0]) < 1e-4 for k in fr for s in inv[k])
L = [r"\begin{tabular}{rrcccc}", r"\toprule",
     r"training demonstrations & transitions & held out $R^2$ of the inverse model & 30 ordinary & 65 without contrastive & all 90 \\", r"\midrule"]
for k in fr:
    m = S["inverse_models"][k]
    L.append(f"{m['n_demos']} & {m['n_train']} & {m['val_r2_mean']:.2f} & ".replace("-", "$-$") +
             " & ".join(ci(inv[k][s]["ic"]) for s in ("30", "65", "90")) + r" \\")
L += [r"\midrule", r"\multicolumn{3}{l}{matched probe, which uses no inverse model} & " +
      " & ".join(ci(prop[s]["matched"]) for s in ("30", "65", "90")) + r" \\", r"\bottomrule", r"\end{tabular}"]
write("extra_invfit.tex", L)
N["inverse_fit"] = {k: dict(n_demos=S["inverse_models"][k]["n_demos"], n_train=S["inverse_models"][k]["n_train"],
                            r2=round(S["inverse_models"][k]["val_r2_mean"], 4),
                            **{s: num(inv[k][s]["ic"]) for s in inv[k]},
                            **{f"matched-ic_{s}": num(inv[k][s]["matched-ic"]) for s in inv[k]}) for k in fr}
L = [r"\begin{tabular}{lrrrrr}", r"\toprule",
     r"copy of A2World & D logged (simulator) & zero action div. & retrieval rank $\downarrow$ & logged chunk ranked first & matched probe \\", r"\midrule"]
for t_, lab in (("identity", "as released"), ("reflect", "reflected response"), ("rotate", "rotated response"), ("scale_4", r"response $\times 4$")):
    p_ = a2["pixels"][t_]
    first = N["a2world"]["pixels|logged_first"] if t_ == "identity" else N["a2world"][f"pixels|{t_}|logged_first"]
    L.append(f"{lab} & {p_['D_log']:+.3f} & {p_['C1']:.3f} & {p_['ret']:.2f} & {first} of {a2['n_positions']} & {p_['matched']:+.3f} \\\\"
             .replace("+-", "-").replace("-0.000", "0.000").replace("+0.000", "0.000"))
L += [r"\bottomrule", r"\end{tabular}"]
write("extra_main_a2w.tex", L)

lead7 = [prop["30"]["matched-ret"], prop["65"]["matched-ret"], prop["90"]["matched-ret"], img["30"]["matched-ret"],
         img["30_5.5M"]["matched-ret"], img["30_26.8M"]["matched-ret"], img["90"]["matched-ret"]]
pl = [(X[s]["matched"][0] - X[s]["ret"][0], X[s]["matched-ret"][1]) for X in (dpl, dpe) for s in PSETS]
cc = [ctl[s][o]["matched"][0] - ctl[s][o]["ret"][0] for s in ("90", "65", "55") for o in ("miss_easy", "miss_hard", "succ_easy", "succ_hard")]
fi = [prop[s]["matched"][0] - inv[k][s]["ic"][0] for k in fr for s in ("30", "65", "90")]
N["consistency"] = {
    "lead_over_retrieval": {"values": num([v[0] for v in lead7]), "n_sets": 7, "n_positive": int(sum(v[0] > 0 for v in lead7)),
                            "n_interval_above_0": int(sum(v[1] > 0 for v in lead7)),
                            "four_sets_of_30": num([lead7[i][0] for i in (0, 3, 4, 5)])},
    "matched_minus_inverse_proprio": {s: num(prop[s]["matched-ic"]) for s in ("30", "55", "65", "90")},
    "matched_minus_val_proprio": {s: num(prop[s]["matched-val"]) for s in ("30", "55", "65", "90")},
    "planner_matched_minus_retrieval": {"values": num([v[0] for v in pl]), "n": len(pl), "n_positive": int(sum(v[0] > 0 for v in pl)),
                                        "n_interval_above_0": int(sum(v[1] > 0 for v in pl))},
    "control_matched_minus_retrieval": {"values": num(cc), "n_cells": len(cc), "n_positive": int(sum(v > 0 for v in cc))},
    "inverse_fit_matched_minus_inverse": {"values": num(fi), "n": len(fi), "n_positive": int(sum(v > 0 for v in fi)),
                                          "min": round(float(min(fi)), 4), "max": round(float(max(fi)), 4)}}
print("consistency:", json.dumps(N["consistency"]))
json.dump(N, open(RR / "numbers.json", "w"), indent=1, default=float)
print("wrote numbers.json")
