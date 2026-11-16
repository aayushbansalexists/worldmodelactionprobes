import argparse
import json
import os
import subprocess

import numpy as np

BASE = {"SIM": "#0b0b0b", "BLUE": "#2a78d6", "ORANGE": "#eb6834", "AQUA": "#1baf7a", "RED": "#e34948"}


def rgb(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], float)


def hexs(v):
    return "#" + "".join(f"{int(round(float(x))):02x}" for x in np.clip(v, 0, 255))


def mix(h, other, t):
    return hexs(rgb(h) * (1 - t) + rgb(other) * t)


def draw(seed, frac):
    rng = np.random.default_rng(seed)
    cols, shifts = {}, {}
    for k, h in BASE.items():
        d = rng.uniform(-frac, frac, 3) * 255.0
        cols[k] = hexs(rgb(h) + d)
        shifts[k] = [int(x) for x in (rgb(cols[k]) - rgb(h))]
    return cols, shifts


def derived(c):
    return {
        "BLUE_LIGHT": mix(c["BLUE"], "#ffffff", 0.83),
        "BLUE_DARK": mix(c["BLUE"], "#000000", 0.47),
        "gtfill": mix(c["AQUA"], "#ffffff", 0.90),
        "mdfill": mix(c["BLUE"], "#ffffff", 0.90),
        "okfill": mix(c["ORANGE"], "#ffffff", 0.90),
    }


def validate(script, cols):
    pal = ",".join(cols[k] for k in ("BLUE", "ORANGE", "AQUA", "RED"))
    r = subprocess.run(["node", script, pal, "--mode", "light"], capture_output=True, text=True)
    out = r.stdout + r.stderr
    return ("[FAIL]" not in out) and ("ALL CHECKS PASS" in out), out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--frac", type=float, default=0.10)
    ap.add_argument("--validator", default=None)
    a = ap.parse_args()
    seed = a.seed if a.seed is not None else int.from_bytes(os.urandom(4), "little")
    first, rejected, report = seed, [], ""
    while True:
        cols, shifts = draw(seed, a.frac)
        if a.validator is None:
            break
        ok, report = validate(a.validator, cols)
        if ok:
            break
        rejected.append(seed)
        seed += 1
    d = derived(cols)
    tex = "\n".join([
        f"\\definecolor{{gtline}}{{HTML}}{{{cols['AQUA'][1:].upper()}}}\\definecolor{{gtfill}}{{HTML}}{{{d['gtfill'][1:].upper()}}}",
        f"\\definecolor{{mdline}}{{HTML}}{{{cols['BLUE'][1:].upper()}}}\\definecolor{{mdfill}}{{HTML}}{{{d['mdfill'][1:].upper()}}}",
        f"\\definecolor{{okline}}{{HTML}}{{{cols['ORANGE'][1:].upper()}}}\\definecolor{{okfill}}{{HTML}}{{{d['okfill'][1:].upper()}}}",
    ])
    json.dump({"rule": f"each RGB channel moved by uniform(-{a.frac}, {a.frac}) x 255, numpy default_rng(seed), "
                       "colours drawn in the order SIM, BLUE, ORANGE, AQUA, RED",
               "first_seed": first, "seed": seed, "rejected_seeds": rejected, "frac": a.frac, "base": BASE,
               "colors": cols, "shift_rgb": shifts, "derived": d, "latex": tex, "validator_report": report},
              open(a.out, "w"), indent=1)
    print("seed", seed, "(first", first, ", rejected", rejected, ")")
    for k in BASE:
        print(f"  {k:7s} {BASE[k]} -> {cols[k]}  shift {shifts[k]}")
    print(report)
    print(tex)
