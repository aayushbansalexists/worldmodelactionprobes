import json
import sys

import numpy as np

d, out = sys.argv[1], sys.argv[2]
a, b = np.load(d + "/matched_v3.npz", allow_pickle=True), np.load(d + "/matched_v3pp.npz", allow_pickle=True)
assert np.array_equal(a["row"], b["row"])
key = lambda z, i: (str(z["task"][i]), str(z["src_demo"][i]), int(z["src_pos"][i]))
fa, fb = {}, {}
for i, r in enumerate(a["row"]):
    fa.setdefault(int(r), key(a, i))
    fb.setdefault(int(r), key(b, i))
same_segment = [fa[r] == fb[r] for r in fa]
same_demo = [fa[r][1] == fb[r][1] for r in fa]
res = dict(positions=len(fa), same_nearest_segment=int(sum(same_segment)), same_nearest_demo=int(sum(same_demo)),
           frac_segment=float(np.mean(same_segment)), frac_demo=float(np.mean(same_demo)),
           rank_aligned_all_pairs=float(np.mean([key(a, i) == key(b, i) for i in range(len(a["row"]))])))
json.dump(res, open(out, "w"), indent=1)
print(res)
