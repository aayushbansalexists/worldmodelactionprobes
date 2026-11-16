import glob
import json
import sys

import numpy as np

d, res_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
z = np.load(d + "/items.npz", allow_pickle=True)
res = json.load(open(res_path))
D = np.array(res["pixels"]["identity"]["per_pos"]["D"])
pred = np.zeros((90, 6, 2, 20, 128, 128, 3), np.uint8)
for f in glob.glob(d + "/pred_*.npz"):
    p = np.load(f)
    s = int(p["start"])
    pred[s:s + len(p["pred"])] = p["pred"]
T = np.zeros((90, 6, 20, 2, 128, 128, 3), np.uint8)
for f in glob.glob(d + "/truth/truth_*.npz"):
    t = np.load(f)
    for j, i in enumerate(t["idx"]):
        T[i] = t["img"][j]
eff = np.linalg.norm((T[:, 1].astype(np.float32) - T[:, 0]).reshape(90, -1), axis=1)
near = np.argsort(np.abs(D - np.median(D)))[:20]
i = int(near[np.argmax(eff[near])])
print("position", i, "task", z["task"][i], "D", D[i], "median D", np.median(D), "effect", eff[i], "median effect", np.median(eff))
np.savez_compressed(out, pos=i, task=str(z["task"][i]), D=D[i], cur_agent=z["hist_agent"][i, -1], cur_wrist=z["hist_wrist"][i, -1],
                    fut_agent=z["fut_agent"][i], fut_wrist=z["fut_wrist"][i], pred_logged=pred[i, 0], pred_offset=pred[i, 1],
                    sim_logged=T[i, 0], sim_offset=T[i, 1], a_logged=z["queries"][i, 0], a_offset=z["queries"][i, 1])
