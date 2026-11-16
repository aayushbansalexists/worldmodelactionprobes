from __future__ import annotations

import inspect

import numpy as np
import torch

import analysis.probe_validity as pv

src = inspect.getsource(pv.score)
old = "        tr = lambda R: R @ T.T"
assert src.count(old) == 1, "analysis.probe_validity.score changed; update this wrapper"
src = src.replace(old, "        Tt = torch.from_numpy(T).to(device)\n"
                       "        tr = lambda R: (torch.from_numpy(np.ascontiguousarray(R, dtype=np.float32)).to(device)"
                       " @ Tt.T).cpu().numpy()", 1)
src = src.replace("def score(", "def score_gpu(", 1)
exec(src, vars(pv))
pv.score = vars(pv)["score_gpu"]
torch.backends.cuda.matmul.allow_tf32 = False

if __name__ == "__main__":
    pv.main()
