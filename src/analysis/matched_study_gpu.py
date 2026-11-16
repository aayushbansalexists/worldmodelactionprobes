from __future__ import annotations

import inspect

import torch

import analysis.matched_study as ms

src = inspect.getsource(ms.main)
old = "            RT = R @ T.T"
assert src.count(old) == 1, "analysis.matched_study.main changed; update this wrapper"
src = src.replace(old, "            RT = (torch.from_numpy(R).cuda() @ torch.from_numpy(T).cuda().T).cpu().numpy()", 1)
exec(src, vars(ms))
torch.backends.cuda.matmul.allow_tf32 = False

if __name__ == "__main__":
    ms.main()
