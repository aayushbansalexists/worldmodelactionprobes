from __future__ import annotations

import inspect

import torch

import analysis.inverse_probe as ip

src = inspect.getsource(ip.scores)
old = "    PT = P_ref + (P_q - P_ref) @ T.T"
assert src.count(old) == 1, "analysis.inverse_probe.scores changed; update this wrapper"
src = src.replace(old, "    PT = P_ref + (torch.from_numpy((P_q - P_ref).astype('float32')).to(device)"
                       " @ torch.from_numpy(T.astype('float32')).to(device).T).cpu().numpy()", 1)
exec(src, vars(ip))
torch.backends.cuda.matmul.allow_tf32 = False

if __name__ == "__main__":
    ip.main()
