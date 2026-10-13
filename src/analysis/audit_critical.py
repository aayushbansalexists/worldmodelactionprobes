from __future__ import annotations

import numpy as np
import torch

PASS, FAIL = "PASS", "FAIL"
out = []


def check(name, ok, detail=""):
    out.append((name, bool(ok)))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f"   {detail}" if detail else ""))


def audit_null():
    n_cand = 16
    rng = np.random.default_rng(0)
    T, D, ntraj = 40, 8, 30
    trajs = []
    for _ in range(ntraj):
        x = np.zeros((T, D)); s = rng.standard_normal(D)
        for t in range(T):
            s = 0.97 * s + 0.03 * rng.standard_normal(D)
            x[t] = s
        trajs.append(x)

    def preds_for(_):
        return rng.standard_normal((n_cand, D)) * 0.3

    for pool_mode in ("growing_same_traj", "global_shuffled"):
        hits, tot, pool = 0, 0, []
        for tr in trajs:
            for p in range(0, T - 2, 4):
                pr = preds_for(p)
                if pool:
                    for _ in range(8):
                        alt = pool[int(rng.integers(0, len(pool)))]
                        d = ((pr - alt) ** 2).mean(-1)
                        hits += int(np.argmin(d) == 0); tot += 1
                pool.append(tr[p + 1].copy())
            if pool_mode == "global_shuffled":
                rng.shuffle(pool)
        null = hits / max(tot, 1)
        check(f"empirical null is unbiased ({pool_mode})",
              abs(null - 1 / n_cand) < 0.02, f"null={null:.4f} vs 1/n={1/n_cand:.4f}")

    hits, tot, pool = 0, 0, []
    for tr in trajs:
        for p in range(0, T - 2, 4):
            base = rng.standard_normal(D)
            pr = np.repeat(base[None], n_cand, 0)
            pr[1:] += rng.standard_normal((n_cand - 1, D)) * 0.8
            if pool:
                for _ in range(8):
                    alt = pool[int(rng.integers(0, len(pool)))]
                    d = ((pr - alt) ** 2).mean(-1)
                    hits += int(np.argmin(d) == 0); tot += 1
            pool.append(tr[p + 1].copy())
    null_centred = hits / max(tot, 1)
    check("empirical null DETECTS the candidate asymmetry (this is the point of it)",
          null_centred > 1.5 / n_cand, f"null={null_centred:.4f} >> 1/n={1/n_cand:.4f}")


def audit_cf_alignment():
    L, H, A = 16, 8, 7
    consumed = []

    class Recorder(torch.nn.Module):
        cfg = {"data": {"seq_len": L}}

        def forward(self, obs, act):
            consumed.append(act[0, -1].clone())
            return {"pred": obs[:, -1:].clone()}

    from eval.counterfactual_eval import predict_branch
    fut = torch.arange(H * A, dtype=torch.float32).reshape(H, A)
    ctx_a = torch.full((L, A), -1.0)
    ctx_o = torch.zeros(L, 3)
    predict_branch(Recorder(), ctx_o, ctx_a, fut, H, "cpu")
    got = torch.stack(consumed)
    check("counterfactual: model consumes exactly the H simulator actions, in order",
          got.shape[0] == H and torch.allclose(got, fut),
          f"consumed {got.shape[0]} actions; match={torch.allclose(got, fut)}")
    check("counterfactual: no simulator action is skipped or reused",
          len({tuple(r.tolist()) for r in got}) == H)


def audit_policy_leak():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from wmprobe.adversaries import CausalPolicy

    torch.manual_seed(0)
    O, A, T = 6, 4, 12
    pol = CausalPolicy(O, A, hidden=32, layers=1, heads=2).eval()
    obs = torch.randn(1, T, O)
    act = torch.randn(1, T, A)
    with torch.no_grad():
        p1 = pol(obs, act)
        act2 = act.clone(); act2[0, 5] += 10.0
        p2 = pol(obs, act2)
    check("CausalPolicy does not see the action it predicts (position 5)",
          torch.allclose(p1[0, 5], p2[0, 5], atol=1e-6),
          f"max delta at t=5: {(p1[0,5]-p2[0,5]).abs().max().item():.2e}")
    check("CausalPolicy does not see any future action (positions <5 unchanged)",
          torch.allclose(p1[0, :5], p2[0, :5], atol=1e-6))
    check("CausalPolicy DOES use past actions (position 6 moves)",
          not torch.allclose(p1[0, 6], p2[0, 6], atol=1e-6),
          f"delta at t=6: {(p1[0,6]-p2[0,6]).abs().max().item():.2e}")


def audit_adversary():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from wmprobe.adversaries import PolicyAlignedInflation, UnconditionalInflation

    torch.manual_seed(0)
    O, A, T = 6, 4, 10

    class Base(torch.nn.Module):
        def __init__(s):
            super().__init__(); s.l = torch.nn.Linear(O, O)
        def forward(s, o, a): return s.l(o)

    class ExactPolicy(torch.nn.Module):
        def __init__(s, logged): super().__init__(); s.logged = logged
        def forward(s, o, a): return s.logged

    base = Base().eval()
    obs = torch.randn(1, T, O)
    logged = torch.randn(1, T, A)
    adv = PolicyAlignedInflation(base, ExactPolicy(logged), O, A, c=5.0).eval()
    with torch.no_grad():
        check("adversary vanishes at the logged action (identical to base)",
              torch.allclose(adv(obs, logged), base(obs, logged), atol=1e-6),
              f"max delta {(adv(obs,logged)-base(obs,logged)).abs().max().item():.2e}")
        off = logged + 1.0
        moved = (adv(obs, off) - base(obs, off)).abs().max().item()
        check("adversary moves off the logged action (inflation is real)", moved > 1.0,
              f"max delta {moved:.3f}")
        u = UnconditionalInflation(base, O, A, c=3.0).eval()
        check("unconditional adversary does NOT vanish at the logged action",
              not torch.allclose(u(obs, logged), base(obs, logged), atol=1e-3))


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.environ.get("PROJECT_ROOT", "."))
    for fn in [audit_null, audit_cf_alignment, audit_policy_leak, audit_adversary]:
        print((fn.__doc__ or fn.__name__).strip().splitlines()[0])
        try:
            fn()
        except Exception as e:
            check(f"{fn.__name__} RAISED", False, f"{type(e).__name__}: {e}")
        print()
    n = sum(1 for _, ok in out if ok)
    print(f"{n}/{len(out)} critical checks passed")
    bad = [k for k, ok in out if not ok]
    if bad:
        print("FAILED:", *bad, sep="\n  ")
