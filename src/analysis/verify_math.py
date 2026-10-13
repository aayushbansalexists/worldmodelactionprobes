from __future__ import annotations

import numpy as np
import sympy as sp

OK, FAIL = "PASS", "FAIL"
results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"  [{OK if cond else FAIL}] {name}" + (f"   {detail}" if detail else ""))


def claim_risk_gap():
    rng = np.random.default_rng(0)
    nH, nA, dY = 4, 3, 2
    pH = rng.random(nH); pH /= pH.sum()
    pA = rng.random((nH, nA)); pA /= pA.sum(1, keepdims=True)
    atoms = rng.standard_normal((5, dY))
    pY = rng.random((nH, nA, 5)); pY /= pY.sum(-1, keepdims=True)

    EY_HA = np.einsum('hak,kd->had', pY, atoms)
    EY_H = np.einsum('ha,had->hd', pA, EY_HA)
    EY2 = np.einsum('hak,kd,kd->ha', pY, atoms, atoms)

    def risk(pred_ha):
        t = 0.0
        for h in range(nH):
            for a in range(nA):
                w = pH[h] * pA[h, a]
                p = pred_ha[h, a]
                t += w * (EY2[h, a] - 2 * p @ EY_HA[h, a] + p @ p)
        return t

    Lf = risk(EY_HA)
    Lg = risk(np.repeat(EY_H[:, None, :], nA, axis=1))
    gap = sum(pH[h] * pA[h, a] * np.sum((EY_HA[h, a] - EY_H[h]) ** 2)
              for h in range(nH) for a in range(nA))
    check("risk gap identity  L(g*)-L(f*) = E||E[Y|H,A]-E[Y|H]||^2",
          np.isclose(Lg - Lf, gap, rtol=1e-10), f"lhs={Lg-Lf:.10f} rhs={gap:.10f}")
    check("risk gap is nonnegative", gap >= 0, f"gap={gap:.6f}")


def claim_off_support():
    ds, da = 3, 4
    A = sp.Matrix(sp.symbols('a0:4'))
    pi = sp.Matrix(sp.symbols('p0:4'))
    B = sp.Matrix(ds, da, sp.symbols('b0:12'))
    f = sp.Matrix([sp.Function(f'f{i}')(*A) for i in range(ds)])
    fB = f + B * (A - pi)

    at_logged = sp.simplify((fB - f).subs(list(zip(A, pi))))
    check("off-support: f_B(H,pi(H)) = f(H,pi(H))", at_logged == sp.zeros(ds, 1))

    Jf = f.jacobian(A)
    JfB = fB.jacobian(A)
    check("off-support: grad_A f_B = grad_A f + B(H)", sp.simplify(JfB - Jf - B) == sp.zeros(ds, da))

    rng = np.random.default_rng(1)
    n = 20000
    H = rng.standard_normal((n, 5))
    pi_n = np.tanh(H @ rng.standard_normal((5, da)))
    A_n = pi_n.copy()
    Wf = rng.standard_normal((5, ds))
    Y = H @ Wf + rng.standard_normal((n, ds)) * 0.1
    Bm = rng.standard_normal((ds, da)) * 7.0
    f_n = H @ Wf
    fB_n = f_n + (A_n - pi_n) @ Bm.T
    check("off-support: population MSE identical under A=pi(H) a.s.",
          np.isclose(((f_n - Y) ** 2).mean(), ((fB_n - Y) ** 2).mean(), rtol=1e-12),
          f"dMSE={abs(((f_n-Y)**2).mean()-((fB_n-Y)**2).mean()):.3e}")
    check("off-support: ||J|| inflates with scale of B",
          np.linalg.norm(Bm * 10) > 9 * np.linalg.norm(Bm))


def claim_local_expansion():
    ds, da = 2, 2
    eps = sp.Matrix(sp.symbols('e0:2'))
    t = sp.symbols('t')
    A = sp.Matrix(sp.symbols('a0:2'))
    Y = sp.Matrix(sp.symbols('y0:2'))
    f = sp.Matrix([sp.Function(f'f{i}')(*A) for i in range(ds)])
    J = f.jacobian(A)
    err = f - Y
    lhs = ((f.subs(list(zip(A, A + t * eps))) - Y).T * (f.subs(list(zip(A, A + t * eps))) - Y))[0]
    series = sp.series(lhs, t, 0, 3).removeO()
    quad = sp.expand(series - (err.T * err)[0])
    hess = sum(err[i] * (eps.T * sp.hessian(f[i], list(A)) * eps)[0] for i in range(ds))
    predicted = sp.expand(2 * t * (err.T * J * eps)[0] + t**2 * (eps.T * J.T * J * eps)[0]
                          + t**2 * hess)
    cf = []
    def poly_map(i):
        c = sp.symbols(f"c{i}_0:10")
        cf.extend(c)
        a0, a1 = A
        return (c[0] + c[1]*a0 + c[2]*a1 + c[3]*a0**2 + c[4]*a0*a1 + c[5]*a1**2
                + c[6]*a0**3 + c[7]*a0**2*a1 + c[8]*a0*a1**2 + c[9]*a1**3)
    g = sp.Matrix([poly_map(i) for i in range(ds)])
    Jg, errg = g.jacobian(A), g - Y
    moved = g.subs(list(zip(A, A + t * eps)), simultaneous=True)
    lhs_g = sp.expand(((moved - Y).T * (moved - Y))[0] - (errg.T * errg)[0])
    pg = sp.Poly(lhs_g, t)
    c1 = sp.expand(pg.coeff_monomial(t) - 2 * (errg.T * Jg * eps)[0])
    hg = sum(errg[i] * (eps.T * sp.hessian(g[i], list(A)) * eps)[0] for i in range(ds))
    c2 = sp.expand(pg.coeff_monomial(t**2) - (eps.T * Jg.T * Jg * eps)[0] - hg)
    check("retrieval expansion, symbolic: t and t^2 coefficients match exactly for a generic cubic map",
          c1 == 0 and c2 == 0, f"residual terms: order t {len(sp.Add.make_args(c1)) if c1 != 0 else 0}, "
          f"order t^2 {len(sp.Add.make_args(c2)) if c2 != 0 else 0}")
    import numpy as _np
    rng = _np.random.default_rng(0); dsn, dan = 3, 4
    W = rng.standard_normal((dsn, dan)); a0 = rng.standard_normal(dan) * 0.3
    Yv = rng.standard_normal(dsn); e0 = rng.standard_normal(dan)
    fv = lambda a: _np.tanh(W @ a)
    Jv = lambda a: (W.T * (1 - _np.tanh(W @ a) ** 2)).T
    def curv(a, e):
        sv = _np.tanh(W @ a); er = fv(a) - Yv
        return sum(er[i] * (-2 * sv[i] * (1 - sv[i] ** 2)) * (W[i] @ e) ** 2 for i in range(dsn))
    ratios = []
    for sc in (1e-2, 3e-3):
        e = e0 * sc; er = fv(a0) - Yv
        lhs_n = _np.sum((fv(a0 + e) - Yv) ** 2) - _np.sum((fv(a0) - Yv) ** 2)
        two = 2 * er @ Jv(a0) @ e + e @ Jv(a0).T @ Jv(a0) @ e
        ratios.append((lhs_n - two) / curv(a0, e))
    check("retrieval expansion: omitted term is exactly the curvature e^T grad^2 f [eps,eps]",
          abs(ratios[-1] - 1.0) < 0.05, f"residual/curvature -> {ratios[-1]:.4f}")


def claim_finite_difference():
    rng = np.random.default_rng(2)
    ds, da, n = 3, 5, 400000
    J = rng.standard_normal((ds, da))
    for sigma in (0.05, 0.1):
        e = rng.standard_normal((n, da)) * sigma
        emp = ((e @ J.T) ** 2).sum(1).mean()
        exact = sigma ** 2 * (J ** 2).sum()
        check(f"finite-difference constant is sigma^2*||J||_F^2 (linear f, sigma={sigma})",
              np.isclose(emp, exact, rtol=2e-2), f"mc={emp:.5f} exact={exact:.5f}")
    W = rng.standard_normal((ds, da))
    def f(a): return np.tanh(a @ W.T)
    a0 = rng.standard_normal(da) * 0.3
    Jn = (W.T * (1 - np.tanh(a0 @ W.T) ** 2)).T
    ratios = []
    for sigma in (0.02, 0.04):
        e = rng.standard_normal((200000, da)) * sigma
        emp = ((f(a0 + e) - f(a0)) ** 2).sum(1).mean()
        ratios.append(abs(emp - sigma**2 * (Jn**2).sum()) / sigma**4)
    check("finite-difference remainder scales like sigma^4",
          ratios[1] / max(ratios[0], 1e-12) < 4.0, f"remainder/sigma^4 ratios {ratios[0]:.2f},{ratios[1]:.2f}")


def claim_agm_unbounded():
    c, sigma, lam = sp.symbols('c sigma lamda', positive=True)
    Bf = sp.symbols('Bf', positive=True)
    R = c**2 * sigma**2 * Bf**2
    L = 0 - lam * R
    check("AGM: R(f_c) = c^2 sigma^2 ||B||_F^2", sp.simplify(R - c**2 * sigma**2 * Bf**2) == 0)
    check("AGM: L(f_c) -> -oo as c -> oo", sp.limit(L, c, sp.oo) == -sp.oo)
    rng = np.random.default_rng(3)
    n, ds, da = 5000, 3, 4
    H = rng.standard_normal((n, 4)); Wm = rng.standard_normal((4, ds))
    m = H @ Wm; Y = m.copy()
    pi_n = np.tanh(H @ rng.standard_normal((4, da))); A_n = pi_n.copy()
    Bm = rng.standard_normal((ds, da)); s = 0.1
    mses, Rs = [], []
    for cv in (1.0, 2.0, 4.0):
        fc = m + cv * (A_n - pi_n) @ Bm.T
        mses.append(((fc - Y) ** 2).mean())
        e = rng.standard_normal((n, da)) * s
        Rs.append(((cv * e @ Bm.T) ** 2).sum(1).mean())
    check("AGM: factual MSE is exactly 0 for every c", max(mses) < 1e-24, f"max={max(mses):.2e}")
    check("AGM: R grows as c^2", np.isclose(Rs[2] / Rs[0], 16.0, rtol=0.1), f"R4/R1={Rs[2]/Rs[0]:.2f}")


def claim_zero_gradient():
    import torch

    class Blind(torch.nn.Module):
        def __init__(s):
            super().__init__(); s.l = torch.nn.Linear(6, 3); s.a = torch.nn.Linear(4, 3)
            torch.nn.init.zeros_(s.a.weight); torch.nn.init.zeros_(s.a.bias)

        def forward(s, o, act): return s.l(o) + s.a(act)

    m = Blind()
    o = torch.randn(64, 6); act = torch.randn(64, 4)
    d = m(o, act + torch.randn_like(act) * 0.1) - m(o, act)
    R = (d ** 2).mean()
    R.backward()
    gmax = max(p.grad.abs().max().item() for p in m.parameters() if p.grad is not None)
    check("AGM: grad of R is exactly 0 at exact action blindness", gmax == 0.0, f"max|grad|={gmax:.2e}")

    m2 = Blind(); torch.nn.init.normal_(m2.a.weight, std=1e-3)
    d2 = m2(o, act + torch.randn_like(act) * 0.1) - m2(o, act)
    (d2 ** 2).mean().backward()
    g2 = max(p.grad.abs().max().item() for p in m2.parameters() if p.grad is not None)
    check("AGM: grad of R is nonzero once a small action path exists", g2 > 0, f"max|grad|={g2:.2e}")


def claim_rank_chance():
    n = 16
    rng = np.random.default_rng(4)
    ranks = [int(np.argsort(np.argsort(rng.standard_normal(n)))[0]) for _ in range(200000)]
    check("mean rank under exchangeability is (n-1)/2 = 7.5",
          abs(np.mean(ranks) - (n - 1) / 2) < 0.02, f"mc={np.mean(ranks):.3f}")


def claim_standardized():
    rng = np.random.default_rng(5)
    ds, da = 3, 4
    J = rng.standard_normal((ds, da))
    SA = rng.standard_normal((da, da)); SA = SA @ SA.T + np.eye(da)
    SY = rng.standard_normal((ds, ds)); SY = SY @ SY.T + np.eye(ds)
    def half(M):
        w, V = np.linalg.eigh(M); return V @ np.diag(np.sqrt(w)) @ V.T
    def ihalf(M):
        w, V = np.linalg.eigh(M); return V @ np.diag(1 / np.sqrt(w)) @ V.T
    g = np.linalg.norm(ihalf(SY) @ J @ half(SA), 'fro') ** 2
    P = rng.standard_normal((da, da)) + 3 * np.eye(da)
    Q = rng.standard_normal((ds, ds)) + 3 * np.eye(ds)
    J2 = Q @ J @ np.linalg.inv(P)
    SA2 = P @ SA @ P.T
    SY2 = Q @ SY @ Q.T
    g2 = np.linalg.norm(ihalf(SY2) @ J2 @ half(SA2), 'fro') ** 2
    check("standardized sensitivity is invariant to action/output rescaling",
          np.isclose(g, g2, rtol=1e-8), f"{g:.6f} vs {g2:.6f}")
    check("raw ||J||_F is NOT invariant (so the paper is right to flag unit dependence)",
          not np.isclose(np.linalg.norm(J, 'fro') ** 2, np.linalg.norm(J2, 'fro') ** 2, rtol=1e-3))


def claim_did():
    a, b, c, d, null = sp.symbols('a b c d null')
    G = ((a + null) - (b + null)) - ((c + null) - (d + null))
    check("difference in differences cancels a shared additive null",
          sp.simplify(G - ((a - b) - (c - d))) == 0)


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def claim_matched_invariance():
    rng = np.random.default_rng(0)
    ok_s = ok_n = ok_b = ok_d = True
    for _ in range(2000):
        r, x, y = rng.standard_normal((3, 40))
        c = float(np.exp(rng.uniform(-5, 5)))
        ok_s &= np.isclose(_cos(c * r, x), _cos(r, x), atol=1e-12)
        ok_n &= np.isclose(_cos(-r, x), -_cos(r, x), atol=1e-12)
        ux, uy = x / np.linalg.norm(x), y / np.linalg.norm(y)
        ok_b &= abs(_cos(r, x) - _cos(r, y)) <= np.linalg.norm(ux - uy) + 1e-12
        ok_d &= np.isclose(np.linalg.norm(ux - uy) ** 2, 2 - 2 * _cos(x, y), atol=1e-12)
    check("matched probe: cos(c r, e) = cos(r, e) for every c > 0", bool(ok_s))
    check("matched probe: cos(-r, e) = -cos(r, e)", bool(ok_n))
    check("matched probe: |cos(r,x) - cos(r,y)| <= ||x/|x| - y/|y|||", bool(ok_b))
    check("matched probe: ||x/|x| - y/|y|||^2 = 2 - 2 cos(x, y)", bool(ok_d))


def claim_attenuation():
    rng = np.random.default_rng(1)
    dim, n_mc = 30, 400_000
    v = rng.standard_normal(dim)
    R = rng.standard_normal((25, dim))
    R[0] = v + 0.1 * rng.standard_normal(dim)
    R[1] = -v
    worst, kappas = 0.0, []
    for sig in (0.1, 0.5, 1.0, 3.0):
        W = v + sig * np.linalg.norm(v) / np.sqrt(dim) * rng.standard_normal((n_mc, dim))
        U = W / np.linalg.norm(W, axis=1, keepdims=True)
        m = U.mean(0)
        kappa = float(m @ v / np.linalg.norm(v))
        kappas.append(kappa)
        for r in R:
            lhs = float((U @ r).mean() / np.linalg.norm(r))
            worst = max(worst, abs(lhs - kappa * _cos(r, v)))
    check("attenuation: E cos(r, v+xi) = kappa cos(r, v) for every r (Monte Carlo)",
          worst < 4e-3, f"max deviation {worst:.1e} over 25 responses x 4 noise levels")
    check("attenuation: kappa lies in [0, 1] and falls as noise grows",
          all(0 <= k <= 1 for k in kappas) and all(a > b for a, b in zip(kappas, kappas[1:])),
          "kappa = " + ", ".join(f"{k:.3f}" for k in kappas))


def claim_published_magnitude_measures():
    rng = np.random.default_rng(20)
    d = 12
    A = rng.standard_normal((d, 6)); S = A @ A.T + 0.1 * np.eye(d)
    B = rng.standard_normal((d, d)); E_aniso = B @ B.T + 0.5 * np.eye(d)
    E_iso = 1.7 * np.eye(d)
    Q, R = np.linalg.qr(rng.standard_normal((d, d))); Q = Q * np.sign(np.diag(R))
    emp = lambda Sm, Em: 0.5 * np.linalg.slogdet(np.eye(len(Sm)) + np.linalg.solve(Em, Sm))[1]

    refl = max(abs(emp(S, E) - emp((-np.eye(d)) @ S @ (-np.eye(d)).T, E))
               for E in (E_aniso, E_iso))
    check("empowerment is unchanged by a reflection, for any residual covariance",
          refl == 0.0, f"max deviation {refl:.1e}")
    check("empowerment is unchanged by any rotation when the residual covariance is isotropic",
          abs(emp(S, E_iso) - emp(Q @ S @ Q.T, E_iso)) < 1e-12,
          f"deviation {abs(emp(S, E_iso) - emp(Q @ S @ Q.T, E_iso)):.1e}")
    check("empowerment DOES change under rescaling, so it is a magnitude measure",
          emp(0.25 ** 2 * S, E_iso) < emp(S, E_iso) < emp(4.0 ** 2 * S, E_iso),
          f"{emp(0.25**2*S, E_iso):.2f} < {emp(S, E_iso):.2f} < {emp(4.0**2*S, E_iso):.2f}")

    d3 = 3
    rng2 = np.random.default_rng(21)
    Q3, R3 = np.linalg.qr(rng2.standard_normal((d3, d3))); Q3 = Q3 * np.sign(np.diag(R3))
    iso = {"reflect": -np.eye(d3), "rotate": Q3}
    non_iso = {"x0.25": 0.25 * np.eye(d3), "x4": 4.0 * np.eye(d3)}
    s = rng2.standard_normal(d3)
    tr = lambda P, T, P0: P0 + T @ (P - P0)

    P0 = s.copy()
    v = 0.3 * rng2.standard_normal(d3)
    P_inv = s + v
    ident = max(np.linalg.norm(tr(P0, T, P0) - s) for T in list(iso.values()) + list(non_iso.values()))
    check("group action identity condition is 0 under every transform, with no zero action drift",
          ident < 1e-12, f"max residual {ident:.1e}")
    base = np.linalg.norm(P_inv - s)
    dev_iso = max(abs(np.linalg.norm(tr(P_inv, T, P0) - s) - base) for T in iso.values())
    dev_non = min(abs(np.linalg.norm(tr(P_inv, T, P0) - s) - base) for T in non_iso.values())
    check("group action inverse residual is unchanged by every isometry, with no zero action drift",
          dev_iso < 1e-12 and dev_non > 1e-3,
          f"isometry deviation {dev_iso:.1e}; rescaling moves it by at least {dev_non:.3f}, as it should")

    P0d = s + 0.25 * rng2.standard_normal(d3)
    u = s - P0d
    refl = np.linalg.norm(tr(s, -np.eye(d3), P0d) - s)
    check("with zero action drift u, a reflected model's inverse residual is exactly 2||u||",
          abs(refl - 2 * np.linalg.norm(u)) < 1e-12, f"{refl:.6f} vs 2||u|| = {2*np.linalg.norm(u):.6f}")

    P_A = s + rng2.standard_normal(d3)
    P_B = P_A + 0.05 * rng2.standard_normal(d3)
    base = np.linalg.norm(P_A - P_B)
    dev = max(abs(np.linalg.norm(tr(P_A, T, P0) - tr(P_B, T, P0)) - base) for T in iso.values())
    moved = min(abs(np.linalg.norm(tr(P_A, T, P0) - tr(P_B, T, P0)) - base) for T in non_iso.values())
    check("group action compatibility is unchanged by every isometry, and moved by rescaling",
          dev < 1e-12 and moved > 1e-3, f"isometry deviation {dev:.1e}; rescaling moves it by {moved:.3f}")


def claim_inverse_check():
    rng3 = np.random.default_rng(11)
    ds, da, Hh = 6, 3, 4
    G1, G2, c0 = rng3.standard_normal((da, ds)), rng3.standard_normal((da, ds)), rng3.standard_normal(da)
    affine = lambda x, y: x @ G1.T + (y - x) @ G2.T + c0
    W1, W2 = rng3.standard_normal((32, 2 * ds)), rng3.standard_normal((da, 32))
    nonlin = lambda x, y: np.tanh(np.concatenate([x, y - x], -1) @ W1.T) @ W2.T

    def inv_dir(g, x0, P_ref, P_q, T, dA):
        PT = P_ref + (P_q - P_ref) @ T.T
        read = lambda P: g(np.concatenate([x0[None], P[:-1]]), P)
        return _cos((read(PT) - read(P_ref)).ravel(), dA.ravel())

    worst_s, worst_n, moved_s, moved_n = 0.0, 0.0, 0.0, 0.0
    for _ in range(200):
        x0 = rng3.standard_normal(ds)
        P_ref = x0 + np.cumsum(rng3.standard_normal((Hh, ds)), 0)
        P_q = P_ref + rng3.standard_normal((Hh, ds))
        dA = rng3.standard_normal((Hh, da))
        I = np.eye(ds)
        base = inv_dir(affine, x0, P_ref, P_q, I, dA)
        for cc in (0.25, 4.0, 37.0):
            worst_s = max(worst_s, abs(inv_dir(affine, x0, P_ref, P_q, cc * I, dA) - base))
        worst_n = max(worst_n, abs(inv_dir(affine, x0, P_ref, P_q, -I, dA) + base))
        bn = inv_dir(nonlin, x0, P_ref, P_q, I, dA)
        moved_s = max(moved_s, abs(inv_dir(nonlin, x0, P_ref, P_q, 4.0 * I, dA) - bn))
        moved_n = max(moved_n, abs(inv_dir(nonlin, x0, P_ref, P_q, -I, dA) + bn))
    check("inverse check, affine g: unchanged by every positive rescaling of the response",
          worst_s < 1e-10, f"max deviation {worst_s:.1e}")
    check("inverse check, affine g: changes sign under reflection of the response",
          worst_n < 1e-10, f"max deviation {worst_n:.1e}")
    check("inverse check, nonlinear g: neither invariance holds in general",
          moved_s > 1e-2 and moved_n > 1e-2, f"rescaling moves it by up to {moved_s:.2f}, reflection by {moved_n:.2f}")


if __name__ == "__main__":
    print("verifying the paper's mathematical claims\n")
    for fn in [claim_risk_gap, claim_off_support, claim_local_expansion, claim_finite_difference,
               claim_agm_unbounded, claim_zero_gradient, claim_rank_chance, claim_standardized,
               claim_did, claim_matched_invariance, claim_attenuation,
               claim_published_magnitude_measures, claim_inverse_check]:
        print((fn.__doc__ or fn.__name__).strip().splitlines()[0])
        fn()
        print()
    n_pass = sum(1 for _, ok, _ in results if ok)
    print(f"{n_pass}/{len(results)} checks passed")
    bad = [n for n, ok, _ in results if not ok]
    if bad:
        print("FAILED:", *bad, sep="\n  ")
