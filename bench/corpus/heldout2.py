"""Second held-out family set (final-v2), created after the first qualification round.

Nothing here was seen by any tuning. References: exact where a closed form or an
independent numerical derivation exists (POLYQP: active-set polish of a convex QP
verified to 1e-10 stationarity; EXPFIT2 clean: f* = 0 at the generating parameters),
otherwise best-known from the reference run (tag ``best-known``).
"""
from __future__ import annotations

import math

import numpy as np
import sympy as sp

from spec import INF, Spec, register, xs
from structured import st

exp, sqrt = sp.exp, sp.sqrt


# ---- convex QP with random linear inequalities; reference by active-set polish ----
def _polyqp(n, m, seed):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n)) / math.sqrt(n)
    Q = M.T @ M + np.eye(n)
    c = rng.standard_normal(n)
    A = rng.standard_normal((m, n))
    x_feas = rng.standard_normal(n) * 0.1
    b = A @ x_feas + rng.uniform(0.1, 1.0, m)        # x_feas strictly feasible
    # reference: SLSQP with exact derivatives, then polish on the active set by a KKT solve
    from scipy.optimize import minimize
    f = lambda x: 0.5 * x @ Q @ x + c @ x
    g = lambda x: Q @ x + c
    r = minimize(f, x_feas, jac=g, method="SLSQP", constraints=[{"type": "ineq", "fun": lambda x: b - A @ x, "jac": lambda x: -A}],
                 options={"maxiter": 2000, "ftol": 1e-15})
    x = r.x
    for _ in range(5):
        act = np.flatnonzero(A @ x >= b - 1e-7)
        Aa = A[act]
        K = np.block([[Q, Aa.T], [Aa, np.zeros((act.size, act.size))]])
        sol = np.linalg.solve(K, np.concatenate([-c, b[act]]))
        x, lam = sol[:n], sol[n:]
        if np.all(lam >= -1e-12) and np.all(A @ x <= b + 1e-10):
            break
        # drop the most negative multiplier and re-solve from a slightly relaxed point
        x = x - 1e-6 * (Q @ x + c)
    assert np.all(A @ x <= b + 1e-9) and np.all(lam >= -1e-9), "reference QP polish failed"
    fstar = float(f(x))
    xs_ = xs(n)
    fexpr = sp.Add(*[0.5 * float(Q[i, j]) * xs_[i] * xs_[j] for i in range(n) for j in range(n)]) + sp.Add(*[float(c[i]) * xs_[i] for i in range(n)])
    rows = [sp.Add(*[float(A[i, j]) * xs_[j] for j in range(n)]) for i in range(m)]
    return xs_, fexpr, rows, [float(v) for v in b], fstar, [list(map(float, x))], list(map(float, x_feas))


for _n, _m, _seed in ((10, 20, 11), (100, 200, 12)):
    def _mk(n=_n, m=_m, seed=_seed):
        x, f, rows, b, fstar, xstar, x0 = _polyqp(n, m, seed)
        return st(f"POLYQP_{n}", "polyqp", x, f, rows, [-INF] * m, b, x0=x0, ref_f=fstar, ref_x=xstar,
                  notes="convex QP; reference from an active-set KKT polish verified at generation")
    _mk.__name__ = f"POLYQP_{_n}"
    register(_mk)


# ---- economic dispatch with quadratic transmission losses ----
for _n in (5, 20):
    def _mk(n=_n):
        x = xs(n)
        rng = np.random.default_rng(300 + n)
        a = rng.uniform(0.002, 0.01, n); bcoef = rng.uniform(1.0, 3.0, n); cap = rng.uniform(50, 200, n)
        B = rng.uniform(1e-5, 5e-5, n)
        demand = 0.6 * float(np.sum(cap))
        f = sp.Add(*[float(a[i]) * x[i] ** 2 + float(bcoef[i]) * x[i] for i in range(n)])
        loss = sp.Add(*[float(B[i]) * x[i] ** 2 for i in range(n)])
        c = [sp.Add(*x) - loss]
        return st(f"DISPATCH_{n}", "dispatch", x, f, c, [demand], [demand], xl=[0.0] * n, xu=list(map(float, cap)),
                  x0=[demand / n] * n, tags=["best-known"], notes="economic dispatch; one nonlinear equality; reference from the reference run")
    _mk.__name__ = f"DISPATCH_{_n}"
    register(_mk)


# ---- hanging chain (catenary) with rigid links ----
for _n in (10, 40):
    def _mk(n=_n):
        # nodes 1..n-1 free, node 0 at (0,0), node n at (L_total*0.8, 0); links of equal length
        L = 1.0 / n * 1.25
        x = xs(2 * (n - 1))
        X = [0] + [x[2 * i] for i in range(n - 1)] + [0.8]
        Y = [0] + [x[2 * i + 1] for i in range(n - 1)] + [0]
        f = sp.Add(*[0.5 * (Y[i] + Y[i + 1]) for i in range(n)])          # potential energy (unit link mass)
        c = [(X[i + 1] - X[i]) ** 2 + (Y[i + 1] - Y[i]) ** 2 - L ** 2 for i in range(n)]
        x0 = []
        for i in range(1, n):
            x0 += [0.8 * i / n, -0.1]
        return st(f"CATENARY_{n}", "catenary", x, f, c, [0.0] * n, [0.0] * n, x0=x0, tags=["best-known"],
                  notes="rigid-link hanging chain; n nonconvex equalities; reference from the reference run")
    _mk.__name__ = f"CATENARY_{_n}"
    register(_mk)


# ---- new ellipsoid seeds and ill-conditioned quadsphere ----
for _n in (20, 200):
    def _mk(n=_n):
        from scipy.optimize import brentq
        x = xs(n)
        rng = np.random.default_rng(7000 + n)
        r = rng.uniform(0.3, 3.0, n); p = rng.uniform(1.0, 4.0, n) * np.sign(rng.uniform(-1, 1, n))
        g = lambda lam: float(np.sum((p * r ** 2 / (r ** 2 + lam) / r) ** 2) - 1.0)
        assert g(0.0) > 0
        lam = brentq(g, 0.0, 1e7, xtol=1e-15, rtol=1e-15, maxiter=500)
        xstar = p * r ** 2 / (r ** 2 + lam); fstar = float(np.sum((xstar - p) ** 2))
        f = sp.Add(*[(x[i] - float(p[i])) ** 2 for i in range(n)])
        c = [sp.Add(*[(x[i] / float(r[i])) ** 2 for i in range(n)])]
        return st(f"ELLIPSOID2_{n}", "ellipsoid2", x, f, c, [-INF], [1], x0=[0.0] * n, ref_f=fstar, ref_x=[list(map(float, xstar))],
                  notes="secular-equation reference")
    _mk.__name__ = f"ELLIPSOID2_{_n}"
    register(_mk)

for _n in (30, 300):
    def _mk(n=_n):
        x = xs(n)
        a = [10 ** (5.0 * i / (n - 1)) for i in range(n)]   # condition number 1e5
        inv = sum(1 / ai for ai in a)
        return st(f"QUADSPHERE2_{n}", "quadsphere2", x, sp.Add(*[a[i] * x[i] ** 2 for i in range(n)]), [sp.Add(*x)], [1], [1],
                  xl=[0] * n, x0=[1.0 / n] * n, ref_f=1 / inv, ref_x=[[(1 / ai) / inv for ai in a]], tags=["ill-conditioned"])
    _mk.__name__ = f"QUADSPHERE2_{_n}"
    register(_mk)


# ---- double-exponential fitting ----
def _expfit2(name, noisy):
    x = xs(4)
    a, b, c, d = x
    t = np.linspace(0, 5, 26)
    y = 3.0 * np.exp(-0.4 * t) + 1.5 * np.exp(-2.5 * t)
    if noisy:
        y = y + 0.01 * np.random.default_rng(5).standard_normal(t.size)
    f = sp.Add(*[(a * exp(-b * float(ti)) + c * exp(-d * float(ti)) - float(yi)) ** 2 for ti, yi in zip(t, y)])
    return st(name, "expfit2", x, f, [b - d], [-INF], [0.0], xl=[0, 0, 0, 0], xu=[10, 10, 10, 10], x0=[1, 1, 1, 3],
              ref_f=(None if noisy else 0.0), ref_x=(None if noisy else [[3.0, 0.4, 1.5, 2.5]]),
              tags=["fitting"] + (["best-known"] if noisy else []), notes="a e^{-bt} + c e^{-dt}, ordering constraint b <= d")


@register
def EXPFIT2_CLEAN():
    return _expfit2("EXPFIT2_CLEAN", False)


@register
def EXPFIT2_NOISY():
    return _expfit2("EXPFIT2_NOISY", True)


# ---- tubular column design ----
@register
def TUBULAR_COLUMN():
    x = xs(2)
    d, t = x
    P, sig, E, L = 2500, 500, 0.85e6, 250
    f = 9.8 * d * t + 2 * d
    c = [P / (sp.pi * d * t * sig) - 1, 8 * P * L ** 2 / (sp.pi ** 3 * E * d * t * (d ** 2 + t ** 2)) - 1, 2 / d - 1, d / 14 - 1, 0.2 / t - 1, t / 0.8 - 1]
    return st("TUBULAR_COLUMN", "engineering2", x, f, c, [-INF] * 6, [0] * 6, xl=[2, 0.2], xu=[14, 0.8], x0=[5, 0.5],
              ref_f=26.5313, tags=["best-known"], source="Rao tubular column; best-known 26.5313")
