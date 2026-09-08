"""Scalable structured families and engineering-design models.

References: closed form where available (QUADSPHERE, CHAINROSEN), an
independent numerical derivation done at generation time from the KKT system
of a convex problem (LQTRAJ: linear KKT solve; ELLIPSOID: 1-D secular
equation), or a *best-known* published value (engineering designs, tag
``best-known``). Best-known values are targets, not certificates.
"""
from __future__ import annotations

import math

import numpy as np
import sympy as sp

from spec import INF, Spec, register, xs

sqrt, exp = sp.sqrt, sp.exp


def st(name, family, x, f, c=(), cl=(), cu=(), xl=None, xu=None, x0=None, ref_f=None, ref_x=None, tags=(), notes="", source=""):
    n = len(x)
    return Spec(name=name, family=family, x=list(x), f=sp.sympify(f), c=[sp.sympify(e) for e in c],
                cl=list(cl), cu=list(cu), xl=list(xl) if xl is not None else [-INF] * n,
                xu=list(xu) if xu is not None else [INF] * n, x0=list(x0), ref_f=ref_f, ref_x=ref_x,
                source=source or "bench/corpus/structured.py", tags=list(tags), notes=notes)


# ---------------- chained Rosenbrock (bounds; + one linear equality) ----------------
def _chainrosen(n):
    x = xs(n)
    return x, sp.Add(*[100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(n - 1)])


def _rosen_x0(n):
    return [(-1.2 if i % 2 == 0 else 1.0) for i in range(n)]


for _n in (10, 50, 200):
    def _mk_box(n=_n):
        x, f = _chainrosen(n)
        return st(f"CHAINROSEN_BOX_{n}", "chainrosen", x, f, xl=[-2] * n, xu=[2] * n, x0=_rosen_x0(n), ref_f=0.0, ref_x=[[1] * n],
                  tags=["multiple-local-minima"], notes="global minimum f=0 at x=1; a local minimum near x1=-1 exists for n>=4")
    _mk_box.__name__ = f"CHAINROSEN_BOX_{_n}"
    register(_mk_box)

    def _mk_eq(n=_n):
        x, f = _chainrosen(n)
        return st(f"CHAINROSEN_EQ_{n}", "chainrosen", x, f, [sp.Add(*x)], [n], [n], xl=[-2] * n, xu=[2] * n, x0=_rosen_x0(n), ref_f=0.0,
                  ref_x=[[1] * n], tags=["multiple-local-minima"])
    _mk_eq.__name__ = f"CHAINROSEN_EQ_{_n}"
    register(_mk_eq)


# ---------------- diagonal quadratic on the simplex plane: closed form ----------------
for _n in (10, 100, 1000):
    def _mk(n=_n):
        x = xs(n)
        a = [10 ** (3.0 * i / (n - 1)) for i in range(n)]   # condition number 1e3
        f = sp.Add(*[a[i] * x[i] ** 2 for i in range(n)])
        inv = sum(1 / ai for ai in a)
        xstar = [(1 / ai) / inv for ai in a]
        return st(f"QUADSPHERE_{n}", "quadsphere", x, f, [sp.Add(*x)], [1], [1], xl=[0] * n, x0=[1.0 / n + 0.5 * (-1) ** i / n for i in range(n)],
                  ref_f=1 / inv, ref_x=[xstar], notes="x_i = (1/a_i)/sum(1/a_j); bounds inactive")
    _mk.__name__ = f"QUADSPHERE_{_n}"
    register(_mk)


# ---------------- projection onto an ellipsoid: 1-D secular equation ----------------
for _n in (5, 50, 500):
    def _mk(n=_n):
        from scipy.optimize import brentq
        x = xs(n)
        rng = np.random.default_rng(1000 + n)
        r = rng.uniform(0.5, 2.0, n)
        p = rng.uniform(1.0, 3.0, n) * np.sign(rng.uniform(-1, 1, n))
        # KKT: 2(x-p) + 2 lam x / r^2 = 0 -> x_i = p_i r_i^2 / (r_i^2 + lam); find lam >= 0 with sum (x_i/r_i)^2 = 1.
        def g(lam):
            xi = p * r ** 2 / (r ** 2 + lam)
            return float(np.sum((xi / r) ** 2) - 1.0)
        assert g(0.0) > 0, "p must start outside the ellipsoid"
        lam = brentq(g, 0.0, 1e6, xtol=1e-15, rtol=1e-15, maxiter=500)
        xstar = p * r ** 2 / (r ** 2 + lam)
        fstar = float(np.sum((xstar - p) ** 2))
        f = sp.Add(*[(x[i] - float(p[i])) ** 2 for i in range(n)])
        c = [sp.Add(*[(x[i] / float(r[i])) ** 2 for i in range(n)])]
        return st(f"ELLIPSOID_{n}", "ellipsoid", x, f, c, [-INF], [1], x0=[0.0] * n, ref_f=fstar, ref_x=[list(map(float, xstar))],
                  notes="reference from the secular equation solved to 1e-15 at generation time")
    _mk.__name__ = f"ELLIPSOID_{_n}"
    register(_mk)


# ---------------- LQ trajectory: double integrator, minimum-energy rest-to-rest ----------------
for _N in (10, 50, 200):
    def _mk(N=_N):
        T = 1.0
        dt = T / N
        # variables: p_1..p_N, v_1..v_N, u_0..u_{N-1}; p_0 = v_0 = 0 known; p_N = 1, v_N = 0 as fixed variables (bounds).
        x = xs(3 * N)
        P = x[0:N]; V = x[N:2 * N]; U = x[2 * N:3 * N]
        f = sp.Add(*[dt * u ** 2 for u in U])
        c = []
        for k in range(N):
            pk = P[k - 1] if k > 0 else 0
            vk = V[k - 1] if k > 0 else 0
            c.append(P[k] - pk - dt * vk)
            c.append(V[k] - vk - dt * U[k])
        m = len(c)
        xl = [-INF] * (3 * N); xu = [INF] * (3 * N)
        xl[N - 1] = xu[N - 1] = 1.0     # p_N = 1
        xl[2 * N - 1] = xu[2 * N - 1] = 0.0  # v_N = 0
        for k in range(N):
            xl[2 * N + k], xu[2 * N + k] = -50.0, 50.0
        # Independent reference: solve the equality-constrained QP KKT system exactly (fixed vars as equalities).
        n = 3 * N
        H = np.zeros((n, n))
        for k in range(N):
            H[2 * N + k, 2 * N + k] = 2 * dt
        A = np.zeros((m + 2, n)); b = np.zeros(m + 2)
        for k in range(N):
            A[2 * k, k] = 1
            if k > 0:
                A[2 * k, k - 1] = -1; A[2 * k, N + k - 1] = -dt
            A[2 * k + 1, N + k] = 1
            if k > 0:
                A[2 * k + 1, N + k - 1] = -1
            A[2 * k + 1, 2 * N + k] = -dt
        A[m, N - 1] = 1; b[m] = 1.0
        A[m + 1, 2 * N - 1] = 1; b[m + 1] = 0.0
        K = np.block([[H, A.T], [A, np.zeros((m + 2, m + 2))]])
        rhs = np.concatenate([np.zeros(n), b])
        sol = np.linalg.solve(K, rhs)
        xstar = sol[:n]
        fstar = float(xstar @ H @ xstar / 2)
        assert np.max(np.abs(xstar[2 * N:])) < 50, "control bound must be inactive for the closed-form reference"
        x0 = [0.0] * n
        x0[N - 1] = 1.0
        return st(f"LQTRAJ_{N}", "lqtraj", x, f, c, [0.0] * m, [0.0] * m, xl=xl, xu=xu, x0=x0, ref_f=fstar, ref_x=[list(map(float, xstar))],
                  tags=["fixed", "sparse"], notes="minimum-energy double integrator; reference from a direct KKT solve; terminal conditions as fixed variables")
    _mk.__name__ = f"LQTRAJ_{_N}"
    register(_mk)


# ---------------- exponential fitting with bounds (noise-free -> f* = 0; noisy -> best-known) ----------------
def _expfit(name, noisy, constrained):
    x = xs(3)
    a, b, c0 = x
    t = np.linspace(0, 10, 21)
    rng = np.random.default_rng(42)
    y = 2.0 * np.exp(-0.5 * t) + 1.0
    if noisy:
        y = y + 0.01 * rng.standard_normal(t.size)
    f = sum((a * exp(-b * float(ti)) + c0 - float(yi)) ** 2 for ti, yi in zip(t, y))
    cons, cl, cu = ([a + c0], [-INF], [3.0]) if constrained else ([], [], [])
    ref_f = 0.0 if not noisy else None
    ref_x = [[2.0, 0.5, 1.0]] if not noisy else None
    return st(name, "expfit", x, f, cons, cl, cu, xl=[0, 0, -5], xu=[10, 5, 5], x0=[1, 1, 0], ref_f=ref_f, ref_x=ref_x,
              tags=["fitting"] + (["best-known"] if noisy else []), notes="a exp(-b t) + c fitted to 21 points; truth (2, 0.5, 1)")


@register
def EXPFIT_CLEAN():
    return _expfit("EXPFIT_CLEAN", False, False)


@register
def EXPFIT_CLEAN_CON():
    return _expfit("EXPFIT_CLEAN_CON", False, True)


@register
def EXPFIT_NOISY():
    return _expfit("EXPFIT_NOISY", True, False)


# ---------------- engineering design (published best-known values) ----------------
@register
def PRESSURE_VESSEL():
    x = xs(4)
    Ts, Th, R, L = x
    f = 0.6224 * Ts * R * L + 1.7781 * Th * R ** 2 + 3.1661 * Ts ** 2 * L + 19.84 * Ts ** 2 * R
    c = [-Ts + 0.0193 * R, -Th + 0.00954 * R, -sp.pi * R ** 2 * L - sp.Rational(4, 3) * sp.pi * R ** 3 + 1296000, L - 240]
    return st("PRESSURE_VESSEL", "engineering", x, f, c, [-INF] * 4, [0] * 4, xl=[0.0625, 0.0625, 10, 10], xu=[6.1875, 6.1875, 200, 200],
              x0=[1, 1, 50, 100], ref_f=5885.3327736, tags=["best-known"], source="Sandgren 1990 continuous relaxation; best-known 5885.3327736")


@register
def WELDED_BEAM():
    x = xs(4)
    h, l, t, b = x
    P, Lb, E, G = 6000, 14, 30e6, 12e6
    tau1 = P / (sqrt(2) * h * l)
    M = P * (Lb + l / 2)
    R = sqrt(l ** 2 / 4 + ((h + t) / 2) ** 2)
    J = 2 * (sqrt(2) * h * l * (l ** 2 / 12 + ((h + t) / 2) ** 2))
    tau2 = M * R / J
    tau = sqrt(tau1 ** 2 + 2 * tau1 * tau2 * l / (2 * R) + tau2 ** 2)
    sigma = 6 * P * Lb / (b * t ** 2)
    delta = 4 * P * Lb ** 3 / (E * t ** 3 * b)
    Pc = 4.013 * E * (t * b ** 3 / 6) / Lb ** 2  # sqrt(t^2 b^6/36) with t, b > 0 * (1 - t / (2 * Lb) * sqrt(E / (4 * G)))
    f = 1.10471 * h ** 2 * l + 0.04811 * t * b * (14 + l)
    c = [tau - 13600, sigma - 30000, h - b, 0.10471 * h ** 2 + 0.04811 * t * b * (14 + l) - 5, 0.125 - h, delta - 0.25, P - Pc]
    return st("WELDED_BEAM", "engineering", x, f, c, [-INF] * 7, [0] * 7, xl=[0.1, 0.1, 0.1, 0.1], xu=[2, 10, 10, 2], x0=[0.5, 2, 5, 0.5],
              ref_f=1.724852, tags=["best-known"], source="Rao / Coello welded beam; best-known 1.724852")


@register
def SPRING():
    x = xs(3)
    d, D, N = x
    f = (N + 2) * D * d ** 2
    c = [1 - D ** 3 * N / (71785 * d ** 4), (4 * D ** 2 - d * D) / (12566 * (D * d ** 3 - d ** 4)) + 1 / (5108 * d ** 2) - 1,
         1 - 140.45 * d / (D ** 2 * N), (D + d) / 1.5 - 1]
    return st("SPRING", "engineering", x, f, c, [-INF] * 4, [0] * 4, xl=[0.05, 0.25, 2], xu=[2, 1.3, 15], x0=[0.1, 0.5, 10],
              ref_f=0.012665, tags=["best-known"], source="Arora / Belegundu tension-compression spring; best-known 0.012665")


@register
def SPEED_REDUCER():
    x = xs(7)
    x1, x2, x3, x4, x5, x6, x7 = x
    f = (0.7854 * x1 * x2 ** 2 * (3.3333 * x3 ** 2 + 14.9334 * x3 - 43.0934) - 1.508 * x1 * (x6 ** 2 + x7 ** 2)
         + 7.4777 * (x6 ** 3 + x7 ** 3) + 0.7854 * (x4 * x6 ** 2 + x5 * x7 ** 2))
    c = [27 / (x1 * x2 ** 2 * x3) - 1, 397.5 / (x1 * x2 ** 2 * x3 ** 2) - 1, 1.93 * x4 ** 3 / (x2 * x3 * x6 ** 4) - 1,
         1.93 * x5 ** 3 / (x2 * x3 * x7 ** 4) - 1, sqrt((745 * x4 / (x2 * x3)) ** 2 + 16.9e6) / (110 * x6 ** 3) - 1,
         sqrt((745 * x5 / (x2 * x3)) ** 2 + 157.5e6) / (85 * x7 ** 3) - 1, x2 * x3 / 40 - 1, 5 * x2 / x1 - 1, x1 / (12 * x2) - 1,
         (1.5 * x6 + 1.9) / x4 - 1, (1.1 * x7 + 1.9) / x5 - 1]
    return st("SPEED_REDUCER", "engineering", x, f, c, [-INF] * 11, [0] * 11, xl=[2.6, 0.7, 17, 7.3, 7.3, 2.9, 5.0],
              xu=[3.6, 0.8, 28, 8.3, 8.3, 3.9, 5.5], x0=[3, 0.75, 20, 8, 8, 3.5, 5.25], ref_f=2994.4711,
              tags=["best-known"], source="Golinski speed reducer, continuous relaxation; best-known 2994.4711")


@register
def THREEBAR_TRUSS():
    x = xs(2)
    A1, A2 = x
    P, sig, Lt = 2, 2, 100
    f = (2 * sqrt(2) * A1 + A2) * Lt
    c = [(sqrt(2) * A1 + A2) / (sqrt(2) * A1 ** 2 + 2 * A1 * A2) * P - sig, A2 / (sqrt(2) * A1 ** 2 + 2 * A1 * A2) * P - sig,
         1 / (A1 + sqrt(2) * A2) * P - sig]
    return st("THREEBAR_TRUSS", "engineering", x, f, c, [-INF] * 3, [0] * 3, xl=[0, 0], xu=[1, 1], x0=[0.5, 0.5], ref_f=263.8958434,
              tags=["best-known"], source="Nowcki three-bar truss; best-known 263.8958434")


@register
def CANTILEVER():
    x = xs(5)
    f = 0.0624 * sp.Add(*x)
    c = [61 / x[0] ** 3 + 37 / x[1] ** 3 + 19 / x[2] ** 3 + 7 / x[3] ** 3 + 1 / x[4] ** 3 - 1]
    return st("CANTILEVER", "engineering", x, f, c, [-INF], [0], xl=[0.01] * 5, xu=[100] * 5, x0=[5] * 5, ref_f=1.33995636,
              tags=["best-known"], source="Chickermane & Gea cantilever beam; best-known 1.33995636")


# ---------------- Markowitz-style portfolio (convex QP; reference = best-known from the reference run) ----------------
for _n in (20, 100):
    def _mk(n=_n):
        x = xs(n)
        rng = np.random.default_rng(99 + n)
        F = rng.standard_normal((n, 3))
        Sigma = F @ F.T / 3 + np.diag(rng.uniform(0.05, 0.3, n))
        mu = rng.uniform(0.02, 0.15, n)
        f = sp.Add(*[float(Sigma[i, j]) * x[i] * x[j] for i in range(n) for j in range(n) if abs(Sigma[i, j]) > 0])
        c = [sp.Add(*[float(mu[i]) * x[i] for i in range(n)]), sp.Add(*x)]
        r = float(np.mean(mu) + 0.5 * (np.max(mu) - np.mean(mu)))
        return st(f"PORTFOLIO_{n}", "portfolio", x, f, c, [r, 1], [INF, 1], xl=[0] * n, xu=[1] * n, x0=[1.0 / n] * n,
                  tags=["best-known", "convex"], notes="convex QP: any KKT point is global; reference from the reference run")
    _mk.__name__ = f"PORTFOLIO_{_n}"
    register(_mk)
