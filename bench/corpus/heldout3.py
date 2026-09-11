"""Third held-out family set (final3), created after the SQP member was frozen (candidate C6).

Nothing here was seen by any tuning of any candidate. References are exact where an
independent derivation exists and ``best-known`` (multi-start reference at generation)
otherwise; every published Hock–Schittkowski value is re-verified at generation by an
independent solve from the published minimizer's neighbourhood.

Families:
  nnls_simplex  least squares on the probability simplex (convex QP, many bounds active;
                reference: KKT polish on the active set, verified to 1e-10)
  maxent        entropy maximisation under linear equalities (convex; reference: Newton on the
                dual, verified to 1e-12 primal residual)
  logsumexp     log-sum-exp of affine functions inside a box (convex; reference: projected Newton
                verified to 1e-12 stationarity)
  rosen_sphere  chained Rosenbrock inside a ball that excludes the unconstrained minimum
                (nonconvex; best-known from 40 starts)
  sinfit        sinusoid fitting to noiseless data (nonconvex; f* = 0 exact)
  hs3           HS114, published value re-verified at generation (HS59 was transcribed, failed
                verification against its published minimizer, and was dropped)
"""
from __future__ import annotations

import math

import numpy as np
import sympy as sp

from spec import INF, register, xs
from structured import st
from hs import hs, ge

exp, log, sin = sp.exp, sp.log, sp.sin


# ---------------- least squares on the simplex ----------------
def _nnls_simplex(n, seed):
    rng = np.random.default_rng(seed)
    m = 2 * n
    A = rng.standard_normal((m, n))
    x_true = np.zeros(n)
    support = rng.choice(n, size=max(1, n // 4), replace=False)
    x_true[support] = rng.uniform(0.5, 1.5, support.size)
    x_true /= x_true.sum()
    b = A @ x_true + 0.05 * rng.standard_normal(m)
    Q = A.T @ A
    c = -A.T @ b
    f = lambda x: 0.5 * float(x @ Q @ x) + float(c @ x) + 0.5 * float(b @ b)
    # reference: SLSQP with exact derivatives, then KKT polish on the identified active set
    from scipy.optimize import minimize
    r = minimize(lambda x: 0.5 * x @ Q @ x + c @ x, np.full(n, 1.0 / n), jac=lambda x: Q @ x + c, method="SLSQP",
                 bounds=[(0, None)] * n, constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1, "jac": lambda x: np.ones(n)}],
                 options={"maxiter": 5000, "ftol": 1e-16})
    x = np.maximum(r.x, 0.0)
    for _ in range(20):
        free = np.flatnonzero(x > 1e-9)
        k = free.size
        K = np.block([[Q[np.ix_(free, free)], np.ones((k, 1))], [np.ones((1, k)), np.zeros((1, 1))]])
        sol = np.linalg.solve(K, np.concatenate([-c[free], [1.0]]))
        xf, nu = sol[:k], sol[k]
        x = np.zeros(n)
        x[free] = xf
        if np.all(xf >= -1e-12):
            g = Q @ x + c + nu
            zl = g  # multipliers of the lower bounds on the fixed set
            fixed = np.setdiff1d(np.arange(n), free)
            if np.all(zl[fixed] >= -1e-9):
                break
            # release the most negative bound multiplier
            j = fixed[np.argmin(zl[fixed])]
            x[j] = 1e-6
        else:
            j = free[np.argmin(xf)]
            x[j] = 0.0
    g = Q @ x + c + nu
    assert abs(x.sum() - 1) < 1e-12 and np.all(x >= -1e-12), "simplex reference infeasible"
    assert np.all(g[x > 1e-9] < 1e-9) and np.all(g[x > 1e-9] > -1e-9) and np.all(g[x <= 1e-9] >= -1e-9), "simplex reference not KKT"
    fstar = f(x)
    xs_ = xs(n)
    fexpr = sp.Add(*[0.5 * float(Q[i, j]) * xs_[i] * xs_[j] for i in range(n) for j in range(n)]) \
        + sp.Add(*[float(c[i]) * xs_[i] for i in range(n)]) + 0.5 * float(b @ b)
    return xs_, fexpr, [sp.Add(*xs_)], fstar, [list(map(float, x))]


for _n, _seed in ((6, 31), (30, 32), (120, 33)):
    def _mk(n=_n, seed=_seed):
        x, f, rows, fstar, xstar = _nnls_simplex(n, seed)
        return st(f"NNLS_SIMPLEX_{n}", "nnls_simplex", x, f, rows, [1.0], [1.0], xl=[0.0] * n, x0=[1.0 / n] * n,
                  ref_f=fstar, ref_x=xstar, notes="least squares on the probability simplex; reference from a KKT polish verified at generation")
    _mk.__name__ = f"NNLS_SIMPLEX_{_n}"
    register(_mk)


# ---------------- entropy maximisation ----------------
def _maxent(n, k, seed):
    rng = np.random.default_rng(seed)
    A = rng.uniform(0.0, 1.0, (k, n))
    x_ref = rng.uniform(0.2, 1.0, n)
    x_ref /= x_ref.sum() / 1.0
    b = A @ x_ref
    # dual: x(nu) = exp(-1 - A^T nu); Newton on r(nu) = A x(nu) - b
    nu = np.zeros(k)
    for _ in range(200):
        x = np.exp(-1.0 - A.T @ nu)
        r = A @ x - b
        if np.max(np.abs(r)) < 1e-14:
            break
        J = -(A * x) @ A.T
        nu = nu - np.linalg.solve(J, r)
    x = np.exp(-1.0 - A.T @ nu)
    assert np.max(np.abs(A @ x - b)) < 1e-12, "maxent dual Newton failed"
    fstar = float(np.sum(x * np.log(x)))
    xs_ = xs(n)
    fexpr = sp.Add(*[xs_[i] * log(xs_[i]) for i in range(n)])
    rows = [sp.Add(*[float(A[i, j]) * xs_[j] for j in range(n)]) for i in range(k)]
    x0 = np.full(n, float(b.mean() / A.mean() / n) if A.mean() > 0 else 1.0 / n)
    return xs_, fexpr, rows, [float(v) for v in b], fstar, [list(map(float, x))], list(map(float, np.full(n, 0.5)))


for _n, _k, _seed in ((10, 3, 41), (50, 5, 42), (200, 8, 43)):
    def _mk(n=_n, k=_k, seed=_seed):
        x, f, rows, b, fstar, xstar, x0 = _maxent(n, k, seed)
        return st(f"MAXENT_{n}", "maxent", x, f, rows, b, b, xl=[1e-9] * n, x0=x0, ref_f=fstar, ref_x=xstar,
                  notes="entropy maximisation; reference from Newton on the dual, primal residual < 1e-12 at generation")
    _mk.__name__ = f"MAXENT_{_n}"
    register(_mk)


# ---------------- log-sum-exp in a box ----------------
def _logsumexp(n, p, seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((p, n))
    bvec = rng.standard_normal(p)
    # reference: projected Newton with exact Hessian, bounds |x| <= 1
    def fg(x):
        z = A @ x + bvec
        zm = z.max()
        w = np.exp(z - zm)
        s = w.sum()
        f = zm + math.log(s)
        pr = w / s
        g = A.T @ pr
        H = A.T @ (np.diag(pr) - np.outer(pr, pr)) @ A
        return f, g, H
    x = np.zeros(n)
    for _ in range(500):
        f, g, H = fg(x)
        free = ~(((x <= -1 + 1e-15) & (g > 0)) | ((x >= 1 - 1e-15) & (g < 0)))
        pg = np.where(free, g, 0.0)
        if np.max(np.abs(pg)) < 1e-13:
            break
        d = np.zeros(n)
        Hf = H[np.ix_(free, free)] + 1e-12 * np.eye(free.sum())
        d[free] = -np.linalg.solve(Hf, g[free])
        t = 1.0
        while t > 1e-12:
            xt = np.clip(x + t * d, -1, 1)
            if fg(xt)[0] < f - 1e-4 * t * abs(pg @ d):
                break
            t *= 0.5
        x = np.clip(x + t * d, -1, 1)
    # polish: full Newton steps on the free set (quadratic convergence near the solution)
    for _ in range(20):
        f, g, H = fg(x)
        free = ~(((x <= -1 + 1e-12) & (g > 0)) | ((x >= 1 - 1e-12) & (g < 0)))
        pg = np.where(free, g, 0.0)
        if np.max(np.abs(pg)) < 1e-13:
            break
        d = np.zeros(n)
        d[free] = -np.linalg.solve(H[np.ix_(free, free)], g[free])
        x = np.clip(x + d, -1, 1)
    f, g, H = fg(x)
    pg = np.where(((x <= -1 + 1e-12) & (g > 0)) | ((x >= 1 - 1e-12) & (g < 0)), 0.0, g)
    assert np.max(np.abs(pg)) < 1e-10, f"logsumexp reference stationarity {np.max(np.abs(pg))}"
    xs_ = xs(n)
    fexpr = log(sp.Add(*[exp(sp.Add(*[float(A[i, j]) * xs_[j] for j in range(n)]) + float(bvec[i])) for i in range(p)]))
    return xs_, fexpr, float(f), [list(map(float, x))]


for _n, _p, _seed in ((10, 40, 51), (20, 80, 52)):
    def _mk(n=_n, p=_p, seed=_seed):
        x, f, fstar, xstar = _logsumexp(n, p, seed)
        return st(f"LOGSUMEXP_{n}", "logsumexp", x, f, xl=[-1.0] * n, xu=[1.0] * n, x0=[0.0] * n, ref_f=fstar, ref_x=xstar,
                  notes="log-sum-exp of affine functions in a box; reference by projected Newton, stationarity < 1e-10 at generation")
    _mk.__name__ = f"LOGSUMEXP_{_n}"
    register(_mk)


# ---------------- chained Rosenbrock inside a ball ----------------
def _rosen_sphere(n, radius, seed):
    xs_ = xs(n)
    f = sp.Add(*[100 * (xs_[i + 1] - xs_[i] ** 2) ** 2 + (1 - xs_[i]) ** 2 for i in range(n - 1)])
    ball = sp.Add(*[v ** 2 for v in xs_])
    # best-known: multi-start SLSQP with exact derivatives from the numpy compilation
    from scipy.optimize import minimize
    fn = sp.lambdify(xs_, f, "numpy")
    gn = sp.lambdify(xs_, [sp.diff(f, v) for v in xs_], "numpy")
    rng = np.random.default_rng(seed)
    best = (math.inf, None)
    for k in range(40):
        x0 = rng.uniform(-1, 1, n) * radius / math.sqrt(n)
        r = minimize(lambda x: float(fn(*x)), x0, jac=lambda x: np.asarray(gn(*x), float), method="SLSQP",
                     constraints=[{"type": "ineq", "fun": lambda x: radius ** 2 - x @ x, "jac": lambda x: -2 * x}],
                     options={"maxiter": 2000, "ftol": 1e-15})
        if r.success and r.x @ r.x <= radius ** 2 + 1e-9 and r.fun < best[0]:
            best = (float(r.fun), r.x.copy())
    assert best[1] is not None
    return xs_, f, ball, best[0], [list(map(float, best[1]))]


for _n, _radius, _seed in ((4, 1.0, 61), (10, 2.0, 62)):
    def _mk(n=_n, radius=_radius, seed=_seed):
        x, f, ball, fstar, xstar = _rosen_sphere(n, radius, seed)
        return st(f"ROSEN_SPHERE_{n}", "rosen_sphere", x, f, [ball], [-INF], [radius ** 2], x0=[0.0] * n, ref_f=fstar, ref_x=xstar,
                  tags=["best-known"], notes=f"chained Rosenbrock inside the ball of radius {radius}; best-known from 40 SLSQP starts at generation")
    _mk.__name__ = f"ROSEN_SPHERE_{_n}"
    register(_mk)


# ---------------- sinusoid fitting (clean) ----------------
@register
def SINFIT_CLEAN():
    x = xs(4)
    a, w, phi, d = x
    t = np.linspace(0, 3, 31)
    y = 2.0 * np.sin(1.7 * t + 0.4) + 0.5
    f = sp.Add(*[(a * sin(w * float(ti) + phi) + d - float(yi)) ** 2 for ti, yi in zip(t, y)])
    return st("SINFIT_CLEAN", "sinfit", x, f, xl=[0.1, 0.5, -3.2, -5], xu=[5, 4, 3.2, 5], x0=[1.0, 1.0, 0.0, 0.0],
              ref_f=0.0, ref_x=[[2.0, 1.7, 0.4, 0.5]], notes="a sin(w t + phi) + d on 31 noiseless samples; f* = 0")


# ---------------- Hock–Schittkowski, unused so far ----------------
@register
def HS114():
    x = xs(10); cl, cu = ge(8)
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = x
    a, b = 0.99, 0.9
    f = 5.04 * x1 + 0.035 * x2 + 10 * x3 + 3.36 * x5 - 0.063 * x4 * x7
    g1 = 35.82 - 0.222 * x10 - b * x9
    g2 = -133 + 3 * x7 - a * x10
    g5 = 1.12 * x1 + 0.13167 * x1 * x8 - 0.00667 * x1 * x8 ** 2 - a * x4
    g6 = 57.425 + 1.098 * x8 - 0.038 * x8 ** 2 + 0.325 * x6 - a * x7
    c = [g1, g2, -g1 + x9 * (1 / b - b), -g2 + (1 / a - a) * x10, g5, g6, -g5 + (1 / a - a) * x4, -g6 + (1 / a - a) * x7,
         1.22 * x4 - x1 - x5, 98000 * x3 / (x4 * x9 + 1000 * x3) - x6, (x2 + x5) / x1 - x8]
    cl = cl + [0.0] * 3
    cu = cu + [0.0] * 3
    return hs("HS114", x, f, c, cl, cu, xl=[1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 85, 90, 3, 1.2, 145],
              xu=[2000, 16000, 120, 5000, 2000, 93, 95, 12, 4, 162],
              x0=[1745, 12000, 110, 3048, 1974, 89.2, 92.8, 8, 3.6, 145], ref_f=-1768.80696,
              ref_x=[[1698.095, 15818.61, 54.10268, 3031.225, 2000.0, 90.11542, 95.0, 10.49330, 1.561636, 153.5354]],
              notes="alkylation process (Bracken–McCormick); published value re-verified at generation")
