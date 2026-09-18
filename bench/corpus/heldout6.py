"""Sixth held-out family set (final6), created after the tree was frozen for round 6 (``docs/23``
Phase E; wheel ``d57902fa79492531``: ``quadratic_bands='decaying'``, ``zero_step='decrease'``,
``saddle_step='linearized'`` on top of the round-5 candidate and Phases B and C) and before any
run of any benchmark solver on these problems.

Round 5's set separated no solver on five of its seven families, its one discriminating family
was a friction problem with the seed changed, two of its targets were published roundings and one
stored reference point was wrong by 4.6 tolerances (``bench/results/s6v5-final5/README.md``
section 8). The rules here answer that: every family is new in structure (the nearest relative in
the development corpus or the friction audit is named, with what differs), every family is one on
which the solvers are expected to differ (the reason is written below, before any run), every
reference is exact or computed at generation by an independent method and verified to a
stationarity of 1e-10 (relative to the gradient's own scale where the problem's numbers are
large), and ``verify_refs.py`` checks under the corpus model itself that the stored point is
feasible, stationary and reproduces the target to 1e-6. No target is a published rounding.

Families, the nearest relative, and why the solvers should differ:

  expspec    Non-negative exponential spectrum (a discrete Laplace inversion): chi-square fit
             0.5 |(A x - y) / sigma|^2, A_ij = exp(-s_j t_i), x >= 0. The Hessian A'A is dense,
             smooth, without decay, with a condition number beyond 1e16, and most bounds are
             active. Nearest: DECONV (a banded Gaussian Gram matrix in a box) and NNLS_SIMPLEX (a
             well-conditioned random design on the simplex). Expected to differ: interior-point
             methods stall on ill-conditioned bounded least squares (round 5, DECONV_200); the
             quadratic probe's finite-difference Hessian is numerically singular here, so its
             convexity test may decline; active-set SQP methods should identify the bounds.
  loggas     Stieltjes' electrostatic problem: -sum_{i<j} log(x_j - x_i) under sum x^2 <= n. Every
             pair is coupled, the row is nonlinear and active, and the model is undefined outside
             the ordered chamber, which no bound or row protects. Exact: the minimiser is the
             scaled zeros of the Hermite polynomial H_n, multiplier (n - 1) / 4. Nearest: DENSELAP
             (dense pairwise coupling on a hyperplane) and DOMAIN_LOG (a log guarded by bounds).
             Expected to differ: a line search that leaves the chamber sees NaN (NumPy) or a
             complex value (MATLAB); dense quasi-Newton at n = 120 against a first step that must
             stay ordered.
  stiefel    Orthogonality constraints written the way ``X'*X - eye(k)`` is flattened: all k^2
             equalities, k (k - 1) / 2 of them duplicates, so the Jacobian is rank-deficient at
             every point and at the solution. Orthogonal Procrustes (exact by the SVD) and a
             two-dimensional dominant eigenspace (exact by the eigenvalues; the minimisers form an
             orbit, so the reduced Hessian is singular too). Nearest: REDUNDANT_EQ (one plane three
             times, n = 3) and RANKLOSS_JAC. Expected to differ: SLSQP's least-squares subproblem
             and any KKT factorisation without inertia correction fail on dependent rows.
  cstr       A cascade of N stirred reactors, first-order Arrhenius kinetics, sized for 98 %
             conversion: minimise the total residence time. Nonlinear equalities with an
             exponential in the temperature, units of K, s and mol/L in one vector, the second
             instance started far from feasibility (residuals of 1e3). Exact: every reactor at the
             temperature cap and equal residence times (AM-GM). Nearest: infeasible_start_far (a
             linear equality) and HS112. Expected to differ: globalisation from a far-infeasible
             start on rows whose linearisation is poor; variable scales across 1e-2 to 4e2.
  propfair   Proportional-fair rates on a line network: -sum w_i log x_i under link capacities
             A x <= c, x >= 0. The objective is infinite on the bound the user wrote. Reference:
             barrier path, then Newton on the active links' KKT system. Nearest: MAXENT (entropy
             under linear equalities) and DOMAIN_LOG. Expected to differ: methods whose trial
             points land on a bound evaluate log(0); many active linear inequality rows at n = 60.
  tvdenoise  Smoothed total-variation denoising in a box, n = 500 and 1000: strictly convex, not
             quadratic, tridiagonal Hessian with curvature up to lambda / epsilon. Reference:
             projected Newton, then Newton on the final active set. Nearest: OBSTACLE (a
             tridiagonal bounded QP). Expected to differ: with finite differences a dense gradient
             costs n + 1 evaluations and a coloured one four; the clock decides who gets enough.
  gpdesign   Two geometric programs in raw SI units, as a designer writes them: the simple wing of
             Hoburg and Abbeel (nine variables from 4e-3 to 4e6, seven inequality rows) and an
             economic pipe diameter with the flow relations as four equalities (residuals from
             1e-2 to 1e5). Exact: a KKT point of a geometric program is its global optimum (the
             log transform is convex); the pipe has a closed form. Nearest: PRESSURE_VESSEL and
             bad_scaling. Expected to differ: row and variable scaling from a non-zero start.
  stepnoise  A smooth convex problem with a ball row plus piecewise-constant pseudo-random noise of
             amplitude 1e-9 and 1e-7 whose cells are narrower than a finite-difference step: what a
             simulator with a tolerance does. The noise has zero derivative almost everywhere, so
             the exact-derivative track sees the smooth gradient. Reference: the noiseless optimum
             (Newton on its KKT system). Nearest: NOISYQP (a smooth sine, three orders inside the
             tolerance) and noisy_simulator. Expected to differ: a forward difference of step
             1.5e-8 carries a gradient error near amplitude / step, 0.07 and 7; nobody is expected
             to attain the second one at defaults, and an honest exit is the point.
  phasesplit A binary mixture in n cells: double-well free energy per cell, a local field, total
             composition conserved. Every KKT point is enumerated at generation (branch patterns
             of the cubic, one root-find in the multiplier each); the start is a composition ramp
             whose own local minimum is not the global one (checked by a gradient flow). Nearest:
             NCBOXQP (an indefinite QP in a box). Expected to differ: only by the first step; a
             local solver that certifies the start's minimum is correct and not attained.
  winkler    Unilateral contact on a beam on an elastic foundation: 0.5 p'Gp - d'p, 0 <= p <= pmax,
             G the influence matrix exp(-t d)(cos t d + sin t d), t = 1 and t = 0.4 at n = 250. The
             Hessian decays away from the diagonal, the dense build does not fit the clock, and the
             band the probe needs fits its half of the clock at t = 1 and should not at t = 0.4
             (``docs/22`` 7.19). Measured before sealing, the model alone: 1.8 ms and 3.9 ms a call
             where DECONV_200 measures 4.1 ms. With entries falling as exp(-t d), a band of about 20
             (t = 1) or 45 (t = 0.4) should pass the fit test DECONV passed at 23: about 5 000
             evaluations and 9 s against a break-even of 6 ms a call at t = 1, about 11 000 and 44 s
             against a break-even of 2.7 ms at t = 0.4; the dense build (31 625) fits neither.
             Nearest: DECONV (a Gaussian Gram matrix). Expected to differ: this is the honest test
             of ``quadratic_bands='decaying'``, a default with one corpus record behind it.
  longmem    Isotonic generalised least squares under long-memory noise: 0.5 (x - y)'G(x - y),
             x_1 <= ... <= x_n, G_ij = (1 + |i - j|)^-s, s = 0.8 and s = 2, n = 200. Dense, never
             banded; at s = 2 the band-fit error keeps falling by more than a quarter per doubling,
             so the rule is expected to follow a decay that does not end and pay for it; at
             s = 0.8 it should stop at once. n - 1 sparse linear rows, most of them active.
             Reference: pooling and splitting on the block system, multipliers by cumulative sums.
             Nearest: COVQP and POLYQP (dense Hessians without order).
"""
from __future__ import annotations

import math

import numpy as np
import sympy as sp

from spec import INF, register, xs
from structured import st

exp, log, sqrt, sin, cosh = sp.exp, sp.log, sp.sqrt, sp.sin, sp.cosh


class sign(sp.Function):
    """sign(u) for the noise family, with derivative zero: the jumps of a piecewise-constant term
    carry no derivative almost everywhere, which is what the exact-derivative track should see of
    a quantised simulator. The class name is what both code printers emit (``sign``)."""

    nargs = 1
    is_real = True

    @classmethod
    def eval(cls, arg):
        return None

    def fdiff(self, argindex=1):
        return sp.S.Zero


# ---------------- shared checks ----------------
def _proj_stationarity(g, x, lo, hi, tol=1e-10):
    """Largest violation of the bound-constrained first-order conditions for a gradient `g` of the
    Lagrangian (rows already folded in)."""
    lo_a = x <= lo + tol
    up_a = x >= hi - tol
    free = ~(lo_a | up_a)
    worst = float(np.max(np.abs(g[free]))) if free.any() else 0.0
    if lo_a.any():
        worst = max(worst, float(np.max(np.maximum(0.0, -g[lo_a]))))
    if up_a.any():
        worst = max(worst, float(np.max(np.maximum(0.0, g[up_a]))))
    return worst


def _kkt_residual(p, x, act_tol=1e-9):
    """First-order residual of a corpus model `p` (a NumpyProblem) at `x` with least-squares
    multipliers of the right signs on the active set; returns (residual, gradient scale)."""
    from scipy.optimize import nnls
    x = np.asarray(x, float)
    g = p.grad(x)
    cols = []
    for j in range(p.n):
        if np.isfinite(p.xl[j]) and x[j] - p.xl[j] <= act_tol * max(1.0, abs(p.xl[j])):
            e = np.zeros(p.n); e[j] = -1.0; cols.append(e)
        if np.isfinite(p.xu[j]) and p.xu[j] - x[j] <= act_tol * max(1.0, abs(p.xu[j])):
            e = np.zeros(p.n); e[j] = 1.0; cols.append(e)
    if p.m:
        c, J = p.cons(x), p.jac(x)
        for i in range(p.m):
            if p.cl[i] == p.cu[i]:
                cols += [J[i], -J[i]]
            else:
                if np.isfinite(p.cu[i]) and p.cu[i] - c[i] <= act_tol * max(1.0, abs(p.cu[i])):
                    cols.append(J[i])
                if np.isfinite(p.cl[i]) and c[i] - p.cl[i] <= act_tol * max(1.0, abs(p.cl[i])):
                    cols.append(-J[i])
    scale = max(1.0, float(np.max(np.abs(g))))
    if not cols:
        return float(np.max(np.abs(g))), scale
    B = np.stack(cols, axis=1)
    y, _ = nnls(B, -g, maxiter=200 * B.shape[1])
    return float(np.max(np.abs(g + B @ y))), scale


def _newton_kkt(p, x, rows, fixed, iters=40):
    """Newton on the KKT system of `p` with the rows in `rows` held at their active side and the
    variables in `fixed` held at their value. `rows` is a list of (index, value). Returns x, lam."""
    x = np.asarray(x, float).copy()
    n = p.n
    free = np.array([j for j in range(n) if j not in set(fixed)], int)
    ridx = np.array([i for i, _ in rows], int)
    rval = np.array([v for _, v in rows], float)
    lam = np.zeros(p.m)
    if ridx.size:
        J = p.jac(x)[np.ix_(ridx, free)]
        lam[ridx] = np.linalg.lstsq(J.T, -p.grad(x)[free], rcond=None)[0]
    for _ in range(iters):
        g = p.grad(x)
        J = p.jac(x)[np.ix_(ridx, free)] if ridx.size else np.zeros((0, free.size))
        r1 = g[free] + (J.T @ lam[ridx] if ridx.size else 0.0)
        r2 = p.cons(x)[ridx] - rval if ridx.size else np.zeros(0)
        if max(np.max(np.abs(r1), initial=0.0), np.max(np.abs(r2), initial=0.0)) < 1e-14 * max(1.0, float(np.max(np.abs(g)))):
            break
        H = p.hess_lagrangian(x, lam)[np.ix_(free, free)]
        k = free.size
        K = np.block([[H, J.T], [J, np.zeros((ridx.size, ridx.size))]])
        step = np.linalg.lstsq(K, -np.concatenate([r1, r2]), rcond=None)[0]
        x[free] += step[:k]
        lam[ridx] += step[k:]
    return x, lam


def _quad(x, Q, cvec, const):
    """0.5 x'Qx + c'x + const as a SymPy sum over the stored upper triangle."""
    n = len(x)
    return sp.Add(*[float(0.5 * Q[i, i]) * x[i] ** 2 for i in range(n)]) \
        + sp.Add(*[float(Q[i, j]) * x[i] * x[j] for i in range(n) for j in range(i + 1, n) if Q[i, j] != 0.0]) \
        + sp.Add(*[float(cvec[i]) * x[i] for i in range(n)]) + float(const)


# ---------------- expspec: non-negative exponential spectrum ----------------
def _expspec(n, seed):
    rng = np.random.default_rng(seed)
    m = 2 * n
    s = np.logspace(-1.0, 2.0, n)                  # decay rates, 1/s
    t = np.logspace(-2.5, 1.3, m)                  # sampling times, s
    A = np.exp(-np.outer(t, s))
    ls = np.log10(s)
    x_true = 1.0 * np.exp(-0.5 * ((ls + 0.3) / 0.15) ** 2) + 0.6 * np.exp(-0.5 * ((ls - 1.0) / 0.10) ** 2)
    y = A @ x_true
    sigma = 0.01 * float(np.max(y))
    y = y + sigma * rng.standard_normal(m)
    As, ys = A / sigma, y / sigma
    Q = As.T @ As
    Q = 0.5 * (Q + Q.T)
    cvec = -As.T @ ys
    const = 0.5 * float(ys @ ys)
    from scipy.optimize import nnls
    x_ref, _ = nnls(As, ys, maxiter=200 * n)
    # polish on the free set: accurate least squares on the design, then refinement on the
    # corpus model's own normal equations (the two differ by the rounding of Q)
    for _ in range(6):
        free = x_ref > 0.0
        idx = np.flatnonzero(free)
        xf = np.linalg.lstsq(As[:, idx], ys, rcond=None)[0]
        for _r in range(3):
            gf = Q[np.ix_(idx, idx)] @ xf + cvec[idx]
            xf = xf - np.linalg.lstsq(Q[np.ix_(idx, idx)], gf, rcond=None)[0]
        x_ref = np.zeros(n)
        x_ref[idx] = np.maximum(xf, 0.0)
        g = Q @ x_ref + cvec
        release = (~free) & (g < -1e-10 * np.max(np.abs(g)))
        if not release.any() and np.all(xf > 0.0):
            break
        x_ref[release] = 1e-6
    g = Q @ x_ref + cvec
    stat = _proj_stationarity(g, x_ref, np.zeros(n), np.full(n, INF), tol=0.0)
    scale = max(1.0, float(np.max(np.abs(g))))
    assert stat < 1e-10 * scale, f"EXPSPEC_{n}: reference stationarity {stat:.1e} against gradient scale {scale:.1e}"
    assert np.sum(x_ref == 0.0) >= n // 2, f"EXPSPEC_{n}: most bounds should be active"
    f_star = 0.5 * float(x_ref @ Q @ x_ref) + float(cvec @ x_ref) + const
    return Q, cvec, const, x_ref, f_star


for _n, _seed in ((30, 601), (90, 602)):
    def _mk(n=_n, seed=_seed):
        Q, cvec, const, x_ref, f_star = _expspec(n, seed)
        x = xs(n)
        return st(f"EXPSPEC_{n}", "expspec", x, _quad(x, Q, cvec, const), xl=[0.0] * n, x0=[0.1] * n, ref_f=f_star,
                  ref_x=[list(map(float, x_ref))], tags=["convex", "ill-conditioned"],
                  notes="non-negative exponential spectrum (discrete Laplace inversion): chi-square fit of 2n samples of "
                        "sum_j x_j exp(-s_j t), rates 0.1..100 1/s, 1 % noise, x >= 0; the Hessian A'A / sigma^2 is dense, "
                        "smooth and numerically singular (condition number beyond 1e16), most bounds are active; f* is "
                        "unique though x* need not be; reference by Lawson-Hanson NNLS polished on the free set, projected "
                        "gradient < 1e-10 of the gradient's scale")
    _mk.__name__ = f"EXPSPEC_{_n}"
    register(_mk)


# ---------------- loggas: Stieltjes' electrostatic problem ----------------
def _hermite_zeros(n):
    z = np.sort(np.polynomial.hermite.hermgauss(n)[0])
    for _ in range(8):   # Newton on z_i - sum_{j != i} 1 / (z_i - z_j) = 0
        d = z[:, None] - z[None, :]
        np.fill_diagonal(d, 1.0)
        inv = 1.0 / d
        np.fill_diagonal(inv, 0.0)
        F = z - inv.sum(axis=1)
        Jm = inv ** 2
        Jd = 1.0 + Jm.sum(axis=1)
        Jm = -Jm
        np.fill_diagonal(Jm, Jd)
        z = z - np.linalg.solve(Jm, F)
    return z


def _loggas(n):
    z = _hermite_zeros(n)
    assert abs(float(z @ z) - 0.5 * n * (n - 1)) < 1e-9 * n * n, f"LOGGAS_{n}: sum of squared Hermite zeros"
    scale = math.sqrt(n / (0.5 * n * (n - 1)))          # sum x^2 = n
    x_ref = scale * z
    lam = 1.0 / (2.0 * scale ** 2)                        # multiplier of sum x^2 <= n: (n - 1) / 4
    d = x_ref[:, None] - x_ref[None, :]
    np.fill_diagonal(d, 1.0)
    inv = 1.0 / d
    np.fill_diagonal(inv, 0.0)
    gL = -inv.sum(axis=1) + 2.0 * lam * x_ref
    assert np.max(np.abs(gL)) < 1e-10, f"LOGGAS_{n}: stationarity {np.max(np.abs(gL)):.1e}"
    iu = np.triu_indices(n, 1)
    f_star = -float(np.sum(np.log(x_ref[iu[1]] - x_ref[iu[0]])))
    return x_ref, f_star, lam


for _n in (30, 120):
    def _mk(n=_n):
        x_ref, f_star, lam = _loggas(n)
        x = xs(n)
        f = -sp.Add(*[log(x[j] - x[i]) for i in range(n) for j in range(i + 1, n)])
        c = [sp.Add(*[x[i] ** 2 for i in range(n)])]
        return st(f"LOGGAS_{n}", "loggas", x, f, c, [-INF], [float(n)], x0=[float(v) for v in np.linspace(-1.0, 1.0, n)],
                  ref_f=f_star, ref_x=[list(map(float, x_ref))], tags=["convex", "domain", "dense"],
                  notes="Stieltjes' electrostatic problem: -sum_{i<j} log(x_j - x_i) s.t. sum x^2 <= n; the model is "
                        "undefined unless x is increasing, which no bound or row enforces; convex on that chamber with the "
                        f"row active, so the KKT point is the minimum: the zeros of the Hermite polynomial H_{n} scaled to "
                        "the sphere, multiplier (n - 1) / 4 (closed form; zeros by Golub-Welsch and Newton, stationarity "
                        "< 1e-10); start equally spaced in [-1, 1], strictly inside the ball")
    _mk.__name__ = f"LOGGAS_{_n}"
    register(_mk)


# ---------------- stiefel: orthogonality written as all k^2 equalities ----------------
def _rotation(k, rng, angle):
    S = rng.standard_normal((k, k))
    S = S - S.T
    S *= angle / np.linalg.norm(S, 2)
    w, V = np.linalg.eig(S)                          # exp of a skew matrix is a rotation
    R = (V * np.exp(w)) @ np.linalg.inv(V)
    R = np.real(R)
    U, _, Vt = np.linalg.svd(R)
    return U @ Vt


def _orth_rows(X, p_rows, k):
    rows, rhs = [], []
    for i in range(k):
        for j in range(k):
            rows.append(sp.Add(*[X[r][i] * X[r][j] for r in range(p_rows)]))
            rhs.append(1.0 if i == j else 0.0)
    return rows, rhs


def _stiefel_check(name, spec, x_ref, f_star):
    p = spec.numpy()
    assert p.violation(x_ref) < 1e-12, f"{name}: reference feasibility {p.violation(x_ref):.1e}"
    res, scale = _kkt_residual(p, x_ref)
    assert res < 1e-10 * scale, f"{name}: reference stationarity {res:.1e}"
    assert abs(p.f(x_ref) - f_star) < 1e-9 * max(1.0, abs(f_star)), f"{name}: reference value"


def _procrustes(k, seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((2 * k, k))
    R_true = _rotation(k, rng, 1.0)
    B = A @ R_true + 0.3 * rng.standard_normal((2 * k, k))
    M = A.T @ B
    U, S, Vt = np.linalg.svd(M)
    X = U @ Vt
    assert np.linalg.det(X) > 0.5 and S[-1] > 0.1, f"PROCRUSTES_{k}: the global minimiser must be a rotation, well separated"
    f_star = float(np.sum(A * A) + np.sum(B * B) - 2.0 * np.sum(S))
    return A, B, X, f_star, float(4.0 * S[-1])


for _k, _seed in ((3, 611), (5, 612)):
    def _mk(k=_k, seed=_seed):
        A, B, X_ref, f_star, gap = _procrustes(k, seed)
        x = xs(k * k)
        X = [[x[r * k + c] for c in range(k)] for r in range(k)]
        G, M = A.T @ A, A.T @ B
        f = sp.Add(*[float(G[a, b]) * X[a][c] * X[b][c] for c in range(k) for a in range(k) for b in range(k)]) \
            - 2 * sp.Add(*[float(M[a, c]) * X[a][c] for a in range(k) for c in range(k)]) + float(np.sum(B * B))
        rows, rhs = _orth_rows(X, k, k)
        spec = st(f"PROCRUSTES_{k}", "stiefel", x, f, rows, rhs, rhs, x0=[float(v) for v in np.eye(k).ravel()],
                  ref_f=f_star, ref_x=[list(map(float, X_ref.ravel()))],
                  tags=["nonconvex", "rank-deficient", "multiple-local-minima"],
                  notes=f"orthogonal Procrustes |A X - B|_F^2 over {k} x {k} matrices with X'X = I written as all {k * k} "
                        f"entries, so {k * (k - 1) // 2} equalities are duplicates and the Jacobian is rank-deficient "
                        "everywhere; exact: X = U V' from the SVD of A'B, a rotation like the identity start; the other "
                        f"component of the orthogonal group holds one more local minimum, higher by {gap:.3f}")
        _stiefel_check(f"PROCRUSTES_{k}", spec, X_ref.ravel(), f_star)
        return spec
    _mk.__name__ = f"PROCRUSTES_{_k}"
    register(_mk)


@register
def EIGSUB_6X2():
    p_rows, k = 6, 2
    rng = np.random.default_rng(613)
    V, _ = np.linalg.qr(rng.standard_normal((p_rows, p_rows)))
    ev = np.array([5.0, 3.5, 1.2, 0.8, 0.5, 0.2])
    S = (V * ev) @ V.T
    S = 0.5 * (S + S.T)
    w, W = np.linalg.eigh(S)
    X_ref = W[:, ::-1][:, :k]
    f_star = -float(w[-1] + w[-2])
    x = xs(p_rows * k)
    X = [[x[r * k + c] for c in range(k)] for r in range(p_rows)]
    f = -sp.Add(*[float(S[a, b]) * X[a][c] * X[b][c] for c in range(k) for a in range(p_rows) for b in range(p_rows)])
    rows, rhs = _orth_rows(X, p_rows, k)
    spec = st("EIGSUB_6X2", "stiefel", x, f, rows, rhs, rhs, x0=[float(v) for v in np.eye(p_rows)[:, :k].ravel()],
              ref_f=f_star, ref_x=[list(map(float, X_ref.ravel()))], tags=["nonconvex", "rank-deficient", "degenerate"],
              notes="dominant two-dimensional eigenspace of a 6 x 6 symmetric matrix as -trace(X'SX) s.t. X'X = I written "
                    "as all four entries (one duplicate); exact: minus the two largest eigenvalues (5 and 3.5, gap 2.3 to "
                    "the third); the minimisers are an orbit of the orthogonal group, so the reduced Hessian is singular "
                    "along it as well as the Jacobian being rank-deficient")
    _stiefel_check("EIGSUB_6X2", spec, X_ref.ravel(), f_star)
    return spec


# ---------------- cstr: a reactor cascade sized for a conversion ----------------
_K0, _EA_R, _T_MAX, _C_OUT = 1.2e9, 8750.0, 400.0, 0.02      # 1/s, K, K, mol/L (feed 1 mol/L)


def _cstr(N, far):
    kmax = _K0 * math.exp(-_EA_R / _T_MAX)
    ratio = (1.0 / _C_OUT) ** (1.0 / N)
    tau = (ratio - 1.0) / kmax
    x_ref = np.concatenate([ratio ** -np.arange(1, N + 1), np.full(N, tau), np.full(N, _T_MAX)])
    x_ref[N - 1] = _C_OUT
    f_star = N * tau
    x = xs(3 * N)
    C, TAU, T = x[:N], x[N:2 * N], x[2 * N:]
    rows = [C[k] * (1 + _K0 * exp(-_EA_R / T[k]) * TAU[k]) - (C[k - 1] if k else 1.0) for k in range(N)]
    xl = [0.0] * N + [0.1] * N + [300.0] * N
    xu = [1.0] * (N - 1) + [_C_OUT] + [3600.0] * N + [_T_MAX] * N
    if far:
        x0 = [1.0] * (N - 1) + [_C_OUT] + [3600.0] * N + [_T_MAX] * N
    else:
        x0 = [float(v) for v in np.linspace(1.0, _C_OUT, N + 1)[1:]] + [60.0] * N + [350.0] * N
    name = f"CSTR_{N}"
    spec = st(name, "cstr", x, sp.Add(*TAU), rows, [0.0] * N, [0.0] * N, xl=xl, xu=xu, x0=x0, ref_f=f_star,
              ref_x=[list(map(float, x_ref))], tags=["nonconvex", "units"],
              notes=f"{N} stirred reactors in series, first-order kinetics k = 1.2e9 exp(-8750 / T) 1/s, feed 1 mol/L, outlet "
                    "at most 0.02 mol/L (a bound on the last concentration), T in [300, 400] K, residence times in "
                    "[0.1, 3600] s; minimise the total residence time; the balances c_k (1 + k(T_k) tau_k) = c_{k-1} are "
                    "nonlinear equalities; exact: every reactor at 400 K with equal residence times (AM-GM on the "
                    "conversion product), the only KKT point; "
                    + ("start far from feasibility: every reactor at its largest and hottest, concentrations at the feed, "
                       "balance residuals of 1.4e3" if far else
                       "start: a linear concentration profile, 60 s and 350 K per reactor, balance residuals of order 1"))
    p = spec.numpy()
    assert p.violation(x_ref) < 1e-12, f"{name}: reference feasibility {p.violation(x_ref):.1e}"
    res, scale = _kkt_residual(p, x_ref)
    assert res < 1e-10 * scale, f"{name}: reference stationarity {res:.1e}"
    return spec


@register
def CSTR_3():
    return _cstr(3, far=False)


@register
def CSTR_8():
    return _cstr(8, far=True)


# ---------------- propfair: proportional-fair rates on a line network ----------------
def _propfair(n, links, seed):
    rng = np.random.default_rng(seed)
    A = np.zeros((links, n))
    for i in range(n):
        a = int(rng.integers(0, links))
        b = min(links, a + int(rng.integers(1, 5)))
        A[a:b, i] = 1.0
    keep = A.sum(axis=1) > 0
    A = A[keep]
    L = A.shape[0]
    cap = rng.uniform(1.0, 3.0, L)
    w = rng.uniform(0.5, 2.0, n)
    x0 = np.array([0.5 * np.min((cap / A.sum(axis=1))[A[:, i] > 0]) for i in range(n)])
    assert np.all(A @ x0 < cap), "propfair: the start must be strictly feasible"
    # barrier path from the start
    x = x0.copy()
    mu = 1.0
    while mu > 1e-11:
        for _ in range(60):
            s = cap - A @ x
            g = -w / x + mu * (A.T @ (1.0 / s))
            H = np.diag(w / x ** 2) + mu * (A.T * (1.0 / s ** 2)) @ A
            dx = np.linalg.solve(H, -g)
            t = 1.0
            while np.any(x + t * dx <= 0) or np.any(A @ (x + t * dx) >= cap):
                t *= 0.5
            x = x + 0.99 * t * dx if t < 1.0 else x + dx
            if np.max(np.abs(dx)) < 1e-13:
                break
        mu *= 0.2
    lam = mu / 0.2 / (cap - A @ x)
    act = np.flatnonzero(lam > 1e-6)
    lam_a = lam[act]
    for _ in range(30):   # Newton on the active links' KKT system
        Aa = A[act]
        r1 = -w / x + Aa.T @ lam_a
        r2 = Aa @ x - cap[act]
        if max(np.max(np.abs(r1)), np.max(np.abs(r2))) < 1e-14:
            break
        K = np.block([[np.diag(w / x ** 2), Aa.T], [Aa, np.zeros((act.size, act.size))]])
        step = np.linalg.solve(K, -np.concatenate([r1, r2]))
        x = x + step[:n]
        lam_a = lam_a + step[n:]
    inactive = np.setdiff1d(np.arange(L), act)
    assert np.all(lam_a > 1e-4), f"PROPFAIR_{n}: strict complementarity (smallest multiplier {lam_a.min():.1e})"
    assert np.all(x > 1e-3) and (inactive.size == 0 or np.all((cap - A @ x)[inactive] > 1e-4)), f"PROPFAIR_{n}: slack"
    stat = float(np.max(np.abs(-w / x + A[act].T @ lam_a)))
    assert stat < 1e-10 and np.max(np.abs(A[act] @ x - cap[act])) < 1e-13, f"PROPFAIR_{n}: stationarity {stat:.1e}"
    return A, cap, w, x0, x, -float(w @ np.log(x)), act.size


for _n, _links, _seed in ((12, 6, 621), (60, 25, 622)):
    def _mk(n=_n, links=_links, seed=_seed):
        A, cap, w, x0, x_ref, f_star, n_act = _propfair(n, links, seed)
        x = xs(n)
        f = -sp.Add(*[float(w[i]) * log(x[i]) for i in range(n)])
        c = [sp.Add(*[x[i] for i in range(n) if A[l, i] > 0]) for l in range(A.shape[0])]
        return st(f"PROPFAIR_{n}", "propfair", x, f, c, [-INF] * A.shape[0], [float(v) for v in cap], xl=[0.0] * n,
                  x0=[float(v) for v in x0], ref_f=f_star, ref_x=[list(map(float, x_ref))], tags=["convex", "domain"],
                  notes=f"proportional-fair rates: -sum w_i log x_i for {n} flows over contiguous stretches of a line of "
                        f"{A.shape[0]} links, capacities in [1, 3], x >= 0 as the user writes it, so the objective is "
                        f"infinite on the bound; strictly convex; {n_act} links saturated at the optimum with multipliers "
                        "above 1e-4; reference by a barrier path and Newton on the active links' KKT system, stationarity "
                        "< 1e-10; start strictly feasible at half of each flow's equal share")
    _mk.__name__ = f"PROPFAIR_{_n}"
    register(_mk)


# ---------------- tvdenoise: smoothed total variation in a box ----------------
_TV_LAMBDA, _TV_EPS = 0.3, 1e-3


def _tvdenoise(n, seed):
    rng = np.random.default_rng(seed)
    levels = np.array([0.0, 0.7, 0.3, 1.0, 0.0, 0.55, 1.0, 0.2])
    edges = np.linspace(0, n, levels.size + 1).astype(int)
    truth = np.zeros(n)
    for a, b, v in zip(edges[:-1], edges[1:], levels):
        truth[a:b] = v
    d = truth + 0.08 * rng.standard_normal(n)
    lam, eps2 = _TV_LAMBDA, _TV_EPS ** 2

    def fgh(u):
        du = np.diff(u)
        r = np.sqrt(du ** 2 + eps2)
        f = 0.5 * float(np.sum((u - d) ** 2)) + lam * float(np.sum(r))
        t = lam * du / r
        g = u - d
        g[:-1] -= t
        g[1:] += t
        k = lam * eps2 / r ** 3
        H = np.eye(n)
        i = np.arange(n - 1)
        H[i, i] += k
        H[i + 1, i + 1] += k
        H[i, i + 1] -= k
        H[i + 1, i] -= k
        return f, g, H

    u = np.clip(d, 0.0, 1.0)
    for _ in range(400):   # projected Newton (Bertsekas)
        f, g, H = fgh(u)
        bound = ((u <= 0.0) & (g > 0.0)) | ((u >= 1.0) & (g < 0.0))
        free = ~bound
        if _proj_stationarity(g, u, np.zeros(n), np.ones(n), tol=0.0) < 1e-13:
            break
        idx = np.flatnonzero(free)
        step = np.zeros(n)
        step[idx] = np.linalg.solve(H[np.ix_(idx, idx)], -g[idx])
        t = 1.0
        while t > 1e-12:
            trial = np.clip(u + t * step, 0.0, 1.0)
            if fgh(trial)[0] <= f + 1e-4 * float(g @ (trial - u)):
                break
            t *= 0.5
        u = trial
    f, g, _ = fgh(u)
    stat = _proj_stationarity(g, u, np.zeros(n), np.ones(n), tol=0.0)
    assert stat < 1e-10, f"TVDENOISE_{n}: reference stationarity {stat:.1e}"
    n_act = int(np.sum((u <= 0.0) | (u >= 1.0)))
    assert n_act >= 20, f"TVDENOISE_{n}: some bounds must be active ({n_act})"
    return d, u, f, n_act


for _n, _seed in ((500, 631), (1000, 632)):
    def _mk(n=_n, seed=_seed):
        d, u_ref, f_star, n_act = _tvdenoise(n, seed)
        x = xs(n)
        f = sp.Add(*[0.5 * (x[i] - float(d[i])) ** 2 for i in range(n)]) \
            + _TV_LAMBDA * sp.Add(*[sqrt((x[i + 1] - x[i]) ** 2 + _TV_EPS ** 2) for i in range(n - 1)])
        return st(f"TVDENOISE_{n}", "tvdenoise", x, f, xl=[0.0] * n, xu=[1.0] * n, x0=[0.5] * n, ref_f=f_star,
                  ref_x=[list(map(float, u_ref))], tags=["convex", "sparse"],
                  notes="smoothed total-variation denoising 0.5 |u - d|^2 + 0.3 sum sqrt((u_{i+1} - u_i)^2 + 1e-6) of a "
                        "piecewise-constant signal with 8 % noise, 0 <= u <= 1; strictly convex, not quadratic, tridiagonal "
                        f"Hessian with curvature up to 300; {n_act} bounds active; reference by projected Newton and Newton "
                        "on the final active set, projected gradient < 1e-10")
    _mk.__name__ = f"TVDENOISE_{_n}"
    register(_mk)


# ---------------- gpdesign: geometric programs in raw SI units ----------------
@register
def WING_SIMPLE():
    # Hoburg and Abbeel (2014), the simple wing: minimise drag in steady level flight.
    k_form, e_osw, mu_air, rho, tau_w, n_ult = 1.2, 0.95, 1.78e-5, 1.23, 0.12, 3.8
    v_min, cl_max, s_wet, cda0, w0 = 22.0, 1.5, 2.05, 0.031, 4940.0
    x = xs(9)
    A, S, V, W, Re, CD, CL, Cf, Ww = x
    f = 0.5 * rho * S * CD * V ** 2
    c = [CD - (cda0 / S + k_form * Cf * s_wet + CL ** 2 / (sp.pi * A * e_osw)),
         Cf - 0.074 / Re ** 0.2,
         rho * V * sqrt(S / A) / mu_air - Re,
         0.5 * rho * S * CL * V ** 2 - W,
         0.5 * rho * S * cl_max * v_min ** 2 - W,
         W - w0 - Ww,
         Ww - (45.24 * S + 8.71e-5 * n_ult * A ** 1.5 * sqrt(w0 * W * S) / tau_w)]
    mk = lambda ref_f, ref_x: st(  # noqa: E731
        "WING_SIMPLE", "gpdesign", x, f, c, [0.0] * 7, [INF] * 7, xl=[1, 1, 10, 1000, 1e5, 1e-3, 0.01, 1e-4, 100],
        xu=[40, 100, 100, 50000, 1e8, 1, 2, 0.1, 20000], x0=[10, 20, 40, 8000, 3e6, 0.03, 0.5, 0.004, 2500],
        ref_f=ref_f, ref_x=ref_x, tags=["units", "nonconvex"],
        source="the simple wing of Hoburg and Abbeel, Geometric programming for aircraft design optimization (2014)",
        notes="aspect ratio, wing area (m^2), speed (m/s), weight (N), Reynolds number, drag, lift and skin-friction "
              "coefficients, wing weight (N): nine variables from 4e-3 to 4e6 and seven inequality rows in raw SI units "
              "(residuals from 1e-3 to 1e6); a geometric program, so a KKT point is the global optimum (the log "
              "transform is convex); reference by Newton on the KKT system of the active rows, stationarity < 1e-10 of "
              "the gradient's scale; the published optimum is a drag of 303.1 N")
    p = mk(None, None).numpy()
    # the published design, then Newton on the KKT system with every row but the stall row active
    x_pub = np.array([8.46, 16.4, 38.2, 7341.0, 3.68e6, 0.0206, 0.499, 0.0036, 2401.0])
    x_ref = x_pub
    active = [0, 1, 2, 3, 4, 5, 6]
    for _ in range(3):
        x_ref, lam = _newton_kkt(p, x_pub if _ == 0 else x_ref, [(i, 0.0) for i in active], [])
        drop = [i for i in active if lam[i] > 1e-12]          # a lower-side row needs lam <= 0
        slack = p.cons(x_ref)
        if not drop and np.all(slack > -1e-9):
            break
        active = [i for i in active if i not in drop]
    assert p.violation(x_ref) < 1e-8, f"WING_SIMPLE: reference feasibility {p.violation(x_ref):.1e}"
    res, scale = _kkt_residual(p, x_ref)
    assert res < 1e-10 * scale, f"WING_SIMPLE: reference stationarity {res:.1e} against {scale:.1e}"
    f_star = p.f(x_ref)
    assert abs(f_star - 303.1) < 0.5, f"WING_SIMPLE: {f_star} against the published 303.1 N"
    return mk(f_star, [list(map(float, x_ref))])


@register
def PIPE_SIZING():
    # economic diameter of a water main: annualised capital against pumping energy
    q, length, rho, mu_w, eta = 0.05, 1000.0, 998.0, 1.0e-3, 0.7        # m^3/s, m, kg/m^3, Pa s, -
    cap_cost, energy = 900.0, 0.12 * 8000.0 / 1000.0                    # $/(m^2.5 yr) per metre, $/(W yr)
    a = cap_cost * length
    kdp = 0.316 * (math.pi * mu_w / (4 * rho * q)) ** 0.25 * length * rho / 2 * 16 * q ** 2 / math.pi ** 2
    b = energy * q * kdp / eta
    d_star = (4.75 * b / (1.5 * a)) ** (1.0 / 6.25)
    v_star = 4 * q / (math.pi * d_star ** 2)
    re_star = rho * v_star * d_star / mu_w
    fr_star = 0.316 / re_star ** 0.25
    dp_star = fr_star * length / d_star * rho * v_star ** 2 / 2
    x_ref = np.array([d_star, v_star, re_star, fr_star, dp_star])
    f_star = a * d_star ** 1.5 + energy * q * dp_star / eta
    x = xs(5)
    D, V, Re, Fr, dP = x
    f = a * D ** 1.5 + (energy * q / eta) * dP
    c = [V * sp.pi * D ** 2 / 4 - q, Re - rho * V * D / mu_w, Fr - 0.316 / Re ** 0.25, dP - Fr * (length / D) * rho * V ** 2 / 2]
    spec = st("PIPE_SIZING", "gpdesign", x, f, c, [0.0] * 4, [0.0] * 4, xl=[0.05, 0.1, 1e3, 1e-3, 1e2], xu=[1, 10, 1e7, 0.1, 1e7],
              x0=[0.3, 1.0, 1e5, 0.02, 1e4], ref_f=f_star, ref_x=[list(map(float, x_ref))], tags=["units", "nonconvex"],
              notes="economic diameter of a 1 km water main carrying 0.05 m^3/s: annualised capital 900 D^1.5 $/m against "
                    "pumping energy at 0.12 $/kWh, 8000 h/yr, 70 % efficiency; diameter (m), velocity (m/s), Reynolds "
                    "number, Blasius friction factor and pressure drop (Pa) tied by four monomial equalities whose "
                    "residuals at the start run from 2e-3 to 2e5; a geometric program with a closed form: eliminating "
                    "gives a D^1.5 + b D^-4.75 and D* = (4.75 b / 1.5 a)^(1/6.25)")
    p = spec.numpy()
    assert p.violation(x_ref) < 1e-9 * dp_star, f"PIPE_SIZING: reference feasibility {p.violation(x_ref):.1e}"
    assert np.all(x_ref > np.asarray(spec.xl) * 1.01) and np.all(x_ref < np.asarray(spec.xu) * 0.99), "PIPE_SIZING: bound active"
    res, scale = _kkt_residual(p, x_ref)
    assert res < 1e-10 * scale, f"PIPE_SIZING: reference stationarity {res:.1e} against {scale:.1e}"
    return spec


# ---------------- stepnoise: piecewise-constant noise finer than a difference step ----------------
def _stepnoise(n, amp, seed):
    rng = np.random.default_rng(seed)
    w = rng.uniform(1.0, 4.0, n)
    tgt = rng.uniform(-1.5, 1.5, n)
    a = rng.uniform(-1.0, 1.0, n)
    a /= np.linalg.norm(a)
    radius2 = 0.6 * float(tgt @ tgt)                 # the unconstrained minimiser (near tgt) lies outside the ball
    x = xs(n)
    smooth = sp.Add(*[float(w[i]) * cosh(x[i] - float(tgt[i])) for i in range(n)]) \
        + 2.0 * sp.Add(*[float(a[i]) * x[i] for i in range(n)]) ** 2
    dirs = rng.uniform(-1.0, 1.0, (3, n))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    freq = (1.0e9, 1.6180339887e9, 2.4142135624e9)
    noise = sp.Add(*[sign(sin(freq[k] * sp.Add(*[float(dirs[k, i]) * x[i] for i in range(n)]) + float(k + 1))) for k in range(3)])
    row = [sp.Add(*[x[i] ** 2 for i in range(n)])]
    name = f"STEPNOISE_{n}"

    def mk(f, ref_f=None, ref_x=None):
        return st(name, "stepnoise", x, f, row, [-INF], [radius2], xl=[-2.0] * n, xu=[2.0] * n, x0=[0.0] * n, ref_f=ref_f,
                  ref_x=ref_x, tags=["noisy", "convex"],
                  notes=f"sum w_i cosh(x_i - t_i) + 2 (a'x)^2 inside the ball |x|^2 <= {radius2:.4f} (active) and a box, "
                        f"plus {amp:g} / 3 times three square waves sign(sin(1e9 d_k'x)): piecewise-constant noise whose "
                        "cells (3e-9) are narrower than a forward-difference step, with zero derivative almost everywhere, "
                        f"so a difference of step 1.5e-8 carries a gradient error near {amp / 1.5e-8:.2g} while the "
                        "exact-derivative track sees the smooth gradient; reference: the noiseless optimum by Newton on its "
                        f"KKT system, stationarity < 1e-10, which the noise moves by at most {amp:g} in f")

    p = mk(smooth).numpy()                           # the noiseless twin
    from scipy.optimize import minimize as _sp_min
    r = _sp_min(p.f, np.zeros(n), jac=p.grad, method="SLSQP", bounds=[(-2.0, 2.0)] * n,
                constraints=[{"type": "ineq", "fun": lambda z: radius2 - p.cons(z)[0], "jac": lambda z: -p.jac(z)[0]}],
                options={"maxiter": 500, "ftol": 1e-14})
    fixed = [j for j in range(n) if abs(abs(r.x[j]) - 2.0) < 1e-7]
    x_ref, lam = _newton_kkt(p, r.x, [(0, radius2)], fixed)
    assert lam[0] > 1e-3, f"{name}: the ball row must be active (multiplier {lam[0]:.1e})"
    assert p.violation(x_ref) < 1e-12, f"{name}: reference feasibility {p.violation(x_ref):.1e}"
    res, scale = _kkt_residual(p, x_ref)
    assert res < 1e-10 * scale, f"{name}: reference stationarity {res:.1e}"
    return mk(smooth + (amp / 3.0) * noise, p.f(x_ref), [list(map(float, x_ref))])


@register
def STEPNOISE_6():
    return _stepnoise(6, 1e-9, 641)


@register
def STEPNOISE_12():
    return _stepnoise(12, 1e-7, 642)


# ---------------- phasesplit: a conserved double-well mixture, every KKT point enumerated ----------------
_G1_MAX = 8.0 / (3.0 * math.sqrt(3.0))               # largest value of g'(c) = 4 c^3 - 4 c on the middle branch


def _cubic_branches(rhs):
    """Roots of 4 c^3 - 4 c = rhs as (left, middle, right); None where a branch does not exist."""
    roots = np.roots([4.0, 0.0, -4.0, -rhs])
    real = np.sort(roots[np.abs(roots.imag) < 1e-9].real)
    s = 1.0 / math.sqrt(3.0)
    left = [c for c in real if c <= -s + 1e-12]
    right = [c for c in real if c >= s - 1e-12]
    mid = [c for c in real if -s < c < s]
    return (left[0] if left else None, mid[0] if mid else None, right[-1] if right else None)


def _phasesplit(n, seed):
    from itertools import product
    from scipy.optimize import brentq
    rng = np.random.default_rng(seed)
    v = rng.uniform(0.5, 1.5, n)
    h = np.linspace(0.45, -0.45, n) + 0.05 * rng.standard_normal(n)     # the field favours poor cells on the left
    total = 0.1 * float(v.sum())

    def fval(c):
        return float(np.sum(v * ((c ** 2 - 1.0) ** 2 + h * c)))

    def point(lmb, pattern):
        c = np.empty(n)
        for i in range(n):
            b = _cubic_branches(lmb - h[i])[pattern[i]]
            if b is None:
                return None
            c[i] = b
        return c

    grid = np.linspace(-3.0, 3.0, 2401)
    table = np.full((n, 3, grid.size), np.nan)      # v_i c_i on each branch over the multiplier grid
    for i in range(n):
        for q, lmb in enumerate(grid):
            for b_, c_ in enumerate(_cubic_branches(lmb - h[i])):
                if c_ is not None:
                    table[i, b_, q] = v[i] * c_
    found = []
    patterns = [pt for pt in product((0, 2), repeat=n)]
    patterns += [pt[:i] + (1,) + pt[i:] for i in range(n) for pt in product((0, 2), repeat=n - 1)]
    for pt in patterns:
        vals = table[np.arange(n), list(pt), :].sum(axis=0) - total
        for k in np.flatnonzero(np.isfinite(vals[:-1]) & np.isfinite(vals[1:]) & (vals[:-1] * vals[1:] <= 0.0)):
            lmb = brentq(lambda t: float(v @ point(t, pt)) - total, grid[k], grid[k + 1], xtol=1e-15, rtol=1e-15)
            c = point(lmb, pt)
            # second order on the tangent space of v'c = total
            Hd = v * (12.0 * c ** 2 - 4.0)
            Z = np.linalg.svd(v[None, :])[2][1:].T
            if np.min(np.linalg.eigvalsh(Z.T @ (Hd[:, None] * Z))) > 1e-8:
                found.append((fval(c), c, lmb))
    assert len(found) >= 3, f"PHASESPLIT_{n}: expected several local minima, found {len(found)}"
    found.sort(key=lambda t: t[0])
    f_star, c_star, lmb = found[0]
    for _ in range(20):   # Newton on the (n + 1) KKT system
        r1 = v * (4.0 * c_star ** 3 - 4.0 * c_star + h) - lmb * v
        r2 = float(v @ c_star) - total
        K = np.block([[np.diag(v * (12.0 * c_star ** 2 - 4.0)), -v[:, None]], [v[None, :], np.zeros((1, 1))]])
        step = np.linalg.solve(K, -np.concatenate([r1, [r2]]))
        c_star = c_star + step[:n]
        lmb += step[n]
    stat = float(np.max(np.abs(v * (4.0 * c_star ** 3 - 4.0 * c_star + h) - lmb * v)))
    assert stat < 1e-10 and abs(float(v @ c_star) - total) < 1e-13, f"PHASESPLIT_{n}: stationarity {stat:.1e}"
    assert np.max(np.abs(c_star)) < 1.9, f"PHASESPLIT_{n}: the bounds must be inactive"
    # the start: a ramp against the field, moved onto the conservation plane; its own minimum by a gradient flow
    c0 = np.linspace(0.85, -0.85, n)
    c0 = c0 + (total - float(v @ c0)) / float(v @ v) * v
    c = c0.copy()
    P = np.eye(n) - np.outer(v, v) / float(v @ v)
    for _ in range(200000):
        g = P @ (v * (4.0 * c ** 3 - 4.0 * c + h))
        if np.max(np.abs(g)) < 1e-9:
            break
        c = c - 0.02 * g
    f_flow = fval(c)
    match = min(found, key=lambda t: float(np.max(np.abs(t[1] - c))))
    assert np.max(np.abs(match[1] - c)) < 1e-6, f"PHASESPLIT_{n}: the gradient flow must end at an enumerated minimum"
    assert f_flow > f_star + 1e-2 * max(1.0, abs(f_star)), f"PHASESPLIT_{n}: the start's minimum must not be the global one"
    return v, h, total, c0, c_star, fval(c_star), len(found), f_flow


for _n, _seed in ((6, 651), (10, 652)):
    def _mk(n=_n, seed=_seed):
        v, h, total, c0, c_star, f_star, count, f_flow = _phasesplit(n, seed)
        x = xs(n)
        f = sp.Add(*[float(v[i]) * ((x[i] ** 2 - 1) ** 2 + float(h[i]) * x[i]) for i in range(n)])
        c = [sp.Add(*[float(v[i]) * x[i] for i in range(n)])]
        return st(f"PHASESPLIT_{n}", "phasesplit", x, f, c, [total], [total], xl=[-2.0] * n, xu=[2.0] * n,
                  x0=[float(t) for t in c0], ref_f=f_star, ref_x=[list(map(float, c_star))],
                  tags=["multiple-local-minima", "nonconvex"],
                  notes=f"a binary mixture in {n} cells: sum v_i ((c_i^2 - 1)^2 + h_i c_i) with the total composition "
                        f"conserved (one linear equality) and -2 <= c <= 2 inactive; {count} local minima enumerated at "
                        "generation over the branch patterns of the cubic (at most one cell on the middle branch), "
                        f"reference: the global one; the start is a composition ramp against the field whose own minimum "
                        f"(by a gradient flow on the conservation plane) has f = {f_flow:.6f}; a solver may certify that one")
    _mk.__name__ = f"PHASESPLIT_{_n}"
    register(_mk)


# ---------------- box and row QPs: active-set polish ----------------
def _box_qp(Q, cvec, lo, hi, name):
    """min 0.5 x'Qx + c'x in a box, Q positive definite: projected Newton, then the exact solve on
    the final active set; asserts a projected gradient below 1e-10."""
    n = len(cvec)
    x = np.clip(np.zeros(n), lo, hi)
    for _ in range(200):
        g = Q @ x + cvec
        bound = ((x <= lo) & (g > 0.0)) | ((x >= hi) & (g < 0.0))
        idx = np.flatnonzero(~bound)
        if _proj_stationarity(g, x, lo, hi, tol=0.0) < 1e-13:
            break
        step = np.zeros(n)
        step[idx] = np.linalg.solve(Q[np.ix_(idx, idx)], -g[idx])
        f0 = 0.5 * x @ Q @ x + cvec @ x
        t = 1.0
        while t > 1e-14:
            trial = np.clip(x + t * step, lo, hi)
            if 0.5 * trial @ Q @ trial + cvec @ trial <= f0 + 1e-4 * float(g @ (trial - x)):
                break
            t *= 0.5
        x = trial
    g = Q @ x + cvec
    stat = _proj_stationarity(g, x, lo, hi, tol=0.0)
    assert stat < 1e-10, f"{name}: reference stationarity {stat:.1e}"
    return x, 0.5 * float(x @ Q @ x) + float(cvec @ x)


# ---------------- winkler: contact on a beam on an elastic foundation ----------------
def _winkler(n, theta, seed):
    rng = np.random.default_rng(seed)
    dist = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]).astype(float)
    G = np.exp(-theta * dist) * (np.cos(theta * dist) + np.sin(theta * dist))
    G[np.abs(G) < 1e-17] = 0.0                       # below one unit of rounding of the diagonal
    assert np.min(np.linalg.eigvalsh(G)) > 1e-4, f"WINKLER: influence matrix must be positive definite"
    s = np.linspace(0.0, 1.0, n)
    # two rigid punches pressed into the beam, a rough surface between them
    gap = 3.0 - 60.0 * (s - 0.3) ** 2
    gap = np.maximum(gap, 2.2 - 200.0 * (s - 0.75) ** 2)
    gap = gap + 0.15 * rng.standard_normal(n)
    pmax = 1.0
    p_ref, f_q = _box_qp(G, -gap, np.zeros(n), np.full(n, pmax), f"WINKLER_{n}")
    n_lo, n_up = int(np.sum(p_ref <= 0.0)), int(np.sum(p_ref >= pmax))
    assert n_lo >= n // 10 and n_up >= n // 20, f"WINKLER: both bounds must be active ({n_lo}, {n_up})"
    return G, gap, pmax, p_ref, f_q, n_lo, n_up


for _tag, _theta, _seed in (("STIFF", 1.0, 661), ("SOFT", 0.4, 662)):
    def _mk(tag=_tag, theta=_theta, seed=_seed):
        n = 250
        G, gap, pmax, p_ref, f_star, n_lo, n_up = _winkler(n, theta, seed)
        x = xs(n)
        nnz = int(np.count_nonzero(np.triu(G, 1)))
        return st(f"WINKLER_{tag}_{n}", "winkler", x, _quad(x, G, -gap, 0.0), xl=[0.0] * n, xu=[pmax] * n, x0=[0.5 * pmax] * n,
                  ref_f=f_star, ref_x=[list(map(float, p_ref))], tags=["convex"],
                  notes=f"unilateral contact pressures on a beam on an elastic foundation at {n} nodes: 0.5 p'Gp - d'p, "
                        f"0 <= p <= 1, G_ij = exp(-t d)(cos t d + sin t d), d = |i - j|, t = {theta:g} (entries below 1e-17 "
                        f"dropped: {nnz} pairs), two punches and a rough gap; the Hessian decays away from the diagonal "
                        f"at exp(-{theta:g}) per index and is positive definite; {n_lo} nodes out of contact and {n_up} at "
                        "the pressure cap; reference by projected Newton and the exact solve on the final active set, "
                        "projected gradient < 1e-10")
    _mk.__name__ = f"WINKLER_{_tag}_250"
    register(_mk)


# ---------------- longmem: isotonic generalised least squares under long-memory noise ----------------
def _longmem(n, s_exp, seed):
    rng = np.random.default_rng(seed)
    dist = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]).astype(float)
    G = (1.0 + dist) ** (-s_exp)
    assert np.min(np.linalg.eigvalsh(G)) > 1e-3, "LONGMEM: the weight matrix must be positive definite"
    t = np.linspace(0.0, 1.0, n)
    trend = 4.0 * np.minimum(t, 0.6) ** 2 + 0.5 * np.tanh(12.0 * (t - 0.8))
    y = 5.0 * (trend + 0.12 * rng.standard_normal(n))
    Gy = G @ y

    def solve(blocks):
        B = np.zeros((n, len(blocks)))
        for k, (a, b) in enumerate(blocks):
            B[a:b, k] = 1.0
        z = np.linalg.solve(B.T @ G @ B, B.T @ Gy)
        x = B @ z
        mu = -np.cumsum(G @ x - Gy)[:-1]              # multipliers of x_{i+1} - x_i >= 0
        return z, x, mu

    blocks = [(i, i + 1) for i in range(n)]
    for _ in range(20 * n):
        z, x, mu = solve(blocks)
        viol = np.flatnonzero(np.diff(z) < 0.0)
        if viol.size:                                 # pool the worst adjacent violators
            k = int(viol[np.argmin(np.diff(z)[viol])])
            blocks[k:k + 2] = [(blocks[k][0], blocks[k + 1][1])]
            continue
        inner = np.ones(n - 1, bool)
        inner[[b - 1 for _, b in blocks[:-1]]] = False
        if np.all(mu[inner] >= -1e-12 * np.max(np.abs(mu))):
            break
        j = int(np.flatnonzero(inner)[np.argmin(mu[inner])])   # split where the multiplier is most negative
        k = next(q for q, (a, b) in enumerate(blocks) if a <= j < b - 1)
        a, b = blocks[k]
        blocks[k:k + 1] = [(a, j + 1), (j + 1, b)]
    z, x, mu = solve(blocks)
    g = G @ x - Gy
    inner = np.ones(n - 1, bool)
    inner[[b - 1 for _, b in blocks[:-1]]] = False
    mu_full = np.where(inner, mu, 0.0)
    res = g.copy()                                    # g - D'mu with (D'mu)_j = mu_{j-1} - mu_j
    res[1:] -= mu_full
    res[:-1] += mu_full
    scale = max(1.0, float(np.max(np.abs(mu_full))))
    assert np.all(np.diff(x) >= -1e-13) and np.all(mu_full >= -1e-11 * scale), "LONGMEM: primal or dual sign"
    assert np.max(np.abs(res)) < 1e-10 * scale, f"LONGMEM: stationarity {np.max(np.abs(res)):.1e} against {scale:.1e}"
    f_star = 0.5 * float((x - y) @ G @ (x - y))
    return G, y, x, f_star, int(inner.sum()), len(blocks)


for _tag, _s, _seed in (("S08", 0.8, 671), ("S20", 2.0, 672)):
    def _mk(tag=_tag, s_exp=_s, seed=_seed):
        n = 200
        G, y, x_ref, f_star, n_act, n_blocks = _longmem(n, s_exp, seed)
        x = xs(n)
        f = _quad(x, G, -(G @ y), 0.5 * float(y @ G @ y))
        c = [x[i + 1] - x[i] for i in range(n - 1)]
        return st(f"LONGMEM_{tag}_{n}", "longmem", x, f, c, [0.0] * (n - 1), [INF] * (n - 1), x0=[float(np.mean(y))] * n,
                  ref_f=f_star, ref_x=[list(map(float, x_ref))], tags=["convex", "dense"],
                  notes=f"isotonic generalised least squares 0.5 (x - y)'G(x - y), x_1 <= ... <= x_{n}, with the long-memory "
                        f"weights G_ij = (1 + |i - j|)^-{s_exp:g} (dense, positive definite, a power law that never becomes "
                        f"banded) on a noisy increasing trend; {n - 1} sparse linear rows, {n_act} active ({n_blocks} "
                        "blocks); reference by pooling and splitting on the block system with multipliers by cumulative "
                        "sums, stationarity < 1e-10 of the multipliers' scale; start: the constant mean, every row active")
    _mk.__name__ = f"LONGMEM_{_tag}_200"
    register(_mk)
