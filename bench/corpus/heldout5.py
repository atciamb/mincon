"""Fifth held-out family set (final5), created after the round-5 increments I1-I9, S-E, I5 with
its structured build, and I6 were frozen (``docs/22``), and before any run of any of them on
these problems.

Nothing here was seen by any tuning of any candidate. The families come from the round-5 plan
(``docs/22`` §5, gate G3: realistic problems written the way a working researcher writes them)
and from the two gaps the I5 probe opened: a quadratic program with a quadratic *constraint* row,
which the probe declines by design, and a nonconvex quadratic program with several local minima,
which the convexity check declines. References are exact where the mathematics allows (closed
form, or an independent method verified at generation to a stationarity of 1e-10 or better) and
published best-known values for the two design problems, which are tagged ``best-known``.

Families:
  pkfit        two-compartment pharmacokinetic fit, y = A e^{-k1 t} + B e^{-k2 t} with k2/k1 = 1000
               (the closed form of a stiff linear ODE), on a log-spaced grid; clean (f* = 0) and
               with 2 % multiplicative noise (reference: Newton-polished least squares from the
               true parameters, stationarity < 1e-10 relative to f*; tagged best-known since
               global optimality of a noisy fit is not proved)
  logistic     logistic-growth fit y = K / (1 + e^{-r (t - t0)}) (the closed form of the logistic
               ODE); clean and with 2 % noise, as above
  tcport       portfolio with quadratic transaction costs and a variance *row*: the objective is
               quadratic and the risk constraint x' Sigma x <= sigma^2 is quadratic, so the
               quadratic-program probe declines by design (reference: active-set KKT Newton
               polish, stationarity < 1e-10; the variance row is active)
  deconv       bounded deconvolution 0.5 |K x - y|^2, 0 <= x <= 1, Gaussian kernel of width 3
               (the friction problem box_lsq at n = 60 and n = 200; reference: bounded-variable
               least squares, projected gradient < 1e-10)
  noisyqp      a convex box-constrained QP plus a deterministic stand-in for 1e-7 simulator
               noise, 1e-7 (sin(1e3 a'x) + sin(3e4 b'x)) (the corpus needs closed expressions;
               reference: the noiseless QP's optimum, which the perturbation moves by less than
               1e-5 relative in f; tagged noisy)
  ncboxqp      indefinite QP on the box [-1, 1]^n: every local minimum enumerated over the 3^n
               active-set patterns at generation; reference: the global one (tagged
               multiple-local-minima; a solver may certify another)
  engineering3 two design problems with units from the structural-optimisation literature,
               the corrugated bulkhead (n = 4, best-known 6.8429) and the I-beam vertical
               deflection (n = 4, best-known 0.0130741)
"""
from __future__ import annotations

import itertools

import numpy as np
import sympy as sp

from spec import INF, register, xs
from structured import st

exp = sp.exp
sqrt = sp.sqrt


def _proj_stationarity(g, x, lo, hi, tol=1e-10):
    """Largest violation of the bound-constrained first-order conditions for a gradient `g`
    of the Lagrangian (rows already folded in)."""
    lo_a = x <= lo + tol
    up_a = x >= hi - tol
    free = ~(lo_a | up_a)
    worst = float(np.max(np.abs(g[free]))) if free.any() else 0.0
    if lo_a.any():
        worst = max(worst, float(np.max(np.maximum(0.0, -g[lo_a]))))
    if up_a.any():
        worst = max(worst, float(np.max(np.maximum(0.0, g[up_a]))))
    return worst


# ---------------- least-squares fits with closed-form models ----------------
def _polished_lsq(spec_fn, x_true, name):
    """Reference for a noisy fit: Newton with the exact Hessian from the true parameters on the
    unconstrained least-squares problem, verified at stationarity < 1e-10 with every bound and
    row inactive."""
    s = spec_fn(None, None)
    p = s.numpy()
    x = np.asarray(x_true, float)
    lam = np.zeros(s.m)
    # stationarity relative to the objective's magnitude: the gradient of a sum of squares with
    # residuals r and sensitivities J has a rounding floor of order eps |J| |r| per term
    tol = 1e-10 * max(1.0, abs(p.f(x)))
    for _ in range(100):
        g = p.grad(x)
        if np.max(np.abs(g)) < 0.1 * tol:
            break
        H = p.hess_lagrangian(x, lam)
        step = np.linalg.solve(H + 1e-14 * np.eye(len(x)), -g)
        # damped so a far start cannot leave the basin; rounding-level increases are accepted
        t = 1.0
        f0 = p.f(x)
        while t > 1e-8 and p.f(x + t * step) > f0 * (1.0 + 1e-12) + 1e-300:
            t *= 0.5
        x = x + t * step
    g = p.grad(x)
    assert np.max(np.abs(g)) < tol, f"{name}: fit reference stationarity {np.max(np.abs(g)):.1e} against {tol:.1e}"
    assert p.violation(x) == 0.0, f"{name}: the fit reference must satisfy bounds and rows"
    if s.m:
        c = p.cons(x)
        assert np.all(c > np.asarray(s.cl) + 1e-6) and np.all(c < np.asarray(s.cu) + 1e300), f"{name}: row active"
    assert np.all(x > np.asarray(s.xl) + 1e-6) and np.all(x < np.asarray(s.xu) - 1e-6), f"{name}: bound active"
    return float(p.f(x)), [list(map(float, x))]


def _pkfit(noisy):
    a, k1, b, k2 = 2.0, 0.05, 5.0, 50.0            # stiffness ratio k2 / k1 = 1000
    t = np.logspace(-2, np.log10(60.0), 25)
    y = a * np.exp(-k1 * t) + b * np.exp(-k2 * t)
    if noisy:
        y = y * (1.0 + 0.02 * np.random.default_rng(501).standard_normal(t.size))
    x = xs(4)
    A, K1, B, K2 = x
    f = sp.Add(*[(A * exp(-K1 * float(ti)) + B * exp(-K2 * float(ti)) - float(yi)) ** 2 for ti, yi in zip(t, y)])

    def mk(ref_f, ref_x):
        name = "PKFIT_NOISY2" if noisy else "PKFIT_CLEAN"
        return st(name, "pkfit", x, f, [K2 - 10 * K1], [0.0], [INF], xl=[0, 1e-3, 0, 1], xu=[20, 1, 20, 1000],
                  x0=[1.0, 0.5, 1.0, 10.0], ref_f=ref_f, ref_x=ref_x,
                  tags=["fitting"] + (["best-known"] if noisy else []),
                  notes="two-compartment pharmacokinetics A e^{-k1 t} + B e^{-k2 t}, k2 / k1 = 1000 at the truth, log-spaced "
                        "sampling 0.01..60; ordering row k2 >= 10 k1"
                        + ("; 2 % multiplicative noise, reference by Newton-polished least squares from the truth, "
                           "stationarity < 1e-10" if noisy else "; noiseless, f* = 0 at the truth"))
    if not noisy:
        return mk(0.0, [[a, k1, b, k2]])
    fstar, xstar = _polished_lsq(mk, [a, k1, b, k2], "PKFIT_NOISY2")
    return mk(fstar, xstar)


@register
def PKFIT_CLEAN():
    return _pkfit(False)


@register
def PKFIT_NOISY2():
    return _pkfit(True)


def _logistic(noisy):
    kk, r, t0 = 50.0, 0.8, 6.0
    t = np.linspace(0.0, 15.0, 31)
    y = kk / (1.0 + np.exp(-r * (t - t0)))
    if noisy:
        y = y * (1.0 + 0.02 * np.random.default_rng(502).standard_normal(t.size))
    x = xs(3)
    K, R, T0 = x
    f = sp.Add(*[(K / (1 + exp(-R * (float(ti) - T0))) - float(yi)) ** 2 for ti, yi in zip(t, y)])

    def mk(ref_f, ref_x):
        name = "LOGISTIC_NOISY2" if noisy else "LOGISTIC_CLEAN"
        return st(name, "logistic", x, f, xl=[1, 0.01, -5], xu=[200, 5, 25], x0=[10.0, 0.2, 2.0], ref_f=ref_f, ref_x=ref_x,
                  tags=["fitting"] + (["best-known"] if noisy else []),
                  notes="logistic growth K / (1 + e^{-r (t - t0)}) on t = 0..15"
                        + ("; 2 % multiplicative noise, reference by Newton-polished least squares from the truth, "
                           "stationarity < 1e-10" if noisy else "; noiseless, f* = 0 at the truth"))
    if not noisy:
        return mk(0.0, [[kk, r, t0]])
    fstar, xstar = _polished_lsq(mk, [kk, r, t0], "LOGISTIC_NOISY2")
    return mk(fstar, xstar)


@register
def LOGISTIC_CLEAN():
    return _logistic(False)


@register
def LOGISTIC_NOISY2():
    return _logistic(True)


# ---------------- KKT polish for a convex QP with a budget row, one quadratic row and a box ----------------
def _kkt_polish(H, c, a, b, D, rho, lo, hi, x, name):
    """min 0.5 x'Hx + c'x  s.t.  a'x = b (when `a` is not None), x'Dx <= rho (when `D` is not
    None), lo <= x <= hi. Active-set Newton from `x` (an SLSQP point): unknowns x_free, lambda,
    nu; the active set is revised from multiplier signs and bound violations until stable.
    Convex, so the KKT point is the global minimum. Returns x, f, whether the quadratic row is
    active, and its multiplier."""
    n = len(x)
    x = np.clip(np.asarray(x, float), lo, hi)
    risk_a = D is not None and x @ (D @ x) >= rho - 1e-8
    nu = 0.0
    lam = 0.0
    for _outer in range(80):
        lo_a = x <= lo + 1e-8
        up_a = x >= hi - 1e-8
        free = ~(lo_a | up_a)
        idx = np.flatnonzero(free)
        k = idx.size
        has_eq = a is not None
        m = k + (1 if has_eq else 0) + (1 if risk_a else 0)

        def resid(z):
            xx = x.copy()
            xx[idx] = z[:k]
            lam_ = z[k] if has_eq else 0.0
            nu_ = z[-1] if risk_a else 0.0
            g = H @ xx + c + (lam_ * a if has_eq else 0.0) + (2.0 * nu_ * (D @ xx) if risk_a else 0.0)
            rows = [g[idx]]
            if has_eq:
                rows.append([a @ xx - b])
            if risk_a:
                rows.append([xx @ (D @ xx) - rho])
            return np.concatenate(rows), xx, g, lam_, nu_

        z = np.concatenate([x[idx], [lam] if has_eq else [], [nu] if risk_a else []])
        for _newton in range(60):
            rvec, xx, g, lam, nu = resid(z)
            if np.max(np.abs(rvec)) < 1e-13:
                break
            J = np.zeros((m, m))
            J[:k, :k] = H[np.ix_(idx, idx)] + (2.0 * nu * D[np.ix_(idx, idx)] if risk_a else 0.0)
            col = k
            if has_eq:
                J[:k, col] = a[idx]
                J[col, :k] = a[idx]
                col += 1
            if risk_a:
                dx = 2.0 * (D @ xx)[idx]
                J[:k, col] = dx
                J[col, :k] = dx
            z = z - np.linalg.solve(J, rvec)
        rvec, xx, g, lam, nu = resid(z)
        changed = False
        if risk_a and nu < -1e-12:
            risk_a = False
            changed = True
        release_lo = lo_a & (g < -1e-10)
        release_up = up_a & (g > 1e-10)
        out_lo = free & (xx < lo - 1e-12)
        out_up = free & (xx > hi + 1e-12)
        if release_lo.any() or release_up.any() or out_lo.any() or out_up.any():
            changed = True
        x = np.clip(xx, lo, hi)
        if release_lo.any():
            x[release_lo] = lo[release_lo] + 1e-6
        if release_up.any():
            x[release_up] = hi[release_up] - 1e-6
        if D is not None and not risk_a and x @ (D @ x) > rho + 1e-10:
            risk_a = True
            changed = True
        if not changed and np.max(np.abs(rvec)) < 1e-12:
            break
    gL = H @ x + c + (lam * a if a is not None else 0.0) + (2.0 * nu * (D @ x) if risk_a else 0.0)
    stat = _proj_stationarity(gL, x, lo, hi)
    assert stat < 1e-10, f"{name}: KKT polish stationarity {stat:.1e}"
    if a is not None:
        assert abs(a @ x - b) < 1e-12, f"{name}: budget row"
    if D is not None:
        assert x @ (D @ x) <= rho + 1e-10, f"{name}: quadratic row"
        assert (not risk_a) or nu >= -1e-14
    f = 0.5 * float(x @ H @ x) + float(c @ x)
    return x, f, risk_a, nu


def _tcport(n, seed):
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((n, 3))
    Sigma = F @ F.T / 3.0 + np.diag(rng.uniform(0.05, 0.3, n))
    Sigma = 0.5 * (Sigma + Sigma.T)
    mu = rng.uniform(0.02, 0.15, n)
    w = np.full(n, 1.0 / n)                     # current holdings
    d = rng.uniform(0.5, 2.0, n)                # per-asset transaction-cost coefficients
    gamma = 0.05
    cap = 4.0 / n
    lo, hi = np.zeros(n), np.full(n, cap)
    # objective -mu'x + gamma (x - w)' diag(d) (x - w) = 0.5 x'Hx + c'x + const
    H = 2.0 * gamma * np.diag(d)
    c = -mu - 2.0 * gamma * d * w
    const = gamma * float(w @ (d * w))
    a, b = np.ones(n), 1.0
    from scipy.optimize import minimize as _sp_min

    def slsqp(rho):
        cons = [{"type": "eq", "fun": lambda x: a @ x - b, "jac": lambda x: a}]
        if rho is not None:
            cons.append({"type": "ineq", "fun": lambda x: rho - x @ (Sigma @ x), "jac": lambda x: -2.0 * Sigma @ x})
        r = _sp_min(lambda x: 0.5 * x @ H @ x + c @ x, w, jac=lambda x: H @ x + c, method="SLSQP",
                    bounds=list(zip(lo, hi)), constraints=cons, options={"maxiter": 5000, "ftol": 1e-15})
        return r.x

    x_u, _, _, _ = _kkt_polish(H, c, a, b, None, None, lo, hi, slsqp(None), f"TCPORT_{n}")
    rho = 0.6 * float(x_u @ (Sigma @ x_u))
    x_star, f_q, active, nu = _kkt_polish(H, c, a, b, Sigma, rho, lo, hi, slsqp(rho), f"TCPORT_{n}")
    assert active and nu > 1e-8, f"TCPORT_{n}: the variance row must be active at the reference"
    f_star = f_q + const
    return Sigma, mu, w, d, gamma, rho, cap, x_star, f_star


for _n, _seed in ((20, 511), (100, 512)):
    def _mk(n=_n, seed=_seed):
        Sigma, mu, w, d, gamma, rho, cap, x_star, f_star = _tcport(n, seed)
        x = xs(n)
        f = -sp.Add(*[float(mu[i]) * x[i] for i in range(n)]) + float(gamma) * sp.Add(
            *[float(d[i]) * (x[i] - float(w[i])) ** 2 for i in range(n)])
        risk = sp.Add(*[float(Sigma[i, j]) * x[i] * x[j] for i in range(n) for j in range(n)])
        c = [risk, sp.Add(*x)]
        return st(f"TCPORT_{n}", "tcport", x, f, c, [-INF, 1], [float(rho), 1], xl=[0] * n, xu=[float(cap)] * n,
                  x0=[float(v) for v in w], ref_f=f_star, ref_x=[list(map(float, x_star))], tags=["convex"],
                  notes="portfolio with quadratic transaction costs from the current holdings w = 1/n and a variance row "
                        "x' Sigma x <= sigma^2 (active at the solution), budget row, caps; convex QP with a quadratic "
                        "constraint row, so the quadratic-program probe declines by design; reference by active-set KKT "
                        "Newton polish, stationarity < 1e-10")
    _mk.__name__ = f"TCPORT_{_n}"
    register(_mk)


# ---------------- bounded deconvolution (box_lsq at n) ----------------
def _deconv(n, seed):
    rng = np.random.default_rng(seed)
    i = np.arange(n)
    K = np.exp(-0.5 * ((i[:, None] - i[None, :]) / 3.0) ** 2)
    K /= K.sum(axis=1, keepdims=True)
    x_true = np.zeros(n)
    x_true[n // 5: n // 5 + n // 6] = 0.8
    x_true[3 * n // 5: 3 * n // 5 + n // 16] = 1.0
    x_true[4 * n // 5] = 0.6
    y = K @ x_true + 0.02 * rng.standard_normal(n)
    from scipy.optimize import lsq_linear
    ref = lsq_linear(K, y, bounds=(0.0, 1.0), method="bvls", tol=1e-15, max_iter=20000)
    x_ref = np.clip(ref.x, 0.0, 1.0)
    # polish the free block by a Newton step (BVLS stops on its own tolerance)
    for _ in range(5):
        g = K.T @ (K @ x_ref - y)
        lo_a = x_ref <= 1e-12
        up_a = x_ref >= 1.0 - 1e-12
        free = ~(lo_a | up_a)
        if free.any():
            idx = np.flatnonzero(free)
            Kf = K[:, idx]
            r = y - K[:, ~free] @ x_ref[~free]
            xf, *_ = np.linalg.lstsq(Kf, r, rcond=None)
            x_ref[idx] = xf
        x_ref = np.clip(x_ref, 0.0, 1.0)
    g = K.T @ (K @ x_ref - y)
    stat = _proj_stationarity(g, x_ref, np.zeros(n), np.ones(n))
    assert stat < 1e-10, f"DECONV_{n}: reference stationarity {stat:.1e}"
    Q = K.T @ K
    cvec = -K.T @ y
    const = 0.5 * float(y @ y)
    f_star = 0.5 * float(x_ref @ Q @ x_ref) + float(cvec @ x_ref) + const
    return Q, cvec, const, x_ref, f_star


for _n, _seed in ((60, 521), (200, 522)):
    def _mk(n=_n, seed=_seed):
        Q, cvec, const, x_ref, f_star = _deconv(n, seed)
        x = xs(n)
        f = sp.Add(*[float(0.5 * Q[i, i]) * x[i] ** 2 for i in range(n)]) \
            + sp.Add(*[float(Q[i, j]) * x[i] * x[j] for i in range(n) for j in range(i + 1, n) if Q[i, j] != 0.0]) \
            + sp.Add(*[float(cvec[i]) * x[i] for i in range(n)]) + float(const)
        return st(f"DECONV_{n}", "deconv", x, f, xl=[0.0] * n, xu=[1.0] * n, x0=[0.5] * n, ref_f=f_star,
                  ref_x=[list(map(float, x_ref))], tags=["convex"],
                  notes="bounded deconvolution 0.5 |K x - y|^2 with a Gaussian kernel of width 3 and 2 % noise on the "
                        "data, 0 <= x <= 1 (the friction problem box_lsq at this n); dense ill-conditioned Hessian K'K; "
                        "reference by bounded-variable least squares polished on the free block, projected gradient < 1e-10")
    _mk.__name__ = f"DECONV_{_n}"
    register(_mk)


# ---------------- a convex box QP with a deterministic 1e-7 noise stand-in ----------------
@register
def NOISYQP_20():
    n = 20
    rng = np.random.default_rng(531)
    A = rng.standard_normal((n + 4, n))
    Q = A.T @ A / (n + 4) + np.eye(n)
    cvec = rng.uniform(-2.0, 2.0, n)
    lo, hi = -np.ones(n), np.ones(n)
    from scipy.optimize import minimize as _sp_min
    r = _sp_min(lambda x: 0.5 * x @ Q @ x + cvec @ x, np.zeros(n), jac=lambda x: Q @ x + cvec, method="L-BFGS-B",
                bounds=list(zip(lo, hi)), options={"maxiter": 5000, "ftol": 1e-15, "gtol": 1e-12})
    x_star, f_star, _, _ = _kkt_polish(Q, cvec, None, None, None, None, lo, hi, r.x, "NOISYQP_20")
    assert np.sum((x_star <= lo + 1e-10) | (x_star >= hi - 1e-10)) >= 2, "NOISYQP_20: some bounds must be active"
    avec = rng.uniform(-1.0, 1.0, n)
    bvec = rng.uniform(-1.0, 1.0, n)
    x = xs(n)
    f = sp.Add(*[float(0.5 * Q[i, i]) * x[i] ** 2 for i in range(n)]) \
        + sp.Add(*[float(Q[i, j]) * x[i] * x[j] for i in range(n) for j in range(i + 1, n)]) \
        + sp.Add(*[float(cvec[i]) * x[i] for i in range(n)]) \
        + 1e-7 * (sp.sin(1e3 * sp.Add(*[float(avec[i]) * x[i] for i in range(n)]))
                  + sp.sin(3e4 * sp.Add(*[float(bvec[i]) * x[i] for i in range(n)])))
    return st("NOISYQP_20", "noisyqp", x, f, xl=list(lo), xu=list(hi), x0=[0.0] * n, ref_f=f_star,
              ref_x=[list(map(float, x_star))], tags=["noisy"],
              notes="convex box QP plus 1e-7 (sin(1e3 a'x) + sin(3e4 b'x)), a deterministic stand-in for simulator noise "
                    "of 1e-7 (the corpus needs closed expressions); the perturbation's gradient is up to 3e-3, so the "
                    "landscape has wiggle minima every 2e-4 along b; reference: the noiseless QP's optimum by KKT polish, "
                    "which the perturbation moves by less than 1e-5 relative in f")


# ---------------- indefinite box QP: every local minimum enumerated ----------------
def _ncboxqp(n, seed):
    rng = np.random.default_rng(seed)
    V, _ = np.linalg.qr(rng.standard_normal((n, n)))
    eig = rng.uniform(-2.0, 2.0, n)
    eig[: n // 3] = -np.abs(eig[: n // 3])     # at least a third of the directions concave
    Q = (V * eig) @ V.T
    Q = 0.5 * (Q + Q.T)
    cvec = rng.uniform(-1.0, 1.0, n)
    minima = []
    for free_mask in itertools.product([False, True], repeat=n):
        free = np.asarray(free_mask)
        idx_f = np.flatnonzero(free)
        idx_b = np.flatnonzero(~free)
        k = idx_f.size
        if k:
            Qff = Q[np.ix_(idx_f, idx_f)]
            try:
                np.linalg.cholesky(Qff + 1e-12 * np.eye(k))
            except np.linalg.LinAlgError:
                continue          # a local minimum has a positive semidefinite free block
        # every sign pattern of the bound variables at once
        signs = np.array(list(itertools.product([-1.0, 1.0], repeat=idx_b.size))) if idx_b.size else np.zeros((1, 0))
        X = np.zeros((signs.shape[0], n))
        X[:, idx_b] = signs
        if k:
            rhs = -(cvec[idx_f][None, :] + X[:, idx_b] @ Q[np.ix_(idx_b, idx_f)])
            X[:, idx_f] = np.linalg.solve(Qff, rhs.T).T
        G = X @ Q + cvec[None, :]
        ok = np.all(np.abs(X[:, idx_f]) < 1.0 - 1e-12, axis=1) if k else np.ones(signs.shape[0], bool)
        if idx_b.size:
            ok &= np.all(np.where(X[:, idx_b] < 0, G[:, idx_b] >= -1e-12, G[:, idx_b] <= 1e-12), axis=1)
        for xk in X[ok]:
            minima.append((0.5 * float(xk @ Q @ xk) + float(cvec @ xk), xk))
    assert len(minima) >= 2, f"NCBOXQP_{n}: expected several local minima, found {len(minima)}"
    minima.sort(key=lambda t: t[0])
    f_star, x_star = minima[0]
    return Q, cvec, x_star, f_star, len(minima), minima[1][0]


for _n, _seed in ((8, 541), (12, 542)):
    def _mk(n=_n, seed=_seed):
        Q, cvec, x_star, f_star, count, f_second = _ncboxqp(n, seed)
        x = xs(n)
        f = sp.Add(*[float(0.5 * Q[i, i]) * x[i] ** 2 for i in range(n)]) \
            + sp.Add(*[float(Q[i, j]) * x[i] * x[j] for i in range(n) for j in range(i + 1, n)]) \
            + sp.Add(*[float(cvec[i]) * x[i] for i in range(n)])
        return st(f"NCBOXQP_{n}", "ncboxqp", x, f, xl=[-1.0] * n, xu=[1.0] * n, x0=[0.0] * n, ref_f=f_star,
                  ref_x=[list(map(float, x_star))], tags=["multiple-local-minima", "nonconvex"],
                  notes=f"indefinite QP on [-1, 1]^{n} (a third of the eigenvalues negative); {count} local minima "
                        f"enumerated over the 3^n active-set patterns at generation, second best f = {f_second:.6f}; "
                        "reference: the global one; the quadratic-program probe declines (not convex) by design")
    _mk.__name__ = f"NCBOXQP_{_n}"
    register(_mk)


# ---------------- design problems with units ----------------
@register
def CORRUGATED_BULKHEAD():
    # Kim & Lee / Rao: width, depth, length, thickness of a corrugated bulkhead; minimise weight.
    x = xs(4)
    w, h, ln, t = x
    s = sqrt(ln ** 2 - h ** 2)
    f = 5.885 * t * (w + ln) / (w + s)
    c = [h * t * (0.4 * w + ln / 6) - 8.94 * (w + s),
         h ** 2 * t * (0.2 * w + ln / 12) - 2.2 * (8.94 * (w + s)) ** sp.Rational(4, 3),
         t - 0.0156 * w - 0.15,
         t - 0.0156 * ln - 0.15,
         t - 1.05,
         ln - h]
    def mk(ref_f, ref_x):
        return st("CORRUGATED_BULKHEAD", "engineering3", x, f, c, [0] * 6, [INF] * 6, xl=[10, 10, 10, 1.05], xu=[100, 100, 100, 5],
                  x0=[50, 20, 60, 1.5], ref_f=ref_f, ref_x=ref_x, tags=["best-known"],
                  source="corrugated bulkhead (Kim and Lee 1998; Rao); published best-known 6.8429, a rounding of the vertex "
                         "value 6.842958 at (57.692308, 34.147620, 57.692308, 1.05)",
                  notes="units: cm and cm^2 in the constraints, the weight in kg; sqrt(l^2 - h^2) is undefined for h > l, "
                        "which the row l - h >= 0 excludes; reference recomputed for round 6: the optimum is a vertex (t at "
                        "its bound, both thickness rows and the second section row active)")
    # The point usually printed with the published value, (57.692, 34.148, 57.555, 1.05), is not the
    # optimum of this model: its third coordinate gives f = 6.846036, 4.6 scoring tolerances above the
    # target, and the 5e-3 assertion of round 5 let it through (s6v5-final5/README.md, section 8). The
    # optimum is a vertex: t = 1.05, the two thickness rows give w = l = 0.9 / 0.0156, and the second
    # section row fixes h. Four active constraints in four variables, so the multipliers are a 4 x 4 solve.
    from scipy.optimize import brentq
    p = mk(None, None).numpy()
    t_b = 1.05
    w_b = (t_b - 0.15) / 0.0156
    h_b = brentq(lambda v: v ** 2 * t_b * (0.2 * w_b + w_b / 12) - 2.2 * (8.94 * (w_b + np.sqrt(w_b ** 2 - v ** 2))) ** (4.0 / 3.0),
                 30.0, 40.0, xtol=1e-14, rtol=1e-15)
    xb = np.array([w_b, h_b, w_b, t_b])
    active = np.vstack([p.jac(xb)[[1, 2, 3]], [0.0, 0.0, 0.0, 1.0]])
    y = np.linalg.solve(active.T, p.grad(xb))          # grad f = sum y_i grad c_i with y >= 0 at a minimum
    assert np.all(y > 1e-6), f"CORRUGATED_BULKHEAD: multiplier signs {y}"
    assert np.max(np.abs(active.T @ y - p.grad(xb))) < 1e-10, "CORRUGATED_BULKHEAD: reference stationarity"
    assert p.violation(xb) < 1e-10 and abs(p.f(xb) - 6.8429) < 1e-4, "CORRUGATED_BULKHEAD: reference point does not check"
    return mk(float(p.f(xb)), [list(map(float, xb))])


@register
def I_BEAM():
    # Gold & Krishnamurty: minimise the vertical deflection of an I-beam under a cross-section-area
    # and a stress constraint; h, b, tw, tf in cm.
    x = xs(4)
    h, b, tw, tf = x
    inertia = tw * (h - 2 * tf) ** 3 / 12 + b * tf ** 3 / 6 + 2 * b * tf * ((h - tf) / 2) ** 2
    f = 5000 / inertia
    stress = 180000 * h / (tw * (h - 2 * tf) ** 3 + 2 * b * tf * (4 * tf ** 2 + 3 * h * (h - 2 * tf))) \
        + 15000 * b / ((h - 2 * tf) * tw ** 3 + 2 * tf * b ** 3)
    c = [2 * b * tf + tw * (h - 2 * tf), stress]
    spec = st("I_BEAM", "engineering3", x, f, c, [-INF, -INF], [300, 6], xl=[10, 10, 0.9, 0.9], xu=[80, 50, 5, 5],
              x0=[40, 30, 2, 2], ref_f=0.0130741, tags=["best-known"],
              source="I-beam vertical deflection (Gold and Krishnamurty 1997); best-known 0.0130741 at (80, 50, 0.9, 2.3217)",
              notes="units: cm, cm^2 (area <= 300) and kN/cm^2 (stress <= 6); the objective is a deflection of order 1e-2 "
                    "while the constraints are of order 1e2, a scale mismatch of four orders")
    p = spec.numpy()
    xb = np.array([80.0, 50.0, 0.9, 2.3217])
    assert p.violation(xb) < 1e-3 and abs(p.f(xb) - 0.0130741) < 1e-6, "I_BEAM: best-known point does not check"
    return spec
