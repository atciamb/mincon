"""Fourth held-out family set (final4), created after candidate C7 (the curvature-tracking
quasi-Newton rebuild, ``docs/21``) was frozen and before any C7 run on these problems.

Nothing here was seen by any tuning of any candidate. Every reference is exact: computed at
generation by an independent method (Newton on the KKT conditions with the active set
identified, or Newton's method on a strictly convex equality-constrained problem) and
verified to a stationarity of 1e-10 or better, or a closed-form global minimum (f* = 0 for a
noiseless fit). Three of the four families have dense, coupled Hessians at n >= 100, which
the development corpus lacked (``docs/21`` §7, G3). An analytic-centre family (-sum log of dense
rows) was designed first and dropped before any run: its generated MATLAB gradient repeats every
dense row in every entry (69 MB at n = 100), which the corpus codegen cannot carry.

Families:
  covqp       covariance QP with a budget equality, a quadratic risk inequality and caps
              (convex; reference: Newton on the active-set KKT system, stationarity < 1e-10)
  denselap    dense graph-Laplacian coupling of every pair plus an exponential term per variable,
              on a hyperplane (strictly convex, dense Hessian whose curvature changes along the
              path; reference: equality-constrained Newton, stationarity < 1e-10)
  snl         sensor-network localisation from all pairwise and anchor distances, one sensor
              on a circle (nonconvex, dense; f* = 0 exact at the true configuration)
  obstacle    discretised obstacle problem on [0, 1] (bound-constrained convex QP, tridiagonal;
              reference: active-set solve, stationarity < 1e-12)
"""
from __future__ import annotations

import numpy as np
import sympy as sp

from spec import INF, register, xs
from structured import st

log = sp.log


def _stationarity(g, lo_active, up_active, tol):
    """Largest violation of the bound-multiplier signs on the active bounds and of
    stationarity on the free coordinates, for a gradient `g` of the Lagrangian
    (rows already folded in)."""
    free = ~(lo_active | up_active)
    worst = np.max(np.abs(g[free])) if free.any() else 0.0
    if lo_active.any():
        worst = max(worst, float(np.max(np.maximum(0.0, -g[lo_active]))))
    if up_active.any():
        worst = max(worst, float(np.max(np.maximum(0.0, g[up_active]))))
    return worst


# ---------------- covariance QP with a risk constraint ----------------
def _covqp(n, seed):
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((n, 5))
    Sigma = F @ F.T / 5.0 + np.diag(rng.uniform(0.02, 0.2, n))
    Sigma = 0.5 * (Sigma + Sigma.T)
    mu = rng.uniform(0.02, 0.15, n)
    kappa = 0.5
    d = rng.uniform(0.5, 2.0, n)
    cap = 3.0 / n
    lo = np.zeros(n)
    hi = np.full(n, cap)

    def solve(rho):
        # Active-set Newton on the KKT system: unknowns x_free, lambda (budget), nu (risk, when
        # active); bounds fixed at their values; the active set is revised from the multiplier
        # signs until it is stable. Convex, so the KKT point is the global minimum.
        from scipy.optimize import minimize as _sp_min
        cons = [{"type": "eq", "fun": lambda x: x.sum() - 1.0, "jac": lambda x: np.ones(n)}]
        if rho is not None:
            cons.append({"type": "ineq", "fun": lambda x: rho - x @ (d * x), "jac": lambda x: -2.0 * d * x})
        r = _sp_min(lambda x: 0.5 * x @ Sigma @ x - kappa * mu @ x, np.full(n, 1.0 / n),
                    jac=lambda x: Sigma @ x - kappa * mu, method="SLSQP", bounds=list(zip(lo, hi)),
                    constraints=cons, options={"maxiter": 5000, "ftol": 1e-15})
        x = np.clip(r.x, lo, hi)
        for _ in range(60):
            lo_a = x <= lo + 1e-8
            up_a = x >= hi - 1e-8
            free = ~(lo_a | up_a)
            k = int(free.sum())
            risk_a = rho is not None and x @ (d * x) >= rho - 1e-8
            idx_f = np.flatnonzero(free)

            def residual(z):
                xx = x.copy()
                xx[idx_f] = z[:k]
                lam = z[k]
                nu = z[k + 1] if risk_a else 0.0
                g = Sigma @ xx - kappa * mu + lam + 2.0 * nu * d * xx
                rows = [g[idx_f], [xx.sum() - 1.0]]
                if risk_a:
                    rows.append([xx @ (d * xx) - rho])
                return np.concatenate(rows), xx, g, nu

            z = np.concatenate([x[idx_f], [0.0], [0.0] if risk_a else []])
            for _newton in range(50):
                rvec, xx, g, nu = residual(z)
                if np.max(np.abs(rvec)) < 1e-13:
                    break
                m = k + 1 + (1 if risk_a else 0)
                J = np.zeros((m, m))
                J[:k, :k] = Sigma[np.ix_(idx_f, idx_f)] + (2.0 * nu * np.diag(d[idx_f]) if risk_a else 0.0)
                J[:k, k] = 1.0
                J[k, :k] = 1.0
                if risk_a:
                    J[:k, k + 1] = 2.0 * d[idx_f] * xx[idx_f]
                    J[k + 1, :k] = 2.0 * d[idx_f] * xx[idx_f]
                z = z - np.linalg.solve(J, rvec)
            rvec, xx, g, nu = residual(z)
            changed = False
            # Wrong-sign multipliers: release; violated bounds: clip and activate.
            if risk_a and nu < -1e-12:
                x = xx.copy()
                # release the risk constraint by nudging inside; SLSQP's point is already close
                risk_a = False
                changed = True
            release_lo = lo_a & (g < -1e-10)
            release_up = up_a & (g > 1e-10)
            if release_lo.any() or release_up.any():
                changed = True
            out_lo = free & (xx < lo - 1e-12)
            out_up = free & (xx > hi + 1e-12)
            if out_lo.any() or out_up.any():
                changed = True
            x = np.clip(xx, lo, hi)
            if release_lo.any():
                x[release_lo] = lo[release_lo] + 1e-6
            if release_up.any():
                x[release_up] = hi[release_up] - 1e-6
            if not changed and np.max(np.abs(rvec)) < 1e-12:
                break
        lo_a = x <= lo + 1e-10
        up_a = x >= hi - 1e-10
        gL = Sigma @ x - kappa * mu + z[k] + (2.0 * z[k + 1] * d * x if risk_a else 0.0)
        stat = _stationarity(gL, lo_a, up_a, 1e-10)
        assert abs(x.sum() - 1.0) < 1e-12, "budget"
        assert rho is None or x @ (d * x) <= rho + 1e-10, "risk"
        assert stat < 1e-10, f"covqp KKT polish did not converge: stationarity {stat:.1e}"
        assert (not risk_a) or z[k + 1] >= -1e-14
        return x, risk_a

    x_u, _ = solve(None)
    rho = 0.7 * float(x_u @ (d * x_u))
    x_star, risk_active = solve(rho)
    assert risk_active, "the risk constraint must be active at the reference"
    f_star = 0.5 * float(x_star @ Sigma @ x_star) - kappa * float(mu @ x_star)
    return Sigma, mu, kappa, d, rho, cap, x_star, f_star


for _n, _seed in ((30, 401), (120, 402), (300, 403)):
    def _mk(n=_n, seed=_seed):
        Sigma, mu, kappa, d, rho, cap, x_star, f_star = _covqp(n, seed)
        x = xs(n)
        f = sp.Add(*[float(0.5 * Sigma[i, j]) * x[i] * x[j] for i in range(n) for j in range(n)]) \
            - sp.Add(*[float(kappa * mu[i]) * x[i] for i in range(n)])
        c = [sp.Add(*x), sp.Add(*[float(d[i]) * x[i] ** 2 for i in range(n)])]
        return st(f"COVQP_{n}", "covqp", x, f, c, [1, -INF], [1, rho], xl=[0] * n, xu=[cap] * n,
                  x0=[1.0 / n] * n, ref_f=f_star, ref_x=[float(v) for v in x_star], tags=["convex"],
                  notes="covariance QP with budget, quadratic risk inequality (active) and caps; reference from an "
                        "active-set KKT Newton polish, stationarity < 1e-10 at generation")
    _mk.__name__ = f"COVQP_{_n}"
    register(_mk)


# ---------------- dense Laplacian-coupled convex problem ----------------
def _denselap(n, seed):
    rng = np.random.default_rng(seed)
    W = rng.uniform(0.2, 1.0, (n, n)) / n
    W = np.triu(W, 1)
    W = W + W.T                                   # dense symmetric positive weights, zero diagonal
    L = np.diag(W.sum(axis=1)) - W                # dense graph Laplacian
    dvec = rng.uniform(-2.0, 2.0, n)
    box = 3.0
    ones = np.ones(n)
    # f(x) = 0.5 x^T L x + sum(exp(x_i) - d_i x_i), sum(x) = 0, |x| <= box: strictly convex.
    # Reference: equality-constrained Newton from x = 0, then a check that no bound is active
    # (so the KKT point of the equality-constrained problem is the solution of the boxed one).
    x = np.zeros(n)
    for _ in range(200):
        g = L @ x + np.exp(x) - dvec
        H = L + np.diag(np.exp(x))
        K = np.block([[H, ones[:, None]], [ones[None, :], np.zeros((1, 1))]])
        sol = np.linalg.solve(K, np.concatenate([-g, [0.0]]))
        dx = sol[:n]
        lam = -float(g @ ones) / n
        if np.max(np.abs(g + lam * ones)) < 1e-12:
            break
        t = 1.0
        fx = 0.5 * x @ L @ x + np.sum(np.exp(x) - dvec * x)
        while 0.5 * (x + t * dx) @ L @ (x + t * dx) + np.sum(np.exp(x + t * dx) - dvec * (x + t * dx))                 > fx + 0.25 * t * float(g @ dx):
            t *= 0.5
        x = x + t * dx
    x = x - x.mean()
    g = L @ x + np.exp(x) - dvec
    lam = -float(g @ ones) / n
    assert np.max(np.abs(g + lam * ones)) < 1e-10, "denselap Newton did not converge"
    assert np.max(np.abs(x)) < box - 0.1, "a bound is active; the reference assumes none is"
    assert abs(x.sum()) < 1e-11
    f_star = 0.5 * float(x @ L @ x) + float(np.sum(np.exp(x) - dvec * x))
    return W, dvec, box, x, f_star


for _n, _seed in ((100, 411), (250, 412)):
    def _mk(n=_n, seed=_seed):
        W, dvec, box, x_star, f_star = _denselap(n, seed)
        x = xs(n)
        # 0.5 x^T L x written as monomials (cheaper for the code generator than squared differences)
        f = sp.Add(*[float(0.5 * W.sum(axis=1)[i]) * x[i] ** 2 for i in range(n)])             - sp.Add(*[float(W[i, j]) * x[i] * x[j] for i in range(n) for j in range(i + 1, n)])             + sp.Add(*[sp.exp(x[i]) - float(dvec[i]) * x[i] for i in range(n)])
        c = [sp.Add(*x)]
        return st(f"DENSELAP_{n}", "denselap", x, f, c, [0], [0], xl=[-box] * n, xu=[box] * n,
                  x0=[0.0] * n, ref_f=f_star, ref_x=[float(v) for v in x_star], tags=["convex"],
                  notes="dense graph-Laplacian coupling of every pair of variables plus an exponential term per "
                        "variable, on the hyperplane sum x = 0 inside a box that is inactive at the solution; "
                        "strictly convex with a dense Hessian whose curvature changes along the path; reference by "
                        "equality-constrained Newton, stationarity < 1e-10")
    _mk.__name__ = f"DENSELAP_{_n}"
    register(_mk)


# ---------------- sensor-network localisation ----------------
def _snl(N, seed):
    rng = np.random.default_rng(seed)
    P = rng.uniform(-1.0, 1.0, (N, 2))
    anchors = np.array([[-0.8, -0.8], [0.8, -0.8], [-0.8, 0.8], [0.8, 0.8]])
    D2 = np.sum((P[:, None, :] - P[None, :, :]) ** 2, axis=2)
    E2 = np.sum((P[:, None, :] - anchors[None, :, :]) ** 2, axis=2)
    r1 = float(np.sqrt(P[0] @ P[0]))
    x0 = (P + 0.1 * rng.standard_normal((N, 2))).ravel()
    return P, anchors, D2, E2, r1, x0


for _N, _seed in ((12, 421), (30, 422), (75, 423)):
    def _mk(N=_N, seed=_seed):
        P, anchors, D2, E2, r1, x0 = _snl(N, seed)
        n = 2 * N
        x = xs(n)
        px = [x[2 * i] for i in range(N)]
        py = [x[2 * i + 1] for i in range(N)]
        terms = []
        for i in range(N):
            for j in range(i + 1, N):
                terms.append(((px[i] - px[j]) ** 2 + (py[i] - py[j]) ** 2 - float(D2[i, j])) ** 2)
            for a in range(4):
                terms.append(((px[i] - float(anchors[a, 0])) ** 2 + (py[i] - float(anchors[a, 1])) ** 2 - float(E2[i, a])) ** 2)
        f = sp.Add(*terms)
        c = [px[0] ** 2 + py[0] ** 2]
        return st(f"SNL_{n}", "snl", x, f, c, [r1 ** 2], [r1 ** 2], xl=[-1.5] * n, xu=[1.5] * n,
                  x0=[float(v) for v in x0], ref_f=0.0, ref_x=[float(v) for v in P.ravel()],
                  notes="sensor-network localisation from all pairwise and four anchor distances (noiseless), sensor 1 on a "
                        "circle; f* = 0 at the true configuration; every pair couples, so the Hessian is dense; nonconvex")
    _mk.__name__ = f"SNL_{2 * _N}"
    register(_mk)


# ---------------- discretised obstacle problem ----------------
def _obstacle(N, seed):
    rng = np.random.default_rng(seed)
    h = 1.0 / (N + 1)
    t = h * np.arange(1, N + 1)
    load = -8.0 + 2.0 * np.sin(3.0 * np.pi * t) + 0.5 * rng.standard_normal(N)   # pushes the membrane down
    psi = 0.15 - 0.9 * np.abs(t - 0.5) ** 1.5 - 0.05 * np.cos(9.0 * t)             # obstacle from below
    # K = (1/h) tridiag(-1, 2, -1), f(u) = 0.5 u^T K u - h load^T u, u >= psi
    K = (np.diag(2.0 * np.ones(N)) - np.diag(np.ones(N - 1), 1) - np.diag(np.ones(N - 1), -1)) / h
    q = -h * load
    u = np.maximum(psi, 0.0) + 0.1
    for _ in range(200):
        active = u <= psi + 1e-12
        free = ~active
        idx = np.flatnonzero(free)
        rhs = -q[idx] - K[np.ix_(idx, np.flatnonzero(active))] @ psi[active]
        u_f = np.linalg.solve(K[np.ix_(idx, idx)], rhs)
        u_new = psi.copy()
        u_new[idx] = u_f
        g = K @ u_new + q
        viol = free & (u_new < psi - 1e-12)
        wrong = active & (g < -1e-12)
        if not viol.any() and not wrong.any():
            u = u_new
            break
        u = np.maximum(u_new, psi)
        if wrong.any():
            u[wrong] = psi[wrong] + 1e-6
    g = K @ u + q
    active = u <= psi + 1e-10
    stat = _stationarity(g, active, np.zeros(N, dtype=bool), 1e-12)
    assert stat < 1e-12, f"obstacle active-set solve did not converge: {stat:.1e}"
    assert np.all(u >= psi - 1e-14)
    f_star = 0.5 * float(u @ K @ u) + float(q @ u)
    return h, load, psi, u, f_star


for _N, _seed in ((50, 431), (200, 432), (500, 433)):
    def _mk(N=_N, seed=_seed):
        h, load, psi, u_star, f_star = _obstacle(N, seed)
        x = xs(N)
        chain = [0] + list(x) + [0]
        f = sp.Add(*[sp.Rational(1, 2) * (chain[i + 1] - chain[i]) ** 2 / h for i in range(N + 1)]) \
            - sp.Add(*[float(h * load[i]) * x[i] for i in range(N)])
        return st(f"OBSTACLE_{N}", "obstacle", x, f, [], [], [], xl=[float(v) for v in psi], xu=[INF] * N,
                  x0=[float(max(v, 0.0) + 0.1) for v in psi], ref_f=f_star, ref_x=[float(v) for v in u_star],
                  tags=["convex"],
                  notes="discretised obstacle problem: membrane energy under a downward load, u >= obstacle; "
                        "bound-constrained convex QP, tridiagonal; reference by active-set solve, stationarity < 1e-12")
    _mk.__name__ = f"OBSTACLE_{_N}"
    register(_mk)
