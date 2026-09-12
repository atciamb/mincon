"""Realistic user problems for the friction audit (study S-D, docs/22).

Every problem is written the way a user would write it for MATLAB's fmincon:
an objective, a start and the MATLAB constraint pieces (A, b, Aeq, beq, lb,
ub, nonlcon with c <= 0 and ceq == 0). Nothing else is required of the user.
The exact derivatives defined here are used only by the independent oracle
and by the reference computation; no solver run in the audit receives them
unless the problem says so (``wrong_gradient`` deliberately supplies a wrong
objective gradient, because that is the mistake being audited).

References are independent of every solver under test: closed forms,
published optima, or a different algorithm (bounded-variable least squares,
simplex projection, a KKT-verified convex solve), never "best of the
contestants". ``Problem.canonical()`` exposes the model in the harness's
canonical form for ``bench/harness/oracle.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

INF = float("inf")


@dataclass
class Problem:
    name: str
    fun: Callable
    x0: np.ndarray
    grad: Callable                      # exact objective gradient (oracle only)
    A: np.ndarray | None = None
    b: np.ndarray | None = None
    Aeq: np.ndarray | None = None
    beq: np.ndarray | None = None
    lb: np.ndarray | None = None
    ub: np.ndarray | None = None
    nonlcon: Callable | None = None     # (c, ceq) with c <= 0, ceq == 0
    nonlcon_jac: Callable | None = None  # exact (Jc, Jceq), rows = constraints (oracle only)
    target: float = float("nan")
    target_source: str = ""
    x_ref: np.ndarray | None = None
    user_jac: Callable | None = None    # a gradient the *user* supplies (wrong_gradient only)
    args: tuple = ()
    minimum_inputs: str = ""
    story: str = ""
    param_tol: float | None = None      # optional extra check: |x - x_ref|_inf <= param_tol
    tags: tuple = ()
    data: dict = field(default_factory=dict)   # arrays MATLAB needs to build the same problem

    @property
    def n(self) -> int:
        return self.x0.size

    def sizes(self):
        """(#A rows, #Aeq rows, #c, #ceq); nonlinear sizes from one evaluation at x0."""
        na = 0 if self.b is None else np.asarray(self.b).size
        ne = 0 if self.beq is None else np.asarray(self.beq).size
        nc = nceq = 0
        if self.nonlcon is not None:
            c, ceq = self.nonlcon(self.x0, *self.args)
            nc, nceq = np.asarray(c, float).size, np.asarray(ceq, float).size
        return na, ne, nc, nceq

    def canonical(self):
        return Canonical(self)


class Canonical:
    """The harness's canonical form: cl <= c(x) <= cu, xl <= x <= xu, with exact derivatives.
    Rows in order: A x - b (upper 0), Aeq x - beq (equality 0), c(x) (upper 0), ceq(x) (equality 0)."""

    def __init__(self, p: Problem):
        self.p = p
        self.n = p.n
        na, ne, nc, nceq = p.sizes()
        self.m = na + ne + nc + nceq
        self.na, self.ne, self.nc, self.nceq = na, ne, nc, nceq
        self.xl = np.full(self.n, -INF) if p.lb is None else np.asarray(p.lb, float).reshape(-1) * np.ones(self.n)
        self.xu = np.full(self.n, INF) if p.ub is None else np.asarray(p.ub, float).reshape(-1) * np.ones(self.n)
        self.cl = np.concatenate([np.full(na, -INF), np.zeros(ne), np.full(nc, -INF), np.zeros(nceq)])
        self.cu = np.zeros(self.m)
        self.x0 = p.x0.copy()

    def f(self, x):
        return float(self.p.fun(np.asarray(x, float), *self.p.args))

    def grad(self, x):
        return np.asarray(self.p.grad(np.asarray(x, float), *self.p.args), float).reshape(-1)

    def cons(self, x):
        x = np.asarray(x, float)
        parts = []
        if self.na:
            parts.append(np.asarray(self.p.A, float) @ x - np.asarray(self.p.b, float).reshape(-1))
        if self.ne:
            parts.append(np.asarray(self.p.Aeq, float) @ x - np.asarray(self.p.beq, float).reshape(-1))
        if self.nc or self.nceq:
            c, ceq = self.p.nonlcon(x, *self.p.args)
            parts.append(np.asarray(c, float).reshape(-1))
            parts.append(np.asarray(ceq, float).reshape(-1))
        return np.concatenate(parts) if parts else np.zeros(0)

    def jac(self, x):
        x = np.asarray(x, float)
        rows = []
        if self.na:
            rows.append(np.asarray(self.p.A, float).reshape(self.na, self.n))
        if self.ne:
            rows.append(np.asarray(self.p.Aeq, float).reshape(self.ne, self.n))
        if self.nc or self.nceq:
            Jc, Jceq = self.p.nonlcon_jac(x, *self.p.args)
            rows.append(np.asarray(Jc, float).reshape(self.nc, self.n))
            rows.append(np.asarray(Jceq, float).reshape(self.nceq, self.n))
        return np.vstack(rows) if rows else np.zeros((0, self.n))

    def violation(self, x):
        x = np.asarray(x, float)
        v = max(np.max(np.maximum(self.xl - x, 0.0), initial=0.0), np.max(np.maximum(x - self.xu, 0.0), initial=0.0))
        if self.m:
            c = self.cons(x)
            v = max(v, float(np.max(np.maximum(self.cl - c, 0.0))), float(np.max(np.maximum(c - self.cu, 0.0))))
        return float(v)


# ----------------------------------------------------------------------------- helpers

def _simplex_projection(c):
    """argmin 0.5|x - c|^2 s.t. sum x = 1, x >= 0 (sort-based, exact)."""
    u = np.sort(c)[::-1]
    css = np.cumsum(u)
    k = np.nonzero(u * np.arange(1, c.size + 1) > (css - 1))[0][-1]
    tau = (css[k] - 1) / (k + 1)
    return np.maximum(c - tau, 0.0)


def _kkt_verified_convex_reference(p: Problem, x_start, label):
    """Reference for a convex problem by an independent method (SciPy trust-constr with exact
    derivatives, tight tolerances) verified by the oracle's recovered-multiplier stationarity.
    Convexity makes the verified KKT point the global optimum."""
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness"))
    import oracle  # noqa: E402
    from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, minimize
    can = p.canonical()
    cons = []
    if can.na or can.ne:
        rows_a = np.asarray(p.A, float).reshape(can.na, p.n) if can.na else np.zeros((0, p.n))
        rows_e = np.asarray(p.Aeq, float).reshape(can.ne, p.n) if can.ne else np.zeros((0, p.n))
        M = np.vstack([rows_a, rows_e])
        beq = np.asarray(p.beq, float).reshape(-1) if can.ne else np.zeros(0)
        lo = np.concatenate([np.full(can.na, -INF), beq])
        hi = np.concatenate([np.asarray(p.b, float).reshape(-1) if can.na else np.zeros(0), beq])
        cons.append(LinearConstraint(M, lo, hi))
    if can.nc or can.nceq:
        def cf(x):
            c, ceq = p.nonlcon(x)
            return np.concatenate([np.asarray(c, float).reshape(-1), np.asarray(ceq, float).reshape(-1)])

        def cj(x):
            Jc, Jceq = p.nonlcon_jac(x)
            return np.vstack([np.asarray(Jc, float).reshape(can.nc, p.n), np.asarray(Jceq, float).reshape(can.nceq, p.n)])
        cons.append(NonlinearConstraint(cf, np.concatenate([np.full(can.nc, -INF), np.zeros(can.nceq)]),
                                        np.zeros(can.nc + can.nceq), jac=cj))
    r = minimize(p.fun, x_start, jac=p.grad, bounds=Bounds(can.xl, can.xu), constraints=cons, method="trust-constr",
                 options={"gtol": 1e-12, "xtol": 1e-14, "maxiter": 5000})
    x = _active_set_newton_polish(can, np.asarray(r.x, float), tol=1e-4)
    v = oracle.assess(can, x, feas_tol=1e-9, stat_tol=1e-8)
    if not (v.feasible and v.kkt_first_order_recovered):
        raise RuntimeError(f"{p.name}: reference solve not KKT-verified: {v.as_dict()}")
    return can.f(x), x, (f"{label} (trust-constr for the active set at 1e-4, then Newton on the KKT system of that "
                         f"active set); KKT verified independently (stationarity {v.recovered_stationarity:.1e}); "
                         f"convex, so global")


def _active_set_newton_polish(can, x, tol=1e-6):
    """Newton (scipy.optimize.root) on the KKT system of the active set identified at x: free variables and
    multipliers of the active rows are the unknowns; active bounds are fixed at the bound."""
    from scipy.optimize import root
    n, m = can.n, can.m
    at_lo = np.isfinite(can.xl) & (x - can.xl <= tol * np.maximum(1.0, np.abs(can.xl)))
    at_up = np.isfinite(can.xu) & (can.xu - x <= tol * np.maximum(1.0, np.abs(can.xu)))
    fixed = at_lo | at_up
    xfix = np.where(at_lo, can.xl, np.where(at_up, can.xu, x))
    free = np.flatnonzero(~fixed)
    c = can.cons(x)
    act = np.flatnonzero((np.isfinite(can.cl) & (c - can.cl <= tol * np.maximum(1.0, np.abs(can.cl))))
                         | (np.isfinite(can.cu) & (can.cu - c <= tol * np.maximum(1.0, np.abs(can.cu))))) if m else np.zeros(0, int)
    bnd = np.where(np.isfinite(can.cu) & (can.cu - c <= tol * np.maximum(1.0, np.abs(can.cu))), can.cu, can.cl)[act] if m else np.zeros(0)
    k = act.size

    def full(z):
        xx = xfix.copy()
        xx[free] = z[:free.size]
        return xx, z[free.size:]

    def resid(z):
        xx, lam = full(z)
        g = can.grad(xx)
        J = can.jac(xx)
        r1 = g[free] + (J[act][:, free].T @ lam if k else 0.0)
        r2 = can.cons(xx)[act] - bnd if k else np.zeros(0)
        return np.concatenate([r1, r2])
    z0 = np.concatenate([x[free], np.zeros(k)])
    sol = root(resid, z0, method="lm", tol=1e-15)
    xx, _ = full(sol.x)
    return xx


# ----------------------------------------------------------------------------- problems

def chainrosen20() -> Problem:
    n = 20

    def f(x):
        return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))

    def g(x):
        gr = np.zeros(n)
        d = x[1:] - x[:-1] ** 2
        gr[:-1] += -400.0 * x[:-1] * d - 2.0 * (1.0 - x[:-1])
        gr[1:] += 200.0 * d
        return gr
    x0 = np.full(n, -1.2)
    x0[1::2] = 1.0
    return Problem("chainrosen20", f, x0, g, lb=np.full(n, -2.0), ub=np.full(n, 2.0), data=dict(n=n),
                   target=0.0, target_source="closed form: f = 0 at x = 1", x_ref=np.ones(n),
                   minimum_inputs="fun, x0, lb, ub",
                   story="the reviewer's script: 20-D chained Rosenbrock in a box, alternating start",
                   tags=("bounds-only", "nonconvex"))


def _logistic_data():
    t = np.linspace(0.5, 10.0, 12)
    r, K, y0 = 0.8, 5.0, 0.1
    y = K / (1.0 + (K / y0 - 1.0) * np.exp(-r * t))
    return t, y


def odefit() -> Problem:
    """Fit (r, K, y0) of dy/dt = r y (1 - y/K) to 12 clean samples, the ODE integrated numerically
    (what a user does when the model has no closed form). f* = 0 at the true parameters."""
    from scipy.integrate import solve_ivp
    t, data = _logistic_data()

    def simulate(x):
        r, K, y0 = x
        sol = solve_ivp(lambda _t, y: r * y * (1.0 - y / K), (0.0, t[-1]), [y0], t_eval=t,
                        rtol=1e-10, atol=1e-12, method="RK45")
        return sol.y[0]

    def f(x):
        return float(np.sum((simulate(x) - data) ** 2))

    def g(x):
        # gradient for the oracle from the closed-form logistic solution
        r, K, y0 = x
        e = np.exp(-r * t)
        a = K / y0 - 1.0
        den = 1.0 + a * e
        y = K / den
        dy_dr = K * a * t * e / den ** 2
        dy_dK = 1.0 / den - K * (e / y0) / den ** 2
        dy_dy0 = K * (K / y0 ** 2) * e / den ** 2
        res = y - data
        return np.array([2 * np.sum(res * dy_dr), 2 * np.sum(res * dy_dK), 2 * np.sum(res * dy_dy0)])
    return Problem("odefit", f, np.array([0.3, 2.0, 0.5]), g, data=dict(t=t, data=data),
                   lb=np.array([0.01, 0.5, 0.01]), ub=np.array([5.0, 20.0, 2.0]),
                   target=0.0, target_source="clean data: f = 0 at (r, K, y0) = (0.8, 5, 0.1)",
                   x_ref=np.array([0.8, 5.0, 0.1]), param_tol=1e-3,
                   minimum_inputs="fun (calls an ODE integrator), x0, lb, ub",
                   story="parameter fit through a numerical ODE integrator; the objective carries integrator noise near 1e-10",
                   tags=("bounds-only", "simulator"))


def portfolio_risk() -> Problem:
    """Maximise expected return subject to a quadratic risk cap (nonlinear row), full investment and
    position caps. n = 8, convex."""
    rng = np.random.default_rng(7)
    n = 8
    F = rng.standard_normal((n, 3))
    Sigma = F @ F.T / 3 + np.diag(rng.uniform(0.02, 0.1, n))
    mu = rng.uniform(0.02, 0.12, n)
    cap = 0.3
    # a binding variance budget: the equal-weight variance, which lies between the minimum-variance
    # portfolio's and the unconstrained-return vertex's (checked when the reference is verified)
    risk = float(np.ones(n) @ Sigma @ np.ones(n)) / n ** 2

    def f(x):
        return float(-mu @ x)

    def g(x):
        return -mu

    def nonlcon(x):
        return [x @ Sigma @ x - risk], []

    def nonlcon_jac(x):
        return (2 * Sigma @ x).reshape(1, n), np.zeros((0, n))
    p = Problem("portfolio_risk", f, np.full(n, 1.0 / n), g, Aeq=np.ones((1, n)), beq=np.array([1.0]),
                lb=np.zeros(n), ub=np.full(n, cap), nonlcon=nonlcon, nonlcon_jac=nonlcon_jac,
                data=dict(Sigma=Sigma, mu=mu, risk=risk, cap=cap),
                minimum_inputs="fun, x0, Aeq, beq, lb, ub, nonlcon",
                story="Markowitz with a variance cap as a nonlinear row, a budget equality and position caps",
                tags=("mixed", "convex", "qp-like"))
    p.target, p.x_ref, p.target_source = _kkt_verified_convex_reference(p, p.x0, "independent convex solve")
    return p


def pressure_vessel() -> Problem:
    """The classic design problem with engineering units (thicknesses in inches, radius and length
    in inches, a volume constraint of 1 296 000 in^3). Continuous relaxation."""
    def f(x):
        x1, x2, x3, x4 = x
        return float(0.6224 * x1 * x3 * x4 + 1.7781 * x2 * x3 ** 2 + 3.1661 * x1 ** 2 * x4 + 19.84 * x1 ** 2 * x3)

    def g(x):
        x1, x2, x3, x4 = x
        return np.array([0.6224 * x3 * x4 + 2 * 3.1661 * x1 * x4 + 2 * 19.84 * x1 * x3,
                         1.7781 * x3 ** 2,
                         0.6224 * x1 * x4 + 2 * 1.7781 * x2 * x3 + 19.84 * x1 ** 2,
                         0.6224 * x1 * x3 + 3.1661 * x1 ** 2])

    def nonlcon(x):
        x1, x2, x3, x4 = x
        return [-x1 + 0.0193 * x3, -x2 + 0.00954 * x3,
                -math.pi * x3 ** 2 * x4 - 4.0 / 3.0 * math.pi * x3 ** 3 + 1296000.0, x4 - 240.0], []

    def nonlcon_jac(x):
        x1, x2, x3, x4 = x
        Jc = np.array([[-1.0, 0.0, 0.0193, 0.0], [0.0, -1.0, 0.00954, 0.0],
                       [0.0, 0.0, -2 * math.pi * x3 * x4 - 4 * math.pi * x3 ** 2, -math.pi * x3 ** 2],
                       [0.0, 0.0, 0.0, 1.0]])
        return Jc, np.zeros((0, 4))
    p = Problem("pressure_vessel", f, np.array([1.0, 0.5, 50.0, 100.0]), g,
                lb=np.array([0.0625, 0.0625, 10.0, 10.0]), ub=np.array([99.0, 99.0, 200.0, 200.0]),
                nonlcon=nonlcon, nonlcon_jac=nonlcon_jac,
                minimum_inputs="fun, x0, lb, ub, nonlcon",
                story="design with units: a 1.3e6 volume row next to O(1) thickness rows",
                tags=("nonlinear-ineq", "scaling"))
    # the published digits (Ts, Th, R, L) = (0.778168641, 0.384649163, 40.31961872, 200), f = 5885.3327736, are
    # rounded: the volume row is violated by 3e-4 there. Newton on the KKT system of that active set (rows 1-3 and
    # the upper bound on L) recovers the point to machine precision; f agrees with the published value to 1e-9.
    published = np.array([0.778168641, 0.384649163, 40.31961872, 200.0])
    x_ref = _active_set_newton_polish(p.canonical(), published, tol=1e-3)
    assert abs(f(x_ref) - 5885.3327736) < 1e-5 * 5885.33, f(x_ref)
    p.target, p.x_ref = f(x_ref), x_ref
    p.target_source = "published continuous optimum, polished on its active set (f = 5885.3327736)"
    return p


def nan_region() -> Problem:
    """Objective undefined (NaN) outside a disc the constraint keeps the solution inside of; the
    minimiser is on the constraint circle. Closed form."""
    def f(x):
        r2 = 1.5 - x[0] ** 2 - x[1] ** 2
        if r2 < 0:
            return float("nan")
        return float((x[0] - 0.9) ** 2 + (x[1] - 0.9) ** 2 + 0.1 * math.sqrt(r2))

    def g(x):
        r2 = 1.5 - x[0] ** 2 - x[1] ** 2
        s = 0.1 / (2 * math.sqrt(r2)) if r2 > 0 else float("nan")
        return np.array([2 * (x[0] - 0.9) - 2 * x[0] * s, 2 * (x[1] - 0.9) - 2 * x[1] * s])

    def nonlcon(x):
        return [x[0] ** 2 + x[1] ** 2 - 1.0], []

    def nonlcon_jac(x):
        return np.array([[2 * x[0], 2 * x[1]]]), np.zeros((0, 2))
    s = 1 / math.sqrt(2)
    return Problem("nan_region", f, np.array([0.0, 0.0]), g, nonlcon=nonlcon, nonlcon_jac=nonlcon_jac,
                   target=2 * (s - 0.9) ** 2 + 0.1 * s, target_source="closed form on the unit circle at 45 degrees",
                   x_ref=np.array([s, s]),
                   minimum_inputs="fun, x0, nonlcon",
                   story="a model that returns NaN outside its domain; the constraint keeps the solution inside but probes may not be",
                   tags=("nonlinear-ineq", "nan"))


def hs71() -> Problem:
    def f(x):
        return float(x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2])

    def g(x):
        return np.array([x[3] * (x[0] + x[1] + x[2]) + x[0] * x[3], x[0] * x[3], x[0] * x[3] + 1.0,
                         x[0] * (x[0] + x[1] + x[2])])

    def nonlcon(x):
        return [25.0 - x[0] * x[1] * x[2] * x[3]], [x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2 - 40.0]

    def nonlcon_jac(x):
        return (np.array([[-x[1] * x[2] * x[3], -x[0] * x[2] * x[3], -x[0] * x[1] * x[3], -x[0] * x[1] * x[2]]]),
                np.array([[2 * x[0], 2 * x[1], 2 * x[2], 2 * x[3]]]))
    p = Problem("hs71", f, np.array([1.0, 5.0, 5.0, 1.0]), g, lb=np.ones(4), ub=np.full(4, 5.0),
                nonlcon=nonlcon, nonlcon_jac=nonlcon_jac,
                minimum_inputs="fun, x0, lb, ub, nonlcon",
                story="only nonlcon and bounds; the textbook fmincon example",
                tags=("mixed",))
    x_ref = _active_set_newton_polish(p.canonical(), np.array([1.0, 4.7429994, 3.8211503, 1.3794082]), tol=1e-4)
    assert abs(f(x_ref) - 17.0140173) < 1e-6 * 17.0
    p.target, p.x_ref, p.target_source = f(x_ref), x_ref, "published (Hock-Schittkowski 71), polished on its active set"
    return p


def linear_only() -> Problem:
    """A convex quadratic with A/b, Aeq/beq only (n = 6): a production plan."""
    rng = np.random.default_rng(3)
    n = 6
    L = rng.standard_normal((n, n))
    Q = L @ L.T + 0.5 * np.eye(n)
    c = rng.uniform(1.0, 3.0, n)
    A = np.vstack([rng.uniform(0.0, 1.0, (3, n)), -np.eye(n)])   # 3 resource rows and x >= 0 written as rows
    b = np.concatenate([np.array([2.0, 2.5, 1.5]), np.zeros(n)])
    Aeq = np.ones((1, n))
    beq = np.array([3.0])

    def f(x):
        return float(0.5 * x @ Q @ x - c @ x)

    def g(x):
        return Q @ x - c
    p = Problem("linear_only", f, np.full(n, 0.5), g, A=A, b=b, Aeq=Aeq, beq=beq, data=dict(Q=Q, c=c),
                minimum_inputs="fun, x0, A, b, Aeq, beq",
                story="quadratic cost with linear resource rows, a budget equality and non-negativity given as A rows (no lb)",
                tags=("linear", "convex", "qp-like"))
    p.target, p.x_ref, p.target_source = _kkt_verified_convex_reference(p, p.x0, "independent convex solve")
    return p


def with_args() -> Problem:
    """Exponential-decay fit with the data passed through args. f* = 0 (clean data)."""
    t = np.linspace(0.0, 5.0, 25)
    true = np.array([2.0, 0.5, 0.3])
    d = true[0] * np.exp(-true[1] * t) + true[2]

    def model(x, tt):
        return x[0] * np.exp(-x[1] * tt) + x[2]

    def f(x, tt, dd):
        return float(np.sum((model(x, tt) - dd) ** 2))

    def g(x, tt, dd):
        res = model(x, tt) - dd
        e = np.exp(-x[1] * tt)
        return np.array([2 * np.sum(res * e), 2 * np.sum(res * (-x[0] * tt * e)), 2 * np.sum(res)])
    return Problem("with_args", f, np.array([1.0, 1.0, 1.0]), g, lb=np.array([-INF, 0.0, -INF]), args=(t, d),
                   data=dict(t=t, d=d),
                   target=0.0, target_source="clean data: f = 0 at (2, 0.5, 0.3)", x_ref=true, param_tol=1e-3,
                   minimum_inputs="fun(x, t, d), x0, lb; data via args",
                   story="a fit whose objective takes the data as extra arguments; one one-sided bound",
                   tags=("bounds-only", "fit"))


def wrong_gradient() -> Problem:
    """The user supplies an objective gradient with a sign error in one component.
    The audit asks what each solver does on the first try, not whether it converges."""
    n = 4
    tgt = np.arange(1, n + 1, dtype=float)

    def f(x):
        return float(np.sum((x - tgt) ** 2) + 0.1 * np.sum(x ** 4))

    def g(x):
        return 2 * (x - tgt) + 0.4 * x ** 3

    def wrong(x):
        gr = g(x)
        gr[1] = -gr[1]   # the mistake
        return gr
    # reference: per coordinate 2(x - t) + 0.4 x^3 = 0 (monotone, unique)
    from scipy.optimize import brentq
    x_ref = np.array([brentq(lambda v, t=t: 2 * (v - t) + 0.4 * v ** 3, 0.0, t) for t in tgt])
    return Problem("wrong_gradient", f, np.zeros(n), g, lb=np.full(n, -10.0), ub=np.full(n, 10.0), user_jac=wrong,
                   data=dict(tgt=tgt),
                   target=f(x_ref), target_source="closed form per coordinate (monotone cubic)", x_ref=x_ref,
                   minimum_inputs="fun, x0, lb, ub, plus the (wrong) gradient the user chose to supply",
                   story="a supplied gradient with a sign error: does the solver notice, and what does it say?",
                   tags=("bounds-only", "user-error"))


def infeasible_start_far() -> Problem:
    """Projection onto the simplex from a start 1e3 away from the equality. Closed form."""
    n = 10
    c = np.linspace(-1.0, 2.0, n)

    def f(x):
        return float(0.5 * np.sum((x - c) ** 2))

    def g(x):
        return x - c
    x_ref = _simplex_projection(c)
    return Problem("infeasible_start_far", f, np.full(n, 1e3), g, Aeq=np.ones((1, n)), beq=np.array([1.0]), lb=np.zeros(n),
                   data=dict(c=c),
                   target=f(x_ref), target_source="closed form (sort-based simplex projection)", x_ref=x_ref,
                   minimum_inputs="fun, x0, Aeq, beq, lb",
                   story="a start that violates the budget equality by 1e4; bounds are satisfied",
                   tags=("linear", "convex", "far-start"))


def bad_scaling() -> Problem:
    """Variables of magnitude 1e6 (a pressure, Pa) and 1e-6 (an area, m^2), each mattering equally to the
    objective, coupled by a product constraint x0 x1 >= 5. The Hessian's condition number is 1e25."""
    a, bb = 3e6, 1e-6

    def f(x):
        return float((x[0] / a - 1.0) ** 2 + (x[1] / bb - 1.0) ** 2)

    def g(x):
        return np.array([2 * (x[0] / a - 1.0) / a, 2 * (x[1] / bb - 1.0) / bb])

    def nonlcon(x):
        return [5.0 - x[0] * x[1]], []

    def nonlcon_jac(x):
        return np.array([[-x[1], -x[0]]]), np.zeros((0, 2))
    # reference: the target (3e6, 1e-6) has product 3 < 5, so the row is active: x1 = 5/x0, minimise in 1-D
    from scipy.optimize import minimize_scalar
    r = minimize_scalar(lambda u: (u / a - 1.0) ** 2 + (5.0 / u / bb - 1.0) ** 2, bracket=(2e6, 3e6, 6e6), tol=1e-14)
    x_ref = np.array([r.x, 5.0 / r.x])
    return Problem("bad_scaling", f, np.array([1e6, 1e-6]), g, lb=np.array([1.0, 1e-9]), nonlcon=nonlcon,
                   nonlcon_jac=nonlcon_jac,
                   target=f(x_ref), target_source="closed form on the active product constraint (1-D minimisation)",
                   x_ref=x_ref,
                   minimum_inputs="fun, x0, lb, nonlcon",
                   story="unit mismatch of twelve orders of magnitude between two coupled variables (Hessian condition 1e25)",
                   tags=("nonlinear-ineq", "scaling"))


def noisy_simulator() -> Problem:
    """A smooth 4-D problem plus a deterministic 1e-9 'simulator noise' term, with a linear
    budget row and bounds. Target: the noiseless optimum (closed form)."""
    n = 4
    tgt = np.array([0.7, 0.2, 0.5, 0.9])
    w = np.array([1.0, 3.0, 2.0, 5.0])
    primes = np.array([1.0, 2.0, 3.0, 5.0])
    amp = 1e-9

    def smooth(x):
        return float(np.sum(w * (x - tgt) ** 2))

    def f(x):
        return smooth(x) + amp * math.sin(1e5 * float(primes @ x))

    def g(x):
        return 2 * w * (x - tgt)          # oracle: gradient of the smooth part
    # reference: min sum w (x - tgt)^2 s.t. sum x <= 2 (active: sum tgt = 2.3), 0 <= x <= 1: closed form via KKT
    # x_i = tgt_i - lam / (2 w_i), lam from sum x = 2 (interior of the box, checked below)
    lam = (np.sum(tgt) - 2.0) / np.sum(1.0 / (2 * w))
    x_ref = tgt - lam / (2 * w)
    assert np.all(x_ref > 0) and np.all(x_ref < 1)
    return Problem("noisy_simulator", f, np.full(n, 0.25), g, A=np.ones((1, n)), b=np.array([2.0]), lb=np.zeros(n),
                   ub=np.ones(n), data=dict(tgt=tgt, w=w, primes=primes, amp=amp),
                   target=smooth(x_ref), target_source="closed form of the noiseless problem (KKT on the active budget row)",
                   x_ref=x_ref,
                   minimum_inputs="fun, x0, A, b, lb, ub",
                   story="an objective with deterministic noise of 1e-9 (forward-difference gradient error about 0.07)",
                   tags=("linear", "noise"))


def box_lsq() -> Problem:
    """Bounded least-squares deconvolution: min 0.5|Kx - y|^2 with 0 <= x <= 1, n = 50 (a bound-constrained
    convex QP with a dense Hessian). Reference: bounded-variable least squares (BVLS), independent."""
    n = 50
    rng = np.random.default_rng(11)
    i = np.arange(n)
    K = np.exp(-0.5 * ((i[:, None] - i[None, :]) / 3.0) ** 2)
    K /= K.sum(axis=1, keepdims=True)
    x_true = np.zeros(n)
    x_true[10:18] = 0.8
    x_true[30:33] = 1.0
    x_true[40] = 0.6
    y = K @ x_true + 0.02 * rng.standard_normal(n)

    def f(x):
        r = K @ x - y
        return float(0.5 * r @ r)

    def g(x):
        return K.T @ (K @ x - y)
    from scipy.optimize import lsq_linear
    ref = lsq_linear(K, y, bounds=(0.0, 1.0), method="bvls", tol=1e-14, max_iter=10000)
    x_ref = ref.x
    return Problem("box_lsq", f, np.full(n, 0.5), g, lb=np.zeros(n), ub=np.ones(n), data=dict(K=K, y=y),
                   target=f(x_ref), target_source="independent: bounded-variable least squares (scipy lsq_linear, bvls)",
                   x_ref=x_ref,
                   minimum_inputs="fun, x0, lb, ub",
                   story="a 50-variable bound-constrained least-squares deconvolution (convex QP, dense Hessian, many active bounds)",
                   tags=("bounds-only", "convex", "qp-like"))


def equality_circle() -> Problem:
    """Two nonlinear equalities and one inequality in 3-D (a reach problem): keep the hand on a
    sphere and a plane, maximise height with a small effort term. Closed form."""
    def f(x):
        return float(-x[2] + 0.1 * (x[0] ** 2 + x[1] ** 2))

    def g(x):
        return np.array([0.2 * x[0], 0.2 * x[1], -1.0])

    def nonlcon(x):
        return [x[0] - 0.3], [x[0] ** 2 + x[1] ** 2 + x[2] ** 2 - 1.0, x[0] + x[1] - 0.5]

    def nonlcon_jac(x):
        return np.array([[1.0, 0.0, 0.0]]), np.array([[2 * x[0], 2 * x[1], 2 * x[2]], [1.0, 1.0, 0.0]])
    # reference: on the plane x0 + x1 = 0.5 and the sphere, x2 = sqrt(1 - x0^2 - x1^2) is largest and the
    # effort term smallest where x0^2 + x1^2 is smallest on the plane: x0 = x1 = 0.25 (x0 <= 0.3 inactive)
    x_ref = np.array([0.25, 0.25, math.sqrt(1 - 2 * 0.25 ** 2)])
    return Problem("equality_circle", f, np.array([0.5, 0.5, 0.5]), g, nonlcon=nonlcon, nonlcon_jac=nonlcon_jac,
                   target=f(x_ref), target_source="closed form (symmetry on the plane)", x_ref=x_ref,
                   minimum_inputs="fun, x0, nonlcon",
                   story="two nonlinear equalities with an inactive inequality, start infeasible on both equalities",
                   tags=("nonlinear-eq",))


REGISTRY = {p.__name__: p for p in [chainrosen20, odefit, portfolio_risk, pressure_vessel, nan_region, hs71, linear_only,
                                      with_args, wrong_gradient, infeasible_start_far, bad_scaling, noisy_simulator,
                                      box_lsq, equality_circle]}


def get(name: str) -> Problem:
    return REGISTRY[name]()


def all_names():
    return list(REGISTRY)


def export_data(path):
    """Write every problem's arrays, start, bounds and reference to JSON for the MATLAB runner."""
    import json
    out = {}
    for name in all_names():
        p = get(name)
        can = p.canonical()
        d = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in p.data.items()}
        d.update(x0=p.x0.tolist(), lb=[None if not np.isfinite(v) else v for v in can.xl],
                 ub=[None if not np.isfinite(v) else v for v in can.xu],
                 A=None if p.A is None else np.asarray(p.A, float).tolist(), b=None if p.b is None else np.asarray(p.b, float).reshape(-1).tolist(),
                 Aeq=None if p.Aeq is None else np.asarray(p.Aeq, float).tolist(),
                 beq=None if p.beq is None else np.asarray(p.beq, float).reshape(-1).tolist(),
                 target=p.target, x_ref=p.x_ref.tolist(), f0=can.f(p.x0), c0=can.cons(p.x0).tolist())
        out[name] = d
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1)
    return out
