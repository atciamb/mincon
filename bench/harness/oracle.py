"""Independent optimality oracle.

Given a corpus problem (exact derivatives), a returned point and, optionally,
returned multipliers, compute — outside any solver, on the original unscaled
model — the quantities a referee would ask for:

* validity: finite, right shape;
* primal violation (bounds and rows, absolute, in model units);
* stationarity residual  r = grad f + J^T lam - zL + zU  (inf-norm), raw and
  relative to (1 + |grad f|_inf + |J^T lam|_inf);
* multiplier sign violations (lam_i must be >= 0 on an active upper side,
  <= 0 on an active lower side; zL, zU >= 0) and complementarity products;
* an independently recovered least-squares multiplier set from the active
  set (non-negative least squares), with its own residual, when the solver
  supplied no multipliers;
* target attainment against a frozen target value.

Conventions: rows are cl <= c(x) <= cu; stationarity uses the IPOPT/AMPL sign
(lam > 0 pushes against the upper side). fmincon's lambda.ineqnonlin (c <= 0)
maps onto that sign directly; its lambda.eqnonlin sign is determined by the
oracle by trying both (documented in `from_fmincon`). Nothing here trusts a
solver's success flag.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Verdict:
    valid: bool
    f: float
    violation: float
    feasible: bool
    stationarity: float
    stationarity_rel: float
    complementarity: float
    dual_sign_violation: float
    multipliers_supplied: bool
    recovered_stationarity: float
    recovered_stationarity_rel: float
    recovered_cond: float
    lam_norm: float
    active_rows: int
    active_bounds: int
    kkt_first_order: bool
    kkt_first_order_recovered: bool
    target: float | None
    target_attained: bool | None
    target_gap: float | None
    notes: list[str]

    def as_dict(self) -> dict:
        return asdict(self)


def assess(problem, x, lam=None, zl=None, zu=None, *, feas_tol=1e-6, stat_tol=1e-6, target=None,
           target_rel_tol=1e-4, active_tol=None) -> Verdict:
    notes: list[str] = []
    n, m = problem.n, problem.m
    try:
        x = np.asarray(x, dtype=float).reshape(-1)
    except Exception:  # noqa: BLE001
        return _invalid("point is not a numeric vector", target)
    if x.size != n or not np.all(np.isfinite(x)):
        return _invalid(f"point has size {x.size} (expected {n}) or non-finite entries", target)
    try:
        f = float(problem.f(x))
        g = np.asarray(problem.grad(x), float)
        c = np.asarray(problem.cons(x), float)
        J = np.asarray(problem.jac(x), float).reshape(m, n)
    except Exception as exc:  # noqa: BLE001
        return _invalid(f"model evaluation failed at the returned point: {exc}", target)
    if not (np.isfinite(f) and np.all(np.isfinite(g)) and np.all(np.isfinite(c)) and np.all(np.isfinite(J))):
        return _invalid("model or derivatives non-finite at the returned point", target)

    xl, xu, cl, cu = problem.xl, problem.xu, problem.cl, problem.cu
    viol = float(problem.violation(x))
    feasible = viol <= feas_tol
    if active_tol is None:
        active_tol = max(feas_tol, 1e-6) * 10
    act_lo_x = np.isfinite(xl) & (x - xl <= active_tol * np.maximum(1.0, np.abs(xl)))
    act_up_x = np.isfinite(xu) & (xu - x <= active_tol * np.maximum(1.0, np.abs(xu)))
    act_lo_c = np.isfinite(cl) & (c - cl <= active_tol * np.maximum(1.0, np.abs(cl))) if m else np.zeros(0, bool)
    act_up_c = np.isfinite(cu) & (cu - c <= active_tol * np.maximum(1.0, np.abs(cu))) if m else np.zeros(0, bool)
    is_eq = (cl == cu) if m else np.zeros(0, bool)

    supplied = lam is not None
    stat = stat_rel = compl = sign_viol = lam_norm = float("nan")
    if supplied:
        lam = np.asarray(lam, float).reshape(-1)
        zl = np.zeros(n) if zl is None else np.asarray(zl, float).reshape(-1)
        zu = np.zeros(n) if zu is None else np.asarray(zu, float).reshape(-1)
        if lam.size != m or zl.size != n or zu.size != n or not (np.all(np.isfinite(lam)) and np.all(np.isfinite(zl)) and np.all(np.isfinite(zu))):
            notes.append("supplied multipliers have the wrong size or are non-finite; treated as absent")
            supplied = False
    if supplied:
        jt_lam = J.T @ lam if m else np.zeros(n)
        r = g + jt_lam - zl + zu
        stat = float(np.max(np.abs(r)))
        stat_rel = stat / (1.0 + float(np.max(np.abs(g))) + (float(np.max(np.abs(jt_lam))) if m else 0.0))
        lam_norm = float(max(np.max(np.abs(lam)) if m else 0.0, np.max(np.abs(zl)), np.max(np.abs(zu))))
        # signs: zl, zu >= 0; lam_i >= 0 unless the lower side is the active one (then <= 0); free for equalities.
        sign_viol = float(max(np.max(np.maximum(-zl, 0.0)), np.max(np.maximum(-zu, 0.0))))
        if m:
            neg_bad = np.maximum(-lam, 0.0) * (~act_lo_c & ~is_eq)   # negative where lower side is not active
            pos_bad = np.maximum(lam, 0.0) * (~act_up_c & ~is_eq)    # positive where upper side is not active
            sign_viol = float(max(sign_viol, np.max(neg_bad), np.max(pos_bad)))
        # complementarity: products with the slack to whichever side the multiplier pushes against
        cp = [float(np.max(np.abs(zl * np.where(np.isfinite(xl), x - xl, 0.0)))),
              float(np.max(np.abs(zu * np.where(np.isfinite(xu), xu - x, 0.0))))]
        if m:
            slack = np.where(lam >= 0, np.where(np.isfinite(cu), cu - c, 0.0), np.where(np.isfinite(cl), c - cl, 0.0))
            slack = np.where(is_eq, 0.0, slack)
            cp.append(float(np.max(np.abs(lam * slack))))
        compl = max(cp)

    # independent recovery: NNLS on active gradients with sign structure
    rec_stat, rec_rel, rec_cond = _recover(g, J, act_lo_x, act_up_x, act_lo_c, act_up_c, is_eq, n, m)
    active_rows = int(np.sum(act_lo_c | act_up_c | is_eq)) if m else 0
    active_bounds = int(np.sum(act_lo_x | act_up_x))

    kkt = bool(supplied and feasible and stat <= stat_tol and sign_viol <= stat_tol and compl <= max(stat_tol, feas_tol) * max(1.0, lam_norm))
    kkt_rec = bool(feasible and rec_rel <= stat_tol)

    t_att = t_gap = None
    if target is not None and math.isfinite(target):
        t_gap = (f - target) / max(1.0, abs(target))
        t_att = bool(feasible and t_gap <= target_rel_tol)
        if feasible and t_gap < -10 * target_rel_tol:
            notes.append(f"returned objective beats the frozen target by {-t_gap:.2e} relative: target must be revised (versioned)")
    return Verdict(True, f, viol, feasible, stat, stat_rel, compl, sign_viol, supplied, rec_stat, rec_rel, rec_cond,
                   lam_norm, active_rows, active_bounds, kkt, kkt_rec, target, t_att, t_gap, notes)


def _recover(g, J, act_lo_x, act_up_x, act_lo_c, act_up_c, is_eq, n, m):
    """Least-squares multipliers on the active set: min |g + B y| with y >= 0 on inequality columns
    (columns are signed so every inequality multiplier is non-negative) and free on equality columns."""
    cols = []
    free = []
    for j in np.flatnonzero(act_lo_x):
        e = np.zeros(n); e[j] = -1.0; cols.append(e); free.append(False)     # -zL
    for j in np.flatnonzero(act_up_x):
        e = np.zeros(n); e[j] = 1.0; cols.append(e); free.append(False)      # +zU
    for i in range(m):
        if is_eq[i]:
            cols.append(J[i]); free.append(True)
        else:
            if act_up_c[i]:
                cols.append(J[i]); free.append(False)                        # lam >= 0 on the upper side
            if act_lo_c[i]:
                cols.append(-J[i]); free.append(False)                       # lam <= 0 on the lower side
    gn = float(np.max(np.abs(g)))
    if not cols:
        return gn, gn / (1.0 + gn), 1.0
    B = np.stack(cols, axis=1)
    free = np.asarray(free)
    # split free columns into +/- pairs so a single NNLS handles both
    Bp = np.concatenate([B, -B[:, free]], axis=1) if free.any() else B
    try:
        from scipy.optimize import nnls
        y, _ = nnls(Bp, -g, maxiter=50 * Bp.shape[1])
    except Exception:  # noqa: BLE001
        y = np.linalg.lstsq(Bp, -g, rcond=None)[0]
    r = g + Bp @ y
    stat = float(np.max(np.abs(r)))
    rel = stat / (1.0 + gn + float(np.max(np.abs(Bp @ y))))
    sv = np.linalg.svd(B, compute_uv=False)
    cond = float(sv[0] / sv[-1]) if sv.size and sv[-1] > 0 else float("inf")
    return stat, rel, cond


def _invalid(msg, target):
    nan = float("nan")
    return Verdict(False, nan, float("inf"), False, nan, nan, nan, nan, False, nan, nan, nan, nan, 0, 0, False, False,
                   target, False if target is not None else None, None, [msg])


def from_fmincon(problem, x, lam_struct):
    """Map fmincon's lambda struct (from a run of the canonical translation in bench/run_matlab.m:
    rows split as c - cu <= 0 (upper), cl - c <= 0 (lower), c - cl = 0 (eq)) onto canonical multipliers.
    The translation script records the row order it used; see run_matlab.m."""
    raise NotImplementedError("use the row map emitted by run_matlab.m; see adapters.py")
