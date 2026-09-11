"""Second-order classification of returned points (the basin study, docs/19 §4 S-B).

For every scored record that did not attain its target but returned a valid point, this
recomputes the KKT residual with the corpus' exact derivatives, recovers sign-structured
multipliers on the active set, forms the Hessian of the Lagrangian and reports the smallest
eigenvalue of its projection onto the null space of the active constraint gradients:

  * on all active constraints (a necessary condition for a strict local minimum), and
  * on the strongly active ones only (|multiplier| > tol; positive definiteness there is
    sufficient when there are no weakly active constraints).

The same is done for the published reference point when the spec carries one, so the target
itself is checked. Classification:

  strict-local-min   KKT holds and the projected Hessian is positive definite
  degenerate-kkt     KKT holds, projected Hessian singular or a weakly active constraint
                     leaves the cone larger than the subspace checked
  saddle-or-max      KKT holds, projected Hessian has a negative eigenvalue
  not-kkt            the recovered multipliers leave a stationarity residual above tol
  infeasible         the point violates the constraints
  budget             the record ended on its evaluation or time budget

Usage: python -m bench.harness.second_order RESULT_DIR... [--solver S] [--all] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "corpus"))
import spec as corpus  # noqa: E402
import families  # noqa: E402,F401


def null_space(A: np.ndarray, n: int, rtol: float = 1e-9) -> np.ndarray:
    if A.size == 0:
        return np.eye(n)
    u, s, vt = np.linalg.svd(A, full_matrices=True)
    rank = int(np.sum(s > rtol * max(1.0, s[0] if s.size else 1.0)))
    return vt[rank:].T


def recovered_multipliers(p, x, g, J, active_tol):
    """Sign-structured least-squares multipliers on the active set (same convention as the oracle)."""
    n, m = p.n, p.m
    c = p.cons(x)
    xl, xu, cl, cu = p.xl, p.xu, p.cl, p.cu
    act_lo_x = np.isfinite(xl) & (x - xl <= active_tol * np.maximum(1.0, np.abs(xl)))
    act_up_x = np.isfinite(xu) & (xu - x <= active_tol * np.maximum(1.0, np.abs(xu)))
    if m:
        act_lo_c = np.isfinite(cl) & (c - cl <= active_tol * np.maximum(1.0, np.abs(cl)))
        act_up_c = np.isfinite(cu) & (cu - c <= active_tol * np.maximum(1.0, np.abs(cu)))
        is_eq = cl == cu
    else:
        act_lo_c = act_up_c = is_eq = np.zeros(0, bool)
    cols, kinds = [], []
    for j in np.flatnonzero(act_lo_x):
        e = np.zeros(n); e[j] = -1.0; cols.append(e); kinds.append(("xl", j, False))
    for j in np.flatnonzero(act_up_x):
        e = np.zeros(n); e[j] = 1.0; cols.append(e); kinds.append(("xu", j, False))
    for i in range(m):
        if is_eq[i]:
            cols.append(J[i]); kinds.append(("eq", i, True))
        else:
            if act_up_c[i]:
                cols.append(J[i]); kinds.append(("cu", i, False))
            if act_lo_c[i]:
                cols.append(-J[i]); kinds.append(("cl", i, False))
    if not cols:
        return np.zeros(m), np.zeros((0, n)), np.zeros((0, n)), float(np.max(np.abs(g))), []
    B = np.stack(cols, axis=1)
    free = np.asarray([k[2] for k in kinds])
    Bp = np.concatenate([B, -B[:, free]], axis=1) if free.any() else B
    from scipy.optimize import nnls
    try:
        y, _ = nnls(Bp, -g, maxiter=50 * Bp.shape[1])
    except Exception:  # noqa: BLE001
        y = np.linalg.lstsq(Bp, -g, rcond=None)[0]
    if free.any():
        yf = y[:B.shape[1]].copy()
        yf[free] -= y[B.shape[1]:]
    else:
        yf = y
    resid = float(np.max(np.abs(g + B @ yf)))
    lam = np.zeros(m)
    for (kind, i, _), v in zip(kinds, yf):
        if kind in ("eq", "cu"):
            lam[i] += v
        elif kind == "cl":
            lam[i] -= v
    # active gradient rows (in the original orientation) and strongly active subset
    A_all = np.stack([c_ if k[0] not in ("xl", "cl") else -c_ for c_, k in zip(cols, kinds)], axis=0)
    strong = [abs(v) > 1e-8 * max(1.0, float(np.max(np.abs(g)))) or k[2] for k, v in zip(kinds, yf)]
    A_strong = A_all[np.asarray(strong, bool)] if any(strong) else np.zeros((0, n))
    return lam, A_all, A_strong, resid, [(k[0], int(k[1]), float(v)) for k, v in zip(kinds, yf)]


def classify(p, x, *, feas_tol=1e-6, stat_tol=1e-6, active_tol=1e-5):
    x = np.asarray(x, float)
    out = {"f": p.f(x), "violation": float(p.violation(x))}
    if out["violation"] > feas_tol:
        out["class"] = "infeasible"
        return out
    g, J = p.grad(x), p.jac(x)
    lam, A_all, A_strong, resid, mults = recovered_multipliers(p, x, g, J, active_tol)
    gscale = 1.0 + float(np.max(np.abs(g))) + (float(np.max(np.abs(J.T @ lam))) if p.m else 0.0)
    out.update(stationarity=resid, stationarity_rel=resid / gscale, n_active=int(A_all.shape[0]),
               n_strong=int(A_strong.shape[0]), multipliers=mults)
    if resid / gscale > stat_tol:
        out["class"] = "not-kkt"
        return out
    H = p.hess_lagrangian(x, lam)
    Z_all = null_space(A_all, p.n)
    Z_str = null_space(A_strong, p.n)
    ev_all = np.linalg.eigvalsh(Z_all.T @ H @ Z_all) if Z_all.shape[1] else np.array([np.inf])
    ev_str = np.linalg.eigvalsh(Z_str.T @ H @ Z_str) if Z_str.shape[1] else np.array([np.inf])
    hs = max(1.0, float(np.max(np.abs(H))))
    out.update(min_eig_all=float(ev_all.min()), min_eig_strong=float(ev_str.min()),
               dim_null_all=int(Z_all.shape[1]), dim_null_strong=int(Z_str.shape[1]), hess_scale=hs)
    weak = A_all.shape[0] > A_strong.shape[0]
    if ev_str.min() > 1e-8 * hs and (not weak or ev_all.min() > 1e-8 * hs):
        out["class"] = "strict-local-min"
    elif ev_str.min() < -1e-6 * hs:
        out["class"] = "saddle-or-max"
    else:
        out["class"] = "degenerate-kkt"
    if weak:
        out["weakly_active"] = A_all.shape[0] - A_strong.shape[0]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--solver", action="append")
    ap.add_argument("--all", action="store_true", help="classify attained records too")
    ap.add_argument("--track", default="A")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    rows = []
    for d in a.dirs:
        path = os.path.join(d, f"scored-{a.track}.jsonl")
        for line in open(path):
            r = json.loads(line)
            r["_dir"] = os.path.basename(d.rstrip("/"))
            rows.append(r)
    results = []
    ref_done = set()
    for r in rows:
        if r.get("repeat", 0) != 0:
            continue
        if a.solver and r["solver"] not in a.solver:
            continue
        o = r.get("oracle") or {}
        if o.get("target") is None:
            continue
        if o.get("target_attained") and not a.all:
            continue
        if not r.get("x"):
            continue
        p = corpus.get(r["problem"]).numpy()
        try:
            cls = classify(p, r["x"])
        except Exception as exc:  # noqa: BLE001
            cls = {"class": f"error: {exc}"}
        budget = r.get("counts", {}).get("f_model", 0) >= 0.99 * r.get("budget", {}).get("maxfev", 1e18) or \
            r.get("native_status") == 0
        if budget and cls.get("class") in ("infeasible", "not-kkt"):
            cls["class"] = "budget/" + cls["class"]
        results.append(dict(problem=r["problem"], solver=r["solver"], dir=r["_dir"], n=r["n"], m=r["m"],
                            evals=r["counts"]["f_model"], nit=r.get("nit"), status=r.get("native_status"),
                            target=o.get("target"), attained=o.get("target_attained"), **cls))
        if r["problem"] not in ref_done and p.ref_x is not None:
            ref_done.add(r["problem"])
            for k, xr in enumerate(p.ref_x):   # ref_x is a list of known minimizers
                try:
                    rc = classify(p, np.asarray(xr, float))
                except Exception as exc:  # noqa: BLE001
                    rc = {"class": f"error: {exc}"}
                results.append(dict(problem=r["problem"], solver=f"REFERENCE[{k}]", dir="spec", n=r["n"], m=r["m"],
                                    evals=0, nit=0, status=None, target=o.get("target"), attained=None, **rc))
    results.sort(key=lambda z: (z["problem"], not z["solver"].startswith("REFERENCE"), z["solver"]))
    for z in results:
        me = z.get("min_eig_strong")
        me_s = f"{me:.3g}" if me is not None else "-"
        mea = z.get("min_eig_all")
        mea_s = f"{mea:.3g}" if mea is not None else "-"
        st = z.get("stationarity_rel")
        st_s = f"{st:.1e}" if st is not None else "-"
        print(f"{z['problem']:20s} {z['solver']:24s} f={z.get('f', float('nan')):12.6g} target={z['target']:12.6g} "
              f"viol={z.get('violation', float('nan')):.1e} stat={st_s:8s} act={z.get('n_active','-')}/{z.get('n_strong','-')} "
              f"eig(strong)={me_s:9s} eig(all)={mea_s:9s} -> {z['class']}")
    if a.out:
        with open(a.out, "w") as fh:
            for z in results:
                fh.write(json.dumps(z, default=float) + "\n")


if __name__ == "__main__":
    main()


def newton_polish(p, x, *, iters=30, active_tol=1e-4):
    """Lagrange–Newton iterations on the active set identified at x (exact derivatives).

    Returns (x_polished, info). The active set is frozen from the starting point, so this is a local
    polish, not a solve: it answers "is there a KKT point of this active set within a small distance",
    and reports whether the polished point is still feasible with correctly signed multipliers."""
    x = np.asarray(x, float).copy()
    n, m = p.n, p.m
    g, J = p.grad(x), p.jac(x)
    lam, A_all, A_strong, resid, mults = recovered_multipliers(p, x, g, J, active_tol)
    kinds = [(k, i) for k, i, _ in mults]
    # rows of A (oriented as gradients of the active quantities) and their targets
    rows, rhs = [], []
    for k, i in kinds:
        if k == "xl":
            e = np.zeros(n); e[i] = 1.0; rows.append(e); rhs.append(("x", i, p.xl[i]))
        elif k == "xu":
            e = np.zeros(n); e[i] = 1.0; rows.append(e); rhs.append(("x", i, p.xu[i]))
        elif k in ("eq", "cu"):
            rows.append(None); rhs.append(("c", i, p.cu[i]))
        else:
            rows.append(None); rhs.append(("c", i, p.cl[i]))
    q = len(rhs)
    mu = np.zeros(q)
    for it in range(iters):
        g, J, c = p.grad(x), p.jac(x), p.cons(x)
        A = np.zeros((q, n)); r = np.zeros(q)
        for k, (kind, i, tgt) in enumerate(rhs):
            if kind == "x":
                A[k, i] = 1.0; r[k] = x[i] - tgt
            else:
                A[k] = J[i]; r[k] = c[i] - tgt
        # least-squares multipliers for the current point, then Newton on the KKT system
        if q:
            mu = np.linalg.lstsq(A.T, -g, rcond=None)[0]
        lamfull = np.zeros(m)
        for k, (kind, i, tgt) in enumerate(rhs):
            if kind == "c":
                lamfull[i] += mu[k]
        H = p.hess_lagrangian(x, lamfull)
        K = np.block([[H, A.T], [A, np.zeros((q, q))]]) if q else H
        rhs_vec = -np.concatenate([g + (A.T @ mu if q else 0), r]) if q else -g
        try:
            sol = np.linalg.lstsq(K, rhs_vec, rcond=1e-12)[0]
        except np.linalg.LinAlgError:
            break
        dx = sol[:n]
        step = 1.0
        if np.max(np.abs(dx)) > 1.0:
            step = 1.0 / np.max(np.abs(dx))
        x = x + step * dx
        if q:
            mu = mu + step * sol[n:]
        if np.max(np.abs(dx)) < 1e-14 * max(1.0, np.max(np.abs(x))):
            break
    info = classify(p, x, stat_tol=1e-8, active_tol=active_tol)
    info["polish_iters"] = it + 1
    return x, info
