"""Friction audit runner for the Python solvers (study S-D).

    python run.py --out ../results/s7-friction [--solvers mincon-fmincon,scipy-slsqp,scipy-trust-constr] [--problems a,b]

Every solver gets the problem exactly as a user would type it on the first try: the objective, the
start, and the MATLAB-style constraint pieces, translated into that solver's own calling convention
with no options, no derivatives (except ``wrong_gradient``, whose point is the wrong derivative the
user supplied) and no tolerances. Counting is at the model boundary through a one-point cache, as in
``bench/harness/model.py``. A crash is a record (``outcome = "error"``), not a missing row. Every
returned point is judged by the independent oracle (feasible to 1e-6, objective within 1e-4 relative
of the frozen reference); nothing trusts a solver's own success flag.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
import traceback
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "harness"))
import oracle  # noqa: E402
import problems  # noqa: E402


class Counting:
    """One-point cache in front of the user's objective and nonlcon; counts model evaluations and crossings."""

    def __init__(self, p: problems.Problem):
        self.p = p
        self.counts = dict(f_model=0, c_model=0, f_calls=0, c_calls=0, nonfinite_f=0, nonfinite_c=0)
        self.seconds = 0.0
        self._fx = self._fv = self._cx = self._cv = None

    def fun(self, x, *args):
        self.counts["f_calls"] += 1
        x = np.asarray(x, float)
        if self._fx is not None and x.shape == self._fx.shape and np.array_equal(x, self._fx):
            return self._fv
        t = time.perf_counter()
        v = float(self.p.fun(x, *args))
        self.seconds += time.perf_counter() - t
        self.counts["f_model"] += 1
        if not math.isfinite(v):
            self.counts["nonfinite_f"] += 1
        self._fx, self._fv = x.copy(), v
        return v

    def nonlcon(self, x, *args):
        self.counts["c_calls"] += 1
        x = np.asarray(x, float)
        if self._cx is not None and np.array_equal(x, self._cx):
            return self._cv
        t = time.perf_counter()
        c, ceq = self.p.nonlcon(x, *args)
        self.seconds += time.perf_counter() - t
        self.counts["c_model"] += 1
        c = np.asarray(c, float).reshape(-1)
        ceq = np.asarray(ceq, float).reshape(-1)
        if not (np.all(np.isfinite(c)) and np.all(np.isfinite(ceq))):
            self.counts["nonfinite_c"] += 1
        self._cx, self._cv = x.copy(), (c, ceq)
        return c, ceq


def _bounds_arrays(p):
    can = p.canonical()
    return can.xl, can.xu


MINCON_OPTIONS: dict = {}   # set from --options; the default audit passes nothing


def run_mincon_fmincon(p, cnt):
    import mincon
    kw = dict(A=p.A, b=p.b, Aeq=p.Aeq, beq=p.beq, lb=p.lb, ub=p.ub,
              nonlcon=cnt.nonlcon if p.nonlcon is not None else None, args=p.args)
    if p.user_jac is not None:
        kw["jac"] = p.user_jac
    if MINCON_OPTIONS:
        kw["options"] = dict(MINCON_OPTIONS)
    r = mincon.fmincon(cnt.fun, p.x0, **kw)
    return dict(x=np.asarray(r.x, float), f=float(r.fun), status=int(r.status), message=str(r.message),
                reported_success=bool(r.success), usable=bool(r.usable), nit=int(r.nit), nfev_reported=int(r.nfev),
                notes=list(r.notes), algorithm=str(r.get("algorithm", "")), solver_time=float(r.time),
                model_time=float(r.model_time), version=mincon.__version__)


def run_scipy(p, cnt, method):
    import scipy
    from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, minimize
    xl, xu = _bounds_arrays(p)
    bounds = None if (np.all(~np.isfinite(xl)) and np.all(~np.isfinite(xu))) else Bounds(xl, xu)
    na, ne, nc, nceq = p.sizes()
    cons = []
    if method == "SLSQP":
        # what a SciPy user writes: dicts, with fmincon's c <= 0 flipped to SciPy's fun >= 0
        if na:
            A, b = np.asarray(p.A, float), np.asarray(p.b, float).reshape(-1)
            cons.append({"type": "ineq", "fun": lambda x: b - A @ x})
        if ne:
            Aeq, beq = np.asarray(p.Aeq, float), np.asarray(p.beq, float).reshape(-1)
            cons.append({"type": "eq", "fun": lambda x: Aeq @ x - beq})
        if nc:
            cons.append({"type": "ineq", "fun": lambda x, *a: -cnt.nonlcon(x, *a)[0], "args": p.args})
        if nceq:
            cons.append({"type": "eq", "fun": lambda x, *a: cnt.nonlcon(x, *a)[1], "args": p.args})
        r = minimize(cnt.fun, p.x0, args=p.args, jac=p.user_jac, bounds=bounds, constraints=cons, method="SLSQP")
    else:
        if na or ne:
            rows_a = np.asarray(p.A, float).reshape(na, p.n) if na else np.zeros((0, p.n))
            rows_e = np.asarray(p.Aeq, float).reshape(ne, p.n) if ne else np.zeros((0, p.n))
            beq = np.asarray(p.beq, float).reshape(-1) if ne else np.zeros(0)
            lo = np.concatenate([np.full(na, -np.inf), beq])
            hi = np.concatenate([np.asarray(p.b, float).reshape(-1) if na else np.zeros(0), beq])
            cons.append(LinearConstraint(np.vstack([rows_a, rows_e]), lo, hi))
        if nc or nceq:
            def cf(x):
                c, ceq = cnt.nonlcon(x, *p.args)
                return np.concatenate([c, ceq])
            cons.append(NonlinearConstraint(cf, np.concatenate([np.full(nc, -np.inf), np.zeros(nceq)]), np.zeros(nc + nceq)))
        r = minimize(cnt.fun, p.x0, args=p.args, jac=p.user_jac, bounds=bounds, constraints=cons, method="trust-constr")
    return dict(x=np.asarray(r.x, float), f=float(r.fun), status=int(getattr(r, "status", -99)), message=str(r.message),
                reported_success=bool(r.success), usable=None, nit=int(getattr(r, "nit", -1)),
                nfev_reported=int(getattr(r, "nfev", -1)), notes=[], algorithm=method, solver_time=float("nan"),
                model_time=float("nan"), version=scipy.__version__)


SOLVERS = {
    "mincon-fmincon": run_mincon_fmincon,
    "scipy-slsqp": lambda p, c: run_scipy(p, c, "SLSQP"),
    "scipy-trust-constr": lambda p, c: run_scipy(p, c, "trust-constr"),
}


def judge(p, x):
    """Independent verdict on a returned point."""
    can = p.canonical()
    v = oracle.assess(can, x, feas_tol=1e-6, stat_tol=1e-6, target=p.target, target_rel_tol=1e-4)
    d = v.as_dict()
    out = dict(valid=d["valid"], f=d["f"], violation=d["violation"], feasible=d["feasible"],
               kkt_recovered=d["kkt_first_order_recovered"], recovered_stationarity=d["recovered_stationarity"],
               target_attained=d["target_attained"], target_gap=d["target_gap"], oracle_notes=d["notes"])
    if p.x_ref is not None and d["valid"]:
        err = float(np.max(np.abs(np.asarray(x, float) - p.x_ref)))
        out["x_err"] = err
        if p.param_tol is not None:
            out["param_ok"] = err <= p.param_tol
    return out


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, float)):
        return o if math.isfinite(o) else str(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(obj) else str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def run_one(name, solver):
    p = problems.get(name)
    cnt = Counting(p)
    rec = dict(schema="mincon-friction-record/1", problem=name, solver=solver, n=p.n, sizes=p.sizes(),
               minimum_inputs=p.minimum_inputs, story=p.story, tags=list(p.tags), target=p.target,
               target_source=p.target_source, user_supplied_gradient=p.user_jac is not None,
               started=time.time())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t0 = time.perf_counter()
        try:
            out = SOLVERS[solver](p, cnt)
            rec["outcome"] = "ok"
        except Exception as exc:  # noqa: BLE001 - a crash is the data point
            out = None
            rec["outcome"] = "error"
            rec["error"] = f"{type(exc).__name__}: {exc}"
            rec["traceback"] = traceback.format_exc()[-1500:]
        rec["wall"] = time.perf_counter() - t0
    rec["warnings"] = sorted({f"{w.category.__name__}: {w.message}" for w in caught})[:10]
    rec["counts"] = dict(cnt.counts)
    rec["callback_seconds"] = cnt.seconds
    if out is not None:
        rec.update(out)
        rec["verdict"] = judge(p, out["x"])
    return _clean(rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--solvers", default=",".join(SOLVERS))
    ap.add_argument("--problems", default=",".join(problems.all_names()))
    ap.add_argument("--tag", default="", help="suffix for the output file name (e.g. a candidate wheel)")
    ap.add_argument("--options", default="", help="JSON dict of mincon options for a candidate run (default: none)")
    a = ap.parse_args()
    if a.options:
        MINCON_OPTIONS.update(json.loads(a.options))
    os.makedirs(a.out, exist_ok=True)
    env = dict(python=sys.version, platform=platform.platform(), machine=platform.machine())
    try:
        import mincon
        import scipy
        env.update(mincon=mincon.__version__, scipy=scipy.__version__, numpy=np.__version__)
    except ImportError:
        pass
    with open(os.path.join(a.out, "python_env.json"), "w", encoding="utf-8") as fh:
        json.dump(env, fh, indent=1)
    for solver in a.solvers.split(","):
        path = os.path.join(a.out, f"{solver}{a.tag}.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for name in a.problems.split(","):
                rec = run_one(name, solver)
                fh.write(json.dumps(rec, default=_json_default) + "\n")
                fh.flush()
                v = rec.get("verdict", {})
                print(f"{name:22s} {solver:20s} {rec['outcome']:5s} attained={v.get('target_attained')} "
                      f"reported={rec.get('reported_success')} f_model={rec['counts']['f_model']:6d} "
                      f"wall={rec['wall']:.3f}s  {rec.get('message', rec.get('error', ''))[:70]}", flush=True)


if __name__ == "__main__":
    main()
