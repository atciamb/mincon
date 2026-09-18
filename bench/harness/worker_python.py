"""Python solver worker: runs a list of corpus problems through one solver and appends JSONL records.

    python worker_python.py --solver mincon --track A --problems HS71,HS100 --out r.jsonl --progress p.txt

Solvers: mincon (Algorithm auto, the shipped default), mincon-ip, mincon-sqp, mincon-fmincon (fmincon facade),
scipy-slsqp, scipy-trust-constr. Track A: no derivatives supplied; track C: exact objective gradient
and (where the API allows) exact constraint Jacobian.
The supervisor watches `--progress`; each record is flushed before the next problem starts.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "corpus"))
import spec  # noqa: E402
import families  # noqa: E402,F401
from model import CountingModel  # noqa: E402
import schema  # noqa: E402


def parse_overrides(spec_str):
    """'mincon-ip@ftol=1e-6;scaling=none' -> ('mincon-ip', {'ftol': 1e-6, 'scaling': 'none'}).
    Options are separated by ';' (',' also accepted when the string is not going through the
    supervisor, which splits solvers on ',')."""
    if "@" not in spec_str:
        return spec_str, {}
    base, tail = spec_str.split("@", 1)
    over = {}
    for kv in tail.replace(";", ",").split(","):
        if not kv:
            continue
        k, v = kv.split("=", 1)
        try:
            v = int(v) if v.lstrip("-").isdigit() else float(v)
        except ValueError:
            v = {"true": True, "false": False}.get(v.lower(), v)
        over[k] = v
    return base, over


def solve_mincon(model, track, budget, threads, variant, overrides=None):
    import mincon
    p = model.p
    groups = model.row_groups()
    cons = []
    if p.m:
        if groups["eq"].size:
            i = groups["eq"]; cons.append({"type": "eq", "fun": lambda x, i=i: model.c(x)[i] - p.cl[i],
                                           "jac": lambda x, i=i: model.jac(x)[i]})
        if groups["lo"].size:
            i = groups["lo"]; cons.append({"type": "ineq", "fun": lambda x, i=i: model.c(x)[i] - p.cl[i],
                                           "jac": lambda x, i=i: model.jac(x)[i]})
        if groups["hi"].size:
            i = groups["hi"]; cons.append({"type": "ineq", "fun": lambda x, i=i: p.cu[i] - model.c(x)[i],
                                           "jac": lambda x, i=i: -model.jac(x)[i]})
        if track != "C":
            for d in cons:
                d.pop("jac", None)
    bounds = [(None if not np.isfinite(lo) else float(lo), None if not np.isfinite(hi) else float(hi)) for lo, hi in zip(p.xl, p.xu)]
    opts = {"threads": threads}
    if budget.get("maxfev"):
        opts["maxfev"] = int(budget["maxfev"])
    if budget.get("maxtime"):
        opts["maxtime"] = float(budget["maxtime"])
    opts.update({k: v for k, v in (overrides or {}).items() if not k.startswith("_")})  # "_tag=..." keys only label the run
    jac = model.grad if track == "C" else None
    method = {"mincon": "auto", "mincon-ip": "interior-point", "mincon-sqp": "sqp"}.get(variant, "auto")
    t0 = time.perf_counter()
    if variant == "mincon-fmincon":
        # the MATLAB-style facade: nonlcon returns (c <= 0, ceq == 0)
        def nonlcon(x):
            c = model.c(x)
            ineq = np.concatenate([c[groups["hi"]] - p.cu[groups["hi"]], p.cl[groups["lo"]] - c[groups["lo"]]])
            return ineq, c[groups["eq"]] - p.cl[groups["eq"]]

        def nonlcon_jac(x):
            J = model.jac(x)
            return np.concatenate([J[groups["hi"]], -J[groups["lo"]]], axis=0), J[groups["eq"]]
        lb = np.where(np.isfinite(p.xl), p.xl, -np.inf); ub = np.where(np.isfinite(p.xu), p.xu, np.inf)
        r = mincon.fmincon(model.f, p.x0, lb=lb, ub=ub, nonlcon=nonlcon if p.m else None, options=opts, jac=jac,
                           nonlcon_jac=(nonlcon_jac if (p.m and track == "C") else None))
    else:
        r = mincon.minimize(model.f, p.x0, jac=jac, bounds=bounds, constraints=cons or None, method=method, options=opts)
    wall = time.perf_counter() - t0
    lam = np.asarray(r["lambda"], float) if "lambda" in r else None
    # mincon returns multipliers in constraint-block order: eq block, lo block, hi block (each canonical sign already:
    # 'ineq' fun >= 0 is a lower side -> negative multipliers; we map back to canonical row order and canonical sign).
    lam_canon = None
    if lam is not None and p.m == 0:
        # Bounds-only: the bound multipliers z_l, z_u are the whole dual, and the oracle evaluates
        # its first-order test only when multipliers are supplied. Leaving this None was why 90 %
        # of bounds-only records carried NaN stationarity (docs/17 item 20d, September 2026).
        lam_canon = np.zeros(0)
    elif lam is not None and p.m and variant != "mincon-fmincon":
        lam_canon = np.zeros(p.m)
        off = 0
        for key, sign in (("eq", 1.0), ("lo", 1.0), ("hi", -1.0)):
            idx = groups[key]
            if idx.size:
                lam_canon[idx] += sign * lam[off:off + idx.size]
                off += idx.size
    return dict(x=np.asarray(r.x, float), lam=lam_canon, zl=np.asarray(r.get("z_l", []), float), zu=np.asarray(r.get("z_u", []), float),
                native_status=int(r.status), native_message=str(r.message), reported_success=bool(r.success),
                solver_version=mincon.__version__, wall=wall, solver_time=float(r.get("time", float("nan"))),
                nit=int(r.get("nit", -1)), notes=list(r.get("notes", [])), options=dict(opts, method=method),
                trace=list(r.get("trace", [])))


class _BudgetExhausted(Exception):
    """Raised at the model boundary when a SciPy run passes the shared wall or evaluation budget."""


class _BudgetedModel:
    """Protocol v3: SciPy is held to the budget mincon gets through its options and fmincon through its
    OutputFcn. SciPy's methods have no such options, so the model stops the run: the first call after
    the deadline, or after `maxfev` objective evaluations at the model boundary, raises."""

    def __init__(self, model, maxtime, maxfev):
        self._m, self.p = model, model.p
        self.maxtime, self.maxfev, self.t0 = maxtime, maxfev, None

    def _check(self):
        if self.t0 is None:
            return
        if self.maxtime and time.perf_counter() - self.t0 > self.maxtime:
            raise _BudgetExhausted("time")
        if self.maxfev and self._m.counts["f_model"] >= self.maxfev:
            raise _BudgetExhausted("evaluations")

    def f(self, x):
        self._check()
        return self._m.f(x)

    def grad(self, x):
        self._check()
        return self._m.grad(x)

    def c(self, x):
        self._check()
        return self._m.c(x)

    def jac(self, x):
        self._check()
        return self._m.jac(x)

    def row_groups(self):
        return self._m.row_groups()


def solve_scipy(model, track, budget, method):
    from scipy.optimize import minimize, NonlinearConstraint, Bounds
    import scipy
    maxtime, maxfev = float(budget.get("maxtime") or 0.0), int(budget.get("maxfev") or 0)
    model = _BudgetedModel(model, maxtime, maxfev)
    p = model.p
    groups = model.row_groups()
    # the point reported after a budget stop is the last iterate the solver's own callback saw:
    # no invented point and no point from the middle of a line search
    last = {"x": np.asarray(p.x0, float).copy(), "nit": 0}

    def remember(xk, *_state):
        last["x"] = np.array(xk, float)
        last["nit"] += 1
    bounds = Bounds(np.where(np.isfinite(p.xl), p.xl, -np.inf), np.where(np.isfinite(p.xu), p.xu, np.inf))
    jac = model.grad if track == "C" else None
    opts = {}
    if budget.get("maxfev") and method == "SLSQP":
        opts["maxiter"] = 1000
    t0 = model.t0 = time.perf_counter()
    try:
        r = _run_scipy(minimize, NonlinearConstraint, model, p, groups, track, jac, bounds, opts, method, remember)
    except _BudgetExhausted as stop:
        wall = time.perf_counter() - t0
        limit = f"{maxtime:g} s" if str(stop) == "time" else f"{maxfev} objective evaluations"
        return dict(x=last["x"], lam=None, zl=None, zu=None, native_status=-98,
                    native_message=f"stopped by the harness at the shared budget of {limit} (protocol v3); last iterate reported",
                    reported_success=False, solver_version=scipy.__version__, wall=wall, solver_time=float("nan"),
                    nit=last["nit"], notes=[f"budget: {stop}"], options=dict(opts, method=method, budget_enforced=True))
    wall = time.perf_counter() - t0
    return dict(x=np.asarray(r.x, float), lam=None, zl=None, zu=None, native_status=int(getattr(r, "status", -99)),
                native_message=str(getattr(r, "message", "")), reported_success=bool(r.success), solver_version=scipy.__version__,
                wall=wall, solver_time=float("nan"), nit=int(getattr(r, "nit", -1)), notes=[],
                options=dict(opts, method=method, budget_enforced=True))


def _run_scipy(minimize, NonlinearConstraint, model, p, groups, track, jac, bounds, opts, method, remember):
    if method == "trust-constr":
        cons = []
        if p.m:
            cl = np.where(np.isfinite(p.cl), p.cl, -np.inf); cu = np.where(np.isfinite(p.cu), p.cu, np.inf)
            cons = [NonlinearConstraint(model.c, cl, cu, jac=(model.jac if track == "C" else "2-point"))]
        return minimize(model.f, p.x0, jac=jac if jac else "2-point", bounds=bounds, constraints=cons, method="trust-constr",
                        callback=remember, options={"maxiter": 3000, "verbose": 0})
    else:
        cons = []
        if p.m:
            if groups["eq"].size:
                i = groups["eq"]; d = {"type": "eq", "fun": lambda x, i=i: model.c(x)[i] - p.cl[i]}
                if track == "C": d["jac"] = lambda x, i=i: model.jac(x)[i]
                cons.append(d)
            if groups["lo"].size:
                i = groups["lo"]; d = {"type": "ineq", "fun": lambda x, i=i: model.c(x)[i] - p.cl[i]}
                if track == "C": d["jac"] = lambda x, i=i: model.jac(x)[i]
                cons.append(d)
            if groups["hi"].size:
                i = groups["hi"]; d = {"type": "ineq", "fun": lambda x, i=i: p.cu[i] - model.c(x)[i]}
                if track == "C": d["jac"] = lambda x, i=i: -model.jac(x)[i]
                cons.append(d)
        return minimize(model.f, p.x0, jac=jac, bounds=bounds, constraints=cons, method="SLSQP", callback=remember,
                        options=dict(opts, maxiter=1000))


SOLVERS = {
    "mincon": lambda m, t, b, th: solve_mincon(m, t, b, th, "mincon"),
    "mincon-ip": lambda m, t, b, th: solve_mincon(m, t, b, th, "mincon-ip"),
    "mincon-fmincon": lambda m, t, b, th: solve_mincon(m, t, b, th, "mincon-fmincon"),
    "scipy-slsqp": lambda m, t, b, th: solve_scipy(m, t, b, "SLSQP"),
    "scipy-trust-constr": lambda m, t, b, th: solve_scipy(m, t, b, "trust-constr"),
}


def dispatch(solver, model, track, budget, threads):
    base, over = parse_overrides(solver)
    if base.startswith("mincon"):
        return solve_mincon(model, track, budget, threads, base, over)
    return SOLVERS[base](model, track, budget, threads)


def run_one(name, solver, track, budget, threads, experiment, split_of, repeat, keep_trace=False):
    s = spec.get(name)
    rec = schema.new_record(experiment=experiment, track=track, problem=name, family=s.family, split=split_of.get(name),
                            checksum=s.checksum(), n=s.n, m=s.m, solver=solver, derivatives=("exact" if track == "C" else "fd"),
                            threads=threads, budget=budget, repeat=repeat)
    t_build = time.perf_counter()
    p = s.numpy()
    model = CountingModel(p)
    rec["time"]["build"] = time.perf_counter() - t_build
    try:
        out = dispatch(solver, model, track, budget, threads)
        rec.update(x=out["x"], lam=out["lam"], zl=out["zl"], zu=out["zu"], native_status=out["native_status"],
                   native_message=out["native_message"], reported_success=out["reported_success"],
                   solver_version=out["solver_version"], options=out["options"], notes=out["notes"], outcome="ok")
        rec["time"].update(solve_wall=out["wall"], solver_reported=out["solver_time"], callback=model.callback_seconds)
        rec["nit"] = out["nit"]
        if keep_trace and out.get("trace"):
            rec["trace"] = out["trace"]
    except Exception as exc:  # noqa: BLE001 - a crash is a data point
        rec.update(outcome="error", error=f"{type(exc).__name__}: {exc}", notes=[traceback.format_exc()[-2000:]])
        rec["time"].update(callback=model.callback_seconds)
    rec["counts"] = dict(model.counts)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", required=True, help="one of %s, mincon* may carry @key=value overrides" % sorted(SOLVERS))
    ap.add_argument("--track", default="A", choices=["A", "B", "C"])
    ap.add_argument("--problems", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--progress", default=None)
    ap.add_argument("--maxtime", type=float, default=60.0)
    ap.add_argument("--maxfev", type=int, default=100000)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--experiment", default="dev")
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--trace", action="store_true", help="keep the solver's per-iteration trace in the record (mincon only)")
    ap.add_argument("--manifest", default=os.path.join(os.path.dirname(HERE), "corpus", "manifest.json"))
    a = ap.parse_args()
    import json
    split_of = {}
    if os.path.exists(a.manifest):
        split_of = {p["name"]: p.get("split") for p in json.load(open(a.manifest))["problems"]}
    budget = dict(maxtime=a.maxtime, maxfev=a.maxfev)
    names = [n for n in a.problems.split(",") if n]
    with open(a.out, "a") as fh:
        for name in names:
            if a.progress:
                with open(a.progress, "w") as pf:
                    pf.write(f"{name} {time.time()}\n")
            rec = run_one(name, a.solver, a.track, budget, a.threads, a.experiment, split_of, a.repeat, keep_trace=a.trace)
            fh.write(schema.dumps(rec) + "\n")
            fh.flush()
            print(f"{name:<22} {a.solver:<18} {rec['outcome']:<6} status={rec.get('native_status')} f_model={rec['counts']['f_model']} "
                  f"c_model={rec['counts']['c_model']} wall={rec['time'].get('solve_wall', float('nan')):.3f}s", flush=True)
    if a.progress:
        with open(a.progress, "w") as pf:
            pf.write(f"DONE {time.time()}\n")


if __name__ == "__main__":
    main()
