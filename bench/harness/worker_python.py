"""Python solver worker: runs a list of corpus problems through one solver and appends JSONL records.

    python worker_python.py --solver mincon --track A --problems HS71,HS100 --out r.jsonl --progress p.txt

Solvers: mincon (Algorithm auto, the shipped default), mincon-ip, mincon-fmincon (fmincon facade),
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


def solve_mincon(model, track, budget, threads, variant):
    import mincon
    p = model.p
    groups = model.row_groups()
    cons = []
    if p.m:
        if groups["eq"].size:
            i = groups["eq"]; cons.append({"type": "eq", "fun": lambda x, i=i: model.c(x)[i] - p.cl[i]})
        if groups["lo"].size:
            i = groups["lo"]; cons.append({"type": "ineq", "fun": lambda x, i=i: model.c(x)[i] - p.cl[i]})
        if groups["hi"].size:
            i = groups["hi"]; cons.append({"type": "ineq", "fun": lambda x, i=i: p.cu[i] - model.c(x)[i]})
    bounds = [(None if not np.isfinite(lo) else float(lo), None if not np.isfinite(hi) else float(hi)) for lo, hi in zip(p.xl, p.xu)]
    opts = {"threads": threads}
    if budget.get("maxfev"):
        opts["maxfev"] = int(budget["maxfev"])
    if budget.get("maxtime"):
        opts["maxtime"] = float(budget["maxtime"])
    jac = model.grad if track == "C" else None
    method = {"mincon": "auto", "mincon-ip": "interior-point"}.get(variant, "auto")
    t0 = time.perf_counter()
    if variant == "mincon-fmincon":
        # the MATLAB-style facade: nonlcon returns (c <= 0, ceq == 0)
        def nonlcon(x):
            c = model.c(x)
            ineq = np.concatenate([c[groups["hi"]] - p.cu[groups["hi"]], p.cl[groups["lo"]] - c[groups["lo"]]])
            return ineq, c[groups["eq"]] - p.cl[groups["eq"]]
        lb = np.where(np.isfinite(p.xl), p.xl, -np.inf); ub = np.where(np.isfinite(p.xu), p.xu, np.inf)
        r = mincon.fmincon(model.f, p.x0, lb=lb, ub=ub, nonlcon=nonlcon if p.m else None, options=opts, jac=jac)
    else:
        r = mincon.minimize(model.f, p.x0, jac=jac, bounds=bounds, constraints=cons or None, method=method, options=opts)
    wall = time.perf_counter() - t0
    lam = np.asarray(r["lambda"], float) if "lambda" in r else None
    # mincon returns multipliers in constraint-block order: eq block, lo block, hi block (each canonical sign already:
    # 'ineq' fun >= 0 is a lower side -> negative multipliers; we map back to canonical row order and canonical sign).
    lam_canon = None
    if lam is not None and p.m and variant != "mincon-fmincon":
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
                nit=int(r.get("nit", -1)), notes=list(r.get("notes", [])), options=dict(opts, method=method))


def solve_scipy(model, track, budget, method):
    from scipy.optimize import minimize, NonlinearConstraint, Bounds
    p = model.p
    groups = model.row_groups()
    bounds = Bounds(np.where(np.isfinite(p.xl), p.xl, -np.inf), np.where(np.isfinite(p.xu), p.xu, np.inf))
    jac = model.grad if track == "C" else None
    opts = {}
    if budget.get("maxfev") and method == "SLSQP":
        opts["maxiter"] = 1000
    t0 = time.perf_counter()
    if method == "trust-constr":
        cons = []
        if p.m:
            cl = np.where(np.isfinite(p.cl), p.cl, -np.inf); cu = np.where(np.isfinite(p.cu), p.cu, np.inf)
            cons = [NonlinearConstraint(model.c, cl, cu, jac=(model.jac if track == "C" else "2-point"))]
        r = minimize(model.f, p.x0, jac=jac if jac else "2-point", bounds=bounds, constraints=cons, method="trust-constr",
                     options={"maxiter": 3000, "verbose": 0})
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
        r = minimize(model.f, p.x0, jac=jac, bounds=bounds, constraints=cons, method="SLSQP", options=dict(opts, maxiter=1000))
    wall = time.perf_counter() - t0
    import scipy
    return dict(x=np.asarray(r.x, float), lam=None, zl=None, zu=None, native_status=int(getattr(r, "status", -99)),
                native_message=str(getattr(r, "message", "")), reported_success=bool(r.success), solver_version=scipy.__version__,
                wall=wall, solver_time=float("nan"), nit=int(getattr(r, "nit", -1)), notes=[], options=dict(opts, method=method))


SOLVERS = {
    "mincon": lambda m, t, b, th: solve_mincon(m, t, b, th, "mincon"),
    "mincon-ip": lambda m, t, b, th: solve_mincon(m, t, b, th, "mincon-ip"),
    "mincon-fmincon": lambda m, t, b, th: solve_mincon(m, t, b, th, "mincon-fmincon"),
    "scipy-slsqp": lambda m, t, b, th: solve_scipy(m, t, b, "SLSQP"),
    "scipy-trust-constr": lambda m, t, b, th: solve_scipy(m, t, b, "trust-constr"),
}


def run_one(name, solver, track, budget, threads, experiment, split_of, repeat):
    s = spec.get(name)
    rec = schema.new_record(experiment=experiment, track=track, problem=name, family=s.family, split=split_of.get(name),
                            checksum=s.checksum(), n=s.n, m=s.m, solver=solver, derivatives=("exact" if track == "C" else "fd"),
                            threads=threads, budget=budget, repeat=repeat)
    t_build = time.perf_counter()
    p = s.numpy()
    model = CountingModel(p)
    rec["time"]["build"] = time.perf_counter() - t_build
    try:
        out = SOLVERS[solver](model, track, budget, threads)
        rec.update(x=out["x"], lam=out["lam"], zl=out["zl"], zu=out["zu"], native_status=out["native_status"],
                   native_message=out["native_message"], reported_success=out["reported_success"],
                   solver_version=out["solver_version"], options=out["options"], notes=out["notes"], outcome="ok")
        rec["time"].update(solve_wall=out["wall"], solver_reported=out["solver_time"], callback=model.callback_seconds)
        rec["nit"] = out["nit"]
    except Exception as exc:  # noqa: BLE001 - a crash is a data point
        rec.update(outcome="error", error=f"{type(exc).__name__}: {exc}", notes=[traceback.format_exc()[-2000:]])
        rec["time"].update(callback=model.callback_seconds)
    rec["counts"] = dict(model.counts)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", required=True, choices=sorted(SOLVERS))
    ap.add_argument("--track", default="A", choices=["A", "B", "C"])
    ap.add_argument("--problems", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--progress", default=None)
    ap.add_argument("--maxtime", type=float, default=60.0)
    ap.add_argument("--maxfev", type=int, default=100000)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--experiment", default="dev")
    ap.add_argument("--repeat", type=int, default=0)
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
            rec = run_one(name, a.solver, a.track, budget, a.threads, a.experiment, split_of, a.repeat)
            fh.write(schema.dumps(rec) + "\n")
            fh.flush()
            print(f"{name:<22} {a.solver:<18} {rec['outcome']:<6} status={rec.get('native_status')} f_model={rec['counts']['f_model']} "
                  f"c_model={rec['counts']['c_model']} wall={rec['time'].get('solve_wall', float('nan')):.3f}s", flush=True)
    if a.progress:
        with open(a.progress, "w") as pf:
            pf.write(f"DONE {time.time()}\n")


if __name__ == "__main__":
    main()
