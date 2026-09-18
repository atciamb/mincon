"""Check a problem's stored reference under the corpus model itself, one problem per process.

    python verify_refs.py NAME [NAME ...] [--out refs.jsonl]

For each problem: build the Spec and its NumPy model (the same callables the workers use),
evaluate the stored reference point, and require that

* it is feasible to 1e-8 (bounds and rows, model units; the harness scores at 1e-6);
* it reproduces the stored target to 1e-6 relative to max(1, |target|) (round 5 allowed 5e-3
  and shipped a reference point that missed its target by 4.6 tolerances);
* it is a first-order point: the oracle's recovered stationarity (non-negative least-squares
  multipliers on the active set, exact derivatives) is below 1e-10 of max(1, |grad f|_inf).

Exit code 1 if any problem fails. The time of one objective evaluation is reported as well,
because round 6 has families whose design depends on it.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "harness"))
import spec  # noqa: E402
import families  # noqa: E402,F401
from oracle import assess  # noqa: E402

FEAS_TOL, VALUE_TOL, STAT_TOL = 1e-8, 1e-6, 1e-10


def check(name: str) -> dict:
    t0 = time.perf_counter()
    s = spec.get(name)
    t_spec = time.perf_counter() - t0
    t0 = time.perf_counter()
    p = s.numpy()
    p.f(p.x0), p.grad(p.x0)
    if p.m:
        p.cons(p.x0), p.jac(p.x0)
    t_build = time.perf_counter() - t0
    out = dict(name=name, family=s.family, n=s.n, m=s.m, ref_f=s.ref_f, spec_seconds=round(t_spec, 2),
               build_seconds=round(t_build, 2), checksum=s.checksum())
    reps = 20 if s.n <= 100 else 5
    t0 = time.perf_counter()
    for k in range(reps):
        p.f(p.x0 + 1e-9 * (k + 1))
    out["f_call_ms"] = round(1e3 * (time.perf_counter() - t0) / reps, 4)
    out["f_x0"] = p.f(p.x0)
    out["violation_x0"] = p.violation(p.x0)
    if s.ref_x is None or s.ref_f is None:
        out.update(ok=False, why="no stored reference point or value")
        return out
    worst = dict(value=0.0, violation=0.0, stationarity=0.0)
    for xr in s.ref_x:
        xr = np.asarray(xr, float)
        v = assess(p, xr, feas_tol=FEAS_TOL, stat_tol=STAT_TOL, target=s.ref_f, active_tol=1e-9)
        gscale = max(1.0, float(np.max(np.abs(p.grad(xr)))))
        worst["value"] = max(worst["value"], abs(v.f - s.ref_f) / max(1.0, abs(s.ref_f)))
        worst["violation"] = max(worst["violation"], v.violation)
        worst["stationarity"] = max(worst["stationarity"], v.recovered_stationarity / gscale)
        out.update(active_rows=v.active_rows, active_bounds=v.active_bounds, grad_scale=gscale,
                   stationarity_raw=v.recovered_stationarity)
    out.update(value_err=worst["value"], violation=worst["violation"], stationarity=worst["stationarity"])
    out["ok"] = bool(worst["value"] <= VALUE_TOL and worst["violation"] <= FEAS_TOL and worst["stationarity"] <= STAT_TOL)
    return out


def main() -> int:
    args = sys.argv[1:]
    dest = None
    if "--out" in args:
        i = args.index("--out")
        dest = args[i + 1]
        args = args[:i] + args[i + 2:]
    bad = 0
    for name in args:
        rec = check(name)
        line = json.dumps(rec)
        print(line, flush=True)
        if dest:
            with open(dest, "a", newline="\n") as fh:
                fh.write(line + "\n")
        bad += 0 if rec["ok"] else 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
