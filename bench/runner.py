#!/usr/bin/env python3
"""Run solvers over a problem set and emit one JSON record per (problem, solver).

    python runner.py --problems HS71 HS100 --solvers mincon scipy-slsqp
    python runner.py --set constrained-small --solvers all -o results.jsonl
    python profiles.py results.jsonl -o profiles.png

# The one rule that makes a benchmark credible

**Success is decided by this harness, never by the solver.**

Every solver reports its own status with its own conventions, its own
tolerances and its own idea of what "converged" means. Scoring on
`result.success` measures reporting culture. So the harness recomputes, for
every returned point:

* the **maximum constraint violation** in the original problem, from the
  problem's own callables;
* the **objective**, likewise;

and then calls a problem solved by a solver when the point is feasible to
`--feas-tol` *and* its objective is within `--obj-tol` (relative) of the best
objective any solver achieved on that problem. This is the Dolan-Moré and
Mittelmann convention and it is the only way a comparison survives review.

A consequence worth stating plainly: **we can lose by this rule too.** If
`mincon` reports `success=True` at a point another solver beats, the harness
scores it a failure. That is the intent.

# Time is measured, but success is not a function of it

Wall-clock is recorded and used for the timing profiles, but the success
criterion is budget-based (`--maxfev`, `--maxtime`) rather than
time-thresholded, so results are reproducible on a different machine.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import s2mpj_bridge as bridge  # noqa: E402

BIG = 1e20


@dataclass
class Record:
    """One (problem, solver) outcome."""

    problem: str
    solver: str
    n: int
    m: int
    classification: str
    ok: bool  # the run completed without an exception
    f: float
    maxcv: float
    nfev: int
    nit: int
    seconds: float
    reported_success: bool
    reported_status: str
    error: str = ""
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Solver adapters. Each takes a BenchProblem and a budget, and returns
# (x, nfev, nit, reported_success, reported_status, notes).
# --------------------------------------------------------------------------


def _counting(fn):
    """Wrap a callable so the harness counts evaluations itself.

    Solvers disagree about what counts as a function evaluation — some include
    finite-difference probes, some do not, some count vector evaluations as
    one. Counting here makes the numbers comparable.
    """
    state = {"n": 0}

    def wrapped(x):
        state["n"] += 1
        return fn(x)

    wrapped.count = state
    return wrapped


def run_mincon(p, budget, method=None, options=None):
    import mincon

    f = _counting(p.f)
    cons = []
    if p.m:
        # Split each two-sided row into the pieces mincon's Python API takes.
        eq = np.isclose(p.cl, p.cu)
        lo = np.isfinite(p.cl) & ~eq & (p.cl > -BIG)
        hi = np.isfinite(p.cu) & ~eq & (p.cu < BIG)
        if eq.any():
            idx = np.flatnonzero(eq)
            cons.append({"type": "eq", "fun": lambda x, i=idx: p.cons(x)[i] - p.cl[i]})
        if lo.any():
            idx = np.flatnonzero(lo)
            cons.append({"type": "ineq", "fun": lambda x, i=idx: p.cons(x)[i] - p.cl[i]})
        if hi.any():
            idx = np.flatnonzero(hi)
            cons.append({"type": "ineq", "fun": lambda x, i=idx: p.cu[i] - p.cons(x)[i]})

    opts = {"maxfev": budget["maxfev"], "maxtime": budget["maxtime"]}
    opts.update(options or {})
    r = mincon.minimize(
        f,
        p.x0,
        bounds=list(zip(_none_inf(p.xl), _none_inf(p.xu, upper=True))),
        constraints=cons or None,
        method=method,
        options=opts,
    )
    return r.x, f.count["n"], int(r.nit), bool(r.success), str(r.status), list(r.notes)


def run_scipy(p, budget, method):
    from scipy.optimize import minimize as smin

    f = _counting(p.f)
    cons = []
    if p.m:
        eq = np.isclose(p.cl, p.cu)
        lo = np.isfinite(p.cl) & ~eq & (p.cl > -BIG)
        hi = np.isfinite(p.cu) & ~eq & (p.cu < BIG)
        if eq.any():
            idx = np.flatnonzero(eq)
            cons.append({"type": "eq", "fun": lambda x, i=idx: p.cons(x)[i] - p.cl[i]})
        if lo.any():
            idx = np.flatnonzero(lo)
            cons.append({"type": "ineq", "fun": lambda x, i=idx: p.cons(x)[i] - p.cl[i]})
        if hi.any():
            idx = np.flatnonzero(hi)
            cons.append({"type": "ineq", "fun": lambda x, i=idx: p.cu[i] - p.cons(x)[i]})

    bounds = list(zip(_none_inf(p.xl), _none_inf(p.xu, upper=True)))
    opts = {"maxiter": 3000}
    if method == "SLSQP":
        opts["maxiter"] = 1000
    r = smin(f, p.x0, method=method, bounds=bounds, constraints=cons, options=opts)
    return (
        np.asarray(r.x, dtype=float).ravel(),
        f.count["n"],
        int(getattr(r, "nit", 0)),
        bool(r.success),
        str(getattr(r, "status", "")),
        [],
    )


def run_ipopt(p, budget):
    import cyipopt

    f = _counting(p.f)

    class Wrapper:
        def objective(self, x):
            return f(x)

        def gradient(self, x):
            return p.grad(x)

        def constraints(self, x):
            return p.cons(x)

        def jacobian(self, x):
            return p.jac(x).ravel()

    nlp = cyipopt.Problem(
        n=p.n,
        m=p.m,
        problem_obj=Wrapper(),
        lb=p.xl,
        ub=p.xu,
        cl=p.cl if p.m else None,
        cu=p.cu if p.m else None,
    )
    nlp.add_option("print_level", 0)
    nlp.add_option("max_iter", 3000)
    nlp.add_option("max_cpu_time", float(budget["maxtime"]))
    x, info = nlp.solve(p.x0)
    return (
        np.asarray(x, dtype=float).ravel(),
        f.count["n"],
        0,
        info["status"] == 0,
        str(info["status"]),
        [],
    )


def _none_inf(a, upper=False):
    return [None if (not np.isfinite(v) or abs(v) >= BIG) else float(v) for v in a]


SOLVERS: dict[str, Callable] = {
    "mincon": lambda p, b: run_mincon(p, b),
    "mincon-ip": lambda p, b: run_mincon(p, b, method="interior-point"),
    "mincon-fmincon-profile": lambda p, b: run_mincon(
        p, b, method="interior-point", options={"scaling": "none", "finite_diff": "forward"}
    ),
    "scipy-slsqp": lambda p, b: run_scipy(p, b, "SLSQP"),
    "scipy-trust-constr": lambda p, b: run_scipy(p, b, "trust-constr"),
    "scipy-cobyla": lambda p, b: run_scipy(p, b, "COBYLA"),
    "ipopt": run_ipopt,
}

PROBLEM_SETS = {
    "smoke": ["HS71", "HS100", "HS35", "HS43"],
    "hs": None,  # filled in below from the classification filter
    "constrained-small": None,
    "constrained-medium": None,
    "unconstrained-small": None,
}


def resolve_set(name: str, directory) -> list[str]:
    if name == "smoke":
        return PROBLEM_SETS["smoke"]
    names = bridge.list_problems(directory)
    if not names:
        raise SystemExit(
            "No S2MPJ problems found. Run:  python -c "
            "'import s2mpj_bridge as b; b.ensure_s2mpj()'"
        )
    if name == "hs":
        return [n for n in names if n.startswith("HS")]
    if name == "constrained-small":
        return bridge.filter_problems(names, directory, constrained=True, max_n=100, max_m=100)
    if name == "constrained-medium":
        return bridge.filter_problems(
            names, directory, constrained=True, max_n=1000, max_m=1000, min_n=101
        )
    if name == "unconstrained-small":
        return bridge.filter_problems(names, directory, constrained=False, max_n=100)
    if name == "all":
        return names
    raise SystemExit(f"unknown problem set '{name}'")


def run_one(problem, solver_name, budget) -> Record:
    fn = SOLVERS[solver_name]
    base = dict(
        problem=problem.name,
        solver=solver_name,
        n=problem.n,
        m=problem.m,
        classification=problem.classification,
    )
    t0 = time.perf_counter()
    try:
        x, nfev, nit, ok_reported, status, notes = fn(problem, budget)
        seconds = time.perf_counter() - t0
        x = np.asarray(x, dtype=float).ravel()
        if x.size != problem.n or not np.all(np.isfinite(x)):
            return Record(
                **base,
                ok=False,
                f=float("inf"),
                maxcv=float("inf"),
                nfev=nfev,
                nit=nit,
                seconds=seconds,
                reported_success=False,
                reported_status=status,
                error="returned a non-finite or wrongly sized point",
            )
        # Recompute both quantities ourselves.
        try:
            fval = float(problem.f(x))
        except Exception:
            fval = float("inf")
        cv = problem.violation(x)
        return Record(
            **base,
            ok=True,
            f=fval if np.isfinite(fval) else float("inf"),
            maxcv=cv,
            nfev=nfev,
            nit=nit,
            seconds=seconds,
            reported_success=ok_reported,
            reported_status=status,
            notes=notes,
        )
    except Exception as exc:  # noqa: BLE001 - a crashed solver is a data point
        return Record(
            **base,
            ok=False,
            f=float("inf"),
            maxcv=float("inf"),
            nfev=0,
            nit=0,
            seconds=time.perf_counter() - t0,
            reported_success=False,
            reported_status="exception",
            error=f"{type(exc).__name__}: {exc}",
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problems", nargs="*", help="explicit problem names")
    ap.add_argument("--set", default="smoke", help="named set: " + ", ".join(PROBLEM_SETS) + ", all")
    ap.add_argument("--solvers", nargs="+", default=["mincon"], help="'all' or names: " + ", ".join(SOLVERS))
    ap.add_argument("-o", "--output", default="results.jsonl")
    ap.add_argument("--maxfev", type=int, default=100_000)
    ap.add_argument("--maxtime", type=float, default=60.0)
    ap.add_argument("--dir", default=str(bridge.DEFAULT_DIR))
    ap.add_argument("--limit", type=int, default=None, help="cap the number of problems")
    args = ap.parse_args()

    directory = Path(args.dir)
    names = args.problems or resolve_set(args.set, directory)
    if args.limit:
        names = names[: args.limit]
    solvers = list(SOLVERS) if args.solvers == ["all"] else args.solvers
    for s in solvers:
        if s not in SOLVERS:
            raise SystemExit(f"unknown solver '{s}'; choose from {list(SOLVERS)}")

    budget = {"maxfev": args.maxfev, "maxtime": args.maxtime}
    out = Path(args.output)
    written = 0
    header = {
        "kind": "meta",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "problems": len(names),
        "solvers": solvers,
        "budget": budget,
    }
    with out.open("w") as fh:
        fh.write(json.dumps(header) + "\n")
        for i, name in enumerate(names, 1):
            try:
                problem = bridge.load(name, directory)
            except Exception as exc:  # noqa: BLE001
                print(f"[{i}/{len(names)}] {name}: could not load ({exc})", file=sys.stderr)
                continue
            line = f"[{i}/{len(names)}] {name} (n={problem.n}, m={problem.m})"
            print(line, file=sys.stderr, flush=True)
            for s in solvers:
                rec = run_one(problem, s, budget)
                fh.write(json.dumps(asdict(rec)) + "\n")
                fh.flush()
                written += 1
                flag = "ok " if rec.ok else "ERR"
                print(
                    f"      {s:<24} {flag} f={rec.f: .6e} cv={rec.maxcv:.1e} "
                    f"nfev={rec.nfev:<6} {rec.seconds:6.3f}s {rec.error}",
                    file=sys.stderr,
                    flush=True,
                )
    print(f"\nwrote {written} records to {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        traceback.print_exc()
        raise SystemExit(130) from None
