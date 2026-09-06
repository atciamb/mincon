#!/usr/bin/env python3
"""Turn a results file into performance profiles, data profiles and a table.

    python profiles.py results.jsonl -o profiles.png
    python profiles.py results.jsonl --table

# Performance profiles (Dolan & Moré, Math. Prog. 91(2), 2002)

For each problem `p` and solver `s`, let `t(p,s)` be the cost (evaluations or
seconds) and define the **performance ratio**

    r(p,s) = t(p,s) / min over solvers of t(p,·)

The profile is `rho_s(tau) = fraction of problems with r(p,s) <= tau`. Read it
as: `rho_s(1)` is how often `s` was the *fastest*, and `rho_s(tau -> inf)` is
how often it solved the problem at all — the **robustness** number, which is
the one that matters most here and the one `fmincon` currently wins.

Unsolved problems get `r = inf`, so they never contribute, which is the whole
point of the construction: a solver cannot buy a good profile by failing fast.

# Data profiles (Moré & Wild, SIAM J. Optim. 20(1), 2009)

Performance profiles hide how much *budget* a solver needed. A data profile
plots, against a budget measured in units of `n+1` evaluations, the fraction of
problems solved within that budget. This is the right view when the model is
expensive, which is the situation nearly every real user is in.

# A caution about reading these

A performance profile compares solvers *on the set you ran*. Change the set and
the picture changes. Two rules keep it honest:

1. Report the problem set, the tolerances and the budget alongside every plot.
2. Never tune parameters on the same set you report. `bench/` therefore
   distinguishes a **development set** and a **held-out set**; see README.md.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

FEAS_TOL = 1e-5
OBJ_TOL = 1e-4


def load(path: Path):
    meta = {}
    records = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("kind") == "meta":
            meta = d
        else:
            records.append(d)
    return meta, records


def score(records, feas_tol=FEAS_TOL, obj_tol=OBJ_TOL):
    """Decide, per (problem, solver), whether the problem was solved.

    Solved means: the returned point is feasible to `feas_tol`, and its
    objective is within `obj_tol` relative of the best objective any solver
    reached on that problem while feasible.
    """
    by_problem = defaultdict(list)
    for r in records:
        by_problem[r["problem"]].append(r)

    solved = {}
    best_f = {}
    for prob, rs in by_problem.items():
        feasible = [r for r in rs if r["ok"] and r["maxcv"] <= feas_tol and math.isfinite(r["f"])]
        if not feasible:
            best_f[prob] = None
            for r in rs:
                solved[(prob, r["solver"])] = False
            continue
        fbest = min(r["f"] for r in feasible)
        best_f[prob] = fbest
        threshold = fbest + obj_tol * (1.0 + abs(fbest))
        for r in rs:
            solved[(prob, r["solver"])] = (
                r["ok"] and r["maxcv"] <= feas_tol and math.isfinite(r["f"]) and r["f"] <= threshold
            )
    return solved, best_f


def performance_profile(records, metric="nfev", feas_tol=FEAS_TOL, obj_tol=OBJ_TOL):
    solved, _ = score(records, feas_tol, obj_tol)
    solvers = sorted({r["solver"] for r in records})
    problems = sorted({r["problem"] for r in records})
    cost = {(r["problem"], r["solver"]): r[metric] for r in records}

    ratios = {s: [] for s in solvers}
    for p in problems:
        best = min(
            (cost.get((p, s), math.inf) for s in solvers if solved.get((p, s))),
            default=math.inf,
        )
        best = max(best, 1e-12)
        for s in solvers:
            if solved.get((p, s)):
                ratios[s].append(cost.get((p, s), math.inf) / best)
            else:
                ratios[s].append(math.inf)

    taus = np.logspace(0, 3, 300)
    curves = {}
    for s in solvers:
        arr = np.array(ratios[s], dtype=float)
        curves[s] = np.array([(arr <= t).mean() for t in taus])
    return taus, curves, solvers


def data_profile(records, feas_tol=FEAS_TOL, obj_tol=OBJ_TOL, max_units=200):
    solved, _ = score(records, feas_tol, obj_tol)
    solvers = sorted({r["solver"] for r in records})
    problems = sorted({r["problem"] for r in records})
    info = {(r["problem"], r["solver"]): r for r in records}

    units = np.linspace(1, max_units, 300)
    curves = {}
    for s in solvers:
        vals = []
        for u in units:
            hit = 0
            for p in problems:
                r = info.get((p, s))
                if r is None or not solved.get((p, s)):
                    continue
                if r["nfev"] <= u * (r["n"] + 1):
                    hit += 1
            vals.append(hit / max(len(problems), 1))
        curves[s] = np.array(vals)
    return units, curves, solvers


def table(records, feas_tol=FEAS_TOL, obj_tol=OBJ_TOL) -> str:
    solved, _ = score(records, feas_tol, obj_tol)
    solvers = sorted({r["solver"] for r in records})
    problems = sorted({r["problem"] for r in records})
    lines = []
    w = max(len(s) for s in solvers) + 2
    lines.append(
        f"{'solver':<{w}} {'solved':>8} {'rate':>7} {'lied':>6} {'median nfev':>12} {'median s':>9}"
    )
    lines.append("-" * (w + 48))
    for s in solvers:
        rs = [r for r in records if r["solver"] == s]
        ok = [r for r in rs if solved.get((r["problem"], s))]
        # "Lied" = claimed success but the harness disagrees. The single most
        # important column: a solver that lies is worse than one that fails.
        lied = [
            r
            for r in rs
            if r.get("reported_success") and not solved.get((r["problem"], s)) and r["maxcv"] > feas_tol
        ]
        nfev = sorted(r["nfev"] for r in ok) or [0]
        secs = sorted(r["seconds"] for r in ok) or [0.0]
        lines.append(
            f"{s:<{w}} {len(ok):>8} {len(ok)/max(len(problems),1)*100:>6.1f}% "
            f"{len(lied):>6} {nfev[len(nfev)//2]:>12} {secs[len(secs)//2]:>9.3f}"
        )
    lines.append("")
    lines.append(f"{len(problems)} problems, feasibility tol {feas_tol:g}, objective tol {obj_tol:g}")
    lines.append(
        "'lied' counts runs where the solver reported success but returned an infeasible point."
    )
    return "\n".join(lines)


def plot(records, out: Path, feas_tol=FEAS_TOL, obj_tol=OBJ_TOL):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for metric, ax, title in [
        ("nfev", axes[0], "Performance profile (objective evaluations)"),
        ("seconds", axes[1], "Performance profile (wall clock)"),
    ]:
        taus, curves, solvers = performance_profile(records, metric, feas_tol, obj_tol)
        for s in solvers:
            ax.semilogx(taus, curves[s], label=s, linewidth=2)
        ax.set_xlabel(r"$\tau$  (within $\tau\times$ the best solver)")
        ax.set_ylabel(r"fraction of problems, $\rho_s(\tau)$")
        ax.set_title(title)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")

    units, curves, solvers = data_profile(records, feas_tol, obj_tol)
    for s in solvers:
        axes[2].plot(units, curves[s], label=s, linewidth=2)
    axes[2].set_xlabel(r"budget in units of $(n+1)$ evaluations")
    axes[2].set_ylabel("fraction of problems solved")
    axes[2].set_title("Data profile")
    axes[2].set_ylim(0, 1.02)
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results")
    ap.add_argument("-o", "--output", default="profiles.png")
    ap.add_argument("--table", action="store_true", help="print the summary table only")
    ap.add_argument("--feas-tol", type=float, default=FEAS_TOL)
    ap.add_argument("--obj-tol", type=float, default=OBJ_TOL)
    args = ap.parse_args()

    meta, records = load(Path(args.results))
    if not records:
        raise SystemExit("no records in that file")
    print(table(records, args.feas_tol, args.obj_tol))
    if meta:
        print(f"\nrun on {meta.get('platform', '?')} with budget {meta.get('budget')}")
    if not args.table:
        try:
            plot(records, Path(args.output), args.feas_tol, args.obj_tol)
        except ImportError:
            print("\n(matplotlib not installed; skipping plots)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
