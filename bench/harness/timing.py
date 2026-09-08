"""Paired wall-time analysis with replicates.

    python timing.py results/s6-timing/*.A.jsonl --baseline fmincon-interior-point --solver mincon

For every (problem, solver) the median solve wall time over repeats is taken; the paired ratio
solver/baseline is summarised by its geometric mean with a family-clustered bootstrap interval,
plus 50/90/max percentiles and the callback-time share. Only problems attained by both solvers
in every repeat enter the ratio; all others are listed. Startup/import time is not included:
each worker process solves many problems, and the first problem of a worker (MATLAB JIT warm-up)
is flagged separately.
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "corpus"))
import schema  # noqa: E402
import targets as T  # noqa: E402
from score import score_records  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--baseline", default="fmincon-interior-point")
    ap.add_argument("--solver", default="mincon")
    ap.add_argument("--targets", default="v1")
    a = ap.parse_args()
    recs = []
    for pat in a.files:
        for f in glob.glob(pat):
            recs += schema.read_jsonl(f)
    scored = score_records(recs, T.load(a.targets))
    by = defaultdict(lambda: defaultdict(list))   # problem -> solver -> [records]
    for r in scored:
        by[r["problem"]][r["solver"]].append(r)
    rows, excluded = [], []
    ratios_by_fam = defaultdict(list)
    cb_share = {a.solver: [], a.baseline: []}
    for p in sorted(by):
        rs, rb = by[p].get(a.solver, []), by[p].get(a.baseline, [])
        if not rs or not rb:
            excluded.append((p, "missing records")); continue
        if not all(r["oracle"].get("target_attained") for r in rs + rb):
            excluded.append((p, f"not attained in every repeat ({a.solver}: {sum(bool(r['oracle'].get('target_attained')) for r in rs)}/{len(rs)}, {a.baseline}: {sum(bool(r['oracle'].get('target_attained')) for r in rb)}/{len(rb)})"))
            continue
        ws = np.median([r["time"]["solve_wall"] for r in rs]); wb = np.median([r["time"]["solve_wall"] for r in rb])
        rows.append((p, rs[0]["family"], rs[0]["n"], ws, wb, ws / wb, len(rs), len(rb)))
        ratios_by_fam[rs[0]["family"]].append(math.log(ws / wb))
        for name, group in ((a.solver, rs), (a.baseline, rb)):
            for r in group:
                if r["time"].get("solve_wall"):
                    cb_share[name].append(r["time"].get("callback", 0.0) / r["time"]["solve_wall"])
    logs = [v for vs in ratios_by_fam.values() for v in vs]
    gm = math.exp(np.mean(logs)) if logs else float("nan")
    rng = np.random.default_rng(0)
    fams = list(ratios_by_fam)
    boots = []
    for _ in range(2000):
        pick = rng.choice(len(fams), len(fams), replace=True)
        pooled = [v for i in pick for v in ratios_by_fam[fams[i]]]
        boots.append(math.exp(np.mean(pooled)))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    ratios = np.array([r[5] for r in rows])
    print(f"# Wall-time comparison {a.solver} / {a.baseline} ({len(rows)} problems attained by both in every repeat; {len(excluded)} excluded)\n")
    print(f"geo-mean ratio {gm:.4f}  95% family bootstrap [{lo:.4f}, {hi:.4f}]; median {np.median(ratios):.4f}; 90th pct {np.percentile(ratios, 90):.4f}; max {ratios.max():.4f}")
    print(f"median wall: {a.solver} {np.median([r[3] for r in rows]) * 1e3:.2f} ms, {a.baseline} {np.median([r[4] for r in rows]) * 1e3:.2f} ms")
    print(f"callback share of wall (median over runs): {a.solver} {np.median(cb_share[a.solver]):.2f}, {a.baseline} {np.median(cb_share[a.baseline]):.2f}\n")
    print(f"| problem | family | n | {a.solver} median s | {a.baseline} median s | ratio | repeats |\n|---|---|---:|---:|---:|---:|---:|")
    for r in sorted(rows, key=lambda r: -r[5]):
        print(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:.4f} | {r[4]:.4f} | {r[5]:.4f} | {r[6]}/{r[7]} |")
    if excluded:
        print("\nExcluded from the ratio (listed, never dropped silently):")
        for p, why in excluded:
            print(f"- {p}: {why}")


if __name__ == "__main__":
    main()
