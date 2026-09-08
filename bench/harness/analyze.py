"""Paired analysis of scored records.

    python analyze.py results/s2-dev/scored.jsonl --baseline fmincon-interior-point --metric f_model

Outputs (markdown to stdout, JSON next to the input): attainment per solver and family, paired
win/loss/tie tables against the baseline, family-clustered paired bootstrap 95% intervals for the
attainment difference, geometric-mean cost ratios on commonly attained problems with bootstrap
intervals, and Dolan–Moré performance-profile points. Every record is kept; exclusions are listed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import schema  # noqa: E402


def cost_of(r, metric):
    if metric == "wall":
        return float(r.get("time", {}).get("solve_wall", float("nan")))
    if metric == "total_model":
        c = r["counts"]; return float(c.get("f_model", 0) + c.get("c_model", 0))
    return float(r["counts"].get(metric, float("nan")))


def load(path, track=None, repeat=0):
    recs = schema.read_jsonl(path)
    recs = [r for r in recs if (track is None or r["track"] == track) and int(r.get("repeat", 0)) == repeat]
    by = defaultdict(dict)   # problem -> solver -> record (last wins; duplicates reported)
    dups = 0
    for r in recs:
        if r["solver"] in by[r["problem"]]:
            dups += 1
        by[r["problem"]][r["solver"]] = r
    return by, dups


def attained(r):
    return bool(r["oracle"].get("target_attained"))


def boot_ci(values_by_family, fn, B=2000, seed=0):
    """Cluster bootstrap over families: resample families with replacement, recompute fn on the pooled values."""
    fams = list(values_by_family)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(B):
        pick = rng.choice(len(fams), len(fams), replace=True)
        pooled = [v for i in pick for v in values_by_family[fams[i]]]
        if pooled:
            stats.append(fn(pooled))
    if not stats:
        return (float("nan"), float("nan"))
    return (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scored")
    ap.add_argument("--baseline", default="fmincon-interior-point")
    ap.add_argument("--metric", default="total_model")
    ap.add_argument("--track", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    by, dups = load(a.scored, a.track)
    solvers = sorted({s for d in by.values() for s in d})
    problems = sorted(by)
    fam_of = {p: next(iter(by[p].values()))["family"] for p in problems}
    out = {"n_problems": len(problems), "duplicates_overwritten": dups, "solvers": {}, "paired": {}}
    lines = [f"# Analysis of {a.scored}", "", f"{len(problems)} problems, solvers: {', '.join(solvers)}; duplicates overwritten: {dups}", ""]
    # attainment per solver
    lines += ["## Target attainment (feasible to 1e-6 and objective within the frozen relative tolerance)", "",
              "| solver | attained | of | rate | KKT (supplied) | KKT (recovered) | timeouts/crashes/errors |", "|---|---:|---:|---:|---:|---:|---:|"]
    for s in solvers:
        rs = [by[p][s] for p in problems if s in by[p]]
        att = sum(attained(r) for r in rs)
        kk = sum(bool(r["oracle"].get("kkt_first_order")) for r in rs)
        kr = sum(bool(r["oracle"].get("kkt_first_order_recovered")) for r in rs)
        bad = sum(r["outcome"] != "ok" for r in rs)
        scorable = sum(1 for r in rs if r["oracle"].get("target") is not None)
        out["solvers"][s] = dict(attained=att, scorable=scorable, kkt=kk, kkt_recovered=kr, not_ok=bad, n=len(rs))
        lines.append(f"| {s} | {att} | {scorable} | {att / max(1, scorable):.3f} | {kk} | {kr} | {bad} |")
    # per family
    fams = sorted(set(fam_of.values()))
    lines += ["", "## Attainment by family", "", "| family | n | " + " | ".join(solvers) + " |", "|---|---:|" + "---:|" * len(solvers)]
    for f in fams:
        ps = [p for p in problems if fam_of[p] == f]
        row = [f"{sum(attained(by[p][s]) for p in ps if s in by[p])}/{sum(1 for p in ps if s in by[p] and by[p][s]['oracle'].get('target') is not None)}" for s in solvers]
        lines.append(f"| {f} | {len(ps)} | " + " | ".join(row) + " |")
    # paired vs baseline
    b = a.baseline
    if b in solvers:
        lines += ["", f"## Paired comparison against `{b}` (metric for cost: {a.metric})", "",
                  "| solver | both | only solver | only baseline | neither | diff (pp) | 95% CI (family bootstrap) | geo-mean cost ratio solver/baseline (common) | 95% CI | n common |",
                  "|---|---:|---:|---:|---:|---:|---|---:|---|---:|"]
        for s in solvers:
            if s == b:
                continue
            both = only_s = only_b = neither = 0
            diffs_by_fam = defaultdict(list)
            ratios_by_fam = defaultdict(list)
            for p in problems:
                if s not in by[p] or b not in by[p] or by[p][s]["oracle"].get("target") is None:
                    continue
                sa, ba = attained(by[p][s]), attained(by[p][b])
                both += sa and ba; only_s += sa and not ba; only_b += ba and not sa; neither += (not sa) and (not ba)
                diffs_by_fam[fam_of[p]].append(float(sa) - float(ba))
                if sa and ba:
                    cs, cb = cost_of(by[p][s], a.metric), cost_of(by[p][b], a.metric)
                    if cs > 0 and cb > 0 and math.isfinite(cs) and math.isfinite(cb):
                        ratios_by_fam[fam_of[p]].append(math.log(cs / cb))
            n = both + only_s + only_b + neither
            diff = 100 * sum(v for vs in diffs_by_fam.values() for v in vs) / max(1, n)
            lo, hi = boot_ci(diffs_by_fam, lambda v: 100 * float(np.mean(v)))
            logs = [v for vs in ratios_by_fam.values() for v in vs]
            gm = math.exp(float(np.mean(logs))) if logs else float("nan")
            rlo, rhi = boot_ci(ratios_by_fam, lambda v: math.exp(float(np.mean(v)))) if logs else (float("nan"), float("nan"))
            out["paired"][s] = dict(both=both, only_solver=only_s, only_baseline=only_b, neither=neither, diff_pp=diff, diff_ci=[lo, hi],
                                    geo_mean_ratio=gm, ratio_ci=[rlo, rhi], n_common=len(logs))
            lines.append(f"| {s} | {both} | {only_s} | {only_b} | {neither} | {diff:+.1f} | [{lo:+.1f}, {hi:+.1f}] | {gm:.2f} | [{rlo:.2f}, {rhi:.2f}] | {len(logs)} |")
        # performance profile points (Dolan–Moré) on the metric, failures = inf
        lines += ["", f"## Performance profile points ({a.metric}; fraction of problems solved within tau x best)", "",
                  "| tau | " + " | ".join(solvers) + " |", "|---:|" + "---:|" * len(solvers)]
        costs = {s: {} for s in solvers}
        for p in problems:
            for s in solvers:
                r = by[p].get(s)
                costs[s][p] = cost_of(r, a.metric) if (r and attained(r)) else float("inf")
        scorable = [p for p in problems if any(math.isfinite(costs[s][p]) for s in solvers)]
        for tau in (1, 2, 4, 8, 16, 64, 256):
            row = []
            for s in solvers:
                k = 0
                for p in scorable:
                    best = min(costs[t][p] for t in solvers)
                    if math.isfinite(costs[s][p]) and costs[s][p] <= tau * best:
                        k += 1
                row.append(f"{k / max(1, len(scorable)):.2f}")
            lines.append(f"| {tau} | " + " | ".join(row) + " |")
    # exclusions / not-ok list
    lines += ["", "## Non-ok records (timeouts, crashes, errors) — all kept as failures above", ""]
    for p in problems:
        for s in solvers:
            r = by[p].get(s)
            if r and r["outcome"] != "ok":
                lines.append(f"- {p} / {s}: {r['outcome']}: {str(r.get('error'))[:120]}")
    text = "\n".join(lines)
    print(text)
    jpath = a.json or (os.path.splitext(a.scored)[0] + ".analysis.json")
    json.dump(out, open(jpath, "w"), indent=1)
    with open(os.path.splitext(a.scored)[0] + ".analysis.md", "w") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
