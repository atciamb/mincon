"""Score and tabulate the friction-audit records.

    python summarize.py ../results/s7-friction [--tag ""]

Reads every ``<solver>.jsonl`` in the directory (Python and MATLAB records), judges each returned
point with the independent oracle (the Python model; MATLAB definitions are first checked to agree
with it at x0 to 1e-8 relative), and writes ``scored.jsonl`` and ``summary.md``.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "harness"))
import problems  # noqa: E402
from run import judge  # noqa: E402


BASELINE_SOLVERS = {"mincon-fmincon", "scipy-slsqp", "scipy-trust-constr", "fmincon-interior-point", "fmincon-sqp"}


def _num(v):
    if isinstance(v, str):
        return float(v) if v in ("nan", "inf", "-inf", "NaN", "Inf", "-Inf") else float("nan")
    return float(v) if v is not None else float("nan")


def load(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def equivalence(rec, p):
    """MATLAB vs Python model at x0: relative agreement of f and every constraint row."""
    can = p.canonical()
    f0 = can.f(p.x0)
    c0 = can.cons(p.x0)
    fm = _num(rec.get("f0"))
    cm = np.asarray(rec.get("c0", []), float).reshape(-1)
    df = abs(fm - f0) / max(1.0, abs(f0))
    if cm.size != c0.size:
        return df, float("inf")
    dc = float(np.max(np.abs(cm - c0) / np.maximum(1.0, np.abs(c0)))) if c0.size else 0.0
    return df, dc


def score(records):
    out = []
    for rec in records:
        p = problems.get(rec["problem"])
        if rec["solver"].startswith("fmincon"):
            df, dc = equivalence(rec, p)
            rec["equivalence_at_x0"] = dict(f_rel=df, c_rel=dc, ok=bool(df <= 1e-8 and dc <= 1e-8))
        x = rec.get("x")
        if rec.get("outcome") == "ok" and x is not None and len(x) == p.n:
            rec["verdict"] = judge(p, np.asarray(x, float))
        else:
            rec["verdict"] = dict(valid=False, target_attained=False, feasible=False, kkt_recovered=False)
        out.append(rec)
    return out


def _fmt_msg(rec):
    m = rec.get("message") or rec.get("error") or ""
    m = m.strip().splitlines()[0] if m.strip() else ""
    return m[:90]


def table(scored):
    solvers = []
    for r in scored:
        if r["solver"] not in solvers:
            solvers.append(r["solver"])
    names = problems.all_names()
    by = {(r["problem"], r["solver"]): r for r in scored}
    lines = []
    # 1. attainment matrix
    lines.append("| problem | " + " | ".join(solvers) + " |")
    lines.append("|---|" + "---|" * len(solvers))
    tot = {s: 0 for s in solvers}
    for n in names:
        cells = []
        for s in solvers:
            r = by.get((n, s))
            if r is None:
                cells.append("-")
                continue
            v = r["verdict"]
            att = v.get("target_attained")
            tot[s] += bool(att)
            rep = r.get("reported_success")
            fm = r.get("counts", {}).get("f_model", -1)
            mark = "Y" if att else ("err" if r.get("outcome") != "ok" else "n")
            if bool(att) == bool(rep):
                flag = ""
            elif rep and not att:
                # reported success without the reference optimum: a false certificate
                # unless the oracle certifies the point as first-order stationary
                flag = "k" if v.get("kkt_recovered") else "!"
            else:
                flag = "?"
            cells.append(f"{mark}{flag} {fm}")
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    lines.append("| **attained** | " + " | ".join(f"**{tot[s]}/{len(names)}**" for s in solvers) + " |")
    lines.append("")
    lines.append("Cell: Y attained (feasible to 1e-6, objective within 1e-4 relative of the reference) or n / err, "
                 "then model-boundary objective evaluations. `!` marks a false certificate (success reported at a "
                 "point the oracle does not certify); `k` marks success reported at a point the oracle certifies as "
                 "first-order stationary but which is not the reference optimum; `?` marks attainment the solver "
                 "did not report as success.")
    lines.append("")
    # 2. per-solver detail
    for s in solvers:
        lines.append(f"### {s}")
        lines.append("")
        lines.append("| problem | attained | reported | exit | KKT | f_model | c_model | nit | wall s | message |")
        lines.append("|---|---|---|---|---|---:|---:|---:|---:|---|")
        for n in names:
            r = by.get((n, s))
            if r is None:
                continue
            v = r["verdict"]
            c = r.get("counts", {})
            status = r.get("status", r.get("exitflag", ""))
            extra = ""
            if "x_err" in v and v.get("param_ok") is not None:
                extra = f" (param err {v['x_err']:.1e}, {'ok' if v['param_ok'] else 'off'})"
            gap = v.get("target_gap")
            gap_s = "" if gap is None or (isinstance(gap, float) and math.isnan(gap)) else f" gap {gap:.1e}"
            lines.append(f"| {n} | {'Y' if v.get('target_attained') else 'n'}{gap_s}{extra} | {r.get('reported_success')} | "
                         f"{status} | {'Y' if v.get('kkt_recovered') else 'n'} | {c.get('f_model', '')} | {c.get('c_model', '')} | "
                         f"{r.get('nit', r.get('iterations', ''))} | {_num(r.get('wall')):.2f} | {_fmt_msg(r)} |")
        lines.append("")
    # 3. equivalence
    eq = [r for r in scored if "equivalence_at_x0" in r]
    if eq:
        worst = max(max(r["equivalence_at_x0"]["f_rel"], r["equivalence_at_x0"]["c_rel"]) for r in eq)
        bad = [f"{r['problem']}/{r['solver']}" for r in eq if not r["equivalence_at_x0"]["ok"]]
        lines.append(f"MATLAB/Python definition agreement at x0 over {len(eq)} records: worst relative difference "
                     f"{worst:.1e}; records above 1e-8: {bad if bad else 'none'}.")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    recs = []
    for path in sorted(glob.glob(os.path.join(a.dir, "*.jsonl"))):
        base = os.path.basename(path)
        if base.startswith("scored"):
            continue
        if a.tag and not base.endswith(a.tag + ".jsonl"):
            continue
        if not a.tag and base[:-len(".jsonl")] not in BASELINE_SOLVERS:
            continue  # tagged candidate runs are summarised with --tag
        recs.extend(load(path))
    scored = score(recs)
    with open(os.path.join(a.dir, f"scored{a.tag}.jsonl"), "w", encoding="utf-8") as fh:
        for r in scored:
            fh.write(json.dumps(r, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
    md = table(scored)
    with open(os.path.join(a.dir, f"summary{a.tag}.md"), "w", encoding="utf-8") as fh:
        fh.write(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
