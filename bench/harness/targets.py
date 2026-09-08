"""Frozen, versioned objective targets.

    python targets.py init                 -> targets_v1.json from closed-form/published references
    python targets.py update v2 results/*.jsonl
        -> new version: best-known problems get the best independently KKT-verified feasible objective
           across the given records; existing closed-form targets are kept unless a verified point beats
           them by more than the target tolerance (then the target is revised and the revision logged).
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "corpus"))
TARGET_DIR = os.path.join(os.path.dirname(HERE), "corpus")


def path(version):
    return os.path.join(TARGET_DIR, f"targets_{version}.json")


def load(version="v1"):
    return json.load(open(path(version)))


def init():
    import spec, families  # noqa: F401
    t = {}
    for s in spec.all_specs():
        if "diagnostic" in s.tags:
            t[s.name] = dict(target=None, kind="diagnostic", source=s.source)
        elif s.ref_f is not None and "best-known" not in s.tags:
            t[s.name] = dict(target=float(s.ref_f), kind="closed-form-or-published", source=s.source)
        elif s.ref_f is not None:
            t[s.name] = dict(target=float(s.ref_f), kind="best-known-published", source=s.source)
        else:
            t[s.name] = dict(target=None, kind="best-known-pending", source=s.source)
    out = dict(version="v1", created=time.strftime("%Y-%m-%d"), rel_tol=1e-4, feas_tol=1e-6, targets=t, revisions=[])
    json.dump(out, open(path("v1"), "w"), indent=1)
    print(f"wrote {path('v1')}: {sum(1 for v in t.values() if v['target'] is not None)} targets, "
          f"{sum(1 for v in t.values() if v['kind']=='best-known-pending')} pending")


def update(version, files, base="v1", stat_tol=1e-6):
    import schema, spec, families  # noqa: F401
    from oracle import assess
    T = load(base)
    recs = []
    for f in files:
        recs += schema.read_jsonl(f)
    best = {}
    cache = {}
    for r in recs:
        if r.get("outcome") != "ok" or r.get("x") is None:
            continue
        name = r["problem"]
        if name not in cache:
            cache[name] = spec.get(name).numpy()
        p = cache[name]
        v = assess(p, r["x"], r.get("lam"), r.get("zl"), r.get("zu"), feas_tol=T["feas_tol"], stat_tol=stat_tol)
        if v.feasible and (v.kkt_first_order or v.kkt_first_order_recovered):
            if name not in best or v.f < best[name][0]:
                best[name] = (v.f, r["solver"], r["run_id"])
    revisions = list(T["revisions"])
    for name, entry in T["targets"].items():
        if entry["kind"] == "diagnostic":
            continue
        if name in best:
            fbest, solver, rid = best[name]
            if entry["target"] is None:
                entry.update(target=fbest, kind="best-known-verified", source=f"reference run {solver} {rid}")
                revisions.append(dict(version=version, problem=name, action="set", value=fbest, by=solver, run_id=rid))
            elif fbest < entry["target"] - T["rel_tol"] * max(1.0, abs(entry["target"])):
                revisions.append(dict(version=version, problem=name, action="revise", old=entry["target"], value=fbest, by=solver, run_id=rid))
                entry.update(target=fbest, kind=entry["kind"] + "+revised", source=entry["source"] + f"; revised by {solver} {rid}")
    T.update(version=version, created=time.strftime("%Y-%m-%d"), revisions=revisions)
    json.dump(T, open(path(version), "w"), indent=1)
    print(f"wrote {path(version)}: {len([1 for r in revisions if r['version']==version])} changes")


if __name__ == "__main__":
    if sys.argv[1] == "init":
        init()
    else:
        args = sys.argv[3:]
        base = "v1"
        if "--base" in args:
            i = args.index("--base"); base = args[i + 1]; args = args[:i] + args[i + 2:]
        files = [f for pat in args if not pat.startswith("--") for f in glob.glob(pat)]
        update(sys.argv[2], files, base=base)
