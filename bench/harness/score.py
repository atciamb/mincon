"""Attach independent oracle verdicts to raw records, against a frozen target version.

    python score.py --targets v1 results/s2-dev/*.jsonl -o results/s2-dev/scored.jsonl
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "corpus"))
import schema  # noqa: E402
import targets as T  # noqa: E402


def score_records(recs, tv, stat_tol=1e-6):
    import spec, families  # noqa: F401
    from oracle import assess
    cache = {}
    out = []
    for r in recs:
        name = r["problem"]
        entry = tv["targets"].get(name, {})
        tgt = entry.get("target")
        if r.get("outcome") == "ok" and r.get("x") is not None:
            if name not in cache:
                cache[name] = spec.get(name).numpy()
            v = assess(cache[name], r["x"], r.get("lam"), r.get("zl"), r.get("zu"), feas_tol=tv["feas_tol"], stat_tol=stat_tol,
                       target=tgt, target_rel_tol=tv["rel_tol"]).as_dict()
        else:
            v = dict(valid=False, feasible=False, target_attained=(False if tgt is not None else None), kkt_first_order=False,
                     kkt_first_order_recovered=False, f=float("nan"), violation=float("inf"), notes=[r.get("outcome")], target=tgt)
        v["target_kind"] = entry.get("kind")
        v["target_version"] = tv["version"]
        r = dict(r)
        r["oracle"] = v
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--targets", default="v1")
    ap.add_argument("--stat-tol", type=float, default=1e-6)
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    tv = T.load(a.targets)
    recs = []
    for pat in a.files:
        for f in glob.glob(pat):
            if not f.endswith("scored.jsonl"):
                recs += schema.read_jsonl(f)
    scored = score_records(recs, tv, a.stat_tol)
    with open(a.out, "w") as fh:
        for r in scored:
            fh.write(schema.dumps(r) + "\n")
    n_ok = sum(1 for r in scored if r["outcome"] == "ok")
    n_att = sum(1 for r in scored if r["oracle"].get("target_attained"))
    print(f"scored {len(scored)} records ({n_ok} ok); target attained: {n_att}; targets {tv['version']}")


if __name__ == "__main__":
    main()
