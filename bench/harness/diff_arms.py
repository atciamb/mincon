"""List the records that differ between two arms of a scored file.

    python diff_arms.py ../results/abl-X/scored-portfolio.jsonl mincon "mincon@bfgs_rescale=0"

A record counts as changed when its model-boundary evaluation count, its native exit status or
its oracle attainment differs between the candidate and the baseline solver name. Used by every
whole-corpus ablation since abl-i1 to say exactly which problems an increment touched (the two
60 s-limited problems, COVQP_300 and DENSELAP_250, differ between any two runs by machine speed).
"""
import collections
import json
import sys


def main():
    path, cand, base = sys.argv[1], sys.argv[2], sys.argv[3]
    recs = collections.defaultdict(dict)
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        recs[r["problem"]][r["solver"]] = r
    changed = 0
    for p, d in sorted(recs.items()):
        a, b = d.get(cand), d.get(base)
        if not a or not b:
            continue
        ca = a["counts"]["f_model"] + a["counts"]["c_model"]
        cb = b["counts"]["f_model"] + b["counts"]["c_model"]
        va, vb = a["oracle"], b["oracle"]
        if ca != cb or a["native_status"] != b["native_status"] or va.get("target_attained") != vb.get("target_attained"):
            changed += 1
            print(f"{p:22s} cand: status {a['native_status']} evals {ca} att {va.get('target_attained')} "
                  f"wall {a['time']['solve_wall']:.1f} | base: status {b['native_status']} evals {cb} "
                  f"att {vb.get('target_attained')} wall {b['time']['solve_wall']:.1f}")
    print("changed records:", changed, "of", len(recs))


if __name__ == "__main__":
    main()
