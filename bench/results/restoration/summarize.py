"""Regenerate summary.json from the checked-in development tables."""
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_table(name):
    rows = []
    for line in (ROOT / name).read_text(encoding="utf-8-sig").splitlines():
        if not line.startswith(("HS", "TORTURE_")):
            continue
        parts = line.split()
        row = dict(problem=parts[0], n=int(parts[1]), m=int(parts[2]),
                   iterations=int(parts[3]), f_evals=int(parts[4]),
                   reported_optimality=float(parts[5]))
        if parts[6] in {"PASS", "FAIL", "soft", "BETR", "LIE"}:
            row.update(status=None, verdict=parts[6])
        else:
            row.update(status=parts[6], verdict=parts[7])
        rows.append(row)
    assert len(rows) == 54
    return rows


baseline = read_table("baseline-testset.txt")
current = read_table("current-testset.txt")
assert {r["problem"] for r in baseline} == {r["problem"] for r in current}
summary = {
    "baseline_commit": "616cd5edb67ffbf8fdc5ecce7e7137d114aeca7b",
    "scope": "development fixtures; not CUTEst or a matched competitor comparison",
    "baseline_harness": "original scaffold verdicts; not independently rescored",
    "current_harness": "fresh objective/feasibility; separate HS13 status check",
    "baseline_verdicts": dict(collections.Counter(r["verdict"] for r in baseline)),
    "current_verdicts": dict(collections.Counter(r["verdict"] for r in current)),
    "current_statuses": dict(collections.Counter(r["status"] for r in current)),
    "baseline_total_f_evals": sum(r["f_evals"] for r in baseline),
    "current_total_f_evals": sum(r["f_evals"] for r in current),
    "accounting_caveat": "Current counts include failed FD batches; final redundant gradient removed. Baseline omits some failed probes.",
    "baseline": baseline,
    "current": current,
}
(ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps({k: v for k, v in summary.items() if k not in {"baseline", "current"}}, indent=2))
