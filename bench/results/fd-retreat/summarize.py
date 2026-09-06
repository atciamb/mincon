"""Compare the unchanged development harness before and after FD correction."""
from collections import Counter
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.startswith(("HS", "TORTURE_")):
            p = line.split()
            rows.append(dict(problem=p[0], iterations=int(p[3]),
                             f_evals=int(p[4]), status=p[6], verdict=p[7]))
    assert len(rows) == 54
    return rows


before = read(ROOT.parent / "restoration" / "current-testset.txt")
after = read(ROOT / "current-testset.txt")
assert [r["problem"] for r in before] == [r["problem"] for r in after]
summary = {
    "baseline_commit": "e6c95ed",
    "scope": "54 development fixtures, unchanged harness; no held-out comparison",
    "baseline_statuses": dict(Counter(r["status"] for r in before)),
    "current_statuses": dict(Counter(r["status"] for r in after)),
    "baseline_verdicts": dict(Counter(r["verdict"] for r in before)),
    "current_verdicts": dict(Counter(r["verdict"] for r in after)),
    "baseline_total_f_evals": sum(r["f_evals"] for r in before),
    "current_total_f_evals": sum(r["f_evals"] for r in after),
    "changed_rows": [dict(before=a, after=b) for a, b in zip(before, after) if a != b],
}
(ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
