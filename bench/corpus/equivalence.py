"""Compare MATLAB and NumPy evaluations at the shared probe points (f, grad, c, J).

Run generate.py, then equivalence_matlab.m in MATLAB, then this. Exit code 1 on any mismatch.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOL = 1e-10  # relative to max(1, magnitude): both sides are exact arithmetic on the same expressions


def main() -> int:
    py = json.load(open(os.path.join(HERE, "probes.json")))
    ml = json.load(open(os.path.join(HERE, "matlab_probes.json")))
    bad, checked = [], 0
    for name, pts in py.items():
        if name not in ml:
            bad.append((name, "missing in MATLAB output")); continue
        mpts = ml[name]
        if isinstance(mpts, dict):
            mpts = [mpts]
        for j, (a, b) in enumerate(zip(pts, mpts)):
            for key in ("f", "grad", "c", "jac"):
                av = np.atleast_1d(np.asarray(a[key], float)).ravel()
                bv = np.atleast_1d(np.asarray(b.get(key, []), float)).ravel()
                if av.size != bv.size:
                    if av.size == 0 and bv.size == 0:
                        continue
                    bad.append((name, f"probe {j} {key}: size {av.size} vs {bv.size}")); continue
                scale = max(1.0, float(np.max(np.abs(av))) if av.size else 1.0)
                err = float(np.max(np.abs(av - bv))) / scale if av.size else 0.0
                checked += 1
                if not np.isfinite(err) or err > TOL:
                    bad.append((name, f"probe {j} {key}: rel err {err:.2e}"))
    print(f"checked {checked} quantities on {len(py)} problems; mismatches: {len(bad)}")
    for b in bad[:50]:
        print("  ", *b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
