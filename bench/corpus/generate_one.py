"""Generate corpus artefacts one problem per process, and merge them into the committed files.

    python generate_one.py NAME --frag DIR        # matlab/NAME.m, DIR/NAME.manifest.json, DIR/NAME.probes.json
    python generate_one.py --merge DIR NAME ...   # fold the fragments into manifest.json and probes.json

`generate.py` rebuilds every problem in one process: that runs out of memory on the dense
families and lets a different BLAS perturb the last digits of references that are already
committed. Here a new problem is built alone, and the merge rewrites `manifest.json` and
`probes.json` from their parsed contents with the new entries inserted in name order, so every
existing entry keeps its bytes (JSON floats round-trip through `repr`).

Probe points: the start plus four seeded perturbations, as `NumpyProblem.probes`. A model whose
domain is not a box (an ordered chamber, a logarithm on a bound) can be undefined at a
perturbation; each such point is pulled towards the start by halving until the model and its
derivatives are finite, so the MATLAB side is never asked for a complex value.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _finite(p, x) -> bool:
    with np.errstate(all="ignore"):
        vals = [np.atleast_1d(p.f(x)), p.grad(x), p.cons(x), p.jac(x).ravel()]
    return all(np.all(np.isfinite(v)) for v in vals)


def generate(name: str, frag: str) -> None:
    import spec
    import families  # noqa: F401
    import split
    s = spec.get(name)
    os.makedirs(frag, exist_ok=True)
    with open(os.path.join(HERE, "matlab", f"{s.name}.m"), "w", newline="\n", encoding="utf-8") as fh:
        fh.write(s.matlab())
    row = s.manifest_row(split=split.assign(s.name, s.family, s.tags))
    p = s.numpy()
    assert _finite(p, p.x0), f"{name}: the model is not finite at its start"
    pts, shrunk = [], 0
    for x in p.probes(4):
        k = 0
        while not _finite(p, x):
            x = p.x0 + 0.5 * (x - p.x0)
            k += 1
            assert k < 60, f"{name}: no finite probe point near the start"
        shrunk += 1 if k else 0
        pts.append(x)
    probes = [dict(x=list(map(float, x)), f=p.f(x), grad=list(map(float, p.grad(x))), c=list(map(float, p.cons(x))),
                   jac=[list(map(float, r)) for r in p.jac(x)]) for x in pts]
    with open(os.path.join(frag, f"{name}.manifest.json"), "w", newline="\n", encoding="utf-8") as fh:
        json.dump(row, fh, default=_json_num)
    with open(os.path.join(frag, f"{name}.probes.json"), "w", newline="\n", encoding="utf-8") as fh:
        json.dump(probes, fh, default=_json_num)
    print(f"{name}: n = {s.n}, m = {s.m}, split {row['split']}, checksum {row['checksum']}, "
          f"{shrunk} of {len(pts)} probe points pulled into the domain")


def merge(frag: str, names: list[str]) -> None:
    mpath, ppath = os.path.join(HERE, "manifest.json"), os.path.join(HERE, "probes.json")
    man = json.load(open(mpath, encoding="utf-8"))
    rows = {r["name"]: r for r in man["problems"]}
    probes = json.load(open(ppath, encoding="utf-8"))
    for name in names:
        new = json.load(open(os.path.join(frag, f"{name}.manifest.json"), encoding="utf-8"))
        old = rows.get(name)
        if old is not None and old["checksum"] != new["checksum"]:
            raise SystemExit(f"{name}: the model's checksum changed ({old['checksum']} -> {new['checksum']}); not merged")
        rows[name] = new
        probes[name] = json.load(open(os.path.join(frag, f"{name}.probes.json"), encoding="utf-8"))
    man["problems"] = [rows[k] for k in sorted(rows)]
    with open(mpath, "w", newline="\n", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1, default=_json_num)
    with open(ppath, "w", newline="\n", encoding="utf-8") as fh:
        json.dump({k: probes[k] for k in sorted(probes)}, fh, default=_json_num)
    print(f"merged {len(names)} problems: manifest has {len(rows)}, probes {len(probes)}")


def _json_num(v):
    if isinstance(v, float):
        if np.isnan(v):
            return "nan"
        if np.isinf(v):
            return "inf" if v > 0 else "-inf"
    raise TypeError(type(v))


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--merge":
        merge(args[1], args[2:])
    else:
        generate(args[0], args[args.index("--frag") + 1])
