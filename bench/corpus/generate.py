"""Emit MATLAB problem files, the corpus manifest and the equivalence probe file.

    python generate.py            # writes matlab/*.m, manifest.json, probes.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import spec  # noqa: E402
import families  # noqa: E402,F401  (registers every family)
import split  # noqa: E402


def main() -> None:
    mdir = os.path.join(HERE, "matlab")
    os.makedirs(mdir, exist_ok=True)
    manifest, probes = [], {}
    for s in spec.all_specs():
        with open(os.path.join(mdir, f"{s.name}.m"), "w", newline="\n") as fh:
            fh.write(s.matlab())
        manifest.append(s.manifest_row(split=split.assign(s.name, s.family, s.tags)))
        p = s.numpy()
        pts = p.probes(4)
        probes[s.name] = [dict(x=list(map(float, x)), f=p.f(x), grad=list(map(float, p.grad(x))),
                               c=list(map(float, p.cons(x))), jac=[list(map(float, r)) for r in p.jac(x)]) for x in pts]
    with open(os.path.join(HERE, "manifest.json"), "w") as fh:
        json.dump(dict(schema="corpus-manifest/1", problems=manifest), fh, indent=1, default=_json_num)
    with open(os.path.join(HERE, "probes.json"), "w") as fh:
        json.dump(probes, fh, default=_json_num)
    print(f"wrote {len(manifest)} problems to {mdir}, manifest.json and probes.json")


def _json_num(v):
    if isinstance(v, float):
        if np.isnan(v):
            return "nan"
        if np.isinf(v):
            return "inf" if v > 0 else "-inf"
    raise TypeError(type(v))


if __name__ == "__main__":
    main()
