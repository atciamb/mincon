"""Record schema v1 and JSON helpers (standards-compliant JSON: non-finite floats become strings)."""
from __future__ import annotations

import json
import math
import platform
import uuid

SCHEMA = "mincon-bench-record/1"


def new_record(**kw) -> dict:
    rec = dict(schema=SCHEMA, run_id=uuid.uuid4().hex, experiment=None, track=None, problem=None, family=None,
               split=None, checksum=None, n=None, m=None, solver=None, solver_version=None, options={},
               derivatives=None, host=host_info(), threads=None, budget={}, native_status=None, native_message=None,
               reported_success=None, x=None, lam=None, zl=None, zu=None, counts={}, time={}, outcome=None,
               error=None, notes=[], repeat=0)
    rec.update(kw)
    return rec


def host_info() -> dict:
    return dict(platform=platform.platform(), machine=platform.machine(), processor=platform.processor(),
                python=platform.python_version(), node=platform.node())


def _enc(o):
    if isinstance(o, float):
        if math.isnan(o):
            return "nan"
        if math.isinf(o):
            return "inf" if o > 0 else "-inf"
    try:
        import numpy as np
        if isinstance(o, np.ndarray):
            return [_enc(v) for v in o.tolist()]
        if isinstance(o, (np.floating,)):
            return _enc(float(o))
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
    except ImportError:
        pass
    raise TypeError(f"not JSON serializable: {type(o)}")


def _walk(o):
    if isinstance(o, dict):
        return {k: _walk(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_walk(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return _enc(o)
    try:
        import numpy as np
        if isinstance(o, np.ndarray):
            return _walk(o.tolist())
        if isinstance(o, np.generic):
            return _walk(o.item())
    except ImportError:
        pass
    return o


def dumps(rec: dict) -> str:
    return json.dumps(_walk(rec), allow_nan=False)


def loads(line: str) -> dict:
    def fix(o):
        if isinstance(o, dict):
            return {k: fix(v) for k, v in o.items()}
        if isinstance(o, list):
            return [fix(v) for v in o]
        if o == "nan":
            return float("nan")
        if o == "inf":
            return float("inf")
        if o == "-inf":
            return float("-inf")
        return o
    return fix(json.loads(line))


def read_jsonl(path):
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(loads(line))
    return out
