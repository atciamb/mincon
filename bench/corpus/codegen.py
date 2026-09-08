"""Code generation for corpus expressions: NumPy callables and MATLAB source.

Symbols are named ``x_1 .. x_n``. NumPy code indexes one array argument
(``x[0]``), MATLAB code uses ``x(1)``. Long sums are chunked so neither the
Python compiler (recursion depth) nor MATLAB line limits are hit.
"""
from __future__ import annotations

import re

import numpy as np
import sympy as sp
from sympy.printing.numpy import NumPyPrinter
from sympy.printing.octave import OctaveCodePrinter

_PY = NumPyPrinter({"fully_qualified_modules": False})
_ML = OctaveCodePrinter()
_SYM = re.compile(r"\bx_(\d+)\b")
CHUNK = 64


def _chunks(e: sp.Expr):
    if isinstance(e, sp.Add) and len(e.args) > CHUNK:
        args = e.args
        return [sp.Add(*args[i:i + CHUNK]) for i in range(0, len(args), CHUNK)]
    return [e]


def py_expr(e: sp.Expr) -> str:
    return _SYM.sub(lambda m: f"x[{int(m.group(1)) - 1}]", _PY.doprint(e))


def ml_expr(e: sp.Expr) -> str:
    return _SYM.sub(lambda m: f"x({m.group(1)})", _ML.doprint(e))


def make_fn(exprs: list[sp.Expr], lam_syms: list[sp.Symbol] | None = None):
    """Compile ``exprs`` into ``fn(x[, lam]) -> np.ndarray`` (float64, len(exprs))."""
    lines = ["def _fn(x, lam=None):", "    out = np.empty(%d)" % len(exprs)]
    for k, e in enumerate(exprs):
        parts = _chunks(sp.sympify(e))
        lines.append(f"    acc = {py_expr(parts[0])}")
        for p in parts[1:]:
            lines.append(f"    acc = acc + ({py_expr(p)})")
        lines.append(f"    out[{k}] = acc")
    lines.append("    return out")
    src = "\n".join(lines)
    if lam_syms:
        for i, s in enumerate(lam_syms):
            src = re.sub(rf"\b{re.escape(str(s))}\b", f"lam[{i}]", src)
    ns = {"np": np, "numpy": np}
    from numpy import (sqrt, exp, log, sin, cos, tan, pi, arctan, sinh, cosh, tanh, abs, sign)  # noqa: F401
    ns.update(dict(sqrt=sqrt, exp=exp, log=log, sin=sin, cos=cos, tan=tan, pi=pi, arctan=arctan, sinh=sinh, cosh=cosh,
                   tanh=tanh, abs=abs, sign=sign))
    code = compile(src, "<corpus>", "exec")
    exec(code, ns)
    return ns["_fn"]


def ml_assign(target: str, e: sp.Expr) -> list[str]:
    parts = _chunks(sp.sympify(e))
    lines = [f"{target} = {ml_expr(parts[0])};"]
    for p in parts[1:]:
        lines.append(f"{target} = {target} + ({ml_expr(p)});")
    return lines
