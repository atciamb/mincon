"""Hock–Schittkowski problems (1981), transcribed as SymPy expressions.

Book convention g(x) >= 0 becomes rows (0, +inf); h(x) = 0 becomes (0, 0).
``ref_f`` is the published optimum; ``ref_x`` a published minimizer when the
transcription was checked at it. Problems also present in the Rust
``mincon-testset`` crate are cross-checked against it by
``test_equivalence.py`` (objective and constraint values at probe points).
"""
from __future__ import annotations

import math

import sympy as sp

from spec import INF, Spec, register, xs

pi, sqrt, exp, log, sin, cos = sp.pi, sp.sqrt, sp.exp, sp.log, sp.sin, sp.cos
S2, S3 = math.sqrt(2.0), math.sqrt(3.0)


def hs(name, x, f, c=(), cl=(), cu=(), xl=None, xu=None, x0=None, ref_f=None, ref_x=None, notes="", tags=()):
    n = len(x)
    return Spec(name=name, family="hs", x=list(x), f=sp.sympify(f), c=[sp.sympify(e) for e in c],
                cl=list(cl), cu=list(cu), xl=list(xl) if xl is not None else [-INF] * n,
                xu=list(xu) if xu is not None else [INF] * n, x0=list(x0), ref_f=ref_f,
                ref_x=ref_x, source="Hock & Schittkowski 1981" + (f"; {notes}" if notes else ""),
                tags=list(tags), notes=notes)


def ge(k):  # k inequalities g >= 0
    return [0.0] * k, [INF] * k


def eq(k):
    return [0.0] * k, [0.0] * k


def rosen(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


# ---- bound constrained ----
@register
def HS1():
    x = xs(2); return hs("HS1", x, rosen(x), xl=[-INF, -1.5], x0=[-2, 1], ref_f=0.0, ref_x=[[1, 1]])

@register
def HS2():
    x = xs(2); return hs("HS2", x, rosen(x), xl=[-INF, 1.5], x0=[-2, 1], ref_f=0.0504261879, ref_x=[[1.224370749, 1.5]])

@register
def HS3():
    x = xs(2); return hs("HS3", x, x[1] + 1e-5 * (x[1] - x[0]) ** 2, xl=[-INF, 0], x0=[10, 1], ref_f=0.0, ref_x=[[0, 0]])

@register
def HS4():
    x = xs(2); return hs("HS4", x, (x[0] + 1) ** 3 / 3 + x[1], xl=[1, 0], x0=[1.125, 0.125], ref_f=8 / 3, ref_x=[[1, 0]])

@register
def HS5():
    x = xs(2)
    return hs("HS5", x, sin(x[0] + x[1]) + (x[0] - x[1]) ** 2 - 1.5 * x[0] + 2.5 * x[1] + 1,
              xl=[-1.5, -3], xu=[4, 3], x0=[0, 0], ref_f=-1.9132229549810, ref_x=[[-math.pi / 3 + 0.5, -math.pi / 3 - 0.5]])

@register
def HS38():
    x = xs(4)
    f = (100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2 + 90 * (x[3] - x[2] ** 2) ** 2 + (1 - x[2]) ** 2
         + 10.1 * ((x[1] - 1) ** 2 + (x[3] - 1) ** 2) + 19.8 * (x[1] - 1) * (x[3] - 1))
    return hs("HS38", x, f, xl=[-10] * 4, xu=[10] * 4, x0=[-3, -1, -3, -1], ref_f=0.0, ref_x=[[1, 1, 1, 1]])

@register
def HS45():
    x = xs(5)
    return hs("HS45", x, 2 - x[0] * x[1] * x[2] * x[3] * x[4] / 120, xl=[0] * 5, xu=[1, 2, 3, 4, 5], x0=[2] * 5,
              ref_f=1.0, ref_x=[[1, 2, 3, 4, 5]])

@register
def HS110():
    x = xs(10)
    f = sum(log(xi - 2) ** 2 + log(10 - xi) ** 2 for xi in x) - sp.Mul(*x) ** sp.Rational(1, 5)
    return hs("HS110", x, f, xl=[2.001] * 10, xu=[9.999] * 10, x0=[9] * 10, ref_f=-45.77847,
              notes="objective undefined outside the box", tags=["domain"])

# ---- equality constrained ----
@register
def HS6():
    x = xs(2); cl, cu = eq(1)
    return hs("HS6", x, (1 - x[0]) ** 2, [10 * (x[1] - x[0] ** 2)], cl, cu, x0=[-1.2, 1], ref_f=0.0, ref_x=[[1, 1]])

@register
def HS7():
    x = xs(2); cl, cu = eq(1)
    return hs("HS7", x, log(1 + x[0] ** 2) - x[1], [(1 + x[0] ** 2) ** 2 + x[1] ** 2 - 4], cl, cu, x0=[2, 2],
              ref_f=-S3, ref_x=[[0, S3]])

@register
def HS8():
    x = xs(2); cl, cu = eq(2)
    return hs("HS8", x, -1, [x[0] ** 2 + x[1] ** 2 - 25, x[0] * x[1] - 9], cl, cu, x0=[2, 1], ref_f=-1.0,
              notes="constant objective, feasibility only")

@register
def HS9():
    x = xs(2); cl, cu = eq(1)
    return hs("HS9", x, sin(pi * x[0] / 12) * cos(pi * x[1] / 16), [4 * x[0] - 3 * x[1]], cl, cu, x0=[0, 0],
              ref_f=-0.5, notes="infinitely many minimizers x=(12k-3, 16k-4)")

@register
def HS26():
    x = xs(3); cl, cu = eq(1)
    return hs("HS26", x, (x[0] - x[1]) ** 2 + (x[1] - x[2]) ** 4, [(1 + x[1] ** 2) * x[0] + x[2] ** 4 - 3], cl, cu,
              x0=[-2.6, 2, 2], ref_f=0.0, ref_x=[[1, 1, 1]])

@register
def HS27():
    x = xs(3); cl, cu = eq(1)
    return hs("HS27", x, 0.01 * (x[0] - 1) ** 2 + (x[1] - x[0] ** 2) ** 2, [x[0] + x[2] ** 2 + 1], cl, cu,
              x0=[2, 2, 2], ref_f=0.04, ref_x=[[-1, 1, 0]])

@register
def HS28():
    x = xs(3); cl, cu = eq(1)
    return hs("HS28", x, (x[0] + x[1]) ** 2 + (x[1] + x[2]) ** 2, [x[0] + 2 * x[1] + 3 * x[2] - 1], cl, cu,
              x0=[-4, 1, 1], ref_f=0.0, ref_x=[[0.5, -0.5, 0.5]])

@register
def HS39():
    x = xs(4); cl, cu = eq(2)
    return hs("HS39", x, -x[0], [x[1] - x[0] ** 3 - x[2] ** 2, x[0] ** 2 - x[1] - x[3] ** 2], cl, cu, x0=[2] * 4,
              ref_f=-1.0, ref_x=[[1, 1, 0, 0]])

@register
def HS40():
    x = xs(4); cl, cu = eq(3)
    return hs("HS40", x, -x[0] * x[1] * x[2] * x[3], [x[0] ** 3 + x[1] ** 2 - 1, x[0] ** 2 * x[3] - x[2], x[3] ** 2 - x[1]],
              cl, cu, x0=[0.8] * 4, ref_f=-0.25)

@register
def HS42():
    x = xs(4); cl, cu = eq(2)
    return hs("HS42", x, (x[0] - 1) ** 2 + (x[1] - 2) ** 2 + (x[2] - 3) ** 2 + (x[3] - 4) ** 2,
              [x[0] - 2, x[2] ** 2 + x[3] ** 2 - 2], cl, cu, x0=[1] * 4, ref_f=28 - 10 * S2, ref_x=[[2, 2, 0.6 * S2, 0.8 * S2]])

# ---- inequality constrained ----
@register
def HS10():
    x = xs(2); cl, cu = ge(1)
    return hs("HS10", x, x[0] - x[1], [-3 * x[0] ** 2 + 2 * x[0] * x[1] - x[1] ** 2 + 1], cl, cu, x0=[-10, 10],
              ref_f=-1.0, ref_x=[[0, 1]])

@register
def HS11():
    x = xs(2); cl, cu = ge(1)
    return hs("HS11", x, (x[0] - 5) ** 2 + x[1] ** 2 - 25, [-x[0] ** 2 + x[1]], cl, cu, x0=[4.9, 0.1], ref_f=-8.4984642232)

@register
def HS12():
    x = xs(2); cl, cu = ge(1)
    return hs("HS12", x, 0.5 * x[0] ** 2 + x[1] ** 2 - x[0] * x[1] - 7 * x[0] - 7 * x[1], [25 - 4 * x[0] ** 2 - x[1] ** 2],
              cl, cu, x0=[0, 0], ref_f=-30.0, ref_x=[[2, 3]])

@register
def HS13():
    x = xs(2); cl, cu = ge(1)
    return hs("HS13", x, (x[0] - 2) ** 2 + x[1] ** 2, [(1 - x[0]) ** 3 - x[1]], cl, cu, xl=[0, 0], x0=[-2, -2], ref_f=1.0,
              ref_x=[[1, 0]], notes="MFCQ fails at the solution", tags=["degenerate"])

@register
def HS14():
    x = xs(2)
    return hs("HS14", x, (x[0] - 2) ** 2 + (x[1] - 1) ** 2, [x[0] - 2 * x[1] + 1, -0.25 * x[0] ** 2 - x[1] ** 2 + 1],
              [0, 0], [0, INF], x0=[2, 2], ref_f=1.3934646209)

@register
def HS15():
    x = xs(2); cl, cu = ge(2)
    return hs("HS15", x, rosen(x), [x[0] * x[1] - 1, x[0] + x[1] ** 2], cl, cu, xu=[0.5, INF], x0=[-2, 1], ref_f=306.5,
              ref_x=[[0.5, 2]])

@register
def HS16():
    x = xs(2); cl, cu = ge(2)
    return hs("HS16", x, rosen(x), [x[0] + x[1] ** 2, x[0] ** 2 + x[1]], cl, cu, xl=[-2, -INF], xu=[0.5, 1], x0=[-2, 1],
              ref_f=0.25, ref_x=[[0.5, 0.25]], notes="local method from x0 reaches f=3.98206 (verified local minimum)",
              tags=["multiple-local-minima"])

@register
def HS17():
    x = xs(2); cl, cu = ge(2)
    return hs("HS17", x, 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2, [x[1] ** 2 - x[0], x[0] ** 2 - x[1]], cl, cu,
              xl=[-2, -INF], xu=[0.5, 1], x0=[-2, 1], ref_f=1.0, ref_x=[[0, 0]])

@register
def HS18():
    x = xs(2); cl, cu = ge(2)
    return hs("HS18", x, 0.01 * x[0] ** 2 + x[1] ** 2, [x[0] * x[1] - 25, x[0] ** 2 + x[1] ** 2 - 25], cl, cu,
              xl=[2, 0], xu=[50, 50], x0=[2, 2], ref_f=5.0, ref_x=[[math.sqrt(250), math.sqrt(2.5)]])

@register
def HS19():
    x = xs(2); cl, cu = ge(2)
    return hs("HS19", x, (x[0] - 10) ** 3 + (x[1] - 20) ** 3, [(x[0] - 5) ** 2 + (x[1] - 5) ** 2 - 100, -(x[1] - 5) ** 2 - (x[0] - 6) ** 2 + 82.81],
              cl, cu, xl=[13, 0], xu=[100, 100], x0=[20.1, 5.84], ref_f=-6961.81381, ref_x=[[14.095, 0.84296079]])

@register
def HS20():
    x = xs(2); cl, cu = ge(3)
    return hs("HS20", x, rosen(x), [x[0] + x[1] ** 2, x[0] ** 2 + x[1], x[0] ** 2 + x[1] ** 2 - 1], cl, cu,
              xl=[-0.5, -INF], xu=[0.5, INF], x0=[-2, 1], ref_f=38.19872, ref_x=[[0.5, math.sqrt(3) / 2]])

@register
def HS21():
    x = xs(2); cl, cu = ge(1)
    return hs("HS21", x, 0.01 * x[0] ** 2 + x[1] ** 2 - 100, [10 * x[0] - x[1] - 10], cl, cu, xl=[2, -50], xu=[50, 50],
              x0=[-1, -1], ref_f=-99.96, ref_x=[[2, 0]], notes="x0 violates a bound")

@register
def HS22():
    x = xs(2); cl, cu = ge(2)
    return hs("HS22", x, (x[0] - 2) ** 2 + (x[1] - 1) ** 2, [-x[0] - x[1] + 2, -x[0] ** 2 + x[1]], cl, cu, x0=[2, 2],
              ref_f=1.0, ref_x=[[1, 1]])

@register
def HS23():
    x = xs(2); cl, cu = ge(5)
    return hs("HS23", x, x[0] ** 2 + x[1] ** 2,
              [x[0] + x[1] - 1, x[0] ** 2 + x[1] ** 2 - 1, 9 * x[0] ** 2 + x[1] ** 2 - 9, x[0] ** 2 - x[1], x[1] ** 2 - x[0]],
              cl, cu, xl=[-50, -50], xu=[50, 50], x0=[3, 1], ref_f=2.0, ref_x=[[1, 1]])

@register
def HS24():
    x = xs(2); cl, cu = ge(3)
    return hs("HS24", x, ((x[0] - 3) ** 2 - 9) * x[1] ** 3 / (27 * sqrt(3)),
              [x[0] / sqrt(3) - x[1], x[0] + sqrt(3) * x[1], -x[0] - sqrt(3) * x[1] + 6], cl, cu, xl=[0, 0], x0=[1, 0.5],
              ref_f=-1.0, ref_x=[[3, S3]])

@register
def HS29():
    x = xs(3); cl, cu = ge(1)
    return hs("HS29", x, -x[0] * x[1] * x[2], [-x[0] ** 2 - 2 * x[1] ** 2 - 4 * x[2] ** 2 + 48], cl, cu, x0=[1, 1, 1],
              ref_f=-16 * S2, ref_x=[[4, 2 * S2, 2], [-4, -2 * S2, 2], [4, -2 * S2, -2], [-4, 2 * S2, -2]])

@register
def HS30():
    x = xs(3); cl, cu = ge(1)
    return hs("HS30", x, x[0] ** 2 + x[1] ** 2 + x[2] ** 2, [x[0] ** 2 + x[1] ** 2 - 1], cl, cu, xl=[1, -10, -10], xu=[10] * 3,
              x0=[1, 1, 1], ref_f=1.0, ref_x=[[1, 0, 0]])

@register
def HS31():
    x = xs(3); cl, cu = ge(1)
    return hs("HS31", x, 9 * x[0] ** 2 + x[1] ** 2 + 9 * x[2] ** 2, [x[0] * x[1] - 1], cl, cu, xl=[-10, 1, -10], xu=[10, 10, 1],
              x0=[1, 1, 1], ref_f=6.0, ref_x=[[1 / S3, S3, 0]])

@register
def HS32():
    x = xs(3)
    return hs("HS32", x, (x[0] + 3 * x[1] + x[2]) ** 2 + 4 * (x[0] - x[1]) ** 2,
              [6 * x[1] + 4 * x[2] - x[0] ** 3 - 3, 1 - x[0] - x[1] - x[2]], [0, 0], [INF, 0], xl=[0] * 3, x0=[0.1, 0.7, 0.2],
              ref_f=1.0, ref_x=[[0, 0, 1]])

@register
def HS33():
    x = xs(3); cl, cu = ge(2)
    return hs("HS33", x, (x[0] - 1) * (x[0] - 2) * (x[0] - 3) + x[2], [x[2] ** 2 - x[1] ** 2 - x[0] ** 2, x[0] ** 2 + x[1] ** 2 + x[2] ** 2 - 4],
              cl, cu, xl=[0, 0, 0], xu=[INF, INF, 5], x0=[0, 0, 3], ref_f=S2 - 6, ref_x=[[0, S2, S2]])

@register
def HS34():
    x = xs(3); cl, cu = ge(2)
    return hs("HS34", x, -x[0], [x[1] - exp(x[0]), x[2] - exp(x[1])], cl, cu, xl=[0, 0, 0], xu=[100, 100, 10], x0=[0, 1.05, 2.9],
              ref_f=-0.8340324459, ref_x=[[math.log(math.log(10)), math.log(10), 10]], tags=["overflow"])

@register
def HS35():
    x = xs(3); cl, cu = ge(1)
    return hs("HS35", x, 9 - 8 * x[0] - 6 * x[1] - 4 * x[2] + 2 * x[0] ** 2 + 2 * x[1] ** 2 + x[2] ** 2 + 2 * x[0] * x[1] + 2 * x[0] * x[2],
              [3 - x[0] - x[1] - 2 * x[2]], cl, cu, xl=[0] * 3, x0=[0.5] * 3, ref_f=1 / 9, ref_x=[[4 / 3, 7 / 9, 4 / 9]])

@register
def HS36():
    x = xs(3); cl, cu = ge(1)
    return hs("HS36", x, -x[0] * x[1] * x[2], [72 - x[0] - 2 * x[1] - 2 * x[2]], cl, cu, xl=[0] * 3, xu=[20, 11, 42], x0=[10] * 3,
              ref_f=-3300.0, ref_x=[[20, 11, 15]])

@register
def HS37():
    x = xs(3); cl, cu = ge(2)
    return hs("HS37", x, -x[0] * x[1] * x[2], [72 - x[0] - 2 * x[1] - 2 * x[2], x[0] + 2 * x[1] + 2 * x[2]], cl, cu,
              xl=[0] * 3, xu=[42] * 3, x0=[10] * 3, ref_f=-3456.0, ref_x=[[24, 12, 12]])

@register
def HS41():
    x = xs(4); cl, cu = eq(1)
    return hs("HS41", x, 2 - x[0] * x[1] * x[2], [x[0] + 2 * x[1] + 2 * x[2] - x[3]], cl, cu, xl=[0] * 4, xu=[1, 1, 1, 2],
              x0=[2] * 4, ref_f=52 / 27, ref_x=[[2 / 3, 1 / 3, 1 / 3, 2]])

@register
def HS43():
    x = xs(4); cl, cu = ge(3)
    f = x[0] ** 2 + x[1] ** 2 + 2 * x[2] ** 2 + x[3] ** 2 - 5 * x[0] - 5 * x[1] - 21 * x[2] + 7 * x[3]
    c = [8 - x[0] ** 2 - x[1] ** 2 - x[2] ** 2 - x[3] ** 2 - x[0] + x[1] - x[2] + x[3],
         10 - x[0] ** 2 - 2 * x[1] ** 2 - x[2] ** 2 - 2 * x[3] ** 2 + x[0] + x[3],
         5 - 2 * x[0] ** 2 - x[1] ** 2 - x[2] ** 2 - 2 * x[0] + x[1] + x[3]]
    return hs("HS43", x, f, c, cl, cu, x0=[0] * 4, ref_f=-44.0, ref_x=[[0, 1, 2, -1]], notes="Rosen-Suzuki")

@register
def HS44():
    x = xs(4); cl, cu = ge(6)
    f = x[0] - x[1] - x[2] - x[0] * x[2] + x[0] * x[3] + x[1] * x[2] - x[1] * x[3]
    c = [8 - x[0] - 2 * x[1], 12 - 4 * x[0] - x[1], 12 - 3 * x[0] - 4 * x[1], 8 - 2 * x[2] - x[3], 8 - x[2] - 2 * x[3], 5 - x[2] - x[3]]
    return hs("HS44", x, f, c, cl, cu, xl=[0] * 4, x0=[0] * 4, ref_f=-15.0, ref_x=[[0, 3, 0, 4]],
              notes="local minimum f=-13 exists", tags=["multiple-local-minima"])

@register
def HS46():
    x = xs(5); cl, cu = eq(2)
    f = (x[0] - x[1]) ** 2 + (x[2] - 1) ** 2 + (x[3] - 1) ** 4 + (x[4] - 1) ** 6
    c = [x[0] ** 2 * x[3] + sin(x[3] - x[4]) - 1, x[1] + x[2] ** 4 * x[3] ** 2 - 2]
    return hs("HS46", x, f, c, cl, cu, x0=[S2 / 2, 1.75, 0.5, 2, 2], ref_f=0.0, ref_x=[[1, 1, 1, 1, 1]])

@register
def HS47():
    x = xs(5); cl, cu = eq(3)
    f = (x[0] - x[1]) ** 2 + (x[1] - x[2]) ** 3 + (x[2] - x[3]) ** 4 + (x[3] - x[4]) ** 4
    c = [x[0] + x[1] ** 2 + x[2] ** 3 - 3, x[1] - x[2] ** 2 + x[3] - 1, x[0] * x[4] - 1]
    return hs("HS47", x, f, c, cl, cu, x0=[2, S2, -1, 2 - S2, 0.5], ref_f=0.0, ref_x=[[1, 1, 1, 1, 1]])

@register
def HS48():
    x = xs(5); cl, cu = eq(2)
    f = (x[0] - 1) ** 2 + (x[1] - x[2]) ** 2 + (x[3] - x[4]) ** 2
    c = [x[0] + x[1] + x[2] + x[3] + x[4] - 5, x[2] - 2 * (x[3] + x[4]) + 3]
    return hs("HS48", x, f, c, cl, cu, x0=[3, 5, -3, 2, -2], ref_f=0.0, ref_x=[[1, 1, 1, 1, 1]])

@register
def HS49():
    x = xs(5); cl, cu = eq(2)
    f = (x[0] - x[1]) ** 2 + (x[2] - 1) ** 2 + (x[3] - 1) ** 4 + (x[4] - 1) ** 6
    c = [x[0] + x[1] + x[2] + 4 * x[3] - 7, x[2] + 5 * x[4] - 6]
    return hs("HS49", x, f, c, cl, cu, x0=[10, 7, 2, -3, 0.8], ref_f=0.0, ref_x=[[1, 1, 1, 1, 1]])

@register
def HS50():
    x = xs(5); cl, cu = eq(3)
    f = (x[0] - x[1]) ** 2 + (x[1] - x[2]) ** 2 + (x[2] - x[3]) ** 4 + (x[3] - x[4]) ** 2
    c = [x[0] + 2 * x[1] + 3 * x[2] - 6, x[1] + 2 * x[2] + 3 * x[3] - 6, x[2] + 2 * x[3] + 3 * x[4] - 6]
    return hs("HS50", x, f, c, cl, cu, x0=[35, -31, 11, 5, -5], ref_f=0.0, ref_x=[[1, 1, 1, 1, 1]])

@register
def HS51():
    x = xs(5); cl, cu = eq(3)
    f = (x[0] - x[1]) ** 2 + (x[1] + x[2] - 2) ** 2 + (x[3] - 1) ** 2 + (x[4] - 1) ** 2
    c = [x[0] + 3 * x[1] - 4, x[2] + x[3] - 2 * x[4], x[1] - x[4]]
    return hs("HS51", x, f, c, cl, cu, x0=[2.5, 0.5, 2, -1, 0.5], ref_f=0.0, ref_x=[[1, 1, 1, 1, 1]])

@register
def HS52():
    x = xs(5); cl, cu = eq(3)
    f = (4 * x[0] - x[1]) ** 2 + (x[1] + x[2] - 2) ** 2 + (x[3] - 1) ** 2 + (x[4] - 1) ** 2
    c = [x[0] + 3 * x[1], x[2] + x[3] - 2 * x[4], x[1] - x[4]]
    return hs("HS52", x, f, c, cl, cu, x0=[2] * 5, ref_f=1859 / 349)

@register
def HS53():
    x = xs(5); cl, cu = eq(3)
    f = (x[0] - x[1]) ** 2 + (x[1] + x[2] - 2) ** 2 + (x[3] - 1) ** 2 + (x[4] - 1) ** 2
    c = [x[0] + 3 * x[1], x[2] + x[3] - 2 * x[4], x[1] - x[4]]
    return hs("HS53", x, f, c, cl, cu, xl=[-10] * 5, xu=[10] * 5, x0=[2] * 5, ref_f=176 / 43)

@register
def HS55():
    x = xs(6); cl, cu = eq(6)
    f = x[0] + 2 * x[1] + 4 * x[4] + exp(x[0] * x[3])
    c = [x[0] + 2 * x[1] + 5 * x[4] - 6, x[0] + x[1] + x[2] - 3, x[3] + x[4] + x[5] - 2, x[0] + x[3] - 1, x[1] + x[4] - 2, x[2] + x[5] - 2]
    return hs("HS55", x, f, c, cl, cu, xl=[0] * 6, xu=[1, INF, INF, 1, INF, INF], x0=[1, 2, 0, 0, 0, 2], ref_f=19 / 3,
              notes="rank-deficient equality set (rank 5 of 6)", tags=["rank-deficient"])

@register
def HS57():
    x = xs(2); cl, cu = ge(1)
    a = [8, 8, 10, 10, 10, 10, 12, 12, 12, 12, 14, 14, 14, 16, 16, 16, 18, 18, 20, 20, 20, 22, 22, 22, 24, 24, 24, 26, 26, 26, 28, 28, 30, 30, 30, 32, 32, 34, 36, 36, 38, 38, 40, 42]
    b = [0.49, 0.49, 0.48, 0.47, 0.48, 0.47, 0.46, 0.46, 0.45, 0.43, 0.45, 0.43, 0.43, 0.44, 0.43, 0.43, 0.46, 0.45, 0.42, 0.42, 0.43, 0.41, 0.41, 0.40, 0.42, 0.40, 0.40, 0.41, 0.40, 0.41, 0.41, 0.40, 0.40, 0.40, 0.38, 0.41, 0.40, 0.40, 0.41, 0.38, 0.40, 0.40, 0.39, 0.39]
    f = sum((bi - x[0] - (0.49 - x[0]) * exp(-x[1] * (ai - 8))) ** 2 for ai, bi in zip(a, b))
    return hs("HS57", x, f, [0.49 * x[1] - x[0] * x[1] - 0.09], cl, cu, xl=[0.4, -4], x0=[0.42, 5], ref_f=0.02845966,
              notes="least squares data fit", tags=["fitting"])

@register
def HS60():
    x = xs(3); cl, cu = eq(1)
    f = (x[0] - 1) ** 2 + (x[0] - x[1]) ** 2 + (x[1] - x[2]) ** 4
    return hs("HS60", x, f, [x[0] * (1 + x[1] ** 2) + x[2] ** 4 - 4 - 3 * sqrt(2)], cl, cu, xl=[-10] * 3, xu=[10] * 3,
              x0=[2, 2, 2], ref_f=0.03256820025)

@register
def HS61():
    x = xs(3); cl, cu = eq(2)
    f = 4 * x[0] ** 2 + 2 * x[1] ** 2 + 2 * x[2] ** 2 - 33 * x[0] + 16 * x[1] - 24 * x[2]
    c = [3 * x[0] - 2 * x[1] ** 2 - 7, 4 * x[0] - x[2] ** 2 - 11]
    return hs("HS61", x, f, c, cl, cu, x0=[0, 0, 0], ref_f=-143.6461422)

@register
def HS62():
    x = xs(3); cl, cu = eq(1)
    f = (-32.174 * (255 * log((x[0] + x[1] + x[2] + 0.03) / (0.09 * x[0] + x[1] + x[2] + 0.03))
                    + 280 * log((x[1] + x[2] + 0.03) / (0.07 * x[1] + x[2] + 0.03))
                    + 290 * log((x[2] + 0.03) / (0.13 * x[2] + 0.03))))
    return hs("HS62", x, f, [x[0] + x[1] + x[2] - 1], cl, cu, xl=[0] * 3, xu=[1] * 3, x0=[0.7, 0.2, 0.1], ref_f=-26272.51448)

@register
def HS63():
    x = xs(3); cl, cu = eq(2)
    f = 1000 - x[0] ** 2 - 2 * x[1] ** 2 - x[2] ** 2 - x[0] * x[1] - x[0] * x[2]
    c = [8 * x[0] + 14 * x[1] + 7 * x[2] - 56, x[0] ** 2 + x[1] ** 2 + x[2] ** 2 - 25]
    return hs("HS63", x, f, c, cl, cu, xl=[0] * 3, x0=[2, 2, 2], ref_f=961.7151721)

@register
def HS64():
    x = xs(3); cl, cu = ge(1)
    f = 5 * x[0] + 50000 / x[0] + 20 * x[1] + 72000 / x[1] + 10 * x[2] + 144000 / x[2]
    return hs("HS64", x, f, [1 - 4 / x[0] - 32 / x[1] - 120 / x[2]], cl, cu, xl=[1e-5] * 3, x0=[1, 1, 1], ref_f=6299.842428,
              tags=["scaling"])

@register
def HS65():
    x = xs(3); cl, cu = ge(1)
    f = (x[0] - x[1]) ** 2 + (x[0] + x[1] - 10) ** 2 / 9 + (x[2] - 5) ** 2
    return hs("HS65", x, f, [48 - x[0] ** 2 - x[1] ** 2 - x[2] ** 2], cl, cu, xl=[-4.5, -4.5, -5], xu=[4.5, 4.5, 5],
              x0=[-5, 5, 0], ref_f=0.9535288567, notes="x0 outside the box")

@register
def HS66():
    x = xs(3); cl, cu = ge(2)
    return hs("HS66", x, 0.2 * x[2] - 0.8 * x[0], [x[1] - exp(x[0]), x[2] - exp(x[1])], cl, cu, xl=[0] * 3, xu=[100, 100, 10],
              x0=[0, 1.05, 2.9], ref_f=0.5181632741)

@register
def HS71():
    x = xs(4)
    return hs("HS71", x, x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2],
              [x[0] * x[1] * x[2] * x[3] - 25, x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2], [0, 40], [INF, 40],
              xl=[1] * 4, xu=[5] * 4, x0=[1, 5, 5, 1], ref_f=17.0140172935, ref_x=[[1, 4.7429996357, 3.8211499974, 1.3794082939]])

@register
def HS72():
    x = xs(4); cl, cu = ge(2)
    f = 1 + x[0] + x[1] + x[2] + x[3]
    c = [1 - 4 / x[0] - 2.25 / x[1] - 1 / x[2] - 0.25 / x[3], 1 - 0.16 / x[0] - 0.36 / x[1] - 0.64 / x[2] - 0.64 / x[3]]
    return hs("HS72", x, f, c, cl, cu, xl=[0.001] * 4, xu=[4e5, 3e5, 2e5, 1e5], x0=[1] * 4, ref_f=727.67937, tags=["scaling"])

@register
def HS73():
    x = xs(4)
    f = 24.55 * x[0] + 26.75 * x[1] + 39 * x[2] + 40.50 * x[3]
    c = [2.3 * x[0] + 5.6 * x[1] + 11.1 * x[2] + 1.3 * x[3] - 5,
         12 * x[0] + 11.9 * x[1] + 41.8 * x[2] + 52.1 * x[3] - 21 - 1.645 * sqrt(0.28 * x[0] ** 2 + 0.19 * x[1] ** 2 + 20.5 * x[2] ** 2 + 0.62 * x[3] ** 2),
         x[0] + x[1] + x[2] + x[3] - 1]
    return hs("HS73", x, f, c, [0, 0, 0], [INF, INF, 0], xl=[0] * 4, x0=[1, 1, 1, 1], ref_f=29.894378)

@register
def HS74():
    x = xs(4)
    f = 3 * x[0] + 1e-6 * x[0] ** 3 + 2 * x[1] + 2e-6 / 3 * x[1] ** 3
    c = [x[3] - x[2] + 0.55, x[2] - x[3] + 0.55,
         1000 * sin(-x[2] - 0.25) + 1000 * sin(-x[3] - 0.25) + 894.8 - x[0],
         1000 * sin(x[2] - 0.25) + 1000 * sin(x[2] - x[3] - 0.25) + 894.8 - x[1],
         1000 * sin(x[3] - 0.25) + 1000 * sin(x[3] - x[2] - 0.25) + 1294.8]
    return hs("HS74", x, f, c, [0, 0, 0, 0, 0], [INF, INF, 0, 0, 0], xl=[0, 0, -0.55, -0.55], xu=[1200, 1200, 0.55, 0.55],
              x0=[0, 0, 0, 0], ref_f=5126.4981)

@register
def HS75():
    x = xs(4)
    f = 3 * x[0] + 1e-6 * x[0] ** 3 + 2 * x[1] + 2e-6 / 3 * x[1] ** 3
    c = [x[3] - x[2] + 0.48, x[2] - x[3] + 0.48,
         1000 * sin(-x[2] - 0.25) + 1000 * sin(-x[3] - 0.25) + 894.8 - x[0],
         1000 * sin(x[2] - 0.25) + 1000 * sin(x[2] - x[3] - 0.25) + 894.8 - x[1],
         1000 * sin(x[3] - 0.25) + 1000 * sin(x[3] - x[2] - 0.25) + 1294.8]
    return hs("HS75", x, f, c, [0, 0, 0, 0, 0], [INF, INF, 0, 0, 0], xl=[0, 0, -0.48, -0.48], xu=[1200, 1200, 0.48, 0.48],
              x0=[0, 0, 0, 0], ref_f=5174.4127)

@register
def HS76():
    x = xs(4); cl, cu = ge(3)
    f = x[0] ** 2 + 0.5 * x[1] ** 2 + x[2] ** 2 + 0.5 * x[3] ** 2 - x[0] * x[2] + x[2] * x[3] - x[0] - 3 * x[1] + x[2] - x[3]
    c = [5 - x[0] - 2 * x[1] - x[2] - x[3], 4 - 3 * x[0] - x[1] - 2 * x[2] + x[3], x[1] + 4 * x[2] - 1.5]
    return hs("HS76", x, f, c, cl, cu, xl=[0] * 4, x0=[0.5] * 4, ref_f=-4.681818181)

@register
def HS77():
    x = xs(5); cl, cu = eq(2)
    f = (x[0] - 1) ** 2 + (x[0] - x[1]) ** 2 + (x[2] - 1) ** 2 + (x[3] - 1) ** 4 + (x[4] - 1) ** 6
    c = [x[0] ** 2 * x[3] + sin(x[3] - x[4]) - 2 * sqrt(2), x[1] + x[2] ** 4 * x[3] ** 2 - 8 - sqrt(2)]
    return hs("HS77", x, f, c, cl, cu, x0=[2] * 5, ref_f=0.24150513)

@register
def HS78():
    x = xs(5); cl, cu = eq(3)
    f = x[0] * x[1] * x[2] * x[3] * x[4]
    c = [x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2 + x[4] ** 2 - 10, x[1] * x[2] - 5 * x[3] * x[4], x[0] ** 3 + x[1] ** 3 + 1]
    return hs("HS78", x, f, c, cl, cu, x0=[-2, 1.5, 2, -1, -1], ref_f=-2.91970041)

@register
def HS79():
    x = xs(5); cl, cu = eq(3)
    f = (x[0] - 1) ** 2 + (x[0] - x[1]) ** 2 + (x[1] - x[2]) ** 2 + (x[2] - x[3]) ** 4 + (x[3] - x[4]) ** 4
    c = [x[0] + x[1] ** 2 + x[2] ** 3 - 2 - 3 * sqrt(2), x[1] - x[2] ** 2 + x[3] + 2 - 2 * sqrt(2), x[0] * x[4] - 2]
    return hs("HS79", x, f, c, cl, cu, x0=[2] * 5, ref_f=0.0787768209)

@register
def HS80():
    x = xs(5); cl, cu = eq(3)
    f = exp(x[0] * x[1] * x[2] * x[3] * x[4])
    c = [x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2 + x[4] ** 2 - 10, x[1] * x[2] - 5 * x[3] * x[4], x[0] ** 3 + x[1] ** 3 + 1]
    return hs("HS80", x, f, c, cl, cu, xl=[-2.3, -2.3, -3.2, -3.2, -3.2], xu=[2.3, 2.3, 3.2, 3.2, 3.2], x0=[-2, 2, 2, -1, -1],
              ref_f=0.0539498478)

@register
def HS81():
    x = xs(5); cl, cu = eq(3)
    f = exp(x[0] * x[1] * x[2] * x[3] * x[4]) - 0.5 * (x[0] ** 3 + x[1] ** 3 + 1) ** 2
    c = [x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2 + x[4] ** 2 - 10, x[1] * x[2] - 5 * x[3] * x[4], x[0] ** 3 + x[1] ** 3 + 1]
    return hs("HS81", x, f, c, cl, cu, xl=[-2.3, -2.3, -3.2, -3.2, -3.2], xu=[2.3, 2.3, 3.2, 3.2, 3.2], x0=[-2, 2, 2, -1, -1],
              ref_f=0.0539498478)

@register
def HS83():
    x = xs(5)
    a = [85.334407, 0.0056858, 0.0006262, 0.0022053, 80.51249, 0.0071317, 0.0029955, 0.0021813, 9.300961, 0.0047026, 0.0012547, 0.0019085]
    f = 5.3578547 * x[2] ** 2 + 0.8356891 * x[0] * x[4] + 37.293239 * x[0] - 40792.141
    c = [a[0] + a[1] * x[1] * x[4] + a[2] * x[0] * x[3] - a[3] * x[2] * x[4],
         a[4] + a[5] * x[1] * x[4] + a[6] * x[0] * x[1] + a[7] * x[2] ** 2,
         a[8] + a[9] * x[2] * x[4] + a[10] * x[0] * x[2] + a[11] * x[2] * x[3]]
    return hs("HS83", x, f, c, [0, 90, 20], [92, 110, 25], xl=[78, 33, 27, 27, 27], xu=[102, 45, 45, 45, 45],
              x0=[78, 33, 27, 27, 27], ref_f=-30665.53867, tags=["ranged"])

@register
def HS86():
    x = xs(5); cl, cu = ge(10)
    e = [-15, -27, -36, -18, -12]
    C = [[30, -20, -10, 32, -10], [-20, 39, -6, -31, 32], [-10, -6, 10, -6, -10], [32, -31, -6, 39, -20], [-10, 32, -10, -20, 30]]
    D = [4, 8, 10, 6, 2]
    A = [[-16, 2, 0, 1, 0], [0, -2, 0, 0.4, 2], [-3.5, 0, 2, 0, 0], [0, -2, 0, -4, -1], [0, -9, -2, 1, -2.8],
         [2, 0, -4, 0, 0], [-1, -1, -1, -1, -1], [-1, -2, -3, -2, -1], [1, 2, 3, 4, 5], [1, 1, 1, 1, 1]]
    b = [-40, -2, -0.25, -4, -4, -1, -40, -60, 5, 1]
    f = sum(e[j] * x[j] for j in range(5)) + sum(C[i][j] * x[i] * x[j] for i in range(5) for j in range(5)) + sum(D[j] * x[j] ** 3 for j in range(5))
    c = [sum(A[i][j] * x[j] for j in range(5)) - b[i] for i in range(10)]
    return hs("HS86", x, f, c, cl, cu, xl=[0] * 5, x0=[0, 0, 0, 0, 1], ref_f=-32.34867897, notes="Colville #1")

@register
def HS93():
    x = xs(6); cl, cu = ge(2)
    f = 0.0204 * x[0] * x[3] * (x[0] + x[1] + x[2]) + 0.0187 * x[1] * x[2] * (x[0] + 1.57 * x[1] + x[3]) + 0.0607 * x[0] * x[3] * x[4] ** 2 * (x[0] + x[1] + x[2]) + 0.0437 * x[1] * x[2] * x[5] ** 2 * (x[0] + 1.57 * x[1] + x[3])
    c = [0.001 * x[0] * x[1] * x[2] * x[3] * x[4] * x[5] - 2.07, 1 - 0.00062 * x[0] * x[3] * x[4] ** 2 * (x[0] + x[1] + x[2]) - 0.00058 * x[1] * x[2] * x[5] ** 2 * (x[0] + 1.57 * x[1] + x[3])]
    return hs("HS93", x, f, c, cl, cu, xl=[0] * 6, x0=[5.54, 4.4, 12.02, 11.82, 0.702, 0.852], ref_f=135.075961, notes="transformer design")

def _hs95_98(name, ab, ref_f, xl, xu, b=(4.97, -1.88, -29.08, -78.02)):
    x = xs(6); cl, cu = ge(4)
    f = 4.3 * x[0] + 31.8 * x[1] + 63.3 * x[2] + 15.8 * x[3] + 68.5 * x[4] + 4.7 * x[5]
    c = [17.1 * x[0] + 38.2 * x[1] + 204.2 * x[2] + 212.3 * x[3] + 623.4 * x[4] + 1495.5 * x[5] - 169 * x[0] * x[2] - 3580 * x[2] * x[4] - 3810 * x[3] * x[4] - 18500 * x[3] * x[5] - 24300 * x[4] * x[5] - b[0],
         17.9 * x[0] + 36.8 * x[1] + 113.9 * x[2] + 169.7 * x[3] + 337.8 * x[4] + 1385.2 * x[5] - 139 * x[0] * x[2] - 2450 * x[3] * x[4] - 16600 * x[3] * x[5] - 17200 * x[4] * x[5] - b[1],
         -273 * x[1] - 70 * x[3] - 819 * x[4] + 26000 * x[3] * x[4] - b[2],
         159.9 * x[0] - 311 * x[1] + 587 * x[3] + 391 * x[4] + 2198 * x[5] - 14000 * x[0] * x[5] - b[3]]
    return hs(name, x, f, c, cl, cu, xl=xl, xu=xu, x0=[0] * 6, ref_f=ref_f)

@register
def HS95():
    return _hs95_98("HS95", [4.3, 3.1, 0.5], 0.015619525, [0] * 6, [0.31, 0.046, 0.068, 0.042, 0.028, 0.0134])

@register
def HS96():
    return _hs95_98("HS96", [4.3, 3.1, 0.5], 0.015619525, [0] * 6, [0.31, 0.046, 0.068, 0.042, 0.028, 0.0134], b=[4.97, -1.88, -69.08, -118.02])

@register
def HS97():
    return _hs95_98("HS97", [32.5, 21.5, 0.5], 3.1358091, [0] * 6, [0.31, 0.046, 0.068, 0.042, 0.028, 0.0134], b=[32.97, 25.12, -29.08, -78.02])

@register
def HS98():
    return _hs95_98("HS98", [32.5, 21.5, 0.5], 3.1358091, [0] * 6, [0.31, 0.046, 0.068, 0.042, 0.028, 0.0134], b=[32.97, 25.12, -124.08, -173.02])

@register
def HS100():
    x = xs(7); cl, cu = ge(4)
    f = ((x[0] - 10) ** 2 + 5 * (x[1] - 12) ** 2 + x[2] ** 4 + 3 * (x[3] - 11) ** 2 + 10 * x[4] ** 6 + 7 * x[5] ** 2 + x[6] ** 4
         - 4 * x[5] * x[6] - 10 * x[5] - 8 * x[6])
    c = [127 - 2 * x[0] ** 2 - 3 * x[1] ** 4 - x[2] - 4 * x[3] ** 2 - 5 * x[4],
         282 - 7 * x[0] - 3 * x[1] - 10 * x[2] ** 2 - x[3] + x[4],
         196 - 23 * x[0] - x[1] ** 2 - 6 * x[5] ** 2 + 8 * x[6],
         -4 * x[0] ** 2 - x[1] ** 2 + 3 * x[0] * x[1] - 2 * x[2] ** 2 - 5 * x[5] + 11 * x[6]]
    return hs("HS100", x, f, c, cl, cu, x0=[1, 2, 0, 4, 0, 1, 1], ref_f=680.6300573)

@register
def HS104():
    x = xs(8); cl, cu = ge(6)
    f = 0.4 * x[0] ** 0.67 * x[6] ** (-0.67) + 0.4 * x[1] ** 0.67 * x[7] ** (-0.67) + 10 - x[0] - x[1]
    c = [1 - 0.0588 * x[4] * x[6] - 0.1 * x[0], 1 - 0.0588 * x[5] * x[7] - 0.1 * x[0] - 0.1 * x[1],
         1 - 4 * x[2] / x[4] - 2 / (x[2] ** 0.71 * x[4]) - 0.0588 * x[6] / x[2] ** 1.3,
         1 - 4 * x[3] / x[5] - 2 / (x[3] ** 0.71 * x[5]) - 0.0588 * x[7] / x[3] ** 1.3,
         f - 1, 4.2 - f]
    return hs("HS104", x, f, c, cl, cu, xl=[0.1] * 8, xu=[10] * 8, x0=[6, 3, 0.4, 0.2, 6, 6, 1, 0.5], ref_f=3.9511634396)

@register
def HS106():
    x = xs(8); cl, cu = ge(6)
    f = x[0] + x[1] + x[2]
    c = [1 - 0.0025 * (x[3] + x[5]), 1 - 0.0025 * (x[4] + x[6] - x[3]), 1 - 0.01 * (x[7] - x[4]),
         x[0] * x[5] - 833.33252 * x[3] - 100 * x[0] + 83333.333,
         x[1] * x[6] - 1250 * x[4] - x[1] * x[3] + 1250 * x[3],
         x[2] * x[7] - 1250000 - x[2] * x[4] + 2500 * x[4]]
    return hs("HS106", x, f, c, cl, cu, xl=[100, 1000, 1000, 10, 10, 10, 10, 10], xu=[10000, 10000, 10000, 1000, 1000, 1000, 1000, 1000],
              x0=[5000, 5000, 5000, 200, 350, 150, 225, 425], ref_f=7049.330923, notes="heat exchanger design", tags=["scaling"])

@register
def HS108():
    x = xs(9); cl, cu = ge(13)
    f = -0.5 * (x[0] * x[3] - x[1] * x[2] + x[2] * x[8] - x[4] * x[8] + x[4] * x[7] - x[5] * x[6])
    c = [1 - x[2] ** 2 - x[3] ** 2, 1 - x[8] ** 2, 1 - x[4] ** 2 - x[5] ** 2, 1 - x[0] ** 2 - (x[1] - x[8]) ** 2,
         1 - (x[0] - x[4]) ** 2 - (x[1] - x[5]) ** 2, 1 - (x[0] - x[6]) ** 2 - (x[1] - x[7]) ** 2,
         1 - (x[2] - x[4]) ** 2 - (x[3] - x[5]) ** 2, 1 - (x[2] - x[6]) ** 2 - (x[3] - x[7]) ** 2,
         1 - x[6] ** 2 - (x[7] - x[8]) ** 2, x[0] * x[3] - x[1] * x[2], x[2] * x[8], -x[4] * x[8], x[4] * x[7] - x[5] * x[6]]
    return hs("HS108", x, f, c, cl, cu, xl=[-INF] * 8 + [0], x0=[1] * 9, ref_f=-0.8660254038, notes="hexagon; degenerate")

@register
def HS111():
    x = xs(10); cl, cu = eq(3)
    cc = [-6.089, -17.164, -34.054, -5.914, -24.721, -14.986, -24.100, -10.708, -26.662, -22.179]
    ex = [exp(xi) for xi in x]
    tot = sum(ex)
    f = sum(ex[j] * (cc[j] + log(ex[j] / tot)) for j in range(10))
    c = [ex[0] + 2 * ex[1] + 2 * ex[2] + ex[5] + ex[9] - 2, ex[3] + 2 * ex[4] + ex[5] + ex[6] - 1, ex[2] + ex[6] + ex[7] + 2 * ex[8] + ex[9] - 1]
    return hs("HS111", x, f, c, cl, cu, xl=[-100] * 10, xu=[100] * 10, x0=[-2.3] * 10, ref_f=-47.76109026, notes="chemical equilibrium")

@register
def HS112():
    x = xs(10)
    cc = [-6.089, -17.164, -34.054, -5.914, -24.721, -14.986, -24.100, -10.708, -26.662, -22.179]
    tot = sum(x)
    f = sum(x[j] * (cc[j] + log(x[j] / tot)) for j in range(10))
    c = [x[0] + 2 * x[1] + 2 * x[2] + x[5] + x[9] - 2, x[3] + 2 * x[4] + x[5] + x[6] - 1, x[2] + x[6] + x[7] + 2 * x[8] + x[9] - 1]
    return hs("HS112", x, f, c, [0, 0, 0], [0, 0, 0], xl=[1e-6] * 10, x0=[0.1] * 10, ref_f=-47.707579, notes="chemical equilibrium", tags=["domain"])

@register
def HS113():
    x = xs(10); cl, cu = ge(8)
    f = (x[0] ** 2 + x[1] ** 2 + x[0] * x[1] - 14 * x[0] - 16 * x[1] + (x[2] - 10) ** 2 + 4 * (x[3] - 5) ** 2 + (x[4] - 3) ** 2
         + 2 * (x[5] - 1) ** 2 + 5 * x[6] ** 2 + 7 * (x[7] - 11) ** 2 + 2 * (x[8] - 10) ** 2 + (x[9] - 7) ** 2 + 45)
    c = [105 - 4 * x[0] - 5 * x[1] + 3 * x[6] - 9 * x[7], -10 * x[0] + 8 * x[1] + 17 * x[6] - 2 * x[7],
         8 * x[0] - 2 * x[1] - 5 * x[8] + 2 * x[9] + 12, -3 * (x[0] - 2) ** 2 - 4 * (x[1] - 3) ** 2 - 2 * x[2] ** 2 + 7 * x[3] + 120,
         -5 * x[0] ** 2 - 8 * x[1] - (x[2] - 6) ** 2 + 2 * x[3] + 40, -0.5 * (x[0] - 8) ** 2 - 2 * (x[1] - 4) ** 2 - 3 * x[4] ** 2 + x[5] + 30,
         -x[0] ** 2 - 2 * (x[1] - 2) ** 2 + 2 * x[0] * x[1] - 14 * x[4] + 6 * x[5], 3 * x[0] - 6 * x[1] - 12 * (x[8] - 8) ** 2 + 7 * x[9]]
    return hs("HS113", x, f, c, cl, cu, x0=[2, 3, 5, 5, 1, 2, 7, 3, 6, 10], ref_f=24.30620907, notes="Wong #2")

@register
def HS117():
    x = xs(15); cl, cu = ge(5)
    b = [-40, -2, -0.25, -4, -4, -1, -40, -60, 5, 1]
    C = [[30, -20, -10, 32, -10], [-20, 39, -6, -31, 32], [-10, -6, 10, -6, -10], [32, -31, -6, 39, -20], [-10, 32, -10, -20, 30]]
    D = [4, 8, 10, 6, 2]
    A = [[-16, 2, 0, 1, 0], [0, -2, 0, 0.4, 2], [-3.5, 0, 2, 0, 0], [0, -2, 0, -4, -1], [0, -9, -2, 1, -2.8],
         [2, 0, -4, 0, 0], [-1, -1, -1, -1, -1], [-1, -2, -3, -2, -1], [1, 2, 3, 4, 5], [1, 1, 1, 1, 1]]
    e = [-15, -27, -36, -18, -12]
    f = -sum(b[j] * x[j] for j in range(10)) + sum(C[j][k] * x[10 + j] * x[10 + k] for j in range(5) for k in range(5)) + 2 * sum(D[j] * x[10 + j] ** 3 for j in range(5))
    c = [2 * sum(C[k][j] * x[10 + k] for k in range(5)) + 3 * D[j] * x[10 + j] ** 2 + e[j] - sum(A[k][j] * x[k] for k in range(10)) for j in range(5)]
    return hs("HS117", x, f, c, cl, cu, xl=[0] * 15, x0=[0.001] * 6 + [60] + [0.001] * 8, ref_f=32.348679, notes="Colville #2")

@register
def HS118():
    x = xs(15); cl, cu = ge(17)
    f = sum(2.3 * x[3 * k] + 0.0001 * x[3 * k] ** 2 + 1.7 * x[3 * k + 1] + 0.0001 * x[3 * k + 1] ** 2 + 2.2 * x[3 * k + 2] + 0.00015 * x[3 * k + 2] ** 2 for k in range(5))
    c = []
    for j in range(1, 5):
        c += [x[3 * j] - x[3 * j - 3] + 7, 13 - x[3 * j] + x[3 * j - 3],
              x[3 * j + 1] - x[3 * j - 2] + 7, 13 - x[3 * j + 1] + x[3 * j - 2],
              x[3 * j + 2] - x[3 * j - 1] + 7, 13 - x[3 * j + 2] + x[3 * j - 1]]
    # The book states the six difference rows per j as ranged -7 <= d <= 6 (mixed); we keep the one-sided pairs above.
    c += [x[0] + x[1] + x[2] - 60, x[3] + x[4] + x[5] - 50, x[6] + x[7] + x[8] - 70, x[9] + x[10] + x[11] - 85, x[12] + x[13] + x[14] - 100]
    cl, cu = ge(len(c))
    xl = [8, 43, 3] + [0] * 12
    xu = [21, 57, 16] + [90, 120, 60] * 4
    return hs("HS118", x, f, c, cl, cu, xl=xl, xu=xu, x0=[20, 55, 15, 20, 60, 20, 20, 60, 20, 20, 60, 20, 20, 60, 20], ref_f=664.8204500,
              notes="transcription of the difference rows as one-sided pairs; verify against published optimum before use as target",
              tags=["unverified-transcription"])


# ---- added for the second held-out set (unseen by any tuning) ----
@register
def HS25():
    x = xs(3)
    terms = []
    for i in range(1, 100):
        u = 25 + (-50 * math.log(0.01 * i)) ** (2 / 3)
        terms.append((-0.01 * i + exp(-(u - x[1]) ** x[2] / x[0])) ** 2)
    return hs("HS25", x, sp.Add(*terms), xl=[0.1, 0, 0], xu=[100, 25.6, 5], x0=[100, 12.5, 3], ref_f=0.0, ref_x=[[50, 25, 1.5]],
              tags=["fitting", "domain"], notes="99-term data fit; (u_i - x2)^x3 needs u_i > x2, guaranteed by the box")

@register
def HS56():
    x = xs(7); cl, cu = eq(4)
    a = math.asin(math.sqrt(1 / 4.2)); b = math.asin(math.sqrt(5 / 7.2))
    c = [x[0] - 4.2 * sin(x[3]) ** 2, x[1] - 4.2 * sin(x[4]) ** 2, x[2] - 4.2 * sin(x[5]) ** 2,
         x[0] + 2 * x[1] + 2 * x[2] - 7.2 * sin(x[6]) ** 2]
    return hs("HS56", x, -x[0] * x[1] * x[2], c, cl, cu, x0=[1, 1, 1, a, a, a, b], ref_f=-3.456)

@register
def HS84():
    x = xs(5)
    a = [-24345, -8720288.849, 150512.5253, -156.6950325, 476470.3222, 729482.8271, -145421.402, 2931.1506, -40.427932,
         5106.192, 15711.36, -155011.1084, 4360.53352, 12.9492344, 10236.884, 13176.786, -326669.5104, 7390.68412,
         -27.8986976, 16643.076, 30988.146]
    f = -a[0] - a[1] * x[0] - a[2] * x[0] * x[1] - a[3] * x[0] * x[2] - a[4] * x[0] * x[3] - a[5] * x[0] * x[4]
    c = [a[6] * x[0] + a[7] * x[0] * x[1] + a[8] * x[0] * x[2] + a[9] * x[0] * x[3] + a[10] * x[0] * x[4],
         a[11] * x[0] + a[12] * x[0] * x[1] + a[13] * x[0] * x[2] + a[14] * x[0] * x[3] + a[15] * x[0] * x[4],
         a[16] * x[0] + a[17] * x[0] * x[1] + a[18] * x[0] * x[2] + a[19] * x[0] * x[3] + a[20] * x[0] * x[4]]
    return hs("HS84", x, f, c, [0, 0, 0], [294000, 294000, 277200], xl=[0, 1.2, 20, 9, 6.5], xu=[1000, 2.4, 60, 9.3, 7],
              x0=[2.52, 2, 37.5, 9.25, 6.8], ref_f=-5280335.133, tags=["ranged", "scaling"])
