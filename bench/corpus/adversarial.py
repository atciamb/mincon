"""Analytically controlled adversarial problems with independently derived solutions.

Every problem here has a closed-form optimum written down *before* any solver
ran, and `spec.numpy().violation(ref_x)` plus `f(ref_x) == ref_f` is checked
by `test_corpus.py`. Diagnostic-track problems (infeasible, unbounded) carry
the tag ``diagnostic`` and are never part of a success-rate denominator.
"""
from __future__ import annotations

import math

import sympy as sp

from spec import INF, Spec, register, xs

log, sqrt, exp = sp.log, sp.sqrt, sp.exp


def adv(name, x, f, c=(), cl=(), cu=(), xl=None, xu=None, x0=None, ref_f=None, ref_x=None, tags=(), notes=""):
    n = len(x)
    return Spec(name=name, family="adversarial", x=list(x), f=sp.sympify(f), c=[sp.sympify(e) for e in c],
                cl=list(cl), cu=list(cu), xl=list(xl) if xl is not None else [-INF] * n,
                xu=list(xu) if xu is not None else [INF] * n, x0=list(x0), ref_f=ref_f, ref_x=ref_x,
                source="closed form, bench/corpus/adversarial.py", tags=list(tags), notes=notes)


# 1. Fixed variables mixed with free ones. min sum (x_i - i)^2, x_2 fixed at 0, x_4 fixed at 5,
#    s.t. x_1 + x_3 <= 2. Free part: min (x1-1)^2 + (x3-3)^2 s.t. x1+x3<=2 -> x1=0, x3=2, f = 1+1 + (0-2)^2 + (5-4)^2 = 7.
@register
def FIXEDVARS():
    x = xs(4)
    return adv("FIXEDVARS", x, sum((x[i] - (i + 1)) ** 2 for i in range(4)), [x[0] + x[2]], [-INF], [2],
               xl=[-INF, 0, -INF, 5], xu=[INF, 0, INF, 5], x0=[0, 0, 0, 5], ref_f=7.0, ref_x=[[0, 0, 2, 5]], tags=["fixed"])


# 2. Redundant equalities: the same plane three times with different scalings.  min ||x - p||^2 s.t. sum x = 1
#    p = (1,2,3): projection of p onto sum x = 1: x = p - ((sum p - 1)/3) 1 = p - 5/3 -> (-2/3, 1/3, 4/3), f = 3*(5/3)^2 = 25/3.
@register
def REDUNDANT_EQ():
    x = xs(3)
    s = x[0] + x[1] + x[2] - 1
    return adv("REDUNDANT_EQ", x, (x[0] - 1) ** 2 + (x[1] - 2) ** 2 + (x[2] - 3) ** 2, [s, 2 * s, -0.5 * s], [0, 0, 0], [0, 0, 0],
               x0=[0, 0, 0], ref_f=25 / 3, ref_x=[[-2 / 3, 1 / 3, 4 / 3]], tags=["rank-deficient"])


# 3. Jacobian vanishes at the solution: h(x) = (x1 - 1)^2 = 0 (MFCQ fails), min x1^2 + x2^2 -> x = (1, 0), f = 1.
@register
def RANKLOSS_JAC():
    x = xs(2)
    return adv("RANKLOSS_JAC", x, x[0] ** 2 + x[1] ** 2, [(x[0] - 1) ** 2], [0], [0], x0=[2, 1], ref_f=1.0, ref_x=[[1, 0]],
               tags=["degenerate", "rank-deficient"], notes="constraint gradient is zero at the solution")


# 4. Nearly dependent active constraints.  min -(x1 + x2) s.t. x1 + x2 <= 1, x1 + (1+e) x2 <= 1 + e/2 (e=1e-6), x>=0.
#    At the optimum both rows are active: x2 = 1/2, x1 = 1/2, f = -1.
@register
def NEARDEP():
    x = xs(2)
    e = 1e-6
    return adv("NEARDEP", x, -(x[0] + x[1]), [x[0] + x[1], x[0] + (1 + e) * x[1]], [-INF, -INF], [1, 1 + e / 2],
               xl=[0, 0], x0=[0, 0], ref_f=-1.0, ref_x=[[0.5, 0.5]], tags=["ill-conditioned"])


# 5. Narrow box: x2 in [1, 1+1e-8]. min (x1 - 2)^2 + (x2 - 3)^2 -> x1 = 2, x2 = 1+1e-8, f = (2-1e-8)^2.
@register
def NARROWBOX():
    x = xs(2)
    w = 1e-8
    return adv("NARROWBOX", x, (x[0] - 2) ** 2 + (x[1] - 3) ** 2, xl=[-INF, 1], xu=[INF, 1 + w], x0=[0, 1],
               ref_f=(2 - w) ** 2, ref_x=[[2, 1 + w]], tags=["narrow-bounds"])


# 6. Disparate units: x1 ~ 1e-6, x2 ~ 1e6.  min (1e6 x1 - 1)^2 + (1e-6 x2 - 1)^2 s.t. 1e6 x1 + 1e-6 x2 >= 3 -> (1e6 x1, 1e-6 x2) = (1.5, 1.5), f = 0.5.
@register
def UNITS():
    x = xs(2)
    u, v = 1e6 * x[0], 1e-6 * x[1]
    return adv("UNITS", x, (u - 1) ** 2 + (v - 1) ** 2, [u + v], [3], [INF], x0=[0, 0], ref_f=0.5, ref_x=[[1.5e-6, 1.5e6]],
               tags=["scaling"])


# 7. Bad start: Rosenbrock from (1e3, -1e3) inside a unit-disc constraint that is inactive... make it active:
#    min rosenbrock s.t. x1^2 + x2^2 <= 1: known optimum x = (0.7864, 0.6177), f = 0.045674808 (classic).
@register
def BADSTART_DISC():
    x = xs(2)
    return adv("BADSTART_DISC", x, 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2, [x[0] ** 2 + x[1] ** 2], [-INF], [1],
               x0=[1000, -1000], ref_f=0.045674808, notes="reference from the well-known constrained Rosenbrock on the unit disc; ref_x not certified",
               tags=["bad-start"])


# 8. Undefined region: f = -log(x1) - log(x2) + x1 + x2 requires x > 0 (bounds 0); solution x = (1,1), f = 2.
@register
def DOMAIN_LOG():
    x = xs(2)
    return adv("DOMAIN_LOG", x, -log(x[0]) - log(x[1]) + x[0] + x[1], [x[0] + x[1]], [-INF], [3], xl=[0, 0], x0=[0.1, 2.5],
               ref_f=2.0, ref_x=[[1, 1]], tags=["domain"])


# 9. sqrt domain with a bound-active solution: min sqrt(x1) + (x2-1)^2 with x1 >= 0.25 -> x1 = 0.25, f = 0.5.
@register
def DOMAIN_SQRT():
    x = xs(2)
    return adv("DOMAIN_SQRT", x, sqrt(x[0]) + (x[1] - 1) ** 2, xl=[0.25, -INF], x0=[4, 0], ref_f=0.5, ref_x=[[0.25, 1]],
               tags=["domain"])


# 10. LICQ failure: three active constraints in 2D at the solution (0,0): x1>=0, x2>=0, x1+x2>=0; min x1 + x2 + x1^2.
@register
def DEGEN_LICQ():
    x = xs(2)
    return adv("DEGEN_LICQ", x, x[0] + x[1] + x[0] ** 2, [x[0] + x[1]], [0], [INF], xl=[0, 0], x0=[1, 1], ref_f=0.0,
               ref_x=[[0, 0]], tags=["degenerate"])


# 11. Infeasible (diagnostic): x1 + x2 = 1 and x1 + x2 = 3.
@register
def INFEASIBLE_LIN():
    x = xs(2)
    return adv("INFEASIBLE_LIN", x, x[0] ** 2 + x[1] ** 2, [x[0] + x[1], x[0] + x[1]], [1, 3], [1, 3], x0=[0, 0],
               tags=["diagnostic", "infeasible"], notes="no feasible point; correct answer is an infeasibility diagnosis")


# 12. Infeasible nonlinear (diagnostic): x1^2 + x2^2 <= 1 and x1 >= 2.
@register
def INFEASIBLE_NL():
    x = xs(2)
    return adv("INFEASIBLE_NL", x, x[0] + x[1], [x[0] ** 2 + x[1] ** 2], [-INF], [1], xl=[2, -INF], x0=[2, 0],
               tags=["diagnostic", "infeasible"])


# 13. Unbounded (diagnostic): min -x1 s.t. x2 = x1^2 (a parabola going to infinity).
@register
def UNBOUNDED_PAR():
    x = xs(2)
    return adv("UNBOUNDED_PAR", x, -x[0], [x[1] - x[0] ** 2], [0], [0], x0=[0, 0], tags=["diagnostic", "unbounded"])


# 14. Flat objective near an active bound: min 1e-8 (x1 - 5)^2 + (x2 - 1)^4, x1 <= 2  -> x = (2, 1), f = 9e-8.
@register
def FLAT_BOUND():
    x = xs(2)
    return adv("FLAT_BOUND", x, 1e-8 * (x[0] - 5) ** 2 + (x[1] - 1) ** 4, xu=[2, INF], x0=[0, 0], ref_f=9e-8, ref_x=[[2, 1]],
               tags=["flat"])


# 15. Many redundant linear inequalities in 5-D: a_i^T x <= 1 for 40 directions a_i on the unit sphere plus the
#     objective -sum x. The polytope contains the ball of radius 1 and every a_i is a vertex direction of the
#     objective... keep it exact: directions are the 10 coordinate +/- unit vectors and 30 randomized ones with
#     norm > 1 (looser), so the binding rows are x_i <= 1 and the optimum is x = 1, f = -5.
@register
def MANY_INEQ():
    import numpy as np
    x = xs(5)
    rng = np.random.default_rng(7)
    rows = []
    for i in range(5):
        rows.append(x[i]); rows.append(-x[i])
    for _ in range(30):
        a = rng.uniform(-1, 1, 5)
        a = a / (np.sum(np.abs(a)))  # |a|_1 = 1 => a^T x <= 1 holds on the box [-1,1]^5 (Hoelder), so redundant
        rows.append(sum(float(a[j]) * x[j] for j in range(5)))
    return adv("MANY_INEQ", x, -sum(x), rows, [-INF] * 40, [1.0] * 40, x0=[0] * 5, ref_f=-5.0, ref_x=[[1] * 5],
               tags=["redundant"])


# 16. Equality-constrained QP with an exactly known solution (n=6, m=2): min 0.5||x||^2 - e^T x  s.t. sum x = 0, x1 - x6 = 1.
#     KKT: x = e - lam1 1 - lam2 (e1 - e6). sum: 6 - 6 lam1 = 0 -> lam1 = 1; x1 - x6 = -2 lam2 = 1 -> lam2 = -1/2.
#     x = (1/2, 0, 0, 0, 0, -1/2); f = 0.5*0.5 - 0 = 0.25.
@register
def EQQP6():
    x = xs(6)
    return adv("EQQP6", x, sp.Rational(1, 2) * sum(v ** 2 for v in x) - sum(x), [sum(x), x[0] - x[5]], [0, 1], [0, 1],
               x0=[1] * 6, ref_f=0.25, ref_x=[[0.5, 0, 0, 0, 0, -0.5]])


# 17. Huge multiplier: constraint scaled by 1e-6 so lambda ~ 1e6: min (x1-2)^2 s.t. 1e-6 (x1 - 1) <= 0 -> x1 = 1, f = 1, lambda = 2e6.
@register
def BIGMULT():
    x = xs(2)
    return adv("BIGMULT", x, (x[0] - 2) ** 2 + x[1] ** 2, [1e-6 * (x[0] - 1)], [-INF], [0], x0=[0, 1], ref_f=1.0, ref_x=[[1, 0]],
               tags=["scaling"])


# 18. x0 outside bounds and constraint units 1e4: min (x-1)^2 sum, s.t. 1e4 * (x1 + x2) <= 1e4, x in [0,10]; start at (50, -50).
#     Solution x = (0.5, 0.5), f = 0.5.
@register
def START_OUTSIDE():
    x = xs(2)
    return adv("START_OUTSIDE", x, (x[0] - 1) ** 2 + (x[1] - 1) ** 2, [1e4 * (x[0] + x[1])], [-INF], [1e4], xl=[0, 0], xu=[10, 10],
               x0=[50, -50], ref_f=0.5, ref_x=[[0.5, 0.5]], tags=["bad-start", "scaling"])
