"""Counting, caching model wrapper used by every Python adapter.

Counts are taken at the *model boundary*: one objective evaluation per distinct
point, one constraint-vector evaluation per distinct point, regardless of how
many callback crossings a solver's API forces (SciPy and mincon call each
constraint group separately; mincon's fmincon facade calls nonlcon twice).
Both the raw crossings and the model-boundary counts are recorded, so the
crossing overhead is visible rather than hidden. The cache is a single-entry
last-point cache under the declared pure/deterministic callback contract every
corpus problem satisfies.
"""
from __future__ import annotations

import time

import numpy as np


class CountingModel:
    def __init__(self, problem):
        self.p = problem
        self.n, self.m = problem.n, problem.m
        self.counts = dict(f_model=0, c_model=0, g_model=0, j_model=0, f_calls=0, c_calls=0, g_calls=0, j_calls=0, failed=0)
        self.callback_seconds = 0.0
        self._fx = None; self._fv = None
        self._cx = None; self._cv = None
        self._gx = None; self._gv = None
        self._jx = None; self._jv = None

    # ---- objective ----
    def f(self, x):
        self.counts["f_calls"] += 1
        x = np.asarray(x, float)
        if self._fx is not None and x.shape == self._fx.shape and np.array_equal(x, self._fx):
            return self._fv
        t = time.perf_counter()
        try:
            v = float(self.p.f(x))
        except Exception:
            self.counts["failed"] += 1
            raise
        finally:
            self.callback_seconds += time.perf_counter() - t
        self.counts["f_model"] += 1
        self._fx, self._fv = x.copy(), v
        return v

    def grad(self, x):
        self.counts["g_calls"] += 1
        x = np.asarray(x, float)
        if self._gx is not None and np.array_equal(x, self._gx):
            return self._gv.copy()
        t = time.perf_counter()
        try:
            v = np.asarray(self.p.grad(x), float)
        finally:
            self.callback_seconds += time.perf_counter() - t
        self.counts["g_model"] += 1
        self._gx, self._gv = x.copy(), v
        return v.copy()

    # ---- constraints (canonical vector) ----
    def c(self, x):
        self.counts["c_calls"] += 1
        x = np.asarray(x, float)
        if self._cx is not None and np.array_equal(x, self._cx):
            return self._cv.copy()
        t = time.perf_counter()
        try:
            v = np.asarray(self.p.cons(x), float)
        except Exception:
            self.counts["failed"] += 1
            raise
        finally:
            self.callback_seconds += time.perf_counter() - t
        self.counts["c_model"] += 1
        self._cx, self._cv = x.copy(), v
        return v.copy()

    def jac(self, x):
        self.counts["j_calls"] += 1
        x = np.asarray(x, float)
        if self._jx is not None and np.array_equal(x, self._jx):
            return self._jv.copy()
        t = time.perf_counter()
        try:
            v = np.asarray(self.p.jac(x), float)
        finally:
            self.callback_seconds += time.perf_counter() - t
        self.counts["j_model"] += 1
        self._jx, self._jv = x.copy(), v
        return v.copy()

    # ---- row groups in canonical form ----
    def row_groups(self):
        """Index arrays (eq, lower-only, upper-only, ranged) from exact bound comparison."""
        cl, cu = self.p.cl, self.p.cu
        eq = cl == cu
        lo = np.isfinite(cl) & ~eq
        hi = np.isfinite(cu) & ~eq
        return dict(eq=np.flatnonzero(eq), lo=np.flatnonzero(lo), hi=np.flatnonzero(hi))
