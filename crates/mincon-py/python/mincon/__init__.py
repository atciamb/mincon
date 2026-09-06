"""mincon - a robust nonlinear constrained optimizer.

Solves

    minimize    f(x)
    subject to  c_L <= c(x) <= c_U
                x_L <=   x  <= x_U

The API deliberately mirrors :func:`scipy.optimize.minimize`, so an existing
script usually needs only a changed import::

    from mincon import minimize          # instead of scipy.optimize

    res = minimize(rosen, x0, bounds=[(0, None)] * 2,
                   constraints=[{"type": "ineq", "fun": lambda x: 1 - x @ x}])
    print(res.x, res.fun, res.success)

Conventions, spelled out because this is where silent errors live:

* ``{"type": "ineq"}`` means ``fun(x) >= 0``. That is SciPy's convention. It is
  the **opposite** of MATLAB's ``fmincon``, where nonlinear inequalities are
  written ``c(x) <= 0``. Porting from MATLAB means negating them.
* ``res.success`` is ``True`` only when first-order and feasibility tolerances
  are satisfied; this is not a second-order minimum certificate. A solve that
  stopped at a usable but uncertified point has ``res.success == False`` and
  ``res.usable == True``. Check ``res.usable`` before discarding an answer, and
  never treat ``success`` as "the numbers are fine" without also checking
  ``res.maxcv``.
* ``res.maxcv`` is the maximum constraint violation in the *original*,
  unscaled problem, including variable bounds.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from . import _mincon

__all__ = ["minimize", "check_gradients", "OptimizeResult", "ExitFlag", "__version__"]

__version__ = _mincon.__version__


class ExitFlag:
    """Integer exit codes, matching ``fmincon``'s where the meanings coincide."""

    _values = _mincon.exit_flags()
    OPTIMAL = _values["OPTIMAL"]
    STEP_TOLERANCE = _values["STEP_TOLERANCE"]
    FUNCTION_TOLERANCE = _values["FUNCTION_TOLERANCE"]
    ACCEPTABLE = _values["ACCEPTABLE"]
    MAX_REACHED = _values["MAX_REACHED"]
    STOPPED_BY_USER = _values["STOPPED_BY_USER"]
    INFEASIBLE = _values["INFEASIBLE"]
    UNBOUNDED = _values["UNBOUNDED"]
    LOCALLY_INFEASIBLE = _values["LOCALLY_INFEASIBLE"]
    NUMERICAL_FAILURE = _values["NUMERICAL_FAILURE"]


class OptimizeResult(dict):
    """A dict that also allows attribute access, like SciPy's."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - mirrors SciPy behaviour
            raise AttributeError(name) from exc

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self) -> str:
        order = [
            "success",
            "usable",
            "status",
            "message",
            "fun",
            "x",
            "maxcv",
            "optimality",
            "nit",
            "nfev",
            "njev",
            "time",
        ]
        width = max(len(k) for k in order)
        lines = []
        for k in order:
            if k in self:
                lines.append(f"  {k:<{width}} : {self[k]!r}")
        if self.get("notes"):
            lines.append("  notes:")
            lines.extend(f"    - {n}" for n in self["notes"])
        return "OptimizeResult(\n" + "\n".join(lines) + "\n)"


def minimize(
    fun: Callable[[np.ndarray], float],
    x0: Sequence[float],
    args: tuple = (),
    method: str | None = None,
    jac: Callable[[np.ndarray], Sequence[float]] | None = None,
    bounds: Iterable[tuple[float | None, float | None]] | None = None,
    constraints: Mapping | Iterable[Mapping] | None = None,
    tol: float | None = None,
    options: Mapping[str, Any] | None = None,
) -> OptimizeResult:
    """Minimize a scalar function subject to bounds and constraints.

    Parameters
    ----------
    fun : callable
        ``fun(x, *args) -> float``.
    x0 : array_like
        Starting point.
    args : tuple, optional
        Extra arguments passed to ``fun``, ``jac`` and every constraint.
    method : {'auto', 'interior-point', 'sqp'}, optional
        ``'auto'`` (the default) races several configurations and returns the
        best answer. On a single thread it runs them in sequence and stops at
        the first success, so it never costs more than a single solve on a
        problem the default configuration handles.
    jac : callable, optional
        ``jac(x, *args) -> array_like``. Without it the gradient is estimated
        by finite differences, which is supported and tested but costs
        accuracy and evaluations. If you have JAX, PyTorch or CasADi, pass
        their gradient here.
    bounds : sequence of (low, high), optional
        Use ``None`` for an infinite side. Bounds are honoured at **every**
        iterate, including finite-difference probes, so a model that is
        undefined outside its box is safe.
    constraints : dict or sequence of dict, optional
        Each is ``{'type': 'eq'|'ineq', 'fun': callable}``. ``'eq'`` means
        ``fun(x) == 0``; ``'ineq'`` means ``fun(x) >= 0`` (SciPy's convention,
        the opposite of ``fmincon``'s). ``fun`` may return a scalar or a
        vector; the length is fixed by its value at ``x0``.
    tol : float, optional
        Sets the optimality, feasibility and complementarity tolerances at once.
    options : dict, optional
        ``maxiter``, ``maxfev``, ``maxtime`` (seconds), ``ftol``, ``ctol``,
        ``threads``, ``seed``, ``check_derivatives``,
        ``scaling`` in ``{'none', 'gradient', 'equilibration'}``,
        ``finite_diff`` in ``{'forward', 'central', 'adaptive'}``.

    Returns
    -------
    OptimizeResult
        With ``x``, ``fun``, ``success``, ``usable``, ``status``, ``message``,
        ``nit``, ``nfev``, ``njev``, ``maxcv``, ``optimality``, ``con``,
        ``lambda``, ``notes``, ``time``, ``model_time``.

    Notes
    -----
    ``res.notes`` is worth reading when something looks wrong: it records the
    scaling factors applied, the detected sparsity, which portfolio member won,
    and any place the solver had to compromise.
    """
    x0 = np.ascontiguousarray(np.asarray(x0, dtype=np.float64).ravel())

    if args:
        _f = fun
        fun = lambda x: _f(x, *args)  # noqa: E731
        if jac is not None:
            _j = jac
            jac = lambda x: _j(x, *args)  # noqa: E731

    cons = _normalize_constraints(constraints, args)

    opts = dict(options or {})
    if tol is not None:
        opts.setdefault("tol", tol)

    raw = _mincon.minimize(
        fun,
        x0,
        jac=jac,
        bounds=_normalize_bounds(bounds, x0.size),
        constraints=cons,
        method=method,
        options=opts,
    )
    return OptimizeResult(raw)


def check_gradients(
    fun: Callable[[np.ndarray], float],
    x0: Sequence[float],
    jac: Callable[[np.ndarray], Sequence[float]],
    num_points: int = 3,
    tol: float = 1e-5,
) -> OptimizeResult:
    """Compare an analytic gradient against central finite differences.

    A wrong gradient is the most common reason an optimizer "does not work" on
    a model, and it is nearly impossible to diagnose from convergence
    behaviour. Run this first when a solve misbehaves. Unlike ``fmincon``'s
    ``CheckGradients``, this checks several points, not just ``x0``, so a
    gradient that is only wrong past a branch is still caught.
    """
    x0 = np.ascontiguousarray(np.asarray(x0, dtype=np.float64).ravel())
    return OptimizeResult(_mincon.check_gradients(fun, x0, jac, num_points, tol))


def _normalize_bounds(bounds, n):
    if bounds is None:
        return None
    out = []
    for i, b in enumerate(bounds):
        if b is None:
            out.append((None, None))
            continue
        lo, hi = b
        out.append((None if lo is None else float(lo), None if hi is None else float(hi)))
    if len(out) != n:
        raise ValueError(f"bounds has {len(out)} entries but x0 has {n}")
    return out


def _normalize_constraints(constraints, args):
    if constraints is None:
        return None
    if isinstance(constraints, Mapping):
        constraints = [constraints]
    out = []
    for c in constraints:
        if not isinstance(c, Mapping):
            raise TypeError("each constraint must be a dict with 'type' and 'fun'")
        d = {"type": c["type"], "fun": c["fun"]}
        if args:
            _c = c["fun"]
            d["fun"] = lambda x, _c=_c: _c(x, *args)
        out.append(d)
    return out
