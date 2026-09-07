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

__all__ = ["minimize", "fmincon", "check_gradients", "OptimizeResult", "ExitFlag", "__version__"]

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
    method : {'auto', 'interior-point'}, optional
        ``'auto'`` (the default) races several configurations and returns the
        best answer. On a single thread it runs them in sequence and stops at
        the first success. SQP is not implemented in this release.
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
    if x0.size == 0 or not np.all(np.isfinite(x0)):
        raise ValueError("x0 must contain at least one finite number and no NaN or infinity")
    if not callable(fun) or (jac is not None and not callable(jac)):
        raise TypeError("fun and an optional jac must be callable")

    if args:
        _f = fun
        fun = lambda x: _f(x, *args)  # noqa: E731
        if jac is not None:
            _j = jac
            jac = lambda x: _j(x, *args)  # noqa: E731

    cons = _normalize_constraints(constraints, args)

    opts = dict(options or {})
    supported = {"maxiter", "maxfev", "maxtime", "tol", "ftol", "ctol", "threads",
                 "seed", "check_derivatives", "scaling", "finite_diff"}
    unknown = set(opts) - supported
    if unknown:
        raise ValueError(f"unknown options: {sorted(unknown, key=str)}; supported: {sorted(supported)}")
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


def fmincon(fun, x0, A=None, b=None, Aeq=None, beq=None, lb=None, ub=None,
            nonlcon=None, options=None, *, jac=None, args=(), tol=None):
    """Minimize with MATLAB-style constraint inputs and automatic defaults.

    ``A @ x <= b``, ``Aeq @ x == beq``, ``lb <= x <= ub`` and
    ``nonlcon(x, *args) -> (c, ceq)`` with ``c <= 0`` and ``ceq == 0``.
    Every constraint argument is optional. Bounds may be scalars or vectors.
    No derivatives or solver options are required. Options use the Python
    names documented by :func:`minimize`, not MATLAB option names.

    Returns an :class:`OptimizeResult`: use ``r.x``, ``r.fun``, ``r.success``
    and ``r.maxcv``. ``r.multipliers`` groups the MATLAB-sign multipliers as
    ``ineqlin``, ``eqlin``, ``ineqnonlin``, ``eqnonlin``, ``lower``, ``upper``.
    The raw ``con`` and ``lambda`` fields retain the conventions of minimize.
    This is a convenience interface, not MATLAB output-tuple compatibility.

    Example: ``fmincon(lambda x: ((x-1)**2).sum(), [0., 0.],
    nonlcon=lambda x: ([x.sum()-1], []))`` returns approximately ``[.5, .5]``.
    """
    x0 = np.asarray(x0, dtype=float).ravel()
    n = x0.size
    cons = []
    sizes = []
    for matrix, rhs, name, kind in [(A, b, "A", "ineq"), (Aeq, beq, "Aeq", "eq")]:
        if matrix is None and rhs is None:
            sizes.append(0)
            continue
        if matrix is None or rhs is None:
            raise ValueError(f"{name} and its right-hand side must be supplied together")
        mat = np.asarray(matrix, dtype=float)
        vec = np.asarray(rhs, dtype=float).ravel()
        if mat.size == 0 and vec.size == 0:
            sizes.append(0)
            continue
        if mat.ndim != 2 or mat.shape != (vec.size, n):
            raise ValueError(f"{name} must have shape ({vec.size}, {n})")
        if not np.all(np.isfinite(mat)) or not np.all(np.isfinite(vec)):
            raise ValueError(f"{name} and its right-hand side must be finite")
        sign = -1.0 if kind == "ineq" else 1.0
        cons.append({"type": kind, "fun": lambda x, mat=mat, vec=vec, sign=sign: sign*(mat @ x-vec)})
        sizes.append(vec.size)

    def bound(value, default, name):
        if value is None:
            return np.full(n, default)
        a = np.asarray(value, dtype=float)
        if a.size == 0:
            return np.full(n, default)
        if a.ndim == 0:
            a = np.full(n, float(a))
        if a.shape != (n,):
            raise ValueError(f"{name} must be a scalar or a vector of length {n}")
        return a

    bounds = list(zip(bound(lb, -np.inf, "lb"), bound(ub, np.inf, "ub")))
    nonlinear_sizes = [0, 0]
    nonlinear_seen = [False, False]
    if nonlcon is not None:
        if not callable(nonlcon):
            raise TypeError("nonlcon must be callable and return (c, ceq)")

        def component(index):
            def evaluate(x):
                pair = nonlcon(x, *args)
                if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                    raise ValueError("nonlcon must return (c, ceq), with c <= 0 and ceq == 0")
                value = pair[index]
                a = np.asarray([] if value is None else value, dtype=float)
                if a.ndim > 1:
                    raise ValueError("nonlcon components must be scalars or one-dimensional arrays")
                a = a.reshape(-1)
                if nonlinear_seen[index] and nonlinear_sizes[index] != a.size:
                    raise ValueError("nonlcon component lengths must remain constant")
                nonlinear_sizes[index] = a.size
                nonlinear_seen[index] = True
                return -a if index == 0 else a
            return evaluate

        cons.extend([{"type": "ineq", "fun": component(0)},
                     {"type": "eq", "fun": component(1)}])

    # Bind user arguments here so linear constraints do not receive them.
    objective = (lambda x: fun(x, *args)) if args else fun
    gradient = (lambda x: jac(x, *args)) if args and jac is not None else jac
    result = minimize(objective, x0, jac=gradient, bounds=bounds,
                      constraints=cons, tol=tol, options=options)
    multipliers = {}
    start = 0
    for name, size, sign in zip(
        ["ineqlin", "eqlin", "ineqnonlin", "eqnonlin"],
        sizes + nonlinear_sizes, [-1., 1., -1., 1.],
    ):
        multipliers[name] = sign * result["lambda"][start:start+size]
        start += size
    multipliers["lower"] = result.z_l
    multipliers["upper"] = result.z_u
    result.multipliers = multipliers
    return result


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
        low = -np.inf if lo is None else float(lo)
        high = np.inf if hi is None else float(hi)
        if np.isnan(low) or np.isnan(high) or low > high or low == np.inf or high == -np.inf:
            raise ValueError(f"invalid bounds at index {i}: require low <= high with no NaN")
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
