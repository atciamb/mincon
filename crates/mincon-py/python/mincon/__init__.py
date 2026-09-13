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

__all__ = ["minimize", "fmincon", "multistart", "check_gradients", "OptimizeResult", "Multipliers",
           "ExitFlag", "__version__"]

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


class Multipliers(dict):
    """The multiplier groups of :func:`fmincon`, readable as ``m['ineqlin']`` or ``m.ineqlin``."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(f"no multiplier group named '{name}'; groups: {sorted(self)}") from exc

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


_DISPLAY_LEVELS = ("none", "final", "iter")


def _display_level(opts: dict) -> str:
    """Consume ``disp`` / ``display`` from the option dict and return the level."""
    level = "none"
    if "disp" in opts:
        level = "iter" if opts.pop("disp") else "none"
    if "display" in opts:
        level = opts.pop("display")
        if level not in _DISPLAY_LEVELS:
            raise ValueError(f"display must be one of {_DISPLAY_LEVELS}, got {level!r}")
    return level


_ITER_HEADER = (f"{'Iter':>5} {'F-count':>8} {'f(x)':>14} {'Feasibility':>12} {'Optimality':>12} "
                f"{'Step':>10} {'alpha':>8} {'mu':>9}")


def _iteration_line(t: Mapping[str, Any]) -> str:
    step = t.get("step_norm")
    alpha = t.get("alpha")
    mu = t.get("mu")
    return (f"{t['iter']:>5d} {t['nfev']:>8d} {t['f']:>14.6e} {t['maxcv']:>12.3e} {t['optimality']:>12.3e} "
            f"{'---' if step is None else f'{step:.3e}':>10} "
            f"{'---' if alpha is None else f'{alpha:.3f}':>8} "
            f"{'---' if mu is None else f'{mu:.1e}':>9}"
            + ("  restoration" if t.get("in_restoration") else ""))


def _streaming_callback(level: str, user_callback):
    """The per-iteration callback handed to the engine: prints each row as it
    happens when ``level == 'iter'`` (the engine re-acquires the GIL for the
    call) and forwards the row to the user's callback, whose truthy return
    stops the solve."""
    if level != "iter" and user_callback is None:
        return None
    state = {"members": 0}

    def callback(row):
        if level == "iter":
            if row["iter"] == 0:
                state["members"] += 1
                print(_ITER_HEADER if state["members"] == 1 else "  (portfolio: next member)")
            print(_iteration_line(row), flush=True)
        if user_callback is not None:
            return bool(user_callback(row))
        return False
    return callback


def _print_report(result: "OptimizeResult", level: str) -> None:
    """Print the final line(s); the iteration rows were streamed by the callback."""
    if level in ("iter", "final"):
        tag = "converged" if result["success"] else ("usable" if result.get("usable") else "not solved")
        print(f"mincon ({tag}, status {result['status']}): {result['message']}")
        print(f"  f = {result['fun']:.10g}, max constraint violation = {result['maxcv']:.3e}, "
              f"{result['nit']} iterations, {result['nfev']} objective evaluations")


def minimize(
    fun: Callable[[np.ndarray], float],
    x0: Sequence[float],
    args: tuple = (),
    method: str | None = None,
    jac: Callable[[np.ndarray], Sequence[float]] | None = None,
    hess: Callable[[np.ndarray, np.ndarray], Any] | None = None,
    bounds: Iterable[tuple[float | None, float | None]] | None = None,
    constraints: Mapping | Iterable[Mapping] | None = None,
    tol: float | None = None,
    options: Mapping[str, Any] | None = None,
    callback: Callable[[Mapping[str, Any]], Any] | None = None,
    warm_start: Mapping[str, Any] | None = None,
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
        the first success. ``'sqp'`` is the sequential quadratic programming
        member (l1 merit function, damped BFGS, elastic QP): usually the cheapest
        in function evaluations on small dense problems.
    jac : callable, optional
        ``jac(x, *args) -> array_like``. Without it the gradient is estimated
        by finite differences, which is supported and tested but costs
        accuracy and evaluations. If you have JAX, PyTorch or CasADi, pass
        their gradient here. A supplied gradient or Jacobian is checked along
        one direction at ``x0`` (two extra evaluations): a gross disagreement
        raises ``RuntimeError`` naming the component, a mild one is recorded
        in ``res.notes`` (``options={'check_derivatives': ...}`` selects
        ``'auto'`` (this), ``'full'``/``True`` (every entry at three points, a
        failure raises) or ``False``).
    hess : callable, optional
        ``hess(x, lam, *args) -> (n, n)``: the Hessian of the Lagrangian
        ``f(x) + sum(lam * c(x))`` over every constraint row in the order of
        ``constraints`` (equalities and inequalities alike), with ``lam`` in
        the sign convention of ``res['lambda']``. Both algorithms use it in
        place of their quasi-Newton model; the SQP member regularises an
        indefinite Lagrangian Hessian along the constraint normals only, so
        it keeps Newton convergence (HS71: 7 iterations); the interior-point
        member counts the signs of its KKT pivots instead of perturbing them,
        so an indefinite Lagrangian Hessian with a positive reduced Hessian
        needs no inertia correction (HS71: 10 iterations, the same as its
        quasi-Newton model). Only the symmetric part is used.
    bounds : sequence of (low, high), optional
        Use ``None`` for an infinite side. Bounds are honoured at **every**
        iterate, including finite-difference probes, so a model that is
        undefined outside its box is safe.
    constraints : dict or sequence of dict, optional
        Each is ``{'type': 'eq'|'ineq', 'fun': callable, 'jac': callable}``
        with ``'jac'`` optional. ``'eq'`` means ``fun(x) == 0``; ``'ineq'``
        means ``fun(x) >= 0`` (SciPy's convention, the opposite of
        ``fmincon``'s). ``fun`` may return a scalar or a vector; the length is
        fixed by its value at ``x0``. ``jac(x)`` returns the ``(len, n)``
        Jacobian of that block (or ``(n,)`` for a single row). Analytic
        Jacobians are used only when **every** block supplies one; otherwise
        all rows are estimated by finite differences.
    tol : float, optional
        Sets the optimality, feasibility and complementarity tolerances at once.
    options : dict, optional
        ``maxiter``, ``maxfev``, ``maxtime`` (seconds), ``ftol``, ``ctol``,
        ``threads``, ``seed``, ``check_derivatives``,
        ``scaling`` in ``{'none', 'gradient', 'equilibration'}`` (``True``
        keeps the default ``'gradient'``, ``False`` means ``'none'``),
        ``disp`` (bool: stream one line per iteration as it happens and print
        the final line) or ``display`` in ``{'none', 'final', 'iter'}``,
        ``finite_diff`` in ``{'forward', 'central', 'adaptive'}``,
        ``barrier`` in ``{'monotone', 'adaptive', 'adaptive-then-monotone'}``,
        ``fd_error_aware`` (bool, default True: stop at the accuracy the
        finite-difference derivatives can support instead of chasing a tighter
        tolerance), ``bfgs_scaling`` (bool, default True: rescale the initial
        quasi-Newton matrix when the first step had to be cut hard),
        ``scale_variables`` (``'auto'`` (default), ``True`` or ``False``:
        solve in variables divided by their starting magnitudes,
        ``fmincon``'s ``TypicalX`` done for you, the answer mapped back;
        ``'auto'`` does it only when the magnitudes of ``x0`` span a factor
        of 1e4 or more, which is the unit-mismatch case where every
        first-order test is blind; a start of zeros carries no scale, so
        pass ``True`` with a meaningful ``x0`` or scale by hand there),
        ``kkt_pivot_signs`` (``'auto'`` (default), ``'expected'`` or
        ``'free'``: whether the interior-point member's KKT factorisation
        perturbs a pivot whose sign differs from its block's or keeps it and
        counts the inertia; ``'auto'`` counts with a supplied Hessian, whose
        primal block may legitimately be indefinite, and expects with a
        quasi-Newton one), ``quadratic_probe`` (bool, default True: test at
        the start whether the objective is quadratic and every constraint row
        linear, seven evaluations along two lines, and if so build the
        constant Hessian by differencing, check it is convex, and run the SQP
        member with it: a bounded least-squares deconvolution then takes 2
        iterations instead of 166), ``quadratic_build`` (``'structured'``
        (default) or ``'dense'``: with function values only, build that
        Hessian by structure, the diagonal first and then one band at a
        time until the model reproduces the probe's line points, so a
        diagonal Hessian costs 2n evaluations and a tridiagonal one 3n
        instead of n(n+3)/2; a dense one costs the same either way),
        ``bfgs_rescale`` (float, default 0 = off: rebuild the quasi-Newton
        matrix from per-coordinate curvature quotients whenever its curvature
        along an accepted step is off by more than this factor; 10 helps
        separable large problems and hurts dense coupled ones, so it is
        opt-in).

    warm_start : result or mapping, optional
        A previous result for the same problem (``res``): the solve starts
        from its multipliers (``res['lambda']``, ``res.z_l``, ``res.z_u``), its
        quasi-Newton curvature model (``res.hess_approx``, an ``(n, n)`` array
        when the winning member built one) and, for the interior-point member,
        the barrier parameter its trace ended at, instead of rebuilding them.
        Pass ``x0=res.x`` to resume a solve that a budget interrupted.
    callback : callable, optional
        ``callback(row)`` after every iteration with the same dict as a
        ``res.trace`` row (``iter``, ``nfev``, ``f``, ``maxcv``,
        ``optimality``, ``step_norm``, ``alpha``, ``mu``, ...). Return
        ``True`` to stop the solve (``res.status == -1``, the current iterate
        is returned), like ``fmincon``'s ``OutputFcn`` and SciPy's callback.
        An exception raised inside it stops the solve and is re-raised.

    Returns
    -------
    OptimizeResult
        With ``x``, ``fun``, ``success``, ``usable``, ``status``, ``message``,
        ``nit``, ``nfev``, ``njev``, ``maxcv``, ``optimality``, ``con``,
        ``lambda``, ``notes``, ``trace``, ``time``, ``model_time``. ``trace`` is
        a list of per-iteration dicts (``iter``, ``nfev``, ``f``, ``maxcv``,
        ``optimality``, ``step_norm``, ``alpha``, ``mu``, ``in_restoration``)
        for the member that produced the answer.

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
    if hess is not None and not callable(hess):
        raise TypeError("hess must be callable: hess(x, lam) -> (n, n)")
    if callback is not None and not callable(callback):
        raise TypeError("callback must be callable")

    if args:
        _f = fun
        fun = lambda x: _f(x, *args)  # noqa: E731
        if jac is not None:
            _j = jac
            jac = lambda x: _j(x, *args)  # noqa: E731
        if hess is not None:
            _h = hess
            hess = lambda x, lam: _h(x, lam, *args)  # noqa: E731

    cons = _normalize_constraints(constraints, args)

    opts = dict(options or {})
    display = _display_level(opts)
    supported = {"maxiter", "maxfev", "maxtime", "tol", "ftol", "ctol", "threads",
                 "seed", "check_derivatives", "scaling", "finite_diff", "barrier", "fd_error_aware", "bfgs_scaling",
                 "bfgs_rescale", "scale_variables", "kkt_pivot_signs", "quadratic_probe", "quadratic_build"}
    unknown = set(opts) - supported
    if unknown:
        raise ValueError(f"unknown options: {sorted(unknown, key=str)}; "
                         f"supported: {sorted(supported | {'disp', 'display'})}")
    if isinstance(opts.get("scaling"), bool):
        opts["scaling"] = "gradient" if opts["scaling"] else "none"
    if "bfgs_rescale" in opts:
        v = opts["bfgs_rescale"]
        if not (v == 0 or v > 1):
            raise ValueError("bfgs_rescale must be 0 (off) or a factor greater than 1")
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
        hess=hess,
        callback=_streaming_callback(display, callback),
        warm_start=_warm_start_dict(warm_start),
    )
    result = OptimizeResult(raw)
    if display != "none":
        _print_report(result, display)
    return result


def _warm_start_dict(warm_start):
    """A previous result (or any mapping with ``lambda``, ``z_l``, ``z_u`` and optionally
    ``trace`` / ``mu``) -> the dict the engine takes; ``None`` stays ``None``."""
    if warm_start is None:
        return None
    try:
        lam, zl, zu = warm_start["lambda"], warm_start["z_l"], warm_start["z_u"]
    except (KeyError, TypeError) as e:
        raise ValueError("warm_start must be a previous result (or a mapping with 'lambda', 'z_l' and 'z_u')") from e
    mu = warm_start.get("mu") if hasattr(warm_start, "get") else None
    if mu is None:
        trace = warm_start.get("trace") if hasattr(warm_start, "get") else None
        if trace:
            mu = trace[-1].get("mu")
    hess_approx = warm_start.get("hess_approx") if hasattr(warm_start, "get") else None
    return {"lambda": np.asarray(lam, float).ravel(), "z_l": np.asarray(zl, float).ravel(),
            "z_u": np.asarray(zu, float).ravel(), "mu": (None if mu is None or not mu > 0 else float(mu)),
            "hess_approx": None if hess_approx is None else np.asarray(hess_approx, float)}


def fmincon(fun, x0, A=None, b=None, Aeq=None, beq=None, lb=None, ub=None,
            nonlcon=None, options=None, *, jac=None, nonlcon_jac=None, hess=None, args=(), tol=None,
            method=None, callback=None, warm_start=None):
    """Minimize with MATLAB-style constraint inputs and automatic defaults.

    ``A @ x <= b``, ``Aeq @ x == beq``, ``lb <= x <= ub`` and
    ``nonlcon(x, *args) -> (c, ceq)`` with ``c <= 0`` and ``ceq == 0``.
    Every constraint argument is optional. Bounds may be scalars or vectors.
    No derivatives or solver options are required. Options use the Python
    names documented by :func:`minimize`, not MATLAB option names.

    ``jac(x, *args)`` optionally returns the objective gradient and
    ``nonlcon_jac(x, *args)`` optionally returns ``(Jc, Jceq)`` with shapes
    ``(len(c), n)`` and ``(len(ceq), n)`` (rows are constraints, unlike
    MATLAB's transposed ``GC``). Alternatively ``nonlcon`` itself may return
    ``(c, ceq, Jc, Jceq)``; the shape of the return is fixed by its first
    call. Derivatives are used only when supplied; they are checked along one
    direction at ``x0`` by default (see :func:`minimize`), and against finite
    differences entry by entry when ``options={'check_derivatives': True}``.
    ``hess(x, lam, *args)`` optionally returns MATLAB's ``HessianFcn`` matrix,
    ``hess f + sum(lam.ineqnonlin * hess c) + sum(lam.eqnonlin * hess ceq)``,
    where ``lam`` is a :class:`Multipliers` with MATLAB's signs. ``method``
    selects ``'auto'`` (default), ``'interior-point'`` or ``'sqp'``;
    ``callback(row)`` is called after every iteration and stops the solve
    when it returns ``True``.

    Returns an :class:`OptimizeResult`: use ``r.x``, ``r.fun``, ``r.success``
    and ``r.maxcv``. ``r.multipliers`` groups the MATLAB-sign multipliers as
    ``ineqlin``, ``eqlin``, ``ineqnonlin``, ``eqnonlin``, ``lower``, ``upper``,
    readable as ``r.multipliers['eqlin']`` or ``r.multipliers.eqlin``.
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
        if nonlcon_jac is not None and not callable(nonlcon_jac):
            raise TypeError("nonlcon_jac must be callable and return (Jc, Jceq)")

        # Both components are evaluated from one nonlcon call per point: a
        # single-entry cache keyed on the point avoids calling the model twice.
        # nonlcon may return (c, ceq) or (c, ceq, Jc, Jceq); the arity is fixed
        # by the first call so a model cannot silently change its contract.
        cache = {"x": None, "pair": None}
        arity = [None]

        def pair_at(x):
            xa = np.asarray(x, dtype=float)
            if cache["x"] is not None and np.array_equal(xa, cache["x"]):
                return cache["pair"]
            pair = nonlcon(xa, *args)
            if not isinstance(pair, (tuple, list)) or len(pair) not in (2, 4):
                raise ValueError("nonlcon must return (c, ceq) or (c, ceq, Jc, Jceq), with c <= 0 and ceq == 0")
            if arity[0] is None:
                arity[0] = len(pair)
            elif len(pair) != arity[0]:
                raise ValueError(f"nonlcon returned {len(pair)} values after returning {arity[0]} on its first call")
            cache["x"], cache["pair"] = xa.copy(), pair
            return pair

        returns_jacobians = len(pair_at(x0)) == 4
        if returns_jacobians and nonlcon_jac is not None:
            raise ValueError("nonlcon returns its Jacobians; do not also pass nonlcon_jac")

        def component(index):
            def evaluate(x):
                pair = pair_at(x)
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

        blocks = [{"type": "ineq", "fun": component(0)}, {"type": "eq", "fun": component(1)}]
        if nonlcon_jac is not None or returns_jacobians:
            def jac_component(index):
                def evaluate(x):
                    if returns_jacobians:
                        value = pair_at(x)[2 + index]
                    else:
                        pair = nonlcon_jac(x, *args)
                        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                            raise ValueError("nonlcon_jac must return (Jc, Jceq)")
                        value = pair[index]
                    a = np.asarray([] if value is None else value, dtype=float)
                    a = a.reshape(-1, n) if a.size else np.zeros((0, n))
                    return -a if index == 0 else a
                return evaluate
            blocks[0]["jac"] = jac_component(0)
            blocks[1]["jac"] = jac_component(1)
        cons.extend(blocks)

    # Bind user arguments here so linear constraints do not receive them.
    objective = (lambda x: fun(x, *args)) if args else fun
    gradient = (lambda x: jac(x, *args)) if args and jac is not None else jac
    hessian = None
    if hess is not None:
        if not callable(hess):
            raise TypeError("hess must be callable: hess(x, lam) -> (n, n)")

        def hessian(x, lam):
            # The engine's Lagrangian is f + sum(lam_engine * c_engine) over the blocks
            # [A rows, Aeq rows, c, ceq]; c_engine = -c for the inequality blocks and the
            # MATLAB-sign multipliers are -lam_engine there, so MATLAB's HessianFcn formula
            # with these groups is exactly the engine's Lagrangian Hessian.
            groups = Multipliers()
            start = 0
            for name, size, sign in zip(["ineqlin", "eqlin", "ineqnonlin", "eqnonlin"],
                                        sizes + nonlinear_sizes, [-1., 1., -1., 1.]):
                groups[name] = sign * np.asarray(lam, float)[start:start + size]
                start += size
            return hess(x, groups, *args)
    result = minimize(objective, x0, jac=gradient, hess=hessian, bounds=bounds,
                      constraints=cons, tol=tol, options=options, method=method, callback=callback,
                      warm_start=warm_start)
    multipliers = Multipliers()
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
        jac = c.get("jac")
        if jac is not None and not callable(jac):
            raise TypeError("a constraint's 'jac' must be callable (a string estimator is not supported)")
        if jac is not None:
            d["jac"] = jac
        if args:
            _c = c["fun"]
            d["fun"] = lambda x, _c=_c: _c(x, *args)
            if jac is not None:
                d["jac"] = lambda x, _j=jac: _j(x, *args)
        out.append(d)
    return out


def multistart(
    fun: Callable[[np.ndarray], float],
    bounds: Iterable[tuple[float | None, float | None]],
    n_starts: int = 10,
    *,
    x0: Sequence[float] | None = None,
    args: tuple = (),
    method: str | None = None,
    jac: Callable[[np.ndarray], Sequence[float]] | None = None,
    constraints: Mapping | Iterable[Mapping] | None = None,
    tol: float | None = None,
    options: Mapping[str, Any] | None = None,
    seed: int | None = 0,
    workers: int = 1,
) -> OptimizeResult:
    """Run :func:`minimize` from several starting points and return the best result.

    This is local search from ``n_starts`` points, not a global optimizer:
    it finds the best of the basins those starts fall into and nothing
    else. Starts are drawn uniformly inside ``bounds`` (every bound must be
    finite unless ``x0`` is given, in which case unbounded coordinates are
    sampled within ``x0 +- max(1, |x0|)``); ``x0`` itself, when given, is
    the first start. ``seed`` makes the draw reproducible.

    Results are ranked by feasibility first (``maxcv`` within the solver's
    constraint tolerance), then by ``usable``, then by ``fun``. The returned
    :class:`OptimizeResult` is the best run's, with three extra fields:
    ``starts`` (all results, in start order), ``distinct`` (the number of
    distinct objective values among feasible runs, to ``1e-6`` relative) and
    ``nfev_total``. ``workers > 1`` runs starts on a thread pool; the engine
    releases the GIL while it works, so this overlaps the numerical work,
    while Python callbacks still serialize on the GIL.
    """
    if n_starts < 1:
        raise ValueError("n_starts must be at least 1")
    bl = list(bounds)
    lo = np.array([-np.inf if b[0] is None else float(b[0]) for b in bl])
    hi = np.array([np.inf if b[1] is None else float(b[1]) for b in bl])
    n = lo.size
    if np.any(hi < lo):
        raise ValueError("every upper bound must be at least its lower bound")
    center = None if x0 is None else np.asarray(x0, dtype=float).ravel()
    if center is not None and center.size != n:
        raise ValueError(f"x0 has {center.size} entries but bounds has {n}")
    lo_s, hi_s = lo.copy(), hi.copy()
    infinite = ~np.isfinite(lo) | ~np.isfinite(hi)
    if np.any(infinite):
        if center is None:
            raise ValueError("bounds must be finite in every coordinate unless x0 is given")
        spread = np.maximum(1.0, np.abs(center))
        lo_s = np.where(np.isfinite(lo), lo, center - spread)
        hi_s = np.where(np.isfinite(hi), hi, center + spread)
        lo_s = np.maximum(lo_s, lo)
        hi_s = np.minimum(hi_s, hi)
    rng = np.random.default_rng(seed)
    starts = [] if center is None else [center]
    while len(starts) < n_starts:
        starts.append(lo_s + rng.random(n) * (hi_s - lo_s))

    def solve(start):
        return minimize(fun, start, args=args, method=method, jac=jac, bounds=bl,
                        constraints=constraints, tol=tol, options=options)

    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(solve, starts))
    else:
        results = [solve(s) for s in starts]

    ctol = float((options or {}).get("ctol", 1e-6))

    def rank(r):
        feasible = r["maxcv"] <= ctol
        return (not feasible, not r.get("usable", False), r["fun"])

    best = min(results, key=rank)
    feasible_values = sorted(r["fun"] for r in results if r["maxcv"] <= ctol)
    distinct = 0
    last = None
    for v in feasible_values:
        if last is None or abs(v - last) > 1e-6 * max(1.0, abs(last)):
            distinct += 1
            last = v
    out = OptimizeResult(best)
    out["starts"] = results
    out["distinct"] = distinct
    out["nfev_total"] = int(sum(r["nfev"] for r in results))
    return out
