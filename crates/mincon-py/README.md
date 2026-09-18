# mincon

Nonlinear constrained optimization with a Rust solver and a simple Python API.
Supply your objective, starting point and constraints. Derivative estimation,
scaling and solver configuration have automatic defaults.

**Experimental release.** This is research software under active
development, not a completed or proven superior replacement for MATLAB's
`fmincon`. It computes local solutions; it does not guarantee global minima.

## Install

```bash
python -m pip install mincon
```

Prebuilt wheels target standard CPython on Windows x86_64 and Linux x86_64.
NumPy is installed automatically. On platforms without a matching wheel, pip
builds from source and requires Rust 1.83 or newer and a C/C++ toolchain.
macOS wheels and broader runtime coverage remain release work.

## Familiar fmincon inputs

```python
from mincon import fmincon

# Closest point to (1, 1), subject to x[0] + x[1] <= 1.
result = fmincon(
    lambda x: ((x - 1)**2).sum(),
    [0., 0.],
    nonlcon=lambda x: ([x.sum() - 1], []),
)
print(result.x)       # approximately [0.5, 0.5]
print(result.fun)     # approximately 0.5
print(result.success, result.message)
```

No gradients, Hessians, algorithm selection or tolerance settings are required.
Optional constraint arguments use MATLAB's conventions:

| Argument | Meaning |
|---|---|
| `A`, `b` | `A @ x <= b` |
| `Aeq`, `beq` | `Aeq @ x == beq` |
| `lb`, `ub` | Lower/upper bounds, scalars or vectors |
| `nonlcon` | Returns `(c, ceq)` with `c <= 0`, `ceq == 0`; use `[]` for an absent component. May instead return `(c, ceq, Jc, Jceq)` with the Jacobians as `(rows, n)` arrays, or pass `nonlcon_jac` separately |

For example, `fmincon(fun, x0, A=[[1, 1]], b=[1], lb=0)` uses only linear
constraints and bounds. The result is a Python object, not MATLAB's output
tuple. `result.multipliers` groups multipliers with MATLAB's signs, readable
as `result.multipliers["eqlin"]` or `result.multipliers.eqlin`. Options use
Python names such as `options={"maxiter": 500}`; MATLAB's names are accepted
as aliases where an equivalent exists (`MaxIterations`,
`MaxFunctionEvaluations`, `OptimalityTolerance`, `ConstraintTolerance`,
`StepTolerance`, `FiniteDifferenceType`, `Display`, `Algorithm`), and a
MATLAB option with no equivalent raises rather than being ignored.
`options={"disp": True}` streams one line per iteration and prints the final
line (`"display": "final"` prints only the last), and `"scaling": True` /
`False` map to the default gradient scaling / none; `"scale_variables"` is
`"auto"` by default (solve in variables divided by their starting magnitudes
when those span a factor of 1e4 or more, `fmincon`'s `TypicalX` done for you;
`True` always, `False` never). `method=` selects
`'auto'`, `'interior-point'` or `'sqp'`; `callback=` receives every
iteration's row and stops the solve when it returns `True`; `hess=` takes
MATLAB's `HessianFcn(x, lambda)` matrix; `warm_start=previous_result` resumes
from an earlier result's multipliers and quasi-Newton model.

## Several starting points

`mincon.multistart(fun, bounds, n_starts=10, x0=..., jac=..., constraints=...,
seed=0, workers=1)` runs `minimize` from `x0` and random points inside the
bounds and returns the best feasible result, with every run in
`result.starts` and the number of distinct feasible objective values in
`result.distinct`. It is local search from several points, not a global
optimizer; `workers > 1` overlaps the engine's work on a thread pool.

## SciPy-style inputs

```python
from mincon import minimize

result = minimize(
    lambda x: ((x - 1)**2).sum(),
    [0., 0.],
    constraints={"type": "ineq", "fun": lambda x: 1 - x.sum()},
)
```

**Sign convention:** `minimize` inequalities mean `fun(x) >= 0`;
`fmincon` nonlinear inequalities mean `c(x) <= 0`. Equalities are zero in both.
Use `bounds=[(0, None), (0, None)]` with `minimize`; use `lb=0` with `fmincon`.

Supply `jac=` if you have an analytical objective gradient and `hess=` (the
Hessian of the Lagrangian) if you have that too; both algorithms use them in
place of their quasi-Newton models and converge in far fewer evaluations.
Supplied derivatives are checked along one direction at `x0` with two extra
evaluations: a gross disagreement raises with the offending component named,
a mild one is noted in `res.notes`. `args=(...)` passes additional arguments
to your objective and nonlinear constraints. Inspect `help(fmincon)` or
`help(minimize)` for the full interface.

## Interpreting results

- `x`, `fun`: returned point and objective.
- `success`: requested numerical first-order and feasibility checks passed.
  This is not a second-order or global-minimum certificate.
- `maxcv`: constraint violation in your original units.
- `usable`: a usable-point status; it does not replace checking `success`.
- `message`, `notes`: termination reason and solver diagnostics, including
  which portfolio member produced the answer, why the quadratic-program
  probe declined, and every compromise made.
- `limit`: at a budget exit (`status == 0`), which limit bound:
  `'iterations'`, `'evaluations'` or `'time'`; `None` otherwise.
- `nfev`, `nit`: objective calls across the whole portfolio and iterations.
- `trace`: one row per iteration (objective, violation, optimality, step,
  barrier parameter), the cheapest way to see what a solve did.

Always examine the status and feasibility before using the result. An
`Acceptable` or stagnation exit has `success=False`. Finite differences and
noisy model evaluations limit attainable accuracy.

## Current scope

Implemented: a primal-dual interior-point method with feasibility
restoration; a sequential quadratic programming member (l1 merit function,
elastic QP, damped BFGS, second-order corrections, a second-order check that
walks off saddle points) that runs first on problems with at most 20
variables; a portfolio that runs the members in sequence with shared
evaluation and time budgets and stops at the first usable answer; a
quadratic-program probe (seven evaluations at the start) that builds the
constant Hessian by structure and takes Newton steps when the problem turns
out to be a QP; gradient scaling and variable scaling from the start;
bounds-aware finite differences with sparsity detection, graph coloring and
termination at the accuracy the derivatives support; a check of supplied
derivatives; exact objective gradients, constraint Jacobians and Lagrangian
Hessians from Python (dense); an iteration callback and display; warm start;
multistart.

Limits: Python callbacks run serially, one point per call, so a
finite-difference gradient in `n` variables costs `n` model evaluations per
iteration (`2n` with `finite_diff='central'`); pass `jac=` when you have it.
`maxfev`, `maxtime` and a `maxiter` you set are shared across the portfolio
and checked between iterations, so they cannot interrupt a running callback
(the default iteration cap of `400 + 10 n` applies to each member). Linear
rows given as `A`, `b`, `Aeq`, `beq` carry their exact Jacobian; nonlinear
rows use finite differences unless `nonlcon_jac` is supplied. Sparse
Jacobians and `jac_sparsity` from Python, limited-memory curvature (dense
quasi-Newton models make problems beyond a few hundred variables expensive
under finite differences), and batched or parallel model evaluation are not
implemented.

The local development gate is 56/56 fixture outcomes through the default
portfolio (53 strict `Optimal` returns; passes also include accurate points
with non-success statuses and expected infeasible or unbounded diagnostics),
195 Rust tests and 37 Python tests. The measured standing against `fmincon`
and SciPy on a 185-problem corpus, including where they win, is in the
repository README and `docs/17_CLAIM_AUDIT.md`.

## License

MIT OR Apache-2.0. Rust dependency notices are included in the installed
package as `THIRD_PARTY_LICENSES.txt`. Source is included in the source
distribution; no MATLAB installation or license is required.
