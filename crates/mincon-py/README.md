# mincon

Nonlinear constrained optimization with a Rust solver and a simple Python API.
Supply your objective, starting point and constraints. Derivative estimation,
scaling and solver configuration have automatic defaults.

**Experimental 0.1 release.** This is research software under active
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
| `nonlcon` | Returns `(c, ceq)` with `c <= 0`, `ceq == 0`; use `[]` for an absent component |

For example, `fmincon(fun, x0, A=[[1, 1]], b=[1], lb=0)` uses only linear
constraints and bounds. The result is a Python object, not MATLAB's output
tuple. `result.multipliers` groups multipliers with MATLAB's signs. Options
use Python names such as `options={"maxiter": 500}`, not MATLAB option names.

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

Supply `jac=` if you have an analytical objective gradient. It is optional.
`args=(...)` passes additional arguments to your objective and nonlinear
constraints. Inspect `help(fmincon)` or `help(minimize)` for the full interface.

## Interpreting results

- `x`, `fun`: returned point and objective.
- `success`: requested numerical first-order and feasibility checks passed.
  This is not a second-order or global-minimum certificate.
- `maxcv`: constraint violation in your original units.
- `usable`: a usable-point status; it does not replace checking `success`.
- `message`, `notes`: termination reason and solver diagnostics.
- `nfev`, `nit`: objective calls across the portfolio and iteration count.

Always examine the status and feasibility before using the result. An
`Acceptable` or stagnation exit has `success=False`. Finite differences and
noisy model evaluations limit attainable accuracy.

## Current scope

Implemented: primal-dual interior point, feasibility restoration, automatic
gradient scaling, bounded finite differences with retreat, Jacobian sparsity
detection/coloring and a portfolio of interior-point configurations. Python
callbacks run serially. SQP, sparse user Jacobians, limited-memory curvature,
hard per-callback evaluation budgets and progress callbacks are not complete.
`maxfev` and time limits are checked between batches/iterations; they cannot
interrupt an ongoing callback. The derivative checker needs further validation
for failed or unevaluable checks; it is not a certificate.

The local development suite passes 54/54 expected fixture outcomes, with
40 strict Optimal returns. Fixture passes also include accurate points with
non-success statuses and expected infeasible/unbounded diagnostics. Broader
CUTEst and matched competitor comparisons remain outstanding.

## License

MIT OR Apache-2.0. Rust dependency notices are included in the installed
package as `THIRD_PARTY_LICENSES.txt`. Source is included in the source
distribution; no MATLAB installation or license is required.

