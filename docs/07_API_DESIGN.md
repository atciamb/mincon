# Specification: API design

Three audiences, three front doors, one core.

---

## 1. The canonical form

Everything lowers to

```
min f(x)   s.t.   c_L <= c(x) <= c_U,   x_L <= x <= x_U
```

Equalities are `c_L[i] == c_U[i]`; one-sided inequalities use infinite bounds.

Deliberately **not** `fmincon`'s form (`A x <= b`, `Aeq x = beq`, `c(x) <= 0`,
`ceq(x) = 0`) because:

* a range constraint `l <= c(x) <= u` needs one row here and two there, and
  duplicated rows are duplicated work and a rank-deficient Jacobian;
* it removes the equality/inequality branch from every kernel;
* CUTEst, AMPL, `.nl` and S2MPJ all speak it, so the benchmark bridge is a
  memcpy rather than a translation. This is exactly what made
  `bench/s2mpj_bridge.py` a hundred lines instead of a project.

---

## 2. Rust: the `Nlp` trait

```rust
pub trait Nlp: Sync {
    fn dims(&self) -> NlpDims;
    fn x_bounds(&self) -> (&[f64], &[f64]);
    fn c_bounds(&self) -> (&[f64], &[f64]);
    fn x0(&self) -> &[f64];
    fn capabilities(&self) -> Capabilities;
    fn objective(&self, x: &[f64]) -> Result<f64, EvalError>;
    fn constraints(&self, x: &[f64], out: &mut [f64]) -> Result<(), EvalError>;
    // gradient, jacobian, hessian_lagrangian, hessian_vector: defaulted
}
```

Three decisions worth defending:

**Evaluation takes `&self`, not `&mut self`.** Counters live in
`EvalCounters` (atomics) held by the driver. This keeps `Nlp: Sync` cheap,
which is what makes the portfolio possible: two algorithms race on the same
problem in parallel threads. Do not "fix" this by adding `&mut self` — cache
behind a `Mutex`/`OnceLock` inside the implementor.

**`Result<_, EvalError>`, not `f64` with `NaN` as a signal.** Makes the
recoverable case explicit and forces every call site to decide what to do,
which is how the retreat-on-failure behaviour ends up everywhere rather than in
the one place someone remembered.

**Capabilities are declared, not probed.** A model says what it can compute;
the derivative layer fills in the rest. The plug-and-play path — no derivatives
at all — is a first-class citizen with the same code path, because that is the
case we are graded on hardest.

### The builder

```rust
let p = Problem::new(2, |x| 100.0*(x[1]-x[0]*x[0]).powi(2) + (1.0-x[0]).powi(2))
    .start_at(&[-1.2, 1.0])
    .lower_bounds(&[0.0, 0.0])
    .inequality(1, |x, c| c[0] = x[0]*x[0] + x[1]*x[1] - 1.0);
let r = mincon::minimize(&p, &Options::default())?;
```

Note `start_at`, `with_gradient`, `with_constraints`, `with_typical_x` rather
than the obvious names: an inherent method **shadows** the trait method of the
same name, so `problem.x0()` silently resolving to a builder setter is a real
footgun. It cost a compile error to find and would have cost a user an hour.

---

## 3. Python: be a drop-in for SciPy

```python
from mincon import minimize     # instead of scipy.optimize

res = minimize(f, x0, jac=None, bounds=[(0, None)]*n,
               constraints=[{"type": "ineq", "fun": g}], options={...})
res.x, res.fun, res.success, res.maxcv, res.notes
```

The signature, the constraint dict format and the result field names all follow
SciPy, because that is who the caller is. **Where SciPy and `fmincon` disagree
we follow SciPy and document the difference loudly**, and there is exactly one
that matters:

> `{"type": "ineq"}` means `fun(x) >= 0` (SciPy). `fmincon` writes nonlinear
> inequalities as `c(x) <= 0`. **Porting from MATLAB means negating them.**

Result fields beyond SciPy's:

| Field | Why |
|---|---|
| `usable` | `success` is `True` only for a certified optimum; a usable-but-uncertified point has `usable=True`. Users should check this before discarding an answer. |
| `maxcv` | maximum constraint violation, unscaled, including bounds |
| `optimality` | the scaled KKT error |
| `notes` | scaling applied, sparsity detected, which portfolio member won, every compromise made |
| `model_time` | time inside the user's callbacks, so "is it the solver or my model?" is answerable |

`notes` is the field that turns support questions into self-service. It should
grow, not shrink.

### The GIL, and the speedup that is sitting there

Every callback crosses the FFI boundary and takes the GIL, so
`Capabilities::parallel_safe` is `false` for Python models and a
finite-difference gradient in `n` variables makes `n` round trips.

**Built in Phase C (`docs/22` section 7.17):** a batched evaluation hook.
`Nlp::objective_batch` and `Nlp::constraints_batch` take several points and
return one value or one failure per point; their defaults loop, and a model
whose batch calls do better says so with `Capabilities::batch`. The
derivative layer then submits a gradient's probes (and a Jacobian's coloring
groups, and the SQP member's second-difference points) in one call, the same
points as the serial path, with the retreat on refused probes done in rounds.
From Python the hook has two sources. `vectorized=True`: the model accepts a
`(k, n)` array and returns `k` values, so a gradient crosses the boundary
once (measured 7x to 82x fewer crossings on NumPy models); it is declared,
not detected, because a scalar model can return `k` numbers for a `(k, n)`
input by accident, and the declaration is checked by always calling such a
model with a two-dimensional array and refusing any other return shape.
`workers=k` (SciPy's name and meaning; `UseParallel=True` on the facade): the
points of a batch are evaluated on a process pool, or through a map-like
callable the caller supplies, which is what a model that costs seconds per
call needs. A scalar `lambda x: ...` with neither option reaches no new code.

The solve itself releases the GIL (`py.detach`), so a `mincon` call does not
block other Python threads.

### MATLAB-style convenience interface

`fmincon(fun, x0, A=None, b=None, Aeq=None, beq=None, lb=None, ub=None,
nonlcon=None, options=None, *, jac=None, args=(), tol=None)` lowers to the
same automatic solver as `minimize`. `A @ x <= b`, `Aeq @ x == beq`, and
`nonlcon(x, *args) -> (c, ceq)` with `c <= 0`, `ceq == 0` use MATLAB's signs.
Bounds accept scalars or vectors; empty linear pairs and nonlinear components
mean absent constraints. Reject mismatched shapes before calling the model.
Derivatives, scaling and solver configuration remain optional.

Return `OptimizeResult`, not a MATLAB tuple. `multipliers` groups `ineqlin`,
`eqlin`, `ineqnonlin`, `eqnonlin`, `lower`, `upper` using the corresponding
MATLAB signs. The raw `con` and `lambda` fields retain the internal `minimize`
convention; use `multipliers` when porting MATLAB code. Both APIs expose `z_l`
and `z_u`. This is a convenience interface, not full MATLAB option compatibility.
Analytical solutions test inequalities, equalities, bounds and multiplier signs.

### Packaging

`maturin` with `abi3-py39`: one wheel per platform covers every CPython from
3.9 up. Five wheels per release instead of thirty-five. Verified: the wheel
builds and installs, and `from mincon import minimize` solves HS71 to published
accuracy with no derivatives.

---

## 4. Unbuilt front doors, in priority order

1. **AD bridge documentation.** JAX, PyTorch, CasADi — worked examples for
   each. Most users already have exact derivatives and do not know they can
   hand them over. Highest value-per-hour in the project.
2. **Sparse Jacobian from Python.** Accept a SciPy sparse matrix from `jac`,
   and a pattern via `jac_sparsity=`.
3. **Warm starting.** `x0=`, `lambda0=`, `z0=` in and out. Parametric studies,
   sensitivity analysis and MPC all re-solve slightly perturbed problems.
4. **Callbacks.** `callback(x, state) -> bool` for progress and early stopping,
   matching SciPy. The `EvalError::UserAbort` path already exists underneath.
5. **`fmincon` façade: implemented.** The convenience interface above supplies
   familiar constraint inputs and grouped multipliers. Full MATLAB options and
   output tuple compatibility are outside its contract.
6. **C ABI.** So Julia, R, Fortran and C++ can call it. A solver that only
   Rust and Python can reach has given up most of its potential users.

---

## 5. Stability promises

Before 1.0 nothing is promised. From 1.0:

* `Nlp`, `Options`, `SolveReport`, `ExitFlag` are semver-stable.
* **Default option *values* may change in a minor release** if the benchmark
  says they should — that is the whole point of having a benchmark — but every
  such change is called out in the changelog with the numbers that justified it.
* `ExitFlag` integer values never change. Users branch on them.
* `SolveReport` gains `#[non_exhaustive]` at 1.0 so that adding a field stays a
  minor bump. It is not marked today only because the crate's own tests
  construct it literally; move those to a builder first.
