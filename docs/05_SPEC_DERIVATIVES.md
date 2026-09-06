# Specification: derivatives

Implemented in `crates/mincon-diff`.

Derivative quality sets a ceiling on everything above it. Forward differences
give `sqrt(eps) ≈ 1e-8` relative error in the gradient, so a solver using them
cannot honestly converge to `1e-8` optimality no matter how good its
globalization is. This is the layer where the most achievable improvement over
`fmincon` sits, and also the layer where a mistake is invisible: a wrong
derivative does not raise an error, it makes the line search fail three levels
up.

---

## 1. Hierarchy of sources, best first

| Source | Accuracy | Cost per gradient | Status |
|---|---|---|---|
| Analytic, user-supplied | exact | 1 model call | **implemented** |
| AD bridged from the user's framework (JAX, PyTorch, CasADi) | exact | 1 call | **the highest-value unbuilt item** |
| Native reverse-mode AD over a tape | exact | ~3x one call | not built |
| Native forward-mode dual numbers | exact | `n` calls, or `1` per Hessian-vector product | not built |
| Central differences | `eps^(2/3) ≈ 6e-11` | `2n` calls | implemented |
| Forward differences | `sqrt(eps) ≈ 1e-8` | `n` calls | implemented |

**Do not treat finite differences as the destination.** They are the fallback
that must work well, not the plan.

### 1.1 The AD bridge is the priority

Most Python users of a nonlinear solver already have exact derivatives
available and do not know it. A `jax.grad(f)` or a `torch.func.jacrev` handed
to `jac=` costs one call and is exact. The work is not implementing AD; it is:

* documenting it loudly, with a worked example per framework;
* accepting a **sparse** Jacobian from the framework, not just dense;
* making `check_gradients` the first thing a confused user runs.

This is a documentation-and-plumbing task with an enormous payoff, and it
should land before any native AD work.

---

## 2. Finite differences

### 2.1 Bounds-respecting steps — non-negotiable

Textbook finite differences perturb by `+h`. Real models are undefined outside
their box: a thickness that must be positive, a concentration below one, an FEA
that diverges. A solver that steps outside to build a derivative gets `NaN`,
and the user experiences it as "the optimizer crashed on my problem".

`fd::forward_step` therefore:

1. computes `h = rel * max(|x_j|, typical_j, 1)`, signed by `sign(x_j)`;
2. **flips the direction** if `x_j + h` would leave the box — a full-size step
   the other way beats a tiny one toward the bound;
3. shrinks only when both directions are blocked;
4. returns `h = 0` for a variable pinned by `lb == ub`, and the caller writes a
   zero derivative rather than dividing by zero;
5. **recomputes the realized step as `(x_j + h) - x_j`** after rounding, so the
   divisor is the perturbation that actually happened. Skipping this loses
   digits precisely where `x_j` is large.

Defaults match `fmincon`: `sqrt(eps)` forward, `eps^(1/3)` central.

### 2.2 Adaptive escalation

`FdType::Adaptive` starts forward and switches to central when the line search
starts failing — the signature of derivative noise. `fmincon` makes the user
choose; switching automatically recovers most of the accuracy benefit for a
fraction of the cost.

`TORTURE_NOISY` (deterministic `1e-9` noise on the objective, as from an inner
solve with a loose tolerance) exists to keep this honest. Note the corollary
recorded in that problem's notes: **the achievable optimality tolerance is
bounded below by the model's own noise**, so demanding `1e-8` there is a
specification error, not a solver failure.

### 2.3 Retreat on failure

A probe that returns non-finite has its step halved, up to eight times, before
the derivative is given up on. Group probes shrink together so the columns stay
consistent with the divisors used to recover them.

---

## 3. Coloring

Two columns that are **structurally orthogonal** — no row where both are
nonzero — can be perturbed in the same evaluation and read off separately
(Curtis, Powell & Reid, 1974). Partitioning columns into as few such groups as
possible is a graph-coloring problem.

Measured: a tridiagonal Jacobian needs **3 evaluations regardless of `n`**,
against `n` for the dense treatment. `fmincon` does not do this for nonlinear
constraint Jacobians.

**Jacobians need distance-1 coloring** of the column intersection graph.
Implemented greedily with a largest-first ordering, which is within a color or
two of optimal on structured matrices.

**Hessians need star coloring** (Coleman & Moré, 1984), which is strictly
stronger. A distance-1 coloring of a symmetric matrix silently mixes `H[i][j]`
with `H[j][i]` contributions from different columns. `star_coloring` currently
implements the conservative distance-2 coloring, which is always valid and uses
more groups than necessary; replacing it with a true star coloring typically
saves 20–40% of the groups on a mesh Hessian.

**Never change a coloring without `verify_coloring` in the test.** It checks the
orthogonality property directly; a merely valid-looking coloring produces
subtly wrong derivatives.

---

## 4. Sparsity detection

Detecting the sparsity of a function you can only *evaluate* costs `n`
evaluations in the worst case — there is no way to rule out a dependence you
never probed.

`detect::detect_jacobian_sparsity` probes one variable at a time from **two**
base points and takes the union. Two rather than one because a single probe can
land on a stationary point of a genuine dependence (`d/dx x^2` at `x = 0`) and
report a structural zero that is not one — and a wrongly-narrow pattern is a
*silent wrong answer*, not a slow one. The test
`two_base_points_survive_a_stationary_first_probe` pins this.

Gated at `max_probe_variables = 5000`; above that it returns `Dense` with a
reason rather than spending the whole budget. Detection returns `Dense`, never
an error, whenever it is inconclusive: dense is always correct, merely slow.

Honest hierarchy: user declares it (free, exact) > AD derives it (exact) >
probing (`2n` evaluations and only *probably* right).

---

## 5. The derivative checker

A wrong analytic gradient is the most common reason a model "does not work" and
is nearly impossible to diagnose from convergence behaviour. `check_derivatives`
compares against central differences and ranks the worst offenders.

Two improvements over `fmincon`'s `CheckGradients`:

* **Several points, not just `x0`.** A gradient wrong only past a branch is
  invisible at the starting point and catastrophic in the line search.
* **Relative discrepancy with an absolute floor, ranked.** "Component 47 is
  3.2x too large" is actionable; "max discrepancy 1e-3" is not.

Exposed in Python as `mincon.check_gradients`. It should be the first thing
the troubleshooting documentation tells a user to run.

---

## 6. Batched evaluation — the Python speedup

Every Python callback crosses the FFI boundary and takes the GIL. A
finite-difference gradient in `n` variables therefore makes `n` round trips
where one would do.

**The task:** an optional vectorized protocol — the model advertises that it
accepts an `(k, n)` array and returns `k` values — so a whole coloring group,
or a whole gradient, crosses once. For a NumPy model this is close to a free
`n`-fold reduction in interpreter overhead, and it is the largest single
speedup available on the Python side.

Design constraint: it must be *optional* and detected, never required. A user
with a scalar `lambda x: ...` must keep working.
