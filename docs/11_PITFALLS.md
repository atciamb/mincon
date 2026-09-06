# Pitfalls

Failure modes that are expensive because they are quiet. Most produce a solver
that works on textbook problems and falls apart on real ones — which is exactly
the gap between an implementation and `fmincon`.

---

## Category 1: silently wrong answers

The worst class. No error, no crash, plausible output.

### 1.1 Reporting success on an infeasible problem

**The single unforgivable bug.** A user builds on that answer. `fmincon`'s
`-2` is trustworthy after twenty years and matching that is a hard requirement.

*Guard:* `TORTURE_INFEASIBLE`, and the `lied` column in
`bench/profiles.py`'s table. Any non-zero entry is a release blocker.

### 1.2 Wrong multiplier signs

The canonical form's `lambda` is positive at an active **upper** bound
`c(x) <= c_U`; `fmincon`'s `lambda.ineqnonlin` is non-negative for `c(x) <= 0`.
Get the conversion backwards and the multipliers look entirely plausible.

*Guard:* `kkt::tests::assembles_and_solves_a_hand_checkable_kkt_system` pins
the sign on a system solvable by hand. **This test caught a real sign error
during development.**

### 1.3 Forgetting to unscale

Scaling transforms the problem. The objective, the constraint violation and the
multipliers must all be reported in the user's units:

```
lambda_i = lambda_tilde_i * d_c_i / d_f
z        = z_tilde / d_f
```

A user who sees a different objective than their own function returns will not
trust anything else the solver says.

*Guard:* needs a dedicated multiplier test. **Write one.**

### 1.4 Distance-1 coloring of a Hessian

Valid for Jacobians, **wrong for symmetric matrices** — it mixes `H[i][j]` and
`H[j][i]` contributions from different columns. Produces slightly wrong second
derivatives, which appear as "the line search keeps failing" three layers up.

*Guard:* `verify_coloring` in every coloring test; `star_coloring` for
symmetric matrices, never `distance1_coloring`.

### 1.5 Trusting an uncertified inertia

`LDL^T` without pivoting can suffer catastrophic element growth, after which
the signs of `D` describe a matrix that was never computed. Sylvester's law
still holds — for the wrong matrix.

*Guard:* `Factorization::inertia_is_certified()` and
`catastrophic_element_growth_is_detected_not_hidden`. **This guard was added
because a test failed with a solution off by 30 orders of magnitude.**

### 1.6 Narrow sparsity from a single-point probe

`d/dx x^2` at `x = 0` moves nothing, so one probe reports a structural zero
that is not one. A wrongly-narrow pattern is a silent wrong answer, not a slow
one.

*Guard:* two base points, and
`two_base_points_survive_a_stationary_first_probe`.

---

## Category 2: works on paper, fails on real models

### 2.1 Stepping outside the bounds

Engineering models are full of quantities that must be positive. A
finite-difference probe or a line-search trial that leaves the box gets `NaN`,
and the solver is blamed.

*Guard:* `fd_respect_bounds` (default on), `honor_bounds` (default on),
`HS110` and `TORTURE_DOMAIN`.

### 2.2 Treating a non-finite evaluation as fatal

`fmincon`'s `sqp` shortens the step and retries. Most open-source solvers give
up. This single behaviour is worth more in practice than most convergence
theory.

*Guard:* `EvalError::is_retryable`, the retreat loops in `mincon-diff`, and
`TORTURE_NAN_POCKET` — a hole in the domain that is *not* at a bound, lying
directly between the start and the minimum.

### 2.3 An unscaled termination test

On a problem whose multipliers reach `1e6`, an absolute gradient tolerance can
never be met and the solver reports failure at a perfectly good solution.

*Guard:* the `s_d`/`s_c` scaling in `E_mu`; `TORTURE_DEGENERATE` and `HS13`.

### 2.4 Undamped BFGS on a constrained problem

`y` is the change in the gradient of the **Lagrangian**, which is indefinite at
the solution, so `s'y` goes negative routinely and an undamped update destroys
the approximation.

*Guard:* Powell damping,
`powell_damping_keeps_positive_definiteness_on_negative_curvature`.

### 2.5 Dividing by a degenerate bound interval

`lb == ub` makes the barrier interval empty; `Sigma` becomes infinite.

*Guard:* `bound_relax_factor` and `TORTURE_PINNED`.

### 2.6 An unreachable divergence threshold

IPOPT's `diverging_iterates_tol` is an absolute `1e20` on `||x||`. An iterate
growing *linearly* reaches only `3e13` in 420 iterations, so the test never
fires and the user gets `MaxReached`, which tells them nothing. Measured on
`TORTURE_UNBOUNDED`; the relative-growth test fires in 16 iterations.

**General lesson: a safeguard that cannot fire is not a safeguard.** Every
threshold needs a test that makes it trigger.

---

## Category 3: benchmarking mistakes

### 3.1 Scoring on the solver's own status

Every solver has its own tolerances and its own idea of "converged". Scoring on
`result.success` measures reporting culture.

*Guard:* `bench/runner.py` recomputes the objective and the violation from the
problem's own callables. Success is decided by the harness, always.

### 3.2 Counting only the winner's evaluations

The portfolio runs several members. Reporting the winner's count is cheating
and a reviewer will check.

*Guard:* `PortfolioReport::total_f_evals`.

### 3.3 Counting `Acceptable` as success

Quietly loosening tolerances is how benchmark tables get gamed.

*Guard:* `ExitFlag::is_success()` is false for `Acceptable`.

### 3.4 Tuning on the reported set

Anyone can win a performance profile by fitting to it.

*Guard:* the development/held-out split in `bench/README.md`. Keep it.

### 3.5 Assuming the reference value is right

**Two of the bugs found while building the baseline were in the test set**, not
the solver:

* `HS16` has a second local minimum at `f ≈ 3.982` that a local method reaches
  from the *published* starting point. The published `0.25` is the global
  optimum, reachable only by luck. Verified by exhaustive search of the
  feasible neighbourhood.
* `TORTURE_DEGENERATE` was accidentally **unbounded** — I forgot to bound `x2`
  — and the solver correctly reported `Unbounded`.

**When a test fails, first establish whether the test is wrong.** Then fix
whichever is actually broken.

---

## Category 4: performance traps

### 4.1 Re-running the symbolic factorization every iteration

The KKT structure is fixed for the whole solve. Analysing it once and refilling
values is the single largest constant-factor win available.

*Guard:* `KktSystem::new` computes every index map once; `assemble` only
writes values.

### 4.2 Allocating inside the iteration

`Factorization::factor` is allocation-free by construction. Keep it that way.

### 4.3 Rebuilding the whole matrix for a regularization retry

Only the diagonal changes. `set_regularization` shifts it in place.

### 4.4 Refactorizing for a second-order correction

Only the right-hand side changes. Reusing the factors is what makes SOC nearly
free, and it is why `rhs_mut`/`solve_scratch` are separate from
`factor_with_correction`.

### 4.5 One GIL crossing per finite-difference probe

`n` round trips where one would do. See `docs/07_API_DESIGN.md` §3 — the
largest available speedup on the Python side, and it is unbuilt.

---

## Category 5: process

### 5.1 Changing a published constant without measuring

The constants in `docs/02` are other people's experiments. Changing one on
intuition is how a solver quietly gets worse. Change it, run the benchmark,
keep the numbers.

### 5.2 Letting the module docs go stale

Every "not implemented" note in this workspace is load-bearing — it is how a
reader knows what to trust. Delete each one in the same commit that implements
the thing.

### 5.3 Testing a component only against itself

A factorization checked against its own output is checked against nothing. The
inertia test uses a **cyclic Jacobi eigensolver that shares no code with the
factorization**. Every numerical component needs an independent oracle:
an analytic solution, a dense reference, or a different algorithm.
