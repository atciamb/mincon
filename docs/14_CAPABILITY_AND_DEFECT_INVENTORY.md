# Capability and defect inventory at `8ec00bd`

Built from the source and tests on September 7, 2026, before any solver change.
Four states are distinguished: **measured** (implemented and covered by an
independent check), **unqualified** (implemented, but no independent measure
of its effect), **exposed-unsupported** (an option or API exists but does not
do what its name says) and **absent**.

## Capability matrix

| Area | State | Evidence / location |
|---|---|---|
| Primal-dual interior point, filter line search, SOC, fraction-to-boundary, bound-multiplier reset | measured | `crates/mincon-ip/src/solver.rs`; 54-fixture gate, `bench/results/b0-baseline` |
| Inertia correction (Wächter–Biegler IC) | measured on fixtures | `kkt.rs::factor_with_correction` |
| Monotone barrier update | measured | `solver.rs` (`mu` update block) |
| `BarrierUpdate::Adaptive` / `AdaptiveThenMonotone` (the default) | exposed-unsupported: only the monotone rule is wired; the enum value is accepted silently | `solver.rs` uses `kappa_mu`/`theta_mu` only |
| Watchdog (`Options::watchdog`) | exposed-unsupported: never read by the solver | grep shows no consumer |
| `RegularizationMode::InertiaFree` / `Hybrid` | exposed-unsupported: both take the certified-inertia path | `options.rs` doc comments say so |
| `HessianMode::LimitedMemoryBfgs`, `FiniteDifference` | exposed-unsupported: fall back to dense BFGS with a note | `solver.rs::new` |
| Dense damped BFGS | measured on fixtures | `bfgs.rs` |
| Soft + reduced-elastic restoration, `LocallyInfeasible` exit | measured on fixtures | `solver/restoration.rs`, `docs/12` |
| Gradient-based scaling | unqualified as an ablation (portfolio member `ip-unscaled` exists) | `solver.rs::compute_scaling` |
| Sparse LDLᵀ with dynamic regularization, inertia, refinement | measured by unit tests; no randomized indefinite stress test | `mincon-linalg/src/ldlt.rs` |
| `Ordering::Amd` | exposed-unsupported: falls back to RCM | `ordering.rs:53` |
| `LinearSolverKind::DenseLblt` | absent (option accepted) | no dense LBLᵀ implementation |
| Forward/central/adaptive finite differences, bound-aware retreat, boundary stencils | measured (HS33/HS35 analytic KKT tests) | `mincon-diff/src/fd.rs` |
| Jacobian sparsity detection (2 base points, one probe per variable) | unqualified: a hypothesis with no runtime re-verification or fallback | `detect.rs` |
| CPR graph coloring | measured by unit tests | `coloring.rs` |
| Derivative checker | **defective** (see D1) | `check.rs` |
| Portfolio (3 IP members) | unqualified; resource accounting defective (D5) | `crates/mincon/src/portfolio.rs` |
| SQP, QP subsolver | absent (`is_available()` returns false; `Algorithm::Sqp` errors) | `mincon-sqp`, `mincon-qp` |
| Python `minimize` / `fmincon` façade, multipliers grouped MATLAB-style | measured (18 tests, PyPI wheel) | `mincon-py` |
| Sparse Jacobian / `jac_sparsity` from Python; constraint Jacobian callback from Python | absent (only `jac` for the objective) | `mincon-py/src/lib.rs` |
| Hard wall-time / evaluation budget | unqualified: checked once per iteration only; FD probes and line-search trials can overshoot by O(n) evaluations | `solver.rs` termination block |

## Defects confirmed in the source (to be fixed in S3)

D1. **Derivative checker can pass with no evidence.** `CheckReport::passed()`
is `max_relative <= tolerance`. (a) If every `gradient()`/`jacobian()` call
fails, or every objective evaluation fails, no comparison is made and the
check passes. (b) A NaN analytic entry produces `relative = NaN`, and
`f64::max` discards NaN, so `max_relative` stays small and the check passes.
(c) `points_checked` may be zero and still pass. A test
(`a_model_with_no_analytic_derivatives_trivially_passes`) currently encodes
the wrong semantics.

D2. **Python `fmincon` façade evaluates `nonlcon` twice per constraint
evaluation** (`component(0)` and `component(1)` each call the user function),
doubling nonlinear-constraint cost; the `minimize` path also evaluates every
constraint block separately from the objective, so a joint model is called at
least 2–3 times per point.

D3. **Benchmark harness (`bench/runner.py`, `bench/matlab/fmincon_baseline.m`)**:
equality rows classified by `np.isclose(cl, cu)` in Python but `cl == cu` in
MATLAB; constraint calls not counted and each eq/lo/hi group re-evaluates the
full constraint vector; IPOPT receives analytic derivatives, others do not;
SciPy ignores `maxfev`/`maxtime`; MATLAB sets `MaxFunctionEvaluations=1e5`,
`MaxIterations=3000` while the header says defaults; the MATLAB deadline is a
cooperative `OutputFcn` with `persistent t0`; records omit the returned point
and any stationarity evidence; `json.dumps` emits non-standard `Infinity`;
targets are "best objective seen among the contestants", so they move as
contestants are added. The harness is retained as a scaffold only; S1 replaces
it.

D4. **Termination is stricter than documented and platform-sensitive.**
`Optimal` requires the scaled `E_0 <= 1e-8` *and* the unscaled stationarity
∞-norm `<= 1e-8`. HS5 flips between `Optimal` (Linux) and `Acceptable`
(Windows) at identical objective values. HS11/HS31/HS37/HS43 end in
`NumericalFailure` at points with relative objective error `<= 5e-8`.

D5. **Portfolio resources.** With `threads > 1` every member runs to
completion on its own thread regardless of the `threads` value (three members
spawn even for `threads = 2`); there is no cancellation once one member has
converged; `total_f_evals` sums only members that returned `Ok`, so a member
that errored contributes zero; from Python every callback needs the GIL, so
parallel members serialize and the default portfolio costs roughly three
sequential solves in wall time (to be measured in S2).

D6. **Multiplier update.** `lambda += alpha_z * d_lambda` uses the dual step
length; Wächter–Biegler use the primal `alpha`. There is no least-squares
multiplier initialization although `BarrierParams::lambda_init_max` exists.

D7. **Sparsity detection is a hypothesis.** Two base points, entries that are
zero at both are treated as structural zeros for the whole run; no periodic
check, no fallback.

These items are the input to the S2 failure atlas and the S3 repairs; none is
fixed in this commit.

## Added later

D8. **Unbounded objective reported as `Optimal`** (UNBOUNDED_PAR): fixed in
C2 by the far-from-start rule (`docs/16_FAILURE_ATLAS.md`, cluster 5).

D9. **Termination judged in the scaled problem only, with an objective scale
fixed from the gradient at x₀** (found September 11, 2026 by the basin study,
`bench/results/r3-basins/`). The gradient-based scaling divides the objective
by `‖∇f(x₀)‖`-derived factors with no floor; on BADSTART_DISC from
(1000, −1000) the factor is 2.5e-10 and the scaled KKT test `E_0 <= 1e-6`
accepts an unscaled stationarity residual of 87 (exact gradient (87, −50),
constraint slack 3.4e-3, reported multiplier 2165) — a false `Optimal` at
f = 6.375 where the published minimum is 0.0457. The milder form
(QUADSPHERE2_300, factor 0.05) stops at a 1.8e-3 objective gap. With
`scaling = none` both attain. Fix candidates: an unscaled relative
stationarity guard on the termination test, and a drift-triggered rescale
when the current gradient norm has left the range the factor was chosen for.
Fixed in I0 (`bench/results/abl-i0`): unscaled relative stationarity guard
plus drift-triggered rescale; regression test in `mincon-ip`.

D10. **Restoration does not exit at a stationary infeasible point on a
bound** (found September 11, 2026 by the budget study,
`bench/results/r4-budget/`). INFEASIBLE_NL (`x₁ ≥ 2`, `x₁² + x₂² ≤ 1`): from
iteration 10 the restoration phase sits at x = (2, 0), violation 3.0, step
norm 1e-16, for 410 further iterations, then `MaxReached`; every portfolio
member repeats it (9784 evaluations in total). The infeasibility minimizer
lies on the bound `x₁ = 2`, and the restoration's stationarity test does not
account for it. Fixed in I0: projected-gradient stationarity test in
`robust_restoration`; `LocallyInfeasible` after 6 iterations (105 evaluations
across the three portfolio members); regression test in `mincon-ip`.


## Re-audit of the exposed-unsupported rows, September 18, 2026

Phase B of `docs/23`. Each row of the matrix above that promised something the code did not do
was either implemented since, removed, or left with its documentation saying exactly what it
does; nothing silently falls back any more.

| row | disposition |
|---|---|
| `BarrierUpdate::Adaptive` / `AdaptiveThenMonotone` | implemented (round 2); `AdaptiveThenMonotone` is the measured default |
| `Options::watchdog` | removed (never read) |
| `RegularizationMode::InertiaFree` / `Hybrid` | kept: the modes differ in `kkt.rs` (a singularity check, dual regularisation on retry) and every mode requires certified inertia, which their docs and the interior-point module docs now say |
| `HessianMode::LimitedMemoryBfgs`, `FiniteDifference`, `lbfgs_history` | removed (both ran dense BFGS with a note); `Auto` documented as exact-if-supplied else dense BFGS |
| `Ordering::Amd` | removed (ran RCM); the module docs keep the AMD plan for a new variant |
| `LinearSolverKind`, `Options::linear_solver` | removed (never read) |
| `Algorithm::Slqp` | removed (ran SQP) |
| `ScalingMode::Equilibration`, `User` | removed (ran gradient scaling); the Python `'equilibration'` value is an error |
| `Options::display`, `Display` | removed (never read; Python streams through the callback) |
| `Options::restoration` | removed (never read; restoration always runs) |
| SQP, QP subsolver | implemented (rounds 3-5) |
| Sparse Jacobian / `jac_sparsity` from Python | still absent, and still says so (`docs/23` post-1.0) |
| Derivative checker (D1) | fixed (C1) |
| Portfolio resource accounting (D5) | closed here: a counting wrapper charges every member, including one that errors, and the parallel path shares budgets across chunks |
