# Benchmark protocol v2 (preregistered, September 7, 2026)

This replaces the scaffold in `bench/runner.py` (defects listed in
`docs/14_CAPABILITY_AND_DEFECT_INVENTORY.md`, D3). Everything below was fixed
before the first matched run on the development split. Changes after that
point are recorded in the "Revisions" section with dates and reasons; none may
change the held-out (`final`) protocol after any `final` record exists.

## 1. Canonical problem and single-source definitions

Every instance is `min f(x)` s.t. `cl <= c(x) <= cu`, `xl <= x <= xu`, defined
once as SymPy expressions in `bench/corpus/{hs,adversarial,structured}.py`.
`bench/corpus/generate.py` derives from that single definition: NumPy
callables with exact gradients, Jacobians and Lagrangian Hessians
(`spec.NumpyProblem`), MATLAB function files (`bench/corpus/matlab/*.m`), the
manifest with dimensions, row kinds, checksums and split, and probe points.
Row kinds come from exact bound comparison (`cl == cu` is an equality; no
`isclose`). Fixed variables (`xl == xu`) are kept as fixed variables.
MATLAB translation: upper rows `c - cu <= 0`, lower rows `cl - c <= 0`,
equalities `c - cl = 0`; a ranged row contributes one upper and one lower row;
the row map is stored in every MATLAB record so multipliers map back.

Equivalence gate: `equivalence.py` compares `f, grad f, c, J` at `x0` and four
deterministic in-box probes between MATLAB and NumPy to `1e-10` relative. As
of the freeze: 133 problems, 2,660 quantities, 0 mismatches. The 44 problems
shared with the Rust `mincon-testset` crate also agree with that independent
transcription at 176 probe points (`crates/mincon-testset/examples/dump_probes.rs`).

## 2. Corpus and split (frozen; `bench/corpus/split.py`)

| family | instances | reference | split |
|---|---:|---|---|
| hs (Hock–Schittkowski) | 89 | published optimum; 47 with a checked minimizer | 44 exposed -> dev; rest hashed 40/30/30 dev/val/final |
| adversarial | 18 (3 diagnostic) | closed form, derived before any run | hashed 50/25/25; diagnostic track |
| chainrosen (n = 10, 50, 200; box and +equality) | 6 | f* = 0 at x = 1 | dev |
| lqtraj (double integrator, n = 30/150/600, fixed terminal variables) | 3 | direct KKT solve at generation | dev |
| expfit (bounded exponential fitting) | 3 | f* = 0 (clean); best-known (noisy) | dev |
| quadsphere (n = 10/100/1000, cond 1e3) | 3 | closed form | validation |
| engineering (spring, truss, cantilever / vessel, beam, reducer) | 6 | published best-known | 3 val / 3 final |
| ellipsoid projection (n = 5/50/500) | 3 | secular equation at generation | final |
| portfolio (convex QP, n = 20/100) | 2 | best-known from the reference run | final |

Counts: dev 82, validation 24, final 24, diagnostic 3. The final set is small;
the superiority contract in §6 therefore also requires the validation set to
agree in sign, and the plan is to add further held-back families before S6.
The final split is never run until S6. If it is ever used for diagnosis it is
relabelled development and a new final set must be created.

## 3. Targets and outcome definitions (`bench/corpus/targets_v1.json`)

* **Valid point**: finite, correct length.
* **Feasible**: independent maximum violation of bounds and rows, in model
  units, `<= 1e-6` (fmincon's and mincon's default constraint tolerance).
* **Target attained** (primary): feasible and `(f - f*) / max(1, |f*|) <= 1e-4`,
  where `f*` is the frozen target. `1e-4` is chosen so that fmincon's default
  `OptimalityTolerance = 1e-6` and mincon's defaults are both compatible (the
  parity suite showed fmincon-interior-point returning `|f - f*| = 2e-6` at
  defaults). Accuracy profiles at `1e-2 .. 1e-8` are reported as well.
* **First-order convergence**: `oracle.assess` recomputes
  `r = grad f + J^T lam - zL + zU` from exact derivatives with the solver's
  multipliers (`kkt_first_order`, requires `|r|_inf <= 1e-6`, correct signs and
  complementarity) and independently with non-negative least-squares
  multipliers on the active set (`kkt_first_order_recovered`). Both are
  reported; neither is the primary metric because SciPy returns no multipliers
  and fmincon's default accuracy does not guarantee `1e-6` stationarity.
* Targets are versioned. `targets_v1` holds closed-form and published values.
  Best-known targets (portfolio, noisy expfit) are set by `targets.py update`
  from independently KKT-verified feasible points of the reference run; a
  verified point that beats a frozen target by more than the tolerance
  triggers a logged revision and a rescoring of **all** archived records under
  the new version. The scorer never uses "best objective among contestants".
* Diagnostic problems (infeasible, unbounded) are never in the attainment
  denominator; their correct outcome is a diagnosis, reported separately.
* A local solver converging to a different local minimum counts as
  not-attained but is not a false-success claim; `kkt_*` shows whether it is a
  stationary point.

## 4. Tracks

| track | derivatives | options | solvers |
|---|---|---|---|
| A default experience (primary) | none supplied | algorithm defaults; only `Display='off'` and `Algorithm` for fmincon; `threads=1` for mincon; external caps `maxtime=60 s` (cooperative for MATLAB; hard kill by the supervisor at `3 x 60 + 30 s`), `maxfev=100000` for mincon (fmincon keeps its own default evaluation limit, which is part of "defaults") | mincon (auto), mincon-ip, fmincon-interior-point, fmincon-sqp, scipy-slsqp |
| C exact derivatives | exact `grad f`; exact `J` where the API accepts it (fmincon yes; mincon's Python API objective gradient only until Workstream 2) | as A | same |
| D structured large scale, E expensive/noisy | to be defined before their first run | | |

Track B (common tolerances) is deferred until A and C are measured.

## 5. Counting, timing and records

* Counts are taken at the model boundary by a one-point cache
  (`harness/model.py`, `worker_matlab.m`): one objective and one
  constraint-vector evaluation per distinct point. Raw callback crossings are
  recorded too (`*_calls`), exposing API overhead such as the mincon façade's
  double `nonlcon` call (D2). Cost metric for profiles: `f_model + c_model`.
* Time: solver wall time around the solve call, callback time inside the
  model, and problem build time, separately. MATLAB warm-up is excluded
  (worker processes solve many problems; the first problem of each worker is
  still reported and flagged by order). Primary comparison runs single-threaded
  (`maxNumCompThreads(1)`, mincon `threads=1`) on the same Windows laptop,
  Balanced power plan, no other load. Multi-core is a separate track.
* Every record (`harness/schema.py`, `mincon-bench-record/1`) carries the
  returned point, multipliers, native status and message, counts, times,
  options, host, split, checksum, run id and repeat index. Timeouts and
  crashes are schema-complete records with `outcome != ok` and no invented
  point. JSON is standards-compliant (non-finite floats are strings).
* Runs are resumable and checkpointed per record; problem order is a fixed
  seed shuffle per solver, reversed on odd repeats.

## 6. Superiority contract (engineering acceptance targets, not theorems)

Primary comparison: mincon default (`auto`, `threads=1`) vs
`fmincon-interior-point` defaults, track A, on the `final` split, then
confirmed in sign on `validation`.

1. Zero invalid returned-success points in the correctness suite
   (`reported_success` with infeasible or non-finite point) across all splits.
2. Attainment difference `>= +5` percentage points with the family-clustered
   paired bootstrap 95% interval excluding zero. If fmincon attains `>= 95%`
   on the final split, the alternative pre-declared criterion is a `>= 50%`
   reduction in fmincon's failures with the interval excluding zero.
3. On commonly attained problems, geometric-mean ratio of
   `f_model + c_model` (mincon/fmincon) `<= 1.0` with the upper 95% bound
   `< 1.10`; wall-time ratio reported with the same bounds; tail (90th
   percentile) ratios reported.
4. Secondary baselines: fmincon-sqp and scipy-slsqp; if mincon only beats
   interior-point, the claim says so.
5. Gains must appear in at least three families, and the expensive-callback
   track E must not show a callback-count regression that the kernel speed
   hides.

Failing any item narrows the claim; it never changes denominators, tolerances
or budgets after the fact.

## Revisions

(none)
