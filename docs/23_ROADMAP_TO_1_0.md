# Roadmap to 1.0

Written September 18, 2026, after five sealed held-out rounds. It replaces the
original milestone roadmap, which was written before the corpus, the oracle
and the sealed-round protocol existed and which the repository no longer
carries. Order is by expected value under the project's priority: the most
robust solver with the least friction, in Python, with `fmincon`'s "handles
everything" feel. Evaluations and wall time are reported honestly and do not
gate an increment. A milestone is done when its gate is measured, not when the
code looks finished.

## 1. What "complete" means

The thesis of the original mission ("beats `fmincon` on plug-and-play
robustness") cannot be proved at the sample sizes a sealed round affords: the
paired family-bootstrap intervals on eleven to thirteen problems have touched
zero in every round. The honest reading of five rounds is *not worse than
`fmincon` at defaults, never lying, more expensive than SciPy SLSQP*. So 1.0
is defined by gates that can be met and measured.

| # | gate | standing (September 18, 2026, after Phases A to C) |
|---|---|---|
| G1 | **Reliability.** Two consecutive fresh sealed rounds of at least 20 problems in at least 10 families each; track-A attainment at least that of both `fmincon` algorithms; zero false certificates; every miss carrying a diagnosis in `notes` | rounds 3 and 5 meet the attainment half at 12-13 problems; no round has had 20 |
| G2 | **Friction.** Every friction-audit problem attained or refused with an actionable message; the exit message names the limit that bound; every quadratic-probe decision in `notes`; no option that does not do what its name says | 14/15 (`bench/results/s7-friction-d`; the one non-attainment is `wrong_gradient`, which mincon refuses with the offending component named, the outcome this gate asks for); items 20c and 20e of `docs/17` closed and the options of `docs/14` re-audited in Phase B; `UseParallel` does what its name says since Phase C |
| G3 | **Instrument.** The benchmark oracle evaluates its first-order test on every record | met since Phase B: the worker passes bound multipliers on bounds-only problems (`docs/22` section 7.16); records scored before that keep their NaN |
| G4 | **Public surface.** Packaged README accurate; the `fmincon` facade and `minimize` frozen; 0.2.0 on PyPI through `publish.yml` and a Trusted Publisher; wheels for Windows, Linux and macOS; CI green on every push | README rewritten and CI repaired September 18; no Trusted Publisher yet; `publish.yml` builds Windows and Linux only |
| G5 | **Scope statement.** Dense coupled problems at n >= 200 either fixed by an ablated increment or documented as the known cost with numbers | documented (README, known gap 1) |

Not required for 1.0 and kept as post-1.0 work: the CUTEst/S2MPJ run,
limited-memory BFGS, AMD ordering, sparse Jacobians from Python, a C ABI.
(The batched Python evaluation protocol was on this list and was built in
Phase C, because parallel probes needed the same hook.)

## 2. The work, in order

The protocol's ordering rule: a candidate is **frozen before** a held-out set
is generated and sealed, and the set exists **before** any claim changes. So
the increments come first, each ablated on the whole development corpus, then
the freeze, then round 6, then the claims and the release.

### Phase A: the repository honest again (done September 18)

Fix the two private intra-doc links and the `map_or_identity` lint that had
turned every CI run red since September 11 (CI's stable toolchain had moved
ahead of the development machine's); gate the portfolio's fixtures in CI
alongside the interior-point member's; rewrite the packaged README to what the
wheel does; record this roadmap.

### Phase B: instrument and cheap friction (done September 18; `docs/22` section 7.16)

Items 1-8 below are done except `record_trace` from Python (optional, not done) and the
mixed exact/finite-difference Jacobian (a linear row's exact Jacobian is used only when every
block supplies one). The friction audit gained `heatflux_design` and stands at 14/15; the
corpus check `bench/results/abl-b1` changed nothing. One finding goes to Phase D as a fourth
candidate: the SQP member stops at a step-tolerance point next to a degenerate vertex with
many active rows and the interior-point member finishes at 4x fmincon-sqp's cost.

1. The oracle's bounds-only blind spot (G3): pass or recover bound multipliers
   so the first-order test is always evaluated. Settle first why 149 of the
   1520 bounds-only records already carry a finite value.
2. The exit message names the binding limit, with a machine-readable field
   (`limit` in `{'time', 'evaluations', 'iterations'}`), in both members.
3. The quadratic probe's decline reason goes into the report's notes.
4. `maxiter` is a per-member limit while `maxfev` and `maxtime` are shared
   across the portfolio: decide the semantics, document them, and make the
   limit exit say which member and which limit.
5. Linear rows given to the `fmincon` facade (`A`, `b`, `Aeq`, `beq`) carry
   their exact Jacobian instead of being finite-differenced.
6. Scheduler accounting: a member that returned an error contributes its
   evaluations to the shared budget; the parallel path either shares the
   budget or says that it does not.
7. Re-audit the exposed-unsupported options of `docs/14`; implement, remove,
   or make each one error with a message.
8. MATLAB option names accepted as aliases on the facade (`MaxIterations`,
   `MaxFunctionEvaluations`, `OptimalityTolerance`, `ConstraintTolerance`,
   `StepTolerance`, `Display`), documented as aliases.

Gate: the eight development gates green, the friction audit at 13/14 with
every record identical, the Python tests extended for each new field.

### Phase C: the expensive-model path (done September 18; `docs/22` section 7.17)

Both increments are built on one hook (`Nlp::objective_batch` and
`constraints_batch`, advertised by `Capabilities::batch`) and measured.
`workers=k` (or `UseParallel=True`) takes the serial solve's iterates to the
last bit on the 56 fixtures under all three algorithms and on the heat-flux
surrogate through a real process pool; with a 0.2 s model the surrogate's
78 batched calls cost `4 ceil(19 / W) + 1` call-times plus 0.5 to 1.3 s of
pool start, and the whole solve goes from 20.9 s to 9.1 s on 8 workers,
bounded by 26 calls that are made one at a time. Nineteen of those are one
event, the SQP member backtracking off a saddle from a first step of
`max |x|`, which goes to Phase D as a fifth candidate. `vectorized=True`
cuts the crossings into Python 7x to 82x on three NumPy models (wall 3.1x at
n = 50, 1.16x at n = 200 where the solver's dense algebra is the cost) and
takes the scalar model's iterates exactly when the model's row arithmetic
does not depend on the number of rows. Without either option no new code is
reached: the friction audit's fifteen records are identical
(`bench/results/s7-friction-c`).

The case a working scientist brings most often is a model that costs seconds
per call (a PDE solve, a simulation) in a few dozen design variables, where
every iteration's finite-difference gradient is `n` serial calls. Two
increments, each behind an option and measured:

1. **Parallel finite-difference probes.** The model advertises that its calls
   may run concurrently (`options={'workers': k}` or a process pool the
   caller supplies); the derivative layer submits a coloring group's probes
   at once. `fmincon`'s `UseParallel`, done for the plug-and-play path.
   Gate: identical iterates to the serial path; wall time on a model with a
   deliberate 0.2 s cost divided by the worker count minus overhead.
2. **Batched evaluation.** A model that accepts a `(k, n)` array and returns
   `k` values crosses the interpreter boundary once per coloring group
   (`docs/07` section 3). Gate: at least five times fewer callbacks on a
   finite-differenced NumPy model; a scalar `lambda x:` model unchanged.

### Phase D: algorithmic candidates, each behind an option, ablated on all 185 problems

Standing (September 18): all five candidates are measured. Candidates 4 and 5 are defaults
(`docs/22` section 7.18), candidate 1 is a default, candidate 2 was written, ablated and falsified,
and candidate 3 ends without a rule (section 7.19). Candidate 4 turned out not to be the QP: at the degenerate vertex the
QP's step has a predicted decrease below the rounding noise of the merit function, no line search
can verify it, and the termination test was looking at the previous step's multipliers (KKT error
13.2). `zero_step='decrease'` adopts the QP's multipliers there and re-tests once. Its first form
also stopped there and cost six corpus problems their certificate (`bench/results/abl-phased45`);
the form that survived changes sixteen corpus records, none in status or attainment, fifteen for
fewer evaluations (`abl-phased45b`). Candidate 5, `saddle_step='linearized'`, caps the first trial
step off a saddle by the distance to the inactive rows; it changes no corpus record
(`abl-phased45c`) and takes the heat-flux design from 104 evaluations to 87. Together:
`heatflux_design` 477 to 87 (`fmincon-sqp` 120), `noisy_simulator` 202 to 89, friction audit 14/15
with every other record identical (`bench/results/s7-friction-d`).

Candidates 1 to 3 (`docs/22` section 7.19, `bench/results/abl-phased123`, one run with its own
default arm). **1:** the `2n` cap bites on two corpus records, DECONV_200 and COVQP_300. The fit
error at the probe's line points falls at every doubling of the band index on the first (0.61,
0.51, 0.15, 0.002) and wanders around its starting value on the second, so
`quadratic_bands='decaying'` doubles the band search's allowance for as long as that error keeps
falling by a quarter. DECONV_200 goes from the time-limit exit of every earlier run to `Optimal` in
6 639 evaluations, 173 / 172 attained, no other record changes, and it is the default. Its cost is
measured too: with a model slower than 5.5 ms a call the band DECONV_200 needs does not fit the
build's half of a 60 s clock, the probe declines at its deadline and the returned point is 2.5x to
4.7x further from the optimum than before (neither attains). **2:** the rule proposed below has no
corpus record to help: three of 185 interior-point traces end with `mu` fixed above its floor and
none would gain from a cut. The common pattern is the adaptive rule's round trip to the floor (110
of 185). A lower bound tied to the infeasibility (IPOPT's safeguard) was written and ablated:
nothing on the portfolio at either factor, and on the member three certificates lost, two gained
and a cost of 1.18 [1.14, 1.29]. It is not in the tree. **3:** the cost on dense-objective
families is the objective's own dense build, which `quadratic_rows` only allows to happen (the rows
are 240 of the 3 740 evaluations COVQP_120 loses); the rule proposed fires after that cost is paid,
gain and loss fall within one family on either side of `n / 2` quasi-Newton iterations, and no
rule is written. The low-rank-plus-diagonal build buys evaluations and no attainment and moves to
the post-1.0 list.

**Frozen for round 6** (September 18): the tree of the commit that records section 7.19, wheel
`d57902fa79492531` (`quadratic_bands='decaying'`, `zero_step='decrease'`,
`saddle_step='linearized'` on top of the round-5 candidate and Phases B and C).

1. The quadratic probe's band budget (`crates/mincon/src/quadratic.rs`, the
   `2n` cap when the dense build is unaffordable; `docs/22` section 7.15).
   Measure first at the harness's per-evaluation rate and at a slower one;
   falsifier: any attainment loss; DECONV_60 and ELLIPSOID_500 are the records
   to watch.
2. The frozen barrier parameter (`docs/22` section 7.15): force a reduction
   when the monotone schedule has left `mu` unchanged while the subproblem
   error has plateaued above its gate, or damp the adaptive phase's
   oscillation. Trace study on DECONV_200, HS100, HS38, HS32, HS63 first.
3. `quadratic_rows`'s residual cost on dense-objective families, and the
   low-rank-plus-diagonal Hessian build (`docs/22` sections 7.12-7.13).
4. The SQP member on a degenerate vertex with many active rows
   (`heatflux_design`: a `StepTolerance` stop 2.1e-5 above the optimum, then
   the interior-point member at 4x `fmincon-sqp`'s cost; `docs/22` section
   7.16), and whether a step-tolerance point that close is usable to the
   portfolio.
5. The SQP member's step off a saddle point (`docs/22` section 7.17): the
   backtracking starts at `t = max |x|` and pays two evaluations a trial, 19
   evaluations on the heat-flux surrogate for a step of 5.9e-3, none of
   which `workers` can overlap. Size the first trial from the probe's own
   curvature; ask whether the restoration step's second evaluation is
   needed before the merit test has a chance of passing.

Whatever survives is the round-6 candidate. Freeze it, build the wheel, record
its hash.

### Phase E: round 6, designed to discriminate

Round 5's set separated no solver on five of its seven families and its one
discriminating family was a friction problem with the seed changed
(`docs/17` item 21). Rules for `heldout6.py`:

* at least 20 problems in at least 10 families;
* no family a variant of a development or friction problem, the nearest
  relative named in the docstring;
* families where the solvers are expected to differ, with the reason written
  down: bounds-active ill-conditioned least squares with a different
  structure from deconv; dense coupled n = 100-300 with a nonlinear row;
  equality-heavy with rank loss at the solution; far-infeasible starts with
  nonlinear equalities; models undefined outside a region; a banded sparse
  family at n = 500-1000; unit mismatch with a non-zero start; noisy
  derivatives; a nonconvex family whose start's nearest local minimum is not
  the global one; design problems with units and best-known values;
* references exact or verified at stationarity 1e-10, and the stored point
  reproducing the target to 1e-6 under the corpus model (the generation
  assertion tightened from 5e-3); CORRUGATED_BULKHEAD's point recomputed
  before `final5` is reused for anything;
* protocol fairness decided before sealing (the MATLAB worker drops the
  evaluation budget; SciPy runs uncapped) and disclosed as protocol v3 if it
  changes.

Then: MATLAB equivalence, targets v7, seal with hashes, the run on both tracks
with all seven solvers, timing with three repeats, the adversarial pass,
`docs/17` items, the README status block. Negative results are results.

### Phase F: release 0.2.0

Version 0.2.0 in the workspace (the defaults changed since 0.1.0: the
quadratic probe and its structured build and rows, `scale_variables='auto'`,
the KKT pivot signs, the SQP member first at n <= 20, the certificate guards,
warm start, callback, `hess=`, `method=`); a changelog; macOS in the publish
matrix; the Trusted Publisher; a fresh-environment install check; tag
`v0.2.0`; 0.1.0 left un-yanked. After round 6, so the status block shipped on
PyPI is backed by a sealed run of the tree being shipped.

### Post-1.0

A low-rank-plus-diagonal Hessian build for the quadratic probe (covariance
and factor models: about `2n + k (n - 1)` evaluations instead of
`n (n + 3) / 2`; cost only, `docs/22` section 7.19); limited-memory BFGS in
compact form (the route past n of about 1000 with
dense models); sparse Jacobians and `jac_sparsity` from Python; AMD ordering
(only for n > 1000); CUTEst through S2MPJ, the only external validation, when
authorised; a C ABI.

## 3. Rules that do not change

Success is decided by the harness, never by the solver. The portfolio's cost is
the cost of every member. `Acceptable` is not success. Every increment goes
behind an option and is ablated on the whole corpus before it becomes a
default. Claims change only from a sealed held-out round, and a set that has
been examined cannot support a fresh claim. A number produced by the measuring
instrument rather than the solver is not evidence about the solver.
