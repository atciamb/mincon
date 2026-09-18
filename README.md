# mincon

A nonlinear constrained optimizer in Rust, aimed at what MATLAB's `fmincon`
does well — solving the problem a working scientist actually has, without being
told how — with Rust computations and Python wheels built using maturin.
Experimental version 0.1.0 is [available on PyPI](https://pypi.org/project/mincon/0.1.0/).
Install with `pip install mincon`. Prebuilt wheels support Windows and Linux x86_64.
Source and contributions: [atciamb/mincon on GitHub](https://github.com/atciamb/mincon).

```
minimize    f(x)
subject to  c_L <= c(x) <= c_U
            x_L <=   x  <= x_U
```

```python
from mincon import fmincon

r = fmincon(lambda x: ((x - 1)**2).sum(), [0., 0.],
            nonlcon=lambda x: ([x.sum() - 1], []))
print(r.x, r.fun, r.success)  # approximately [0.5, 0.5], 0.5, True
```

```rust
use mincon::{minimize, Options, Problem};

let p = Problem::new(2, |x| 100.0*(x[1]-x[0]*x[0]).powi(2) + (1.0-x[0]).powi(2))
    .start_at(&[-1.2, 1.0])
    .inequality(1, |x, c| c[0] = x[0]*x[0] + x[1]*x[1] - 1.0);
let r = minimize(&p, &Options::default())?;
```

> **Status: experimental, measured.** Matched against the installed MATLAB
> R2025b `fmincon` on a 185-problem single-source corpus with an independent
> KKT oracle (`docs/15_BENCHMARK_PROTOCOL_V2.md`), five rounds of held-out
> qualification. At defaults mincon attained at least as many targets as
> `fmincon-interior-point` on every split (held-out rounds: 19 vs 18 of 22,
> 14 vs 12 of 16, 12 vs 10 of 12) and as many as `fmincon-sqp` on the latest
> round (12 vs 12); on that round it used 0.81× [0.53, 1.05] the model
> evaluations of `fmincon-interior-point` and 1.13× [0.97, 1.38] those of
> `fmincon-sqp`, and returned **13× faster** than the former and 3× faster
> than the latter in wall-clock time (same laptop, single thread), except on
> two problems with n ≥ 120 where its dense BFGS iteration count made it
> slower. The portfolio now has an SQP member (ℓ1 merit, elastic QP, damped
> BFGS, second-order corrections, a second-order check that walks off saddle
> points where `fmincon-sqp` and SLSQP stop) that runs first on small
> problems: on the development corpus it is 0.78× [0.70, 0.86] the
> interior-point member's evaluations. **The preregistered superiority
> contract's evaluation interval is still not met at these sample sizes**;
> see `bench/results/s6v3-final3/README.md`, `docs/17_CLAIM_AUDIT.md` and
> `docs/16_FAILURE_ATLAS.md` for exactly where fmincon wins. Round 4
> (September 12, 2026, eleven sealed problems with dense coupled Hessians,
> `bench/results/s6v4-final4`): reliability against fmincon holds (9/11 vs
> 5/11 with finite differences, 11/11 vs 10/11 with exact derivatives) at
> 0.85× [0.47, 1.94] and 0.53× [0.25, 1.10] of fmincon-interior-point's
> evaluations, but SciPy SLSQP attains all eleven at fewer evaluations than
> mincon, and on these model-bound problems mincon's wall time is 1.4–1.9×
> fmincon's. The curvature-tracking rebuild (`docs/21`) that cut the
> development corpus to 0.94× cost 1.24× on the sealed set and is off by
> default (opt-in `bfgs_rescale`). Round 5 (September 13, 2026, thirteen
> sealed problems in seven families, `bench/results/s6v5-final5`): **no
> attainment difference is resolved and mincon is measurably more expensive
> than SciPy SLSQP.** With finite differences and minimal inputs mincon
> attains 12/13, behind `fmincon-sqp`, SLSQP and mincon's own SQP member
> (13/13 each) and ahead of `fmincon-interior-point` (10/13) and
> trust-constr (11/13); with exact derivatives 13/13. Every paired interval
> touches zero, and the lead over `fmincon-interior-point` is that solver's
> own 3000-evaluation factory default firing after 3 % of the offered
> budget. mincon claimed no false certificate; the run's one false
> certificate is trust-constr's. Evaluations: 1.52× [1.13, 2.04] of SLSQP
> with finite differences and 1.85× [1.47, 2.27] with exact derivatives.
> Wall time on this cheap-model set is 0.28× [0.11, 0.78] `fmincon-sqp` and
> 0.05× [0.027, 0.107] `fmincon-interior-point`, on top of a model-language
> confound of up to 8.8× per evaluation. The round's quadratic-program probe
> changed no attainment outcome on the sealed set, and neither did the
> quadratic-row default that postdates the run
> (`bench/results/s6v5-final5-rows`, a disclosed re-run). mincon's one
> miss is an interior-point stall on a 200-variable deconvolution that its
> own SQP member solves — the portfolio spent the whole 60 s on the wrong
> member.
> Earlier in the round a friction audit of fourteen realistic problems
> (`bench/results/s7-friction`) went from 12/14 to 13/14 as D11 and D12 were
> fixed (`docs/22_ROUND5_ROBUSTNESS_PLAN.md`); a fifteenth problem, a
> heat-flux surface design with 1384 linear rows, was added on September 18
> (`bench/results/s7-friction-b`: 14/15, every solver attains it, mincon at
> 4× fmincon-sqp's evaluations because its SQP member stops short of a
> degenerate vertex and the interior-point member finishes). Phase C of the
> roadmap (September 18, `docs/22` section 7.17) added parallel and batched
> finite-difference probes behind `workers=` and `vectorized=`; without them
> the audit's fifteen records are identical (`bench/results/s7-friction-c`).
> Development
> gate: Rust tests, 56/56 fixtures with independent checks for the portfolio
> and each member (SQP alone: 55/56, HS13 within 4e-4), 50 Python tests.

`fmincon` accepts optional `A, b, Aeq, beq, lb, ub, nonlcon`; its nonlinear
callback returns `(c, ceq)` with `c <= 0`. The existing `minimize` interface
uses SciPy's `ineq >= 0` convention. Both have automatic defaults and return
an `OptimizeResult`. See the [Python guide](crates/mincon-py/README.md) and
[release procedure](docs/13_RELEASE.md). A fresh PyPI installation passes all 18 Python tests.

---

## What works today

* **Interior point** — primal-dual, filter line search, Wächter–Biegler
  Algorithm IC inertia correction, second-order corrections,
  fraction-to-boundary, bound-multiplier resets, scaled `E_mu` termination.
* **Feasibility restoration** — guarded soft steps, then analytically reduced
  elastic minimization with Gauss-Newton curvature and filter re-entry; a
  stationary infeasible point on a bound is diagnosed in a handful of
  iterations.
* **SQP** — ℓ1 merit function with a single penalty shared by the elastic
  QP (so every step is a descent direction even when the linearization is
  inconsistent), damped BFGS with a curvature rescale of the unit start,
  Goldfarb–Idnani dual active-set QP with a hinted warm start, up to four
  second-order corrections, an adaptive step bound, an opt-in
  curvature-tracking rebuild of the model (`bfgs_rescale`, for separable
  large problems), and a second-order probe at termination that leaves
  saddle points. Runs first in the portfolio for n ≤ 20
  (`docs/20_SQP_MATHEMATICS.md`, `docs/21_LARGE_N_CURVATURE_PLAN.md`).
* **Termination you can trust** — the scaled KKT test is guarded by the
  stationarity relative to the gradient in your units (the same measure the
  benchmark's independent oracle uses), budget exits say which limit bound
  (`res.limit`: iterations, evaluations or time) and whether the solve was
  still progressing, and degenerate multipliers are reported as such. An
  iteration cap you set is a limit on the whole solve, shared across the
  portfolio's members like the evaluation and time budgets.
* **Sparse `LDL^T`** written from scratch: dynamic regularization, inertia
  certified by Sylvester's law, element-growth detection, iterative refinement.
  **No HSL, no MUMPS, no Fortran** — which is what makes the wheel possible.
* **Derivatives** — bounds-aware finite differences with realized retreat
  displacements and second-order inward boundary stencils, graph coloring,
  sparsity detection, a multi-point derivative checker that cannot pass
  without evidence; exact objective gradients and constraint Jacobians from
  Python (`jac` in SciPy-style constraint dicts, `nonlcon_jac` in `fmincon`).
  Supplied derivatives are checked along one direction at the start with two
  evaluations: a gross disagreement stops the solve naming the component, a
  mild one is a note (zero false alarms on the 172-problem corpus with exact
  derivatives). `hess=` (the Lagrangian Hessian) gives both members Newton
  convergence: on HS71 the SQP member takes 7 iterations and the interior-point
  member 10, against 5 and 10 with their quasi-Newton models.
* **Expensive and vectorised models** — `workers=k` (or `UseParallel=True`
  on the `fmincon` facade) evaluates every gradient's finite-difference
  probes on `k` worker processes, or through a map-like callable, and takes
  the serial solve's iterates to the last bit (verified on the 56 fixtures
  under each algorithm and through a real process pool); with a 0.2 s model a
  19-variable design solve goes from 20.9 s to 9.1 s on 8 workers, the
  gradients at a sixth of their serial time and the rest bounded by the calls
  a solver makes one after another. `vectorized=True` hands a NumPy model a
  `(k, n)` array once per gradient: 7x to 82x fewer crossings into Python,
  3x in wall time at n = 50. Underneath both is one hook on the `Nlp` trait,
  `objective_batch` / `constraints_batch` (`docs/22` section 7.17).
* **Seeing and steering a run** — `callback=` after every iteration with the
  trace row (return `True` to stop), `disp=True` streams the iteration table,
  `method=` on the `fmincon` facade, `mincon.multistart` for several starts,
  and `warm_start=res` resumes from a previous result's multipliers and
  quasi-Newton model (`res.hess_approx`): resumed at two thirds of its
  evaluations, a solve lands within 5 % of the uninterrupted count on four of
  the ten friction problems where a cold restart costs 1.1x to 1.5x more
  (`bench/results/i6-warmstart`).
* **Quadratic programs are recognised** — by default (`quadratic_probe=True`)
  seven evaluations along two lines at the start tell whether the objective
  is quadratic and every constraint row linear; if so the constant Hessian
  is built by differencing, checked for convexity, and the SQP member takes
  Newton steps with it (a 50-variable bounded least-squares deconvolution:
  8595 evaluations and 166 iterations become 1232 and 2). The Hessian is
  built by structure (`quadratic_build='structured'`): the diagonal first,
  then one band at a time until the model reproduces the probe's line
  points, so a diagonal Hessian costs 2n evaluations, a tridiagonal one 3n
  and a dense one n(n+3)/2. On the corpus one more problem is attained (a
  500-variable obstacle problem that every earlier run left at the budget)
  and the evaluation cost is 0.93 [0.62, 1.04] of the quasi-Newton path
  (`bench/results/abl-i5`, `abl-i5-build`); dense quadratics the
  quasi-Newton path solved in a few iterations still pay 1.1x to 1.7x.
* **Variable scaling from the start** — by default (`scale_variables='auto'`)
  the solve runs in variables divided by their starting magnitudes when those
  span a factor of 1e4 (fmincon's `TypicalX` done for you); on the corpus it
  fires on one problem (HS117, attained either way at 8x the evaluations) and
  changes nothing else (`bench/results/abl-i8`).
* **Gradient-based scaling**, on by default.
* **Adaptive barrier update** (LOQO centrality rule) with a bounded fallback
  to the monotone schedule; **algorithm portfolio** with deterministic
  ranking, sequential early exit for models that cannot evaluate concurrently
  (every Python model) and shared member budgets.
* **Python bindings** — `abi3` wheel, SciPy-compatible `minimize`.
* **Test set** — 56 fixtures, 45 Hock–Schittkowski problems plus 11 torture
  problems, with independent feasibility/reference-value checks and explicit
  failure fixtures; run through the interior-point member alone (the CI gate)
  and through the default portfolio and the SQP member.
* **Benchmark corpus and harness** — 133 problems defined once in SymPy
  (89 Hock–Schittkowski, adversarial closed-form cases, scalable structured
  families, engineering designs), generated NumPy and MATLAB models that agree
  at 2,660 probe quantities, an independent KKT oracle, supervised MATLAB and
  Python workers with hard timeouts, and paired family-bootstrap analysis
  (`bench/harness/README.md`).

### Known gaps, in priority order (measured, see `docs/16_FAILURE_ATLAS.md`)

1. **Dense BFGS at n ≳ 100 on non-separable problems** (ELLIPSOID2_200
   ~400 iterations; CHAINROSEN_BOX_200 not solved within 100 000
   evaluations): hundreds of iterations on long curved valleys for every
   BFGS-based solver. The curvature-tracking rebuild (C7,
   `docs/21_LARGE_N_CURVATURE_PLAN.md`) fixed the *separable* large-n cost
   on development material (MAXENT_200 from 80 000 to 12 400 objective
   evaluations) but cost 1.24× on the sealed round-4 set, whose large
   problems are dense and coupled, so it is **off by default** and available
   as `bfgs_rescale=10` for separable problems. The default large-n
   behaviour is the round-3 one. Dense convex QPs (round 4's covqp and
   obstacle families) are where the SQP member trails SLSQP by 3–5×
   iterations; that QP step is the next candidate (`docs/21` §7). Pass exact
   derivatives where you have them: with them mincon attains every round-4
   problem.
2. **Evaluations against `fmincon-sqp`** are even, not better (1.13×
   [0.97, 1.38] on the last held-out round; 1.05× on n ≤ 50). `fmincon-sqp`'s
   `100·n` evaluation cap stops it inside the target tolerance on the large
   problems where mincon keeps iterating to its optimality tolerance.
3. **Adaptive barrier stalls** on a few problems (HS63: 39 vs 8 iterations
   monotone) before the monotone fallback fires; the SQP member now runs
   first on those sizes, so the barrier rule matters less than it did.
4. **Finite-difference accuracy on sensitive models**: the solver stops at
   the estimated derivative accuracy (HS62: 100 evaluations, was 504) but the
   certificate is only as good as the derivatives; pass `jac` and constraint
   `jac` when you can.
5. **Basins**: from a given start every local method, mincon included, can
   finish at a different local minimum than the published one (HS2, HS16,
   HS20, HS55, HS108 — all verified strict local minima,
   `bench/results/r3-basins`); mincon reports these as local minima, never
   as failures, and has no multi-start.
6. **AMD ordering** falls back to RCM; inertia-free acceptance is not wired.
7. **Exact Hessians** were slower than quasi-Newton in both members as first
   exposed (HS71: SQP 370, interior point 56 iterations). Fixed: the SQP member
   regularises an indefinite Lagrangian Hessian along the constraint normals
   only (7 iterations), and the interior-point member counts the pivot signs of
   the KKT factorisation instead of perturbing a legitimate negative primal
   pivot, which had voided its inertia certificate at every iteration (10
   iterations, `docs/22` §7.8, `bench/results/abl-se`). What remains: a
   Hessian supplied from Python is taken dense; sparse Hessian structure is
   only available through the Rust `Nlp` trait.
8. **Unit mismatches between variables** (a 1e6 pressure next to a 1e-6
   area): every solver in the friction audit, fmincon included, misses the
   optimum, and every first-order certificate is fooled because the gradient
   component along the large variable is 1e-7 relative to the other. mincon no
   longer certifies a non-KKT point there (D11, D12) and, with the default
   `scale_variables='auto'`, reaches the optimum from a start whose magnitudes
   reveal the units; a start of zeros carries no scale (the corpus problem
   UNITS stays unsolved).

---

## Layout

```
docs/                    algorithm specifications, resources, implementation notes, release guide
crates/
  mincon-core/           problem model, options, results
  mincon-linalg/         sparse LDL^T with certified inertia, KKT assembly
  mincon-diff/           finite differences, coloring, sparsity detection, checker
  mincon-ip/             interior point
  mincon-sqp/            SQP (l1 merit, elastic QP, damped BFGS, second-order probe)
  mincon-qp/             dense dual active-set QP subsolver (Goldfarb-Idnani)
  mincon/                public API, algorithm portfolio
  mincon-py/             PyO3 bindings + the Python package
  mincon-testset/        Hock-Schittkowski and torture problems
bench/                   CUTEst harness, performance profiles, fmincon baseline
```

---

## Building

```bash
cargo test --workspace                                                # 203 tests
cargo run --release -p mincon-ip --example run_testset                 # the interior-point regression table (the CI gate)
cargo run --release -p mincon --example run_testset_portfolio -- auto  # the same fixtures through the default portfolio (auto|ip|sqp)
cargo run --release -p mincon-ip --example diagnose HS71               # one problem, in detail

pip install maturin
cd crates/mincon-py && maturin develop --release
```

Benchmarks:

```bash
pip install -e 'crates/mincon-py[bench]'
python bench/runner.py --set smoke --solvers mincon scipy-slsqp
python bench/profiles.py results.jsonl -o profiles.png
```

---

## The honesty rule

This project measures itself against a product with a twenty-year head start,
and the temptation to flatter the numbers is constant. So:

* Benchmark success is decided by the harness, never by the solver's own
  report — a rule that cuts against us as readily as for us.
* The portfolio's cost is reported across **all** members, not the winner's.
* `Acceptable` is not counted as success.
* Known gaps are listed above, not buried.
* When the CUTEst run happens, the README will name the problem classes where
  `fmincon` still wins.

A benchmark with no losses in it is a benchmark nobody believes.

---

## Licence

MIT or Apache-2.0, at your option. Every dependency is MIT, Apache-2.0 or BSD;
nothing here requires a licence, a Fortran toolchain, or HSL.

## Acknowledgements

Standing on: Wächter & Biegler's IPOPT paper, Chiang & Zavala's inertia-free
regularization, Curtis–Powell–Reid coloring, Dolan & Moré's performance
profiles, Gratton & Toint's S2MPJ, and Hock & Schittkowski's test set. Full
annotated bibliography in [`docs/09_RESOURCES.md`](docs/09_RESOURCES.md).
