# S3: termination tolerance, honest exits, fixed-variable multipliers, derivative checks

Measured September 7–8, 2026 with the v2 harness (`bench/harness`) on the
**development split** (82 problems, track A, single thread). Evaluation
counts are deterministic algorithmic quantities and were measured on a Linux
x86_64 container; the matched fmincon numbers come from the Windows laptop run
`bench/results/s2-dev` (same corpus, same scoring). No timing claims here.

## What the development run showed (B0 = `8ec00bd`)

| solver | attained / 81 | paired vs fmincon-ip (wins/losses) | geo-mean evaluations vs fmincon-ip (common) |
|---|---:|---:|---:|
| fmincon-interior-point (defaults) | 73 | — | 1.00 |
| fmincon-sqp (defaults) | 70 | 2 / 5 | 0.73 |
| scipy-slsqp | 72 | 5 / 6 | 0.61 |
| mincon `auto` (B0) | 76 | 5 / 2 | 2.23 [1.87, 4.29] |
| mincon-ip (B0) | 76 | 5 / 2 | 1.87 [1.51, 3.28] |

mincon-ip needed a median 20.5 iterations against fmincon-ip's 12 with
similar evaluations per iteration; 15 of its 82 runs ended `Acceptable` and 7
`NumericalFailure` **after** reaching the target. The trace of HS11 shows the
mechanism: the scaled KKT error reaches 4e-7 at iteration 26, the default
target was 1e-8, forward finite differences cannot deliver it, the solver
escalates to central differences, the line search then fails and restoration
is entered at zero infeasibility, which it reports as a numerical failure.

## Changes in this increment

1. **Default optimality tolerance 1e-6** (was 1e-8; `acceptable_optimality`
   1e-4, was 1e-6). This is fmincon's default and the accuracy forward finite
   differences can support. It is a *default change*, not an algorithmic gain,
   and is reported as such: the fixture gate's strict-`Optimal` count rises
   from 41 to 50 of 54 for that reason alone.
2. **Honest exit when restoration has nothing to restore** (`restoration.rs`):
   a feasible point whose scaled KKT error is within the acceptable tolerance
   is reported `Acceptable`; a feasible point with unverified stationarity is
   `StepTolerance`; only the rest is `NumericalFailure`. HS11/HS31/HS37/HS43
   move from `NumericalFailure` to `Acceptable`/`StepTolerance` at unchanged
   points.
3. **Fixed-variable multipliers** (`solver.rs::finish`): variables with
   `xl == xu` returned huge, equal `z_l = z_u` (e.g. 25.06 each on LQTRAJ_10)
   that fail stationarity in the user's problem; they are now reconstructed
   from `grad f + J^T lambda` using analytic derivatives or two probes off the
   pinned value. Regression test `fixed_variable_multipliers_satisfy_stationarity`.
4. **Derivative checker cannot pass without evidence** (`check.rs`): no
   comparison, a NaN analytic entry or a failed derivative callback is
   inconclusive/failing, never passing; `Options::check_derivatives` is now
   actually consumed by `mincon::minimize` and stops the solve on a failed or
   inconclusive check (fmincon `CheckGradients` semantics). Four regression
   tests; the old test that asserted a trivial pass was inverted.
5. **Constraint Jacobians from Python** (`'jac'` in SciPy-style constraint
   dicts; `nonlcon_jac` in `fmincon`), used only when every block supplies
   one; `nonlcon` is called once per point instead of twice (D2). Three
   Python tests.

## Ablation (development split, 82 problems, track A)

`ablation-tolerance-1e-6.md`: B0 with only `ftol=1e-6` → same 76/81
attained, evaluations **0.62× [0.36, 0.76]** of B0.
`ablation-candidate-c1.md`: candidate C1 (all changes above) → mincon-ip
76/81 at **0.61× [0.37, 0.76]** of B0; `auto` 76/81 at 0.65× [0.37, 0.96].

Against the Windows fmincon-ip records the per-problem geometric-mean
evaluation ratio of C1 mincon-ip on the 71 commonly attained problems is
**1.12** (was 1.87), with equal median iteration count (12 vs 12). The
remaining large ratios are CHAINROSEN_{50,200} (mincon spends hundreds of
BFGS iterations; fmincon fails outright at its 3000-evaluation default),
HS57 (a flat direction where mincon reports `Optimal` at x2 ≈ 7e12), HS37,
HS13, HS74 and HS83. These go to the failure atlas.

Gate: 157 Rust tests, 54/54 fixtures (50 strict Optimal), Clippy clean,
23 Python tests on the rebuilt wheel. Windows rebuild and the installed-wheel
check are recorded in the S3 commit.
