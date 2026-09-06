# Restoration increment — local qualification

Measured September 5, 2026, on Windows x86_64 (MSVC), Rust 1.95.0,
Python 3.12.14, maturin 1.15.0, NumPy 2.5.2, pytest 9.1.1.
The local wheel is `cp39-abi3-win_amd64`; runtime testing used Python 3.12.
Linux, macOS, other Python versions and MSRV were not run locally.
No dependency or published algorithm constant changed.

The scaffold commit is `616cd5edb67ffbf8fdc5ecce7e7137d114aeca7b`.
`baseline-testset.txt` was regenerated from an archive of that commit in an
isolated artifact directory. `current-testset.txt` corresponds to the code in
the commit containing this report. Both use the default single interior-point
configuration, no user derivatives, and the same 54 development fixtures.
The baseline used its original harness; the current harness independently
evaluates the returned objective and constraints and gives HS13 an explicit
degenerate-exit check. Neither is a held-out robustness benchmark.

## Results

| Check | Scaffold | Current |
|---|---:|---:|
| Development fixture passes | 52/54 | 54/54 |
| HS13 | NumericalFailure, inaccurate objective | Acceptable, f=1.00000352 |
| Inconsistent affine fixture | NumericalFailure | LocallyInfeasible |
| New Python wheel regression tests | 3/6 | 6/6 |
| Reported objective calls across testset | 6,758 | 7,930 |

**54 fixture passes are not 54 convergence certificates.** Current statuses:
39 Optimal, 6 Acceptable, 2 StepTolerance, 5 NumericalFailure,
1 LocallyInfeasible and 1 Unbounded. The five NumericalFailure returns
(HS11, HS31, HS33, HS37, HS43) satisfy the fixture's independent feasibility
and reference-objective checks; their strict convergence is unresolved.
`Optimal` means the requested first-order and feasibility tests passed, not
a second-order or global minimum certificate. No independently infeasible
`Optimal` return was detected in these fixtures.

Reported objective cost increases by about **17%** in aggregate. Some decreases
come from removing an unnecessary final gradient. Counting failed FD batches
also changes accounting, so this is not a clean algorithm-only cost comparison.
Material individual regressions remain: HS31 237→1,181 calls, HS42 169→490,
HS5 32→111; improvements include HS100 1,104→679 and HS29 210→79.
No general speed advantage is claimed. During development, enforcing certified
inertia initially drove HS6 to 420 iterations; the tested primal-first ordering
fallback restored it to 10, equal to the scaffold's iteration count.

## Validation

The required sequence completed successfully:

```text
cargo test --workspace
cargo run --release -p mincon-ip --example run_testset
cargo clippy --workspace --all-targets -- -D warnings
```

`cargo test --workspace`: **136 passed** (130 unit, 4 integration, 2 doctests).
Two additional tests of the example's independent verdicts passed with
`cargo test -p mincon-ip --example run_testset`.
`cargo fmt --all -- --check` and documentation compilation also passed.

The wheel was built with maturin, installed into the project venv, and tested
using `python -m pytest crates/mincon-py/tests -q`: **6 passed**. Tests cover
HS71 without derivatives, inconsistent equalities, strict bounds including
Python setup, HS13, multiplier signs/callback counts, and incorrect-gradient
detection. The same tests were first run against the untouched scaffold wheel;
HS13, restoration diagnostics and strict Python setup failed there.

Rust analytical tests additionally cover elastic elimination, full restoration
re-entry, retryable objective and derivative failures, user abort, parent
limits, a poisoned BFGS model, mixed-unit and rank-deficient KKT systems,
scaled nonzero constraint bounds, and multiplier unscaling.

Raw tables and `rust-tests.txt` are included. Run
`python bench/results/restoration/summarize.py` to regenerate `summary.json`.
CI now runs the wheel regressions and requires all 54 fixture outcomes.

## Limits and next work

M1 remains **partial**: the required CUTEst constrained-small improvement of
8 percentage points is unmeasured. S2MPJ acquisition was declined; no alternate
download was attempted. There is no new fmincon, IPOPT or SciPy comparison.

1. Resolve the accurate-point numerical failures and expensive near-solution
   iterations with derivative-accuracy and curvature diagnostics. Audit the
   existing finite-difference retreat divisors: objective probes can retreat
   while their caller still divides by the original step. Add a domain-limited
   analytical derivative oracle before changing this code.
2. Qualify the inertia-free acceptance/retry loop. All current mode names use
   certified inertia; they do not implement the advertised curvature fallback.
3. Benchmark sparse fill and factorization costs. Natural-order fallback may
   increase fill substantially; dense BFGS and the current ordering still limit
   scale. Reconsider faer's existing sparse LDLT/Bunch–Kaufman APIs before
   assuming another custom backend is necessary.
4. Enforce hard callback-level budgets and unify abort/derivative-error handling
   outside restoration. FD batches can overshoot a limit; Python's constraint
   size probe is outside Rust counters. A recovery error can retain inner trace
   rows while returning the pre-restoration point and iteration count.
5. Continue the sparse Jacobian/AD bridge, limited-memory curvature, SQP,
   cross-platform wheel and release work on the roadmap. No PyPI publication
   was performed.

The implemented mathematical variant and source reference are in
[`docs/12_RESTORATION_IMPLEMENTATION.md`](../../../docs/12_RESTORATION_IMPLEMENTATION.md).
