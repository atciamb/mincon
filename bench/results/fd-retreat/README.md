# Finite-difference correctness increment

Measured September 6, 2026, Windows x86_64/MSVC, against `e6c95ed`.
No solver constants, tolerances, dependencies or fixture verdicts changed.
The preceding table is `../restoration/current-testset.txt`; run
`python bench/results/fd-retreat/summarize.py` to regenerate the comparison.

## Corrected behavior

Objective and Jacobian differences divide by the displacement actually taken
after retreat and rounding. Unequal central steps use the derivative of the
quadratic interpolant, evaluated as a secant with an asymmetry correction.
Colored groups can mix central and one-sided columns without probing outside
the box. A surviving central side provides a first-order fallback. An abort
propagates without retreat or opposite-side fallback. Fixed coordinates need
no calls; a nonfixed perturbation that rounds to zero returns an error.
Narrow boxes use whichever side has room for a representable inward step.

Eleven new Rust tests use affine/quadratic analytical derivatives, including
unequal floating-point spacings at `2^53`, dense and sparse Jacobians, serial
and parallel execution, exact callback accounting, fixed coordinates and
cancellation. Six of the initial seven tests failed before implementation;
the additional narrow-box test also failed before its correction. Existing
derivative tests remain green.

The new Python test first failed against the preceding installed wheel:
the checker estimated `0.5624993` for an analytical derivative of `3` after
asymmetric domain retreat. It passes with the rebuilt wheel at `1e-10`
relative tolerance.

## Validation and measured limits

The required sequence passed: `cargo test --workspace` (**147 tests**),
`cargo run --release -p mincon-ip --example run_testset` (**54/54 fixture
outcomes; zero detected false successes**), then Clippy with warnings denied.
The Windows `cp39-abi3` wheel was built with maturin, installed locally and
passed **7 Python tests** on Python 3.12. Other platforms remain untested here.

This increment is a correctness repair, not a demonstrated robustness or speed
gain. Strict `Optimal` returns changed **39 → 38**; HS35 now returns
`NumericalFailure` at an independently feasible, reference-accurate point.
Other statuses are 6 Acceptable, 2 StepTolerance, 6 NumericalFailure,
1 LocallyInfeasible and 1 Unbounded. Objective calls increased
**7,930 → 8,281 (4.4%)**. HS13 increased 773 → 1,018; HS100 679 → 733.
Full changes are in `summary.json`. No held-out or competitor benchmark ran.

Diagnosis of HS35 exposed a separate gap: accepted soft-restoration points
can meet the main stopping tolerances while filter acceptance remains blocked;
soft restoration does not test convergence. At the returned point the exact
quadratic gradient and returned multipliers give stationarity residual
`5.85e-9`. A separate test-first change should apply the existing complete
stopping tests inside soft restoration, including raw stationarity and
complementarity, without weakening filter acceptance for continued iteration.

M1's CUTEst improvement gate is still unmeasured. Hard callback budgets,
derivative-checker validation of failed/empty checks, and the remaining roadmap
items also remain open. Nothing was published to PyPI.
