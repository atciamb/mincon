# Restoration stopping and boundary derivative accuracy

Measured September 6, 2026, Windows x86_64/MSVC. This completes the follow-up
identified in `../fd-retreat/README.md`. No tolerances, published solver
constants, dependencies or fixture verdicts changed.

## Changes and independent evidence

Soft restoration now recognizes an accepted point satisfying the original
problem's complete stopping test even when the filter blocks re-entry. The
test includes feasibility, complementarity and stationarity before multiplier
normalization. Continuing iterations still requires the original filter and
residual safeguards. Elastic-restoration multipliers cannot earn this exit.

An analytical quadratic with a deliberately dominating filter entry reached
its exact minimizer but previously returned `MaxReached`. The new regression
requires `Optimal` after one accepted step. Separate residual tests retain
the strict complementarity request and guard against large multipliers making
the normalized stationarity test misleadingly small.

Independent checking of HS33 exposed a prerequisite: the first-order boundary
fallback used the central step size, leaving a roughly `3.6e-5` error in the
returned bound multiplier despite a small numerical KKT residual. Central
differences now use two inward probes when opposite-sided probes cannot fit.
The same quadratic interpolant handles their realized displacements, including
retreat. A single surviving or coincident probe still has a first-order fallback.
No probe crosses the box. An analytical quadratic regression failed with
derivatives `[3.15625, -2.3125]` instead of `[3, -2]` before this correction.

HS33 and HS35 now have an integration test that independently differentiates
their polynomial objectives and constraints, checks stationarity against the
**requested `1e-8` tolerance**, and checks multiplier signs, complementarity,
feasibility and reference objective values. It passes in debug and release.
It uses neither finite differences nor the solver's reported optimality as its
oracle. HS13 remains a non-success degenerate exit.

## Results

| Check | Start (`e6c95ed`) | FD correction (`2121f55`) | Current |
|---|---:|---:|---:|
| Independent fixture passes | 54/54 | 54/54 | 54/54 |
| Reported strict Optimal | 39 | 38 | 40 |
| Reported objective calls | 7,930 | 8,281 | 7,922 |
| Rust workspace tests | 136 | 147 | 151 |
| Python wheel tests | 6 | 7 | 7 |

Current non-Optimal statuses: 6 Acceptable, 2 StepTolerance,
4 NumericalFailure (HS11, HS31, HS37, HS43), 1 LocallyInfeasible and
1 Unbounded. Fixture passes include accurate points and expected failure
diagnostics; they are not 54 convergence certificates. The harness detected
zero invalid/infeasible success returns. Analytical KKT checks were added for
HS33 and HS35, not for every fixture; approximate derivatives remain an
accuracy limitation for general models.

Total objective cost is effectively unchanged from the start (8 fewer calls),
with individual regressions including HS100 679 -> 733 and HS35 164 -> 173.
HS33 improves from a numerical failure at 173 calls to Optimal at 144 calls.
No general speed or held-out robustness gain is claimed. All three tables use
the same single interior-point configuration, starting points and independent
harness. Run `python bench/results/restoration-stopping/summarize.py` to
regenerate the comparison from the stored raw tables.

## Validation and next work

The required sequence passed: workspace tests, release development gate, then
workspace/all-target Clippy with warnings denied. Formatting checks passed for
the touched Rust files; the pre-existing user edits in `solver.rs` were
preserved. The new KKT integration test also passed in release mode.

Maturin built `mincon-0.1.0-cp39-abi3-win_amd64.whl`, installed into the local
Python 3.12 environment and passed all 7 Python regressions. The wheel is in
`bench/artifacts/restoration-stopping/wheels/` (ignored build output).
Other platforms, other Python versions and publication remain release work.

M1 stays partial: its CUTEst +8 percentage-point gate is unmeasured, and S2MPJ
acquisition was previously declined. No competitor comparison was run.
The remaining near-solution failures need accuracy/curvature diagnostics.
Hard callback budgets and consistent abort handling remain open, as does a
derivative-checker audit: failed/empty checks must not be reported as passing.
Then continue the sparse Jacobian/AD bridge and subsequent roadmap milestones.
