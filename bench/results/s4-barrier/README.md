# S4 increment C2: adaptive barrier update, portfolio policy, divergence note

Development split, track A, evaluation counts (Linux container; deterministic).

## Adaptive barrier (LOQO centrality rule with monotone fallback)

`BarrierUpdate::AdaptiveThenMonotone` was the documented default but only the
monotone schedule was wired. It now runs the Vanderbei–Shanno/LOQO rule
`mu = sigma * avg(d_j z_j)`, `sigma = 0.1 * min(0.05 (1 - xi)/xi, 2)^3`,
`xi = min(d_j z_j)/avg`, clamped to `[tol/10, max(mu_init, mu)]`, and falls
back to the monotone schedule permanently when the scaled KKT error has not
improved by 10% for 5 iterations or when a line search fails. The filter is
reset whenever `mu` changes (as the monotone code already did).

`ablation-adaptive-vs-monotone.md` (C2 vs C2 with `barrier=monotone`):
77/81 vs 76/81 attained (HS57 now reaches the published minimum instead of
the asymptotic stationary point at x2 ≈ 7e12), evaluations **0.93× [0.85,
0.98]**, median iterations 11 vs 12, fallback to monotone in 25 of 82 runs.
Regressions: HS100 (3.2×, 57 iterations, `Acceptable`), HS38 (1.6×), HS32
(1.35×). Combined with C1 the expected track-A ratio against
fmincon-interior-point is about 1.04; the Windows re-run in `s4-c2-dev`
measures it directly.

## Portfolio policy (D5)

* Models that are not parallel-safe (every Python model) run the members
  sequentially with early exit regardless of `threads`; a plain
  `fmincon(...)` call from Python previously ran all three members on
  threads with the GIL serializing them and no early exit (about 3× the
  work).
* Early exit now triggers on any usable, feasible answer, not only `Optimal`.
* Later members receive only the evaluation/time budget the earlier ones left.
* With `threads = k > 1` on a parallel-safe model at most `k` members run at
  once.
Tests: `a_non_parallel_safe_model_costs_one_solve_when_the_first_member_succeeds`,
`the_portfolio_never_exceeds_the_evaluation_budget_in_total`.

## Far-from-start note

A usable point with `||x||_inf > 1e6 max(1, ||x0||_inf)` now carries a note
that the objective may be asymptotically flat (HS57 under the monotone
schedule).

Python `options={'barrier': ...}` exposes the schedule for ablations.
Gate: 159 Rust tests, 54/54 fixtures, Clippy/fmt clean, 23 Python tests.
