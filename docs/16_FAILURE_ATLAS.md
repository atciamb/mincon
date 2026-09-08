# Failure atlas (development split, tracks A and C, September 7–8, 2026)

Each cluster: how often it appeared in the matched development run
(`bench/results/s2-dev`, B0) and after the S3/S4 increments
(`bench/results/s4-c2-dev`, C2), its cost, the mechanism established from
traces, and the remedy (done or open). "Attained" means the independent
target test of `docs/15_BENCHMARK_PROTOCOL_V2.md`; local minima are
distinguished from false claims by the oracle's KKT columns.

| # | cluster | B0 frequency / cost | mechanism | remedy | status |
|---|---|---|---|---|---|
| 1 | Termination chases 1e-8 stationarity with forward-difference derivatives | 15 `Acceptable` + 7 `NumericalFailure` exits after the target was reached; median 20.5 vs 12 iterations; 1.87× fmincon-ip evaluations | scaled KKT error reaches ~1e-7 (HS11: 4e-7 at iteration 26), forward FD cannot go lower, escalation to central FD doubles cost, 15 "acceptable" iterations are burned, then the line search fails | default optimality tolerance 1e-6 (fmincon's), acceptable 1e-4 | fixed (C1); 0.62× evaluations, same attainment |
| 2 | Restoration entered at zero infeasibility reports `NumericalFailure` | HS11, HS31, HS37, HS43, HS74, HS86, HS100, HS113, FIXEDVARS (all attained) | line-search failure at a feasible near-KKT point routes to restoration, which has nothing to restore | classify by the point's status: `Acceptable` / `StepTolerance` / `NumericalFailure` | fixed (C1) |
| 3 | Fixed variables kept in a 2e-10 relaxed box | LQTRAJ_10: 46 iterations vs 7 (fmincon), 1930 vs 248 evaluations; returned `z_l = z_u = 25.06` failing stationarity | barrier terms of a pinned variable dominate; multipliers meaningless; FD gives the pinned variable a zero derivative | multipliers reconstructed from stationarity (C1); after the tolerance fix LQTRAJ_200 costs 1.41× in track A, 1.0× in track C | multipliers fixed; **open**: reversible presolve elimination of fixed variables |
| 4 | Quasi-Newton on long curved valleys (chained Rosenbrock, n ≥ 50) | mincon 130–490 iterations; fmincon-ip fails at its default 3000-evaluation cap (n ≥ 50); with exact gradients every BFGS code needs 230–980 iterations (SLSQP 935, fmincon-ip 772+) | curvature model, not a mincon defect; mincon-ip solves BOX_200 in track C (1.8 s), not in track A within 100k evaluations | none in this increment | **open**: exact/AD Hessians from Python, FD-Hessian option, better BFGS scaling; large-n wall time (dense BFGS ~25 ms/iteration at n = 200) |
| 5 | Asymptotically flat objective (HS57, exponential fit) | B0/C1 report `Optimal` at x2 ≈ 7e12 with f = 0.0306 (published 0.02846) | gradient → 0 along x2 → ∞; first-order conditions hold in the limit | far-from-start note (C2); the adaptive barrier happens to reach the published minimum | note added; **open**: divergence detection without an objective drop |
| 6 | Basin selection from the given start | HS2 (fmincon-ip, SLSQP elsewhere), HS16 (mincon, sqp, SLSQP at the verified local minimum f = 3.98), HS20 (every solver at f = 40.199, published 38.199), HS44, HS55 (rank-deficient: different KKT points), BADSTART_DISC (validation: mincon at a KKT point with f = 6.375 from (1000, −1000); fmincon-ip reaches 0.0457) | legitimate local minima; oracle confirms KKT | none; reported as not attained, never as false success | by design; BADSTART_DISC start handling worth a look |
| 7 | Degenerate constraints | HS13 (MFCQ fails): mincon `Acceptable` at the solution after 80–250 iterations, fmincon-ip stops at f = 1.355; RANKLOSS_JAC (∇h = 0 at x*): fmincon/SLSQP stop at f = 1.028 (h = 1e-4² ≈ tolerance), mincon attains; REDUNDANT_EQ: 29 iterations in B0, 3 after C1 | unbounded multipliers / rank loss; the 1e-6 tolerance on `h = (x−1)^2` allows |x − 1| ≈ 1e-3 | none | acceptable; HS13 cost is high |
| 8 | Adaptive barrier stalls before its fallback fires | HS100 (57 iterations, 3.2× monotone), HS38 (1.6×), HS32 (1.35×) | LOQO rule keeps mu high on badly centred iterates; the 5-iteration KKT-stall detector triggers late | fallback to monotone (C2) | **open**: earlier fallback criterion (e.g. barrier-subproblem progress), ablate on validation |
| 9 | fmincon's own default evaluation cap | fmincon-ip fails QUADSPHERE_100/1000, CHAINROSEN_50/200 with `MaxFunctionEvaluations = 3000` and finite differences (n = 1000 allows 3 gradients) | a "defaults" limitation of the competitor, counted as such in track A | none (protocol keeps defaults) | recorded |
| 10 | Portfolio waste | `auto` = 3 full members when the first is not `Optimal`; from Python all three ran on threads under the GIL with no early exit (≈3× wall time); CHAINROSEN_BOX_200: 30 s vs 10.7 s | no early exit on `Acceptable`; parallel path ignores `parallel_safe` | sequential early exit on any usable feasible answer; shared budgets; concurrency bounded by `threads` | fixed (C2) |
| 11 | Harness/adapter faults (would have flattered or penalised a solver) | `np.isclose` equality rows, uncounted constraint calls, IPOPT with analytic derivatives on a black-box board, no returned point in records, moving targets; my own track-C adapter passed no constraint Jacobian to mincon (2.45× artefact) | — | protocol v2, oracle, schema; constraint `jac` support and re-run | fixed |
| 12 | Derivative checker passing with no evidence; `check_derivatives` option unused | latent | NaN discarded by `f64::max`; empty comparison sets | conclusive/inconclusive verdicts; option wired | fixed (C1) |

| 13 | Adaptive barrier collapses `mu` on a centred but infeasible iterate (HS63) | 39 vs 8 iterations, 303 vs 40 evaluations | LOQO rule gives `sigma = 0` when `xi = 1`; `mu` falls to its floor at iteration 1, then oscillates for 30 iterations | primal-infeasibility floor tried: fixes HS63, neutral overall (1.00× [0.98, 1.10]) — rejected; HS63 added as a fixture | **open** |
| 14 | Finite-difference-limited stationarity (HS62, HS100, HS74, HS83) | 250–1800 evaluations vs 13–50 with exact derivatives | the KKT residual floor is the derivative error; the solver kept iterating and exited `Acceptable` | error-aware termination (C4): estimate the error on the steepest coordinates, stop there, escalate to central first | fixed (0.20–0.68× on those; 0.97× overall) |
| 15 | Dense BFGS from the identity at large n (ELLIPSOID2_200, QUADSPHERE2_300, round 2) | ~800 iterations / 166k evaluations; 42k evaluations without attaining | initial `B = I` is far from the curvature; each BFGS update fixes one direction | unguarded initial scaling rejected (`s4-bfgs-scaling-rejected`); guarded variant untested | **open** |

## Where fmincon still wins (development + validation + held-out, track A)

* fmincon-ip attains HS2, HS16 and BADSTART_DISC's global basins where mincon
  lands in another (cluster 6); it reaches the RANKLOSS_JAC-type degenerate
  points with fewer iterations by stopping earlier.
* fmincon-sqp and SLSQP use 0.6–0.75× the evaluations of any interior-point
  code on the small dense problems that dominate this corpus (cluster: no
  SQP in mincon yet; `docs/03_SPEC_SQP.md`).
* On HS38, HS56, HS63 and ELLIPSOID2_20 fmincon-ip needs fewer evaluations than C4 (clusters 8, 13).
* At n ≥ 200 with finite differences fmincon-ip fails at its evaluation cap while mincon spends a large budget (cluster 15); fmincon-sqp attains QUADSPHERE2_300 where mincon does not.

## Where mincon wins

* Robustness at defaults: dev 77/81 vs 73/81 (5 wins / 1 loss), validation
  23/24 vs 21/24 (3 wins / 1 loss), with fmincon's losses concentrated in
  its evaluation cap (cluster 9) and degenerate HS13.
* Wall time on the same host, single thread: geo-mean ratio 0.008 on the
  development split (median 0.8 ms vs 120 ms); still faster at n = 600
  (LQTRAJ_200, 1.6 s vs 2.3 s).
