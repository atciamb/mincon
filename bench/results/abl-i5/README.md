# abl-i5: the quadratic-program probe with its convexity check (I5), whole-corpus ablation, September 12, 2026

Increment I5 of `docs/22` (§7.7, §7.9): at the start the objective is tested for being quadratic
and every constraint row for being linear along two random bounds-respecting lines (seven
evaluations when both pass, three when the first test fails), the constant Hessian is built by
differencing (function values, `n (n + 3) / 2` evaluations, or the supplied gradient, `n`
calls), checked for convexity (Cholesky of `H + 1e-10 max|H_ii| I`) and for reproducing the
six line points, and the SQP member runs with it as an exact Lagrangian Hessian before the
ordinary portfolio; only an `Optimal` exit settles the portfolio. The variant without the
convexity check lost HS44 to another certified local minimum (`abl-i5-rejected`).

Run: track A, all 172 corpus problems including the three diagnostics, defaults otherwise,
single thread, 60 s / 100 000 evaluations, targets v5, wheel `wheels-i5c` (the S-E tree plus
the probe with its convexity check and the no-scale note), one arm `mincon@quadratic_probe=true`
against the current default's records (`abl-c7` and `s6v4-final4` rule-off arms).

| arm | attained candidate / baseline (of 166 scorable) | cost ratio [95 % family bootstrap] | what changed |
|---|---|---|---|
| `mincon@quadratic_probe=true` | **158 / 158** | 1.04 [0.90, 1.16] on 158 | 35 problems detected as convex quadratic programs, every one `Optimal`; HS44 now declines (nonconvex) and keeps the global minimum at 86 against 58 evaluations (the probe's 7 plus the 14 of the build it discarded); every other problem pays the probe's 4 to 8 evaluations; the diagnostics keep their diagnosis (INFEASIBLE_LIN -4, INFEASIBLE_NL -4, UNBOUNDED_PAR -3); the two 60 s exits and HS117 differ as in every run of this tree |

Where the exact Hessian pays and where it costs, model-boundary evaluations candidate / baseline
(wall seconds in brackets where they matter):

| gains | | losses | |
|---|---|---|---|
| NNLS_SIMPLEX_30 | 695 / 3084 | LQTRAJ_50 (n = 150, 100 rows) | 12 257 / 1855 |
| QUADSPHERE_10 | 167 / 614 | MANY_INEQ (n = 5, 40 rows) | 92 / 36 |
| QUADSPHERE2_30 | 757 / 2604 | NARROWBOX, DEGEN_LICQ, NEARDEP (n = 2) | +11 to +19 |
| OBSTACLE_50 | 1485 / 4246 | QUADSPHERE_100 | 5770 / 3030 |
| OBSTACLE_200 | 20 910 / 51 656 (2.0 s / 6.1 s) | QUADSPHERE2_300 | 47 872 / 27 092 (3.1 s / 1.7 s) |
| NNLS_SIMPLEX_120 | 8120 / 14 770 | LQTRAJ_10 | 677 / 381 |
| HS118 | 223 / 515 | EQQP6, FIXEDVARS, HS21 | +17 to +41 |
| HS28, HS35, HS48, HS51, HS52, HS53, HS76, HS3 | 0.56x to 0.79x | | |
| POLYQP_100 | 5770 / 7498, **`Optimal`** in 1 iteration (7.9 s) where the default stopped on the step tolerance (status 6, 15.6 s) | | |
| POLYQP_10, PORTFOLIO_20 | 205 / 330, 370 / 784 | | |

The pattern is the one §7.7 predicted: the build costs `n (n + 3) / 2` evaluations, so it pays
where the quasi-Newton path needed more than about `n / 2` iterations (bound-constrained and
row-constrained QPs with active-set churn) and costs where it needed a handful (well-conditioned
dense quadratics such as QUADSPHERE and LQTRAJ). A build that used a supplied gradient (`n`
calls) or a sparsity pattern (OBSTACLE's Hessian is tridiagonal) would move the break-even; both
are follow-ups to measure, not rules to write now.

Verdict: the falsifier of `docs/22` H4 / §7.7 (no attainment loss on the corpus) holds; the
evaluation cost is a wash with a wide interval; the robustness gains are the certified
POLYQP_100 and box_lsq at 0.17x in the friction audit. **`quadratic_probe` becomes the default.**

The no-scale note fired on UNITS, ROSEN_SPHERE_4 and ROSEN_SPHERE_10 in this run's wheel (span
threshold 1e6 with the finite-difference floor); the committed rule raises the threshold to 1e8
for finite-difference gradients, which by construction keeps UNITS (span 1e12) and drops the two
ROSEN_SPHERE fires (span 1e6, the truncation error of a forward difference at a coordinate where
the exact gradient is zero). The count of the committed rule on this corpus is therefore one.

Files as in `abl-i1` (`cmp-probe`, `scored-probe` and its analysis, `diff-probe.txt`).
