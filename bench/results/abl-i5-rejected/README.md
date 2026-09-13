# abl-i5-rejected: the quadratic-program probe as first built (no convexity check), September 12, 2026

The first whole-corpus ablation of increment I5 (`docs/22` §7.9): at the start the objective is
tested for being quadratic and every constraint row for being linear along two random lines,
the constant Hessian is built by differencing, and the SQP member runs with it as an exact
Lagrangian Hessian before the ordinary portfolio. This variant handed over *any* quadratic,
convex or not.

Run: track A, all 172 corpus problems including the three diagnostics, defaults otherwise,
single thread, 60 s / 100 000 evaluations, targets v5, wheel `wheels-i5` (the S-E tree plus the
probe and the no-scale note), one arm `mincon@quadratic_probe=true` against the current
default's records (`abl-c7` and `s6v4-final4` rule-off arms).

| arm | attained candidate / baseline (of 166 scorable) | cost ratio [95 % family bootstrap] | what changed |
|---|---|---|---|
| `mincon@quadratic_probe=true` | **157 / 158** | 1.04 [0.89, 1.16] on 157 | 36 problems detected as quadratic programs, every one `Optimal`; **HS44 lost**: a nonconvex QP with two local minima, where the quasi-Newton path reaches the global one (f = -15) from the zero start and the Newton path with the exact indefinite Hessian certifies the other (f = -13, oracle: first-order stationary, relative stationarity 5e-15) in 278 against 58 evaluations; every other problem pays the probe's 4 to 8 evaluations |

Where the exact Hessian pays and where it costs (model-boundary evaluations, candidate against
baseline): OBSTACLE_50 1485 / 4246, OBSTACLE_200 20 910 / 51 656, NNLS_SIMPLEX_30 695 / 3084,
NNLS_SIMPLEX_120 8120 / 14 770, QUADSPHERE2_30 757 / 2604, QUADSPHERE_10 167 / 614, HS118 223 / 515,
POLYQP_10 205 / 330, POLYQP_100 5770 / 7498 and certified (`Optimal` in 1 iteration, 8.3 s) where
the default stopped on the step tolerance (status 6, 15.6 s), PORTFOLIO_20 370 / 784; against
LQTRAJ_50 12 257 / 1855, QUADSPHERE2_300 47 872 / 27 092, QUADSPHERE_100 5770 / 3030, LQTRAJ_10
677 / 381, MANY_INEQ 92 / 36, and the small HS problems by the build's few evaluations. The
diagnostics keep their diagnosis (INFEASIBLE_LIN -4, INFEASIBLE_NL -4, UNBOUNDED_PAR -3).
The no-scale note fired on 11 problems: UNITS, where it is right, and ten false alarms (HS14,
HS22, HS32, HS53, HS60, HS77, HS79, FLAT_BOUND, ROSEN_SPHERE_4, ROSEN_SPHERE_10) whose exact
gradient at the start has zero components that finite differences turn into 1e-9 noise; the
note now ignores components below the difference's own accuracy (`abl-i5-hint`).

Verdict: **rejected as built**. The falsifier of `docs/22` H4 / §7.7 was any attainment loss on
the corpus, and HS44 is one, of the basin kind: a nonconvex QP's Newton path can end at a
different certified local minimum from the quasi-Newton path's. The fix that follows is a
restriction, not a tuning: the Hessian is handed over only when it is positive semidefinite
(Cholesky of `H + 1e-10 max|H_ii| I`), since every KKT point of a convex QP is its global
minimum and the basin cannot change. The re-ablation is `abl-i5`.

Files as in `abl-i1` (`cmp-probe`, `scored-probe` and its analysis, `diff-probe.txt`).
