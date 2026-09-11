# Ablation I0: correctness fixes from the basin and budget studies

Compared: the C5 build (master `8a2d3f8`, records `bench/results/r4-budget/`)
against the same build plus (D9) the unscaled relative stationarity guard with
drift-triggered objective rescale, (D10) the projected-gradient exit for a
stationary infeasible point in restoration, and the progress verdict at budget
exits. Whole corpus (149 problems, 143 scorable), track A, default options,
100 000-evaluation / 120 s budget, one run each on the same cloud host. Wall
times are not used (not the benchmark host).

| | C5 | I0 |
|---|---|---|
| attained (of 143) | 134 | 135 (+BADSTART_DISC; no loss) |
| paired evaluation ratio I0 / C5, 134 jointly attained | — | geo-mean 1.014, median 1.000, bootstrap 95 % [1.002, 1.029] |
| problems with a ratio outside [0.77, 1.3] | — | HS1 72 → 111, QUADSPHERE2_30 810 → 1271, QUADSPHERE2_300 6630 → 13 245 |
| status changes | — | none among scorable problems |
| INFEASIBLE_NL (diagnostic, infeasible) | `MaxReached`, 419 iterations, 9784 evaluations | `LocallyInfeasible`, 6 iterations, 105 evaluations (three members) |

The three costlier problems are the ones where the objective scale factor
chosen at x₀ became stale (gradient norm fell by 24×, 640× and 3000×); each
now finishes with a rescale and a few more iterations at the correct scale,
and QUADSPHERE2_300's answer moves from a 1.8e-3 relative gap (not attained)
to 4e-5 (attained). On QUADSPHERE2_300 the C5 result was a false `Optimal`.

Budget-exit verdicts on this run: CHAINROSEN_BOX_200 and ELLIPSOID_500 both
receive "still making steady progress" notes with the last-20-iteration
movement quoted; no attained run produced a verdict (they did not exhaust the
budget).

Decision: accepted. The 1.4 % evaluation cost buys one attainment, removes
two false `Optimal` claims, and turns a 9784-evaluation non-answer into a
105-evaluation diagnosis.
