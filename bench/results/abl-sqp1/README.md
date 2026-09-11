# Ablation SQP-1: the SQP member alone against the interior-point member alone

Build: I0 (`9a5530d`) plus `mincon-qp` and `mincon-sqp`. Whole corpus (149
problems, 143 scorable), track A (finite differences, defaults), 100 000
evaluations / 120 s, cloud host, one run per solver. Records:
`mincon-ip.A.jsonl` (`method = "interior-point"`), `mincon-sqp.A.jsonl`
(`method = "sqp"`), scored with `targets_v3`.

| | interior point | SQP |
|---|---|---|
| attained (of 143) | 135 | 134 |
| attained only by this member | CATENARY_40, QUADSPHERE2_300, QUADSPHERE_1000 | HS108, HS25 |
| paired evaluation ratio SQP / IP, 132 jointly attained | — | geo-mean 0.781, median 0.875, bootstrap 95 % [0.702, 0.860] |

By problem size (geo-mean of the paired ratio on jointly attained problems):

| n | problems | IP attained | SQP attained | SQP / IP evaluations |
|---|---|---|---|---|
| 1–5 | 101 | 97 | 98 | 0.72 (SQP cheaper on 68 of 97) |
| 6–10 | 21 | 19 | 20 | 0.88 |
| 11–20 | 5 | 5 | 5 | 0.76 (5 of 5) |
| 21–50 | 5 | 5 | 5 | 1.12 |
| 51–100 | 3 | 3 | 2 | 3.2 |
| > 100 | 8 | 6 | 4 | 1.13 |

By constraint count (n ≤ 50): m = 0: 0.81; 1–3: 0.80; 4–10: 0.61; > 10: 0.53
— the more active constraints, the larger SQP's advantage, as §1 of
`docs/20_SQP_MATHEMATICS.md` predicts (no barrier to drive to zero).

Where SQP loses: the dense-BFGS QP on n ≥ 50 with many bounds active at the
solution (QUADSPHERE_100 12×, CHAINROSEN_EQ_200 5×) and CATENARY_40
(step-tolerance exit at f = −10.3489 with 2e-5 relative gap). Where it wins
outright: HS25 (flat start — the second-order probe finds the descent
direction the gradient cannot), HS108 (reaches the published −0.866 where
interior point finds −0.675).

Deviations recorded during development (all fixed before this run except the
last): converging to a saddle on HS33 (second-order probe and escape added);
"Optimal" with multipliers of 2e10 on the degenerate HS13 (now `Acceptable`
with a degeneracy note); zero-step QP multipliers not used for the
termination test (re-test added); a unit initial BFGS matrix rejecting every
trial on UNITS (curvature rescale from the rejected trials); the penalty
parameter exploding from a nearly feasible point (rule restricted to
significant violations, capped at 100× per iteration); Maratos-type
rejections on HS46 (up to four second-order corrections); 300 iterations at
α = 3e-5 on HS106 (adaptive step bound on the QP). Remaining: on HS13 the SQP
member alone stops at f = 0.99962 (step tolerance, violation 7e-12 through
the cubic constraint), a 3.7e-4 relative gap the fixture gate counts as a
failure; the portfolio's interior-point member covers it.

Decision (gate G1 of the plan): met. Attainment equal to the interior-point
member's within one problem; evaluations 0.78× with the interval excluding
1; on n ≤ 20, 0.72–0.88×.
