# S5 increment C4: error-aware finite-difference termination (+ two rejected variants)

Development material = all 130 scorable problems of the dev, validation and
former-final splits (the round-1 final split was diagnosed and is no longer
held out). Linux evaluation counts, Linux-vs-Linux ablation.

## Kept: error-aware termination (`Options::fd_error_aware`, default on)

Mechanism (HS62, HS100, HS74, HS83): with finite-difference derivatives the
scaled KKT residual reaches ~1e-5 and cannot go lower, so the solver spends
tens of iterations trying, then exits `Acceptable`. Now, once the iterate has
lingered within 100× the optimality tolerance for three iterations, the
derivative error is estimated on the four steepest coordinates by
re-differencing with twice the step (a few evaluations, at most every ten
iterations); the stationarity target becomes the estimated error (capped at
the acceptable tolerance), forward differences that are too inaccurate
escalate to central first, and the report notes that the certificate is
bounded by the derivative accuracy.

`ablation-vs-off.md`: same 119/127 attained; evaluations **0.97× [0.96,
1.01]** of the same build with the option off; HS62 0.20×, HS100 0.22×,
HS74 0.52×, HS83 0.68×; worst regressions PRESSURE_VESSEL and WELDED_BEAM
1.17× (escalation to central). Against fmincon-interior-point on these 130
problems: 119 vs 112 attained (+5.5 pp [+1.3, +32.3]), evaluations 1.06×
[0.90, 1.08].

## Rejected: barrier floor tied to primal infeasibility

HS63's stall (39 iterations vs 8 monotone) is the LOQO rule collapsing `mu`
to its floor at iteration 1 while the iterate is still infeasible. A floor of
1–10% of the primal infeasibility fixes HS63 (303 → 43 evaluations) but is
neutral overall (1.00× [0.98, 1.10]) with a wider tail; a floor tied to the
full KKT error made the fixture set 15–25% more expensive. Both reverted;
HS63 is added to the Rust fixture set so the case stays visible.

## Rejected earlier this round: BFGS initial scaling (`../s4-bfgs-scaling-rejected`).
