# abl-i5-rows: convex quadratic constraint rows in the quadratic-program probe, whole-corpus ablation, September 13, 2026

Increment I5's remaining extension (`docs/22` §7.9 follow-ups, §7.13). Until now the probe
declined any problem with a constraint row that is not linear, by design: only the objective's
Hessian was built, so a quadratic row had no curvature in the model. `Options::quadratic_rows`
(`QuadraticRows::{Off, Jacobian, Values}`, Python `quadratic_rows`) lets the probe accept a
quadratic row when the row **keeps the feasible set convex** — a convex function bounded above
or a concave one bounded below, never an equality and never a ranged row. The curvature sign is
checked first along the two probe lines, so an equality or a wrong-sign row declines before
anything is built; the row Hessians are then built together (one constraint or Jacobian
evaluation gives every row at once) by the same structured differencing as the objective's, and
each is checked by a banded Cholesky on the side its bound needs. The model's Lagrangian Hessian
sums the objective's with each row's times its multiplier. `Jacobian` builds rows only from a
supplied Jacobian (`n + 1` Jacobian calls for all rows); `Values` also builds from constraint
values (`2n` evaluations for diagonal rows, `n (n + 3) / 2` for dense ones).

Measured first, before the option was written: of the corpus's problems with a quadratic
objective and at least one quadratic row, the ones the quasi-Newton path found expensive
(ELLIPSOID_50 128 iterations, ELLIPSOID2_200 412, ELLIPSOID_500 196) all have a **diagonal** row
Hessian, so the row build costs `2n`, not `n (n + 3) / 2`.

Run: all 185 corpus problems including the three diagnostics, defaults otherwise, single thread,
60 s / 100 000 evaluations, targets v6, wheel `wheels-i5rows`. Two arms on track A
(`mincon` as the control, `mincon@quadratic_rows=values`) and three on track C (adding
`mincon@quadratic_rows=jacobian`). The control arm is the same wheel with the option off, so
the comparison is exactly the increment.

| track | arm | attained / control (of 179 scorable) | cost ratio [95 % family bootstrap] | records that changed |
|---|---|---|---|---|
| A | `quadratic_rows=values` | **172 / 171** | 0.99 [0.92, 1.01] on 171 | 22; rows fired on 18 problems |
| C | `quadratic_rows=values` | 175 / 175 | 0.91 [0.67, 1.00] objective+constraint calls, 0.98 [0.85, 1.06] with derivative calls | 19; rows fired on 19 |
| C | `quadratic_rows=jacobian` | 175 / 175 | identical to `values` on every record | 19 |

On track C the corpus supplies Jacobians, so the two modes build the same Hessians from the same
`n + 1` Jacobian calls and their records are identical; the distinction only matters for a model
that supplies no Jacobian.

**The gain.** ELLIPSOID_500 (n = 500, a projection onto an ellipsoid) is newly attained on
track A: the control ends the 100 000-evaluation budget at an **infeasible** point
(f = 1995.49, violation 0.85), while the candidate returns f = 2034.13 at violation 8.5e-14,
which is the reference to 1.6e-14, in 26 iterations instead of 196. ELLIPSOID2_200 goes from a
step-tolerance exit at 170 824 evaluations to `Optimal` at 9308 — **0.05×, and certified where
it was not** — in 19 iterations instead of 412. ELLIPSOID_50, ELLIPSOID2_20 and ELLIPSOID_5 fall
from 128, 58 and 16 iterations to 23, 23 and 7. On track C the same family is 5169 → 483 model
calls (ELLIPSOID2_200) and 14 265 → 1083 (ELLIPSOID_500).

**The cost.** The build pays where the row's curvature was not what the quasi-Newton path
needed: COVQP_120 1.59×, ELLIPSOID2_20 1.45×, HS10 1.48× and the two-variable problems HS12,
HS14, HS22 at 1.23× to 1.32× on track A; on track C with the derivative-inclusive count
COVQP_300 143 → 657 calls, COVQP_120 115 → 297 and TCPORT_100 121 → 245, because those have a
**dense** objective Hessian already and the rows add `n + 1` Jacobian calls for little. Against
that, HS113 0.63×, ELLIPSOID_5 0.67×, HS43 0.78× and COVQP_30 0.82× on track A.

**Diagnostics are preserved or improved.** INFEASIBLE_LIN keeps `-4` on both tracks;
UNBOUNDED_PAR keeps `-3` on track A; INFEASIBLE_NL keeps `-4` on track A (190 → 325 evaluations)
and on track C **improves from `-5` (a numerical failure) to `-4` (the documented locally
infeasible diagnosis)** at 820 → 1768 evaluations.

Verdict: the falsifier (no attainment loss, diagnostics unchanged) holds, and the increment
gains one attainment and one certification while costing nothing measurable in the aggregate on
either track or either metric. **`quadratic_rows='values'` becomes the default.** The residual
losses on the dense-Hessian families (covqp, tcport) are reported, not tuned away; a rule that
skips the row build when the objective's Hessian came back dense is the obvious next thing to
measure, and is not written now.

**Friction audit at the tree with the default flipped** (`s7-friction/mincon-fmincon-i5rows.jsonl`,
`summary-i5rows.md`, the control being the structured-build tree `mincon-fmincon-i5f.jsonl`):
13/14 as before, every attained record `Optimal`, and exactly one record changes -- the one
friction problem with a quadratic constraint row, `portfolio_risk`, from 248 model calls in 11
iterations to **166 in 3**. The other thirteen are identical evaluation for evaluation, including
box_lsq at 1232 and noisy_simulator at 200.

Files: `mincon.{A,C}.jsonl` (the control arm, this wheel with the option off),
`mincon_quadratic_rows-{values,jacobian}.{A,C}.jsonl`, `cmp-{A,C}.jsonl`, `scored-{A,C}.jsonl`
with their analyses, `diff-*.txt`, `experiment.json`. The supervisor and scoring logs stayed
local, as in every earlier ablation directory.
