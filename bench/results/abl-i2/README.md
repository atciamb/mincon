# abl-i2: the infeasibility diagnosis (D12), whole-corpus ablation, September 12, 2026

Increment I2 of `docs/22`: the SQP member declares `LocallyInfeasible` only when the linearised
constraint violation cannot be reduced with the QP step bound lifted; when it can, the bound is
reset, the penalty raised to what the merit function needs for that step, and the iteration
retried (at most three times, then `NumericalFailure` with a message that says the step was
rejected and that this is not an infeasibility diagnosis). Mechanism and the record that found
it: `bench/results/s7-friction` (bad_scaling under the SQP member, also fmincon-sqp's failure
there), atlas cluster 20.

Run: track A, all 172 corpus problems **including the three diagnostic ones** (INFEASIBLE_LIN,
INFEASIBLE_NL, UNBOUNDED_PAR, whose correct outcome is the diagnosis I2 must not lose),
defaults, single thread, 60 s / 100 000 evaluations, targets v5, wheel `wheels-i2` of the
I1 + I2 tree (so the I1 guard is on in every arm; its own ablation is `abl-i1`). Baseline for
the 169 scorable problems: the current default's records (`abl-c7` and `s6v4-final4` rule-off
arms). Diagnostics are checked by their exit status.

| arm | attained candidate / baseline (of 166 scorable) | cost ratio [95 % family bootstrap] | records that changed | diagnostics |
|---|---|---|---|---|
| portfolio (`mincon`) | 158 / 158 | 1.00 [1.00, 1.01] on 158 | the two 60 s budget exits (COVQP_300, DENSELAP_250) | INFEASIBLE_LIN -4, INFEASIBLE_NL -4, UNBOUNDED_PAR -3 |
| IP member (`mincon-ip`) | 156 / 156 | 1.00 [1.00, 1.00] on 156 | the same five as `abl-i1` (two 60 s exits, HS16 +16, HS17 +18, UNITS `Optimal` -> `Acceptable`): I1's effect, no I2 code runs here | -4, -4, -3 |
| SQP member (`mincon-sqp`) | 154 / 154 | 1.00 [1.00, 1.00] on 154 | the five 60 s budget exits (ELLIPSOID_500, MAXENT_200 (attained either way; the baseline's run ended `Optimal` at 55 s, this one at the 60 s limit), OBSTACLE_500, QUADSPHERE2_300, QUADSPHERE_1000) | INFEASIBLE_NL -4 in 81 evaluations through the new path ("the linearised violation cannot be reduced without the step bound either"), INFEASIBLE_LIN -4 in 28; UNBOUNDED_PAR ends at its budget (status 0, 2806 evaluations) exactly as under the SQP member before I2 (`abl-sqp1`, 2806): the unboundedness diagnosis is the interior-point member's and the portfolio's, not touched here |

Verdict: H2 of `docs/22` stands. No attained record changed in any arm, no diagnosis was lost,
and the friction problem that motivated the fix (bad_scaling under the SQP member) converges
instead of declaring a feasible problem infeasible (`s7-friction/mincon-fmincon-i3.jsonl`,
`docs/22` §7.2). The refusal path did not fire on any corpus problem (no record changed), so
its cost on the corpus is zero; its benefit is confined, so far, to the friction problem.

Files as in `abl-i1`.
