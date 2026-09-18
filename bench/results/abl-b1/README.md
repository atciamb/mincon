# abl-b1: the Phase B wheel on the whole corpus, track A (September 18, 2026)

A check, not an ablation of an increment: Phase B (`docs/22_ROUND5_ROBUSTNESS_PLAN.md` section
7.16, `docs/23` Phase B) changed no algorithm, but one budget-accounting correction could touch a
record. The quadratic member (the SQP member on a problem the probe found to be a quadratic
program) used to run with the caller's full clock and no iteration cap; it now gets what the
probe left, like every other member. Only a record that stops on the clock inside the quadratic
member could move. The other Phase B changes cannot reach a harness record: the shared iteration
cap applies only when the caller sets `max_iterations` (the harness does not), the counting
wrapper changes a note, the exact linear Jacobian lives in the `fmincon` facade the harness does
not use, and the removed options were never set.

Candidate: the mincon arm at the Phase B tree (wheel sha256 `29ff4239ff5bc8a7`), all 185
problems, track A (finite differences, minimal inputs), 60 s / 100 000 evaluations, single thread.
Baseline: the round-5 default arm `abl-i5-rows/mincon_quadratic_rows-values.A.jsonl` (the same
defaults at the tree of `3aac1e5`), scored together against targets v6 (`cmp-A.jsonl`,
`scored-A.jsonl`, `scored-A.analysis.md`, `diff-A.txt`).

## Result

| | attained | paired difference | cost ratio (common) |
|---|---|---|---|
| candidate vs baseline | 173 / 172 of 185 | +0.6 pp [+0.0, +3.4] | 1.00 [1.00, 1.01] on 172 |

**Three records changed, and they are the three clock-bound problems:**

| problem | candidate | baseline |
|---|---|---|
| COVQP_300 | limit, 15 059 evaluations, 60.5 s, attained | limit, 11 447 evaluations, 62.2 s, not attained |
| DECONV_200 | limit, 18 895 evaluations, 60.6 s, not attained | limit, 14 473 evaluations, 60.3 s, not attained |
| DENSELAP_250 | limit, 22 730 evaluations, 61.2 s, attained | limit, 17 120 evaluations, 60.2 s, attained |

Every one of them stopped on the 60 s clock in both arms, and the candidate got 31-33 % more
evaluations out of the same clock on every one of them: the machine was faster on the day, not
the solver. **COVQP_300's new attainment is clock luck and is not claimed.** None of the three is
a quadratic-member record (the probe declines all three: DECONV_200 on the band budget,
COVQP_300 and DENSELAP_250 on the dense build's size), so the accounting correction did not act
on any record in the corpus. The 182 other records are identical to the evaluation.

The verdict for Phase B on the corpus is therefore: no change, as designed.
