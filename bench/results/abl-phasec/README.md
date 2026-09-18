# abl-phasec: the Phase C wheel on the whole corpus, track A (September 18, 2026)

A check, not an ablation of an increment. Phase C (`docs/22_ROUND5_ROBUSTNESS_PLAN.md` section
7.17) adds batched and parallel finite-difference probes behind `workers=` and `vectorized=`; the
harness uses neither, so no harness record should move. What could move one by accident: the
constraint Jacobian's per-group arithmetic was moved into two closures shared by the serial and the
batched path, the SQP member's second-order verdict into a function shared by both, and the
facade's closures became classes (the harness does not use the facade).

Candidate: the mincon arm at the Phase C tree (the audit wheel `2f878929e60ce53e`'s extension
module, byte for byte that of the wheel of record `444c3ac506ce0ccc`), all 185 problems, track A
(finite differences, minimal inputs), 60 s / 100 000 evaluations, single thread. Baseline: the
round-5 default arm `abl-i5-rows/mincon_quadratic_rows-values.A.jsonl`, scored together against
targets v6 (`cmp-A.jsonl`, `scored-A.jsonl`, `scored-A.analysis.md`, `diff-A.txt`).

## Result

| | attained | paired difference | cost ratio (common) |
|---|---|---|---|
| candidate vs baseline | 172 / 172 of 185 | +0.0 pp [+0.0, +0.0] | 1.00 [0.99, 1.00] on 172 |

**Three records changed, and they are the three clock-bound problems**, as in `../abl-b1`:

| problem | candidate | baseline |
|---|---|---|
| COVQP_300 | limit, 12 049 evaluations, 63.4 s, not attained | limit, 11 447 evaluations, 62.2 s, not attained |
| DECONV_200 | limit, 17 086 evaluations, 60.3 s, not attained | limit, 14 473 evaluations, 60.3 s, not attained |
| DENSELAP_250 | limit, 14 602 evaluations, 60.7 s, attained | limit, 17 120 evaluations, 60.2 s, attained |

All three stopped on the 60 s clock in both arms; what differs is how many evaluations the clock
bought on the day (COVQP_300, which `abl-b1` attained on a faster clock, is back to not attained:
the same clock luck read the other way). The 182 other records are identical to the evaluation.

The verdict for Phase C on the corpus: no change without the option, as designed. The directory
name avoids `abl-c1` ... `abl-c8`, which are candidate ablations of earlier rounds.
