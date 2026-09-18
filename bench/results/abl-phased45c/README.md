# abl-phased45c: `saddle_step='linearized'` on rows only, and both options together (September 18, 2026)

Phase D candidates 4 and 5 (`docs/22_ROUND5_ROBUSTNESS_PLAN.md` section 7.18), all 185 problems,
track A, 60 s / 100 000 evaluations, single thread, wheel `5608cd02d4819f9b`, the default arm in the
same run. The combination arm is the behaviour that became the default afterwards.

| arm | attained | paired difference | cost ratio (common) |
|---|---|---|---|
| `mincon@saddle_step=linearized` | 172 / 172 of 185 | +0.0 pp [+0.0, +0.0] | 1.00 [1.00, 1.01] on 172 |
| `mincon@zero_step=decrease;saddle_step=linearized` | 172 / 172 of 185 | +0.0 pp [+0.0, +0.0] | 0.99 [0.99, 1.00] on 172 |

`saddle_step=linearized` changes no record but the clock-bound ones: on the corpus the step off a
saddle happens on HS25 and HS33 only, both accept their first trial, and with the ratio test on the
constraint rows alone neither is touched. Its benefit is not visible here and is measured on the
heat-flux design (`docs/22` 7.18: 104 evaluations to 87). The combination changes exactly the
sixteen records `zero_step=decrease` changes alone (`../abl-phased45b`), with the same counts, no
exit status changed and no attainment changed.

One disclosed accident: the default arm's COVQP_300 (clock-bound, not attained in any arm) was
solved while another job loaded the machine; it got 4 816 objective evaluations out of its 60 s
instead of about 12 000 and the supervisor's 210 s hard limit added a `timeout` record next to the
worker's `ok` one. The raw file keeps both; `cmp-A.jsonl` drops the duplicate `timeout` line. No
conclusion rests on that record.
