# abl-phased45b: `zero_step='decrease'` as it survived, and the first form of `saddle_step='linearized'` (September 18, 2026)

Phase D candidates 4 and 5 (`docs/22_ROUND5_ROBUSTNESS_PLAN.md` section 7.18), all 185 problems,
track A (finite differences, minimal inputs), 60 s / 100 000 evaluations, single thread, wheel
`cedb828390625616`, the default arm in the same run. Scored against targets v6 (`cmp-A.jsonl`,
`scored-A.jsonl`, `scored-A.analysis.md`, `diff-zero_step.txt`, `diff-saddle_step.txt`). The fourth
arm (both options) was stopped after 5 records when the `saddle_step` arm showed its rule needed
changing; its partial file is in `stopped/`.

| arm | attained | paired difference | cost ratio (common) |
|---|---|---|---|
| `mincon@zero_step=decrease` | 172 / 172 of 185 | +0.0 pp [+0.0, +0.0] | 0.99 [0.99, 1.00] on 172 |
| `mincon@saddle_step=linearized` (bounds and rows) | 172 / 172 of 185 | +0.0 pp [+0.0, +0.0] | 1.00 [1.00, 1.00] on 172 |

**`zero_step=decrease`** (once per point at a feasible point, when the QP step's predicted decrease
is below the rounding noise of the merit function: adopt the QP's multipliers and re-run the
termination test; never a reason to stop): 16 records changed besides two clock-bound ones, all
attained in both arms, none with a different exit status, fifteen with fewer evaluations
(ELLIPSOID2_20 3976 to 3318, HS93 504 to 448, HS95 and HS96 92 to 64, ...) and ELLIPSOID_500 with 6
more of 202 186. The six problems the first form harmed (`../abl-phased45`) keep their certificate.

**`saddle_step=linearized` with the bounds in the ratio test**: one record changed, HS25 (bounds
only), 190 to 209 evaluations. A trial point is projected onto the bounds anyway, so the cap only
shortened a good step; the rule was narrowed to the constraint rows and re-run (`../abl-phased45c`).
