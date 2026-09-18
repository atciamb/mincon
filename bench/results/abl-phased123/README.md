# Phase D, candidates 1 to 3: one ablation with its own default arm (September 18, 2026)

All 185 problems, track A (finite differences), 60 s / 100 000 evaluations, one thread, targets v6,
wheel `4f06f8a3d5be6473` (the tree before `quadratic_bands='decaying'` became the default, with both
options off by default). Six arms in one run, the first the default. Write-up:
`docs/22_ROUND5_ROBUSTNESS_PLAN.md` section 7.19.

| arm | attained | paired difference | cost ratio (common) | records changed |
|---|---|---|---|---|
| `mincon@quadratic_bands=decaying` | 173 / 172 of 185 | +0.6 pp [+0.0, +2.7] | 1.00 [1.00, 1.00] on 172 | 3, the clock-bound ones |
| `mincon@barrier_safeguard=0.1` | 172 / 172 | +0.0 pp [+0.0, +0.0] | 1.00 [1.00, 1.00] on 172 | 14 and the clock-bound ones, none in status |
| `mincon@barrier_safeguard=1` | 172 / 172 | +0.0 pp [+0.0, +0.0] | 1.00 [1.00, 1.01] on 172 | the same 14, none in status |
| `mincon-ip@barrier_safeguard=1` against `mincon-ip` | 167 / 168 | -0.6 pp [-0.9, +0.0] | 1.18 [1.14, 1.29] on 166 | 139 |

`quadratic_bands='decaying'`: DECONV_200 goes from a time-limit exit (16 282 evaluations, not
attained) to `Optimal` in 6 639 evaluations and 24.6 s, the Hessian built at half-bandwidth 27 in
5 422. COVQP_300 and DENSELAP_250 stop on the 60 s clock in every arm and differ only in the
evaluations that clock bought; COVQP_300's probe declines after the same 1 204 evaluations. The
other 182 records are identical to the evaluation. It became the default.

`barrier_safeguard` (a lower bound on the barrier parameter during the adaptive phase, the factor
times the scaled infeasibility relative to the first iterate's; IPOPT's
`adaptive_mu_safeguard_factor`): no status or attainment change on the portfolio. On the
interior-point member alone, three certificates lost (HS19, HS55, LOGISTIC_NOISY2), two gained
(HS100, POLYQP_100), attainment lost on HS13 and HS57 and gained on HS25, and 101 of the 119 records
that are `Optimal` both ways and changed are dearer. Falsified; the option was removed from the tree
and exists only in this wheel.

Files: the six arms' records, `cmp-A.jsonl` and `scored-A.*` (the four portfolio arms),
`cmp-ip-A.jsonl` and `scored-ip-A.*` (the member pair), and one `diff-*.txt` per candidate arm.
The raw native status of every arm was diffed against the default arm before scoring.
