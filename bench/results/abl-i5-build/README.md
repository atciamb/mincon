# abl-i5-build: the quadratic-program probe's structured Hessian build, whole-corpus ablation, September 13, 2026

Follow-up to I5 (`docs/22` §7.9, §7.12). The probe's evaluation jumps in `abl-i5` (LQTRAJ_50
1855 -> 12 257, QUADSPHERE_100 3030 -> 5770, QUADSPHERE2_300 27 092 -> 47 872, MANY_INEQ 36 -> 92)
were measured against the baseline quasi-Newton iteration counts per `n` before anything was
written: every loss has `nit / n <= 0.40` and every gain `nit / n >= 0.48`, no size rule
separates them (OBSTACLE_200 at n = 200 is a gain, QUADSPHERE_100 at n = 100 a loss), and the
loss problems' Hessians are diagonal (QUADSPHERE, QUADSPHERE2, LQTRAJ, EQQP6) or zero
(MANY_INEQ), OBSTACLE's tridiagonal. `Options::quadratic_build` (`QuadraticBuild::{Structured,
Dense}`, Python `quadratic_build`) reorders the function-value build by structure: the diagonal
first (`2n` evaluations), then one off-diagonal band at a time, the model checked against the
probe's six line points after each band and handed over at the first structure that both
reproduces them and passes the convexity check. A dense Hessian costs the same as before.

Run: track A, all 172 corpus problems including the three diagnostics, defaults otherwise,
single thread, 60 s / 100 000 evaluations, targets v5, wheel `wheels-i5f` (the HEAD tree
`d1d0946` plus the option; the first candidate `wheels-i5e` declined box_lsq "not convex" after a
band model that fitted at half-bandwidth 23 failed the 1e-10 convexity check, and was fixed
before the ablation by continuing the search until the model is also convex), one arm
`mincon@quadratic_build=structured`, scored twice: against the current default's records
(`abl-c7` and `s6v4-final4` rule-off arms, the recipe baseline) and against the `abl-i5` arm,
which is the dense build this option replaces.

| comparison | attained candidate / baseline (of 166 scorable) | cost ratio [95 % family bootstrap] | records that changed |
|---|---|---|---|
| against the recipe baseline (`mincon@bfgs_rescale=0`) | **159 / 158** | 0.93 [0.62, 1.04] on 158 | +1 attained: OBSTACLE_500; every other record differs by the probe's evaluations as in `abl-i5` |
| against the dense build (`abl-i5`, `mincon@quadratic_probe=true`) | **159 / 158** | **0.90 [0.65, 0.97]** on 158 | 31, every one cheaper or equal, none lost (`diff-vs-i5.txt`) |

Where the structure pays, total model evaluations, structured / dense build / quasi-Newton baseline:

| problem | structure found | structured | dense (`abl-i5`) | baseline |
|---|---|---:|---:|---:|
| OBSTACLE_500 (n = 500) | half-bandwidth 1 | **3009, `Optimal`** | 100 347, budget exit | 100 283, budget exit |
| OBSTACLE_200 | half-bandwidth 1 | 1209 | 20 910 | 51 656 |
| OBSTACLE_50 | half-bandwidth 1 | 309 | 1485 | 4246 |
| QUADSPHERE2_300 | diagonal | 3022 | 47 872 | 27 092 |
| QUADSPHERE_1000 (n = 1000, above the old dense limit) | diagonal | 8020 | 34 048 (declined, quasi-Newton) | 34 048 |
| QUADSPHERE_100 | diagonal | 820 | 5770 | 3030 |
| LQTRAJ_50 | diagonal | 1082 | 12 257 | 1855 |
| LQTRAJ_200 (n = 600) | diagonal | 4232 | 7873 (declined) | 7873 |
| LQTRAJ_10 | diagonal | 242 | 677 | 381 |
| QUADSPHERE2_30 | diagonal | 322 | 757 | 2604 |
| HS118 | diagonal | 118 | 223 | 515 |
| MANY_INEQ | zero (linear objective) | 82 | 92 | 36 |
| EQQP6, FIXEDVARS, HS48, HS51, HS52, HS53 | diagonal | 6 to 15 fewer | | |
| two-variable problems (BIGMULT, DEGEN_LICQ, HS21, NARROWBOX, NEARDEP, START_OUTSIDE, UNITS) | diagonal | 1 fewer | | |
| NNLS_SIMPLEX, POLYQP, PORTFOLIO, HS44, COVQP | dense, or declined as before | unchanged | | |

The residual losses against the quasi-Newton baseline are the dense QPs the quasi-Newton path
solved in few iterations (PORTFOLIO_100 1.71x, POLYQP_100 1.47x but certified, NNLS_SIMPLEX_120
1.05x) and the small problems' probe cost (MANY_INEQ, EQQP6); they are reported, not tuned away.
The 60 s exits (COVQP_300, DENSELAP_250) and HS117 differ as in every run of this tree; the
diagnostics keep their diagnosis (INFEASIBLE_LIN -4, INFEASIBLE_NL -4, UNBOUNDED_PAR -3).

Verdict: the falsifier of the hand-off ("no loss of the I5 gains, no attainment change") holds
with a margin: no gain is given back, OBSTACLE_500 is a new certified attainment, and the four
jumps the owner named fall below their quasi-Newton baselines. **`quadratic_build='structured'`
becomes the default.** Friction audit with the option (`s7-friction/mincon-fmincon-i5f.jsonl`,
`summary-i5f.md`, 13/14 as before, every record `Optimal`): box_lsq 1485 -> 1232, infeasible_start_far
116 -> 71 (the projection onto the simplex has the identity as its Hessian, a diagonal build),
noisy_simulator 107 -> 200 (a quadratic plus 1e-9 noise: the diagonal model fits the noisy line
points where the dense one misfit, the SQP member runs with that constant model and still certifies
in 12 iterations, at 200 evaluations against the quasi-Newton path's 107), every other record
identical.

Files as in `abl-i1` (`cmp-structured`, `scored-structured` and its analysis, `diff-structured.txt`
against the recipe baseline; `cmp-vs-i5`, `scored-vs-i5` and its analysis, `diff-vs-i5.txt` against
the `abl-i5` arm; `score-*.log`, `run.log`).
