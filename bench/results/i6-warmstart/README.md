# i6-warmstart: resuming an interrupted solve warm against cold (I6), September 12, 2026

Increment I6 of `docs/22` (§7.11): `warm_start=res` on `mincon.minimize` and `mincon.fmincon`
(Rust `Options::warm_start`) starts a solve from a previous result's multipliers, its
quasi-Newton curvature model (`res.hess_approx`, carried by every report for `n <= 1000`) and,
for the interior-point member, the barrier parameter its trace ended at.

Study: `bench/friction/warm_start_study.py`. Each friction problem is solved at defaults
(`quadratic_probe` off, so the quasi-Newton path is what is measured); the same solve is then
interrupted by an evaluation budget at a third and at two thirds of that count and resumed from
the interrupted point twice, warm (`x0=res.x, warm_start=res`) and cold (`x0=res.x` only).
Model-boundary objective evaluations, interrupted run plus resumed run; every run attains the
reference (`measure.json` has the records).

| problem | uninterrupted | cut at 2/3: warm | cold | cut at 1/3: warm | cold |
|---|---:|---:|---:|---:|---:|
| box_lsq | 8595 | 8391 | 11561 | 8947 | 9459 |
| chainrosen20 | 2483 | 2504 | 2725 | 2793 | 2572 |
| odefit | 96 | 100 | 147 | 125 | 125 |
| pressure_vessel | 57 | 68 | 95 | 68 | 77 |
| portfolio_risk | 111 | 127 | 171 | 120 | 120 |
| linear_only | 125 | 132 | 136 | 132 | 169 |
| hs71 | 32 | 37 | 42 | 37 | 37 |
| with_args | 99 | 103 | 133 | 103 | 90 |
| infeasible_start_far | 99 | 177 | 273 | 110 | 55 |
| equality_circle | 26 | 30 | 30 | 30 | 30 |

What was measured on the way: with the multipliers alone (the first build) warm and cold
restarts spent identical evaluations on every SQP-first problem, since the SQP member re-derives
its multipliers from the QP each iteration, and differed by noise on box_lsq; the curvature
model is what a restart loses. With it handed over, a late resume beats the cold restart on eight
problems of ten (equal on two) and comes within 5 % of the uninterrupted run on four; an early
resume is better on three, equal on four and worse on three, where a model built from a few
pairs is a worse start than the identity.

No default changes (the warm start runs only when a previous result is passed), so no corpus
ablation; the increment is the API and this measurement.
