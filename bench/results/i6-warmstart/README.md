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

## The early resume, re-measured: no blending rule (September 13, 2026)

The table above said the warm start is worth it late and mixed early, so the open question was
whether an early resume should **blend** the handed-over model toward an identity instead of using
it raw. `bench/friction/warm_start_blend.py` measures that with no code change: `warm_start`
accepts a mapping, so a modified `hess_approx` is passed straight through. Ten problems, six cut
fractions, five arms, 60 resumes, every one of them attaining the reference (`blend.json`).

| arm | vs cold, all 60 | better / equal / worse | vs the shipped warm start | cuts at 1/3 or earlier | cuts at 1/2 or later |
|---|---:|---|---:|---:|---:|
| `warm` (shipped: the model as handed over) | **0.906** | 35 / 21 / 4 | 1.000 | 1.004 | **0.817** |
| `blend_i` = 0.5 H + 0.5 I | 0.927 | 29 / 25 / 6 | 1.023 | 1.003 | 0.856 |
| `blend_s` = 0.5 H + 0.5 (tr H / n) I | 0.977 | 26 / 24 / 10 | 1.079 | 1.019 | 0.938 |
| `scale` = (tr H / n) I | 1.018 | 16 / 26 / 18 | 1.124 | 1.074 | 0.966 |

Geometric means of total model evaluations (interrupted run plus resumed run).

**No blending rule is written.** Every blend is worse than handing the model over raw -- 1.02x,
1.08x and 1.12x overall -- and the halving is not even a wash where it was supposed to help: on
the thirty cuts at a third or earlier, `blend_i` is 1.003 against cold where the raw model is
1.004, a difference of one evaluation in three hundred, while on the thirty late cuts it gives
back a fifth of the raw model's gain (0.856 against 0.817). Dropping the shape entirely (`scale`)
is worse than cold on 18 of 60. The more of the model that survives, the better the resume, at
every cut fraction measured.

**And nothing identifies the four cuts the warm start loses.** Their statistics sit inside the
other fifty-six on every measure taken: iterations at the cut 1-35 against 0-133, condition number
5-91 against 1-9.1e7, distance from the scaled identity 0.25-0.69 against 0.00-0.85. Two of the
four are the same cut of `infeasible_start_far`, where the cold restart costs 55 evaluations --
*less than the uninterrupted solve's 99* -- so what is being measured there is a lucky cold start,
not a bad warm model; blending improves it to 99 and still loses. The third, `with_args` at 1/3,
gets **worse** under `blend_i` (103 -> 106) while the fourth, `chainrosen20` at 1/3, improves
(2793 -> 2484). A rule keyed on the model would have to fire on two of those four and not the
other two, with no statistic to tell them apart.

So the shipped behaviour stands unchanged: hand the model over whole, whenever the user passes a
previous result. Reproduce with `python warm_start_blend.py` in `bench/friction`.
