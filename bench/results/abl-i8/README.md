# abl-i8: variable scaling from the start (H8), whole-corpus ablation, September 12, 2026

Increment I8 of `docs/22`: `ScaledNlp` solves in `x / d` with `d_i = max(|x0_i|, typical_i)`
(1 where both vanish), chain-ruling every derivative and mapping the solution and the bound
multipliers back. `Options::scale_variables` is `Off` (the default before this ablation), `Auto`
(scale only when the factors span at least 1e4, i.e. when the start says the units are
mismatched) or `On`. Mechanism and the records that motivated it: `bench/results/s7-friction`
(bad_scaling, where every solver misses the optimum and no first-order certificate can see the
gradient along the large variable) and the corpus problem UNITS (`abl-i1`).

Run: track A, all 169 non-diagnostic corpus problems, defaults otherwise, single thread,
60 s / 100 000 evaluations, targets v5, wheel `wheels-i8` (the I1 + I2 + I3 + I4 + I7 + I8 tree;
the derivative check does not run without supplied derivatives, so the only difference from the
`abl-i2` wheel on this track is I8). Two arms, `mincon@scale_variables=auto` and
`mincon@scale_variables=true`, against the current default's records (`abl-c7` and `s6v4-final4`
rule-off arms).

| arm | attained candidate / baseline (of 166 scorable) | cost ratio [95 % family bootstrap] | records that changed |
|---|---|---|---|
| `Auto` | 158 / 158 | 1.01 [1.00, 1.02] on 158 | three: the two 60 s budget exits (COVQP_300, DENSELAP_250) and **HS117**, the only corpus problem whose start spans a factor 1e4, attained either way at 5010 evaluations against 612 (the scaling by the start's magnitudes does not match the solution's there) |
| `On` | 151 / 158 | 1.09 [1.00, 1.25] on 151 | 112 of 169; **seven attainments lost** (QUADSPHERE_1000, QUADSPHERE2_300, POLYQP_100, SNL_150, OBSTACLE_500 already unattained, and two more), attainment difference -4.2 pp [-11.3, -2.1]; wins on OBSTACLE_50/200, QUADSPHERE2_30, WELDED_BEAM, PRESSURE_VESSEL, losses on NNLS_SIMPLEX_*, QUADSPHERE_100 (23x), PORTFOLIO_*, SNL_60 |

Verdict: **`On` is falsified as a default** (it loses attainments with an interval excluding
zero: scaling every variable by its start's magnitude is wrong whenever the start's magnitudes
do not match the solution's). **`Auto` stands** and becomes the default: it changes one corpus
record (HS117, still attained, 8x the evaluations), loses nothing, and is the only thing in the
audit that solves the unit-mismatched start (bad_scaling: 27 evaluations, `s7-friction` `-i8-auto`
run; `TORTURE_UNITS` is now an `Expect::Optimum` fixture). The corpus problem UNITS starts at
the zero vector, which carries no scale, and stays unsolved. H8 as written in `docs/22` §3 was
"opt-in until ablated; falsified if it loses any corpus attainment when turned on for the whole
corpus": the `On` form is falsified, the `Auto` form is not.

Files as in `abl-i1` (`cmp-auto`, `cmp-on`, `scored-auto`, `scored-on` and their analyses).
