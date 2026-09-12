# Rejected: shape-route guard for the C7 curvature-tracking rebuild

September 11, 2026, evening, after the review of `abl-c7`. Probe only (eight
problems, track A, defaults, single thread), not a whole-corpus ablation: the
variant was falsified on the problems it was written for, so it never reached
gate G4 of `docs/21` §7.

## What was tried

The C7 rule rebuilds the quasi-Newton matrix from the per-coordinate
quotients `yᵢ/sᵢ` when one of two tests admits it. The *shape* test validates
the model's own diagonal (it explains the pair at least twice as well as the
full matrix, in log terms) and then installs the quotients, a matrix the test
never examined. The guard tried here: in a shape-route rebuild, keep the
model's diagonal entry wherever the quotient disagrees with it by more than
the factor (10). Route counters (`DenseBfgs::rebuild_routes`, reported in the
notes as "k by scale, j by shape") were added at the same time and are kept.

## Result — guard on vs off, same build

Objective evaluations (`f_model`) and iterations; `-` means no rebuild.

| problem | member | off: rebuilds | off: iterations / f_model | on: rebuilds | on: iterations / f_model |
|---|---|---|---:|---|---:|
| PORTFOLIO_100 | SQP | 1 scale | 175 / 17 793 (`Acceptable`) | 1 scale | **identical** |
| MAXENT_200 | SQP | 1 scale, 1 shape | 46 / 9 600 | 1 scale, 1 shape | identical |
| ELLIPSOID2_200 | SQP | 1 scale, 1 shape | 64 / 13 240 | 2 shape | **118 / 24 299** |
| QUADSPHERE_100 | SQP | 1 scale | 2 / 303 | 1 scale | identical |
| ELLIPSOID_500 | SQP | 1 scale | 63 / 32 231 | 1 scale | identical |
| MAXENT_200 | IP | 1 scale, 1 shape | 59 / 12 382 | 1 scale, 2 shape | **96 / 19 953** |
| ELLIPSOID2_200 | IP | 1 shape | 424 / 88 167 | 1 shape | identical |
| ELLIPSOID_500 | IP | 1 shape | 123 / 62 878 | 3 shape | **138 / 70 522** |
| HS114 | IP | 1 shape | 105 / 1 381 (`StepTolerance`) | 1 shape | 105 / 1 456 (`Optimal`) |
| HS118, DISPATCH_20 | both | scale only | | | identical |

## Why it is rejected

* PORTFOLIO_100's rebuild — the blemish the guard was written for — comes
  through the **scale** route, so the guard cannot touch it. The mechanism in
  the `abl-c7` README ("repeated rebuilds", "shape test's weight threshold")
  was wrong on both counts and has been corrected there: one rebuild, scale
  route.
* Where the guard does act (shape-route rebuilds on MAXENT_200 under the IP
  member, ELLIPSOID2_200 under the SQP member, ELLIPSOID_500 under the IP
  member) it makes things worse by 1.1–1.9×: the quotients it refused to
  install *were* the right local curvature — on those problems the shape test
  fires when the diagonal is roughly right in scale and the off-diagonal is
  wrong, and the quotients refine the diagonal further.

The option was removed from the code; the route counters stay. The open
question is the scale route on a dense coupled problem, `docs/21` §7.2.

## Files

`mincon-{ip,sqp}.jsonl` guard off (the C7 default with route counters),
`mincon-{ip,sqp}_bfgs_shape_guard_true.jsonl` guard on. Same wheel, built from
the working tree at the time (guard behind an option, since removed).
