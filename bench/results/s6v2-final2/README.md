# S6 round 2: held-out qualification of candidate C4 on `final2`

Frozen before the run: candidate C4 (commit `1c952c0`, Windows wheel SHA-256
`dceae711…7384e`), the `final2` set (16 problems never used for any tuning:
HS25, HS56, HS84, POLYQP_10/100, DISPATCH_5/20, CATENARY_10/40,
ELLIPSOID2_20/200, QUADSPHERE2_30/300, EXPFIT2_CLEAN/NOISY, TUBULAR_COLUMN),
targets `v3` (best-known entries set from a separate exact-derivative
reference run of all four solvers; TUBULAR_COLUMN's published 26.5313 was
beaten by a verified KKT point at 26.4995 and revised with a log entry).
Same laptop, single thread, MATLAB R2025b.

## Track A (defaults, finite differences) — `analysis-A.txt`

| solver | attained / 16 | wins / losses vs fmincon-ip | Δ (pp) [95% CI] | evaluations vs fmincon-ip, common [95% CI] |
|---|---:|---:|---|---|
| fmincon-interior-point | 12 | — | — | 1.00 |
| fmincon-sqp | 13 | 1 / 0 | +6.2 [0, +20] | 0.67 [0.53, 0.83] |
| scipy-slsqp | 14 | 3 / 1 | +12.5 [−10.5, +37.5] | 0.58 [0.51, 0.68] |
| **mincon** | **14** | **2 / 0** | **+12.5 [0, +28.6]** | **0.88 [0.68, 1.10]** |

## Track C (exact derivatives) — `analysis-C.txt`

| solver | attained / 16 | wins / losses | evaluations ratio |
|---|---:|---:|---|
| fmincon-interior-point | 15 | — | 1.00 |
| fmincon-sqp | 15 | 0 / 0 | 0.95 [0.64, 1.49] |
| scipy-slsqp | 13 | 0 / 2 | 0.56 [0.49, 0.65] |
| **mincon** | **15** | **0 / 0** | **0.82 [0.62, 1.06]** |

## Wall time (`../s6v2-timing`, 3 repeats) — `timing.md`

Geometric mean mincon / fmincon-ip **0.077 [0.032, 0.234]** on the 12
problems both attained in every repeat; median 0.089; worst 5.45
(POLYQP_100: a 10,000-term Python objective that MATLAB's JIT evaluates about
five times faster — callback cost, 7,498 evaluations at ~1.5 ms). Median
solve 5.6 ms vs 109 ms.

## Per-problem notes

* HS25: every solver stops at the start (gradient ≈ 0 there); nobody attains.
* ELLIPSOID2_200 (n = 200, one quadratic constraint): mincon attains but
  needs ~800 iterations / 166k evaluations (8 s); fmincon-ip and sqp fail;
  SLSQP attains with 117k. QUADSPHERE2_300: mincon 42k evaluations, not
  attained; both fmincon algorithms and SLSQP: sqp and SLSQP attain. This is
  the dense-BFGS-from-identity cluster (atlas #4/#13): the *guarded* initial
  scaling remains the candidate fix.
* HS56 (1.76×), ELLIPSOID2_20 (1.50×, `Acceptable`), POLYQP_100 (1.24×)
  are the remaining losses in evaluation count; EXPFIT2 (0.48×),
  QUADSPHERE2_30 (0.62×), HS84 (0.77×), POLYQP_10 (0.81×) the wins.

## Verdict against the contract

| criterion | round 1 (`s6-final`, C2) | round 2 (`s6v2-final2`, C4) |
|---|---|---|
| no invalid returned success on scorable problems | met (one on the diagnostic track, fixed) | met |
| attainment ≥ +5 pp, interval excluding 0 | +4.5, touches 0 — not met | +12.5, touches 0 — **direction met, interval not** (n = 16) |
| evaluations ≤ 1.0, upper bound < 1.10 | 1.43 [1.04, 1.59] — not met | 0.88 [0.68, **1.10**] — point met, bound at the limit |
| wall time | 0.033 | 0.077 |
| beats SQP-type methods on evaluations | no | no (0.67 / 0.58) |

Across all four evaluations (dev 1.06, validation 0.91, final-1 1.43,
final-2 0.88) the evaluation ratio against fmincon-interior-point now
straddles 1.0 with mincon never less robust. The claim that survives:
**at defaults mincon is at least as reliable as fmincon-interior-point, costs
about the same number of model evaluations (within ±15% on three of four
sets, +43% on the fourth), and answers an order of magnitude faster on
problems up to a few hundred variables; fmincon-sqp and SciPy SLSQP remain
0.6–0.7× cheaper in evaluations on small dense problems.** A statistically
separable superiority claim needs a larger held-out set than 16–24 problems.
