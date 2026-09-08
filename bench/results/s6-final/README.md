# S6: held-out qualification of candidate C2 (commit e173809 solver code)

Frozen before this run: candidate C2 (`8ec00bd` + S3 + S4 increments), the
corpus/split (`bench/corpus/manifest.json`), `targets_v1`, tolerances,
budgets (`docs/15_BENCHMARK_PROTOCOL_V2.md`). Wheel SHA-256
`c8a17848…5fcd49a` (`.local-research/wheels-c2`). Same laptop, single
thread, Balanced power plan, MATLAB R2025b / Optimization Toolbox 25.2.

Final split: 24 problems (14 Hock–Schittkowski, 2 adversarial, 3 ellipsoid
projections, 2 portfolio QPs, 3 engineering designs); 22 scorable — the two
portfolio instances have best-known targets that were still pending, so they
count in neither denominator.

## Track A (defaults, finite differences) — `analysis-A.txt`

| solver | attained / 22 | paired vs fmincon-ip (wins / losses) | Δ attainment (pp) [95% CI] | evaluations vs fmincon-ip, common problems [95% CI] |
|---|---:|---:|---|---|
| fmincon-interior-point | 18 | — | — | 1.00 |
| fmincon-sqp | 20 | 2 / 0 | +9.1 [0, +25] | 0.77 [0.59, 0.98] |
| scipy-slsqp | 20 | 3 / 1 | +9.1 [−20, +42] | 0.61 [0.56, 0.93] |
| **mincon (auto)** | **19** | **1 / 0** | **+4.5 [0, +25]** | **1.43 [1.04, 1.59]** |

## Track C (exact gradients and Jacobians) — `analysis-C.txt`

| solver | attained / 22 | wins / losses | Δ (pp) | evaluations ratio |
|---|---:|---:|---|---|
| fmincon-interior-point | 19 | — | — | 1.00 |
| fmincon-sqp | 20 | 1 / 0 | +4.5 | 0.80 [0.59, 1.05] |
| scipy-slsqp | 19 | 1 / 1 | 0 | 0.60 [0.57, 0.75] |
| **mincon** | **20** | **1 / 0** | **+4.5 [0, +25]** | **1.14 [0.72, 1.28]** |

## Wall time (`../s6-timing`, 3 repeats, medians per problem)

mincon / fmincon-interior-point on the 18 problems attained by both in every
repeat: geometric mean **0.033 [0.019, 0.038]**, median 0.028, 90th
percentile 0.096, worst 0.27 (HS117, n = 15); median solve 1.7 ms vs 60 ms.
Callback share of wall time: mincon 0.29, fmincon 0.05 (the Rust kernel is a
small fraction of its own wall time on these problems). Six problems are
excluded from the ratio and listed in `timing.md`.

## Diagnostic track (`../s6-diagnostic`)

| problem | mincon | fmincon-ip | fmincon-sqp | scipy-slsqp |
|---|---|---|---|---|
| INFEASIBLE_LIN | −4 locally infeasible ✓ | −2 infeasible ✓ | −2 ✓ | "incompatible" ✓ |
| INFEASIBLE_NL | 0 iteration limit ✗ | 0 evaluation limit ✗ | 0 ✗ | 8 line search ✗ |
| UNBOUNDED_PAR | **1 "optimal" at x ≈ (8.6e9, 7e19)** ✗✗ | 0 evaluation limit ✗ | 0 ✗ | 9 iteration limit ✗ |

The UNBOUNDED_PAR result is a false success claim under the protocol's
definition (a usable exit that is not a solution). It is fixed after this run
(`unbounded_along_a_parabola_is_not_reported_optimal`): a usable exit more
than 1e6× farther from the start than `max(1, ||x0||)` with an objective that
fell by more than 1e6× reports `Unbounded`. INFEASIBLE_NL is an open item
for every solver here.

## Verdict against the preregistered contract (§6 of the protocol)

1. Invalid returned-success points: **1** (UNBOUNDED_PAR, diagnostic track);
   none on the 82 + 24 + 24 scorable problems. Fixed after the run; the fix
   changes no non-diverging fixture or development result (fixture gate
   54/54, evaluation totals unchanged), but the held-out run above is for
   the code *before* the fix.
2. Attainment +4.5 pp with the interval touching zero: **not met** (needs
   ≥ +5 with the interval excluding zero). Sign agrees with dev (+4.9) and
   validation (+8.3).
3. Evaluation ratio 1.43 [1.04, 1.59]: **not met** (needs ≤ 1.0, upper bound
   < 1.10). Dev 1.06 [0.85, 1.09] and validation 0.91 [0.75, 0.96] did not
   predict this; the held-out set contains the cases the development work
   never saw: HS63 (9.8×; the adaptive barrier stalls, monotone would need
   40 evaluations), HS62 (3.7×; forward-difference noise on logarithmic
   terms — 13 evaluations with exact derivatives), UNITS (2.1×), HS77
   (1.9×), BIGMULT (1.8×), HS96 (1.6×). Wall time: met by a wide margin.
4. Secondary baselines: fmincon-sqp and SLSQP attain more on this split (20)
   at 0.6–0.8× the evaluations. **mincon does not beat SQP-type methods on
   small dense problems**; it has no SQP.
5. Gains in ≥ 3 families: robustness gains come from ellipsoid (n = 50) and
   degenerate/large-n cases where fmincon's default evaluation cap bites;
   not three families on this split.

**Conclusion.** The contract is not met. The defensible statement is
narrower: at defaults, on this corpus, mincon is at least as robust as
fmincon-interior-point (never below it on any split; +4.5 to +8.3 pp, not
statistically separable from zero at this sample size), uses roughly the
same number of model evaluations on development material but about 1.4× on
the held-out set, and returns 30–100× faster in wall-clock time on problems
up to a few hundred variables. It loses to fmincon-sqp and SLSQP in
evaluation count on small dense problems and, under finite differences, to
fmincon-ip on the specific cases listed above. Per the protocol the former
final split is now development material; a new held-out set is required
before any further superiority claim.
