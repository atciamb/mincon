# Evaluation-budget study (plan §4 S-C), September 11, 2026

Question: does a budget beyond fmincon's 3000 evaluations buy real solutions,
how large a budget, and how should the solver tell the user it is making no
progress instead of silently spending it?

Data: (1) the scored track-A records of every earlier run (143 scorable
problems); (2) a fresh run of the current build (C5, master `8a2d3f8` plus the
`trace` field in the Python result) over the whole corpus with per-iteration
traces, on the cloud host (`mincon-trace.A.jsonl`, `scored-A.jsonl`; wall times
are not comparable to the benchmark host and are not used); (3) fmincon at
raised caps — pending the Windows runner (§4).

## 1. Attainment as a function of the cap (mincon, current build, track A)

| cap on objective evaluations | attained (of 143) |
|---|---|
| 100·n (fmincon-sqp's default rule) | 128 |
| 200 | 103 |
| 3000 (fmincon-ip's default) | 125 |
| 15 000 | 130 |
| 30 000 | 133 |
| 100 000 (the protocol's budget) | 134 |

The nine attainments beyond 3000 evaluations are all n ≥ 50 with finite
differences (CATENARY_40 19 102, CHAINROSEN_BOX_50 13 638, CHAINROSEN_EQ_200
15 280, CHAINROSEN_EQ_50 3472, ELLIPSOID2_200 85 561, ELLIPSOID_50 6928,
LQTRAJ_200 6598, QUADSPHERE2_300 6630, QUADSPHERE_1000 16 016). The median
attained solve costs 15 gradients (evaluations ÷ (n+1)); at n = 200 one
gradient is 201 evaluations, so 3000 evaluations is 15 gradients — enough for
a quadratic, not for a curved valley. The cap is a finite-difference artefact:
in track C the same problems take 10–1000 gradient evaluations and fmincon-ip
attains most of them.

**Answer to Q2, part 1: yes — 3000 → 15 000 buys five solutions, 15 000 →
100 000 buys four more, all on large finite-difference problems.** No cap at
all buys nothing further on this corpus: the two remaining budget failures
(below) are not stalls.

## 2. What the traces show about the long runs

Per-iteration traces of every run (`mincon-trace.A.jsonl`, field `trace`):

* **CHAINROSEN_BOX_200** (not attained, 100 158 evaluations, 496 iterations):
  f falls from 50 336 to 191 in 41 iterations, then *linearly* by ≈ 9 per 41
  iterations (191 → 91), full steps (α = 1) throughout, KKT error ≈ 1. This is
  steady slow progress along a 200-dimensional curved valley with a dense
  BFGS model, not a stall; CHAINROSEN_BOX_50 shows the same linear phase and
  finishes after 260 iterations. Extrapolated, n = 200 needs ~1000 iterations
  (~200 000 evaluations). Remedy is a better curvature model (atlas cluster
  4/15), not a budget rule.
* **ELLIPSOID_500** (not attained, 100 390 evaluations, 196 iterations):
  infeasible throughout, violation 90 → 0.85 with α = 1/16 steps, monotone
  decrease. Slow, steady, not a stall. Track C attains it in 761 iterations.
* **ELLIPSOID2_200** (attained after 413 iterations): between iterations 270
  and 340 the objective oscillates within 1 % while the violation creeps from
  6e-3 to 1e-4 with α ≤ 0.03; it then converges in 40 iterations.
* **CATENARY_40, ELLIPSOID_50, HS13** (attained): each has a phase of 30–100
  iterations where neither the objective nor the violation improves by more
  than a few per cent, followed by convergence.

A window-based no-progress detector (W = 30 iterations, objective not
improved by 1e-4 relative, violation not halved) fires on CATENARY_40 (at
iteration 95 of 233), ELLIPSOID2_200 (114 of 413), ELLIPSOID_50 (87 of 128)
and HS13 (63 of 80) — four attained problems — and also on the two budget
failures. **Hypothesis H3 is falsified: any stall rule that stops the two
budget failures early also loses at least four attainments.** The filter
method's long plateaus are part of how it succeeds.

The genuine stalls in the corpus are of a different kind:

* **INFEASIBLE_NL** (diagnostic; the problem is infeasible): from iteration
  10 the restoration phase sits at the stationary point of the infeasibility
  (x = (2, 0), violation 3.0) with step norm 1e-16, for 410 more iterations,
  then reports `MaxReached`; all three portfolio members repeat this (9784
  evaluations). The restoration phase converged and did not say so. This is a
  defect (**D10**, restoration exit on a stationary infeasible point whose
  minimizer lies on a bound), not a budget question, and it is the
  "INFEASIBLE_NL not diagnosed" item of the atlas.
* **UNBOUNDED_PAR**: already caught by the far-from-start rule (`Unbounded`).

## 3. Policy decided

1. **No default evaluation cap** (keep `max_evaluations = None`, iteration cap
   3000): a cap of any size loses solutions on this corpus and no cap loses
   none. The user's cap remains a user option, as does `max_seconds`.
2. **A progress verdict at every budget exit** (`MaxReached`, user cap, or
   time): the note classifies the last window as *steady progress* (objective
   or violation still decreasing at a stated rate: "increase the budget or
   supply derivatives") or *no progress* (neither moved: "likely stuck; the
   returned point is the best feasible iterate"). Implemented as a note, not a
   new exit flag, because the classification is a heuristic and the exit
   reason is the budget.
3. **Fix D10** so that a stationary infeasible point ends the solve with
   `LocallyInfeasible` in tens, not thousands, of evaluations.
4. The remaining budget consumers (dense BFGS at n ≥ 200 with finite
   differences) are a curvature-model problem — the SQP member does not change
   that; exact/AD Hessians from Python or a limited-memory/structured update
   would. Recorded, not addressed in this phase.

## 4. Pending: fmincon at raised caps

Whether fmincon-ip itself would attain its 15 capped problems given 15 000
or 100 000 evaluations is a MATLAB experiment (solver tags
`fmincon-interior-point@MaxFunctionEvaluations=15000` and `=100000`, with
`MaxIterations=10000`; diagnostic, non-default). The MATLAB worker now accepts
such overrides; the run needs the Windows job runner, which is not alive at
the time of writing. It is scheduled for the next runner session and its
result goes in this directory as `fmincon-cap15k.A.jsonl` /
`fmincon-cap100k.A.jsonl`.
