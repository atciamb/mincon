# Round 5 held-out qualification (final5), September 13, 2026

Thirteen problems in seven families sealed before any candidate run (`SEAL.md`: pkfit clean and
noisy, logistic clean and noisy, tcport 20/100, deconv 60/200, noisyqp 20, ncboxqp 8/12,
corrugated bulkhead, I-beam). Seven solvers, tracks A (finite differences, the minimal-input
first try) and C (exact derivatives), single thread, 60 s / 100 000 evaluations, targets v6,
benchmark laptop. Candidate: the tree of commit `f0d847d` (round-5 I1-I9, S-E, the
quadratic-program probe with its structured Hessian build, the no-scale note, I6), wheel
`0c47a714…`. Then a timing run with three repeats for mincon, fmincon-interior-point and
fmincon-sqp.

Everything below is as measured. Every claim in this file was drafted from the records and then
attacked by an independent adversarial pass over the same records; §8 lists what that pass
struck out, including things this project would have liked to claim.

**The short version.** On the metric this project puts first — attain the target on the first
try with minimal inputs — mincon is **12/13, behind fmincon-sqp, SciPy SLSQP and mincon's own
SQP member (all 13/13) and ahead of fmincon-interior-point (10/13) and trust-constr (11/13)**,
with every paired interval touching zero. With exact derivatives mincon is 13/13. mincon
claimed no false certificate anywhere. Its evaluation cost is not resolved against fmincon-sqp,
is below fmincon-interior-point's on the problems that solver finished, and is **clearly above
SLSQP's on both tracks**. Wall time on this cheap-model set is 0.28× fmincon-sqp's and 0.05×
fmincon-interior-point's, but that comparison is confounded by the model, not the solver
(§5). The round's own increment, the quadratic-program probe, **changed no attainment outcome
in this run**.

## 1. Attainment

Attained = feasible to 1e-6 and the objective within the frozen tolerance of the sealed target,
judged by the oracle, not by the solver.

| track | mincon | mincon-ip | mincon-sqp | fmincon-ip | fmincon-sqp | SLSQP | trust-constr |
|---|---:|---:|---:|---:|---:|---:|---:|
| A (finite differences) | 12/13 | 12/13 | 13/13 | 10/13 | 13/13 | 13/13 | 11/13 |
| C (exact derivatives) | 13/13 | 12/13 | 13/13 | 13/13 | 13/13 | 13/13 | 12/13 |

Paired family-bootstrap differences for mincon on track A: +15.4 pp [+0.0, +33.4] against
fmincon-interior-point, −7.7 pp [−23.1, +0.0] against fmincon-sqp and against SLSQP, +7.7 pp
[+0.0, +23.1] against trust-constr. **Every interval has an endpoint pinned at zero: no
attainment difference in this round is resolved**, neither the ones that flatter mincon nor the
ones that do not. The resampling unit is the seven families, not the thirteen problems.

Three qualifications belong with that table, all of them unfavourable:

* **The +2 over fmincon-interior-point is fmincon's factory default, not a capability gap.**
  All three of its track-A misses exit on `options.MaxFunctionEvaluations = 3.000000e+03` after
  3015, 3050 and 3034 model evaluations and 3.1 s, 0.2 s and 0.8 s — about 3 % of the wall
  budget and 3 % of the evaluation budget the harness offered. The MATLAB worker never applies
  the protocol's evaluation budget at all (`worker_matlab.m` takes `maxfev` and drops it), so
  fmincon is the only solver here running at stock defaults. On track C, where those defaults
  are enough, fmincon-interior-point goes to 13/13.
* **The solvers were not held to the same stopping rule.** mincon: 100 000 evaluations and 60 s.
  fmincon: MATLAB defaults plus a 60 s deadline. SLSQP: `maxiter=1000`, no evaluation cap and no
  time limit. trust-constr: `maxiter=3000`, likewise uncapped — it ran 210.7 s on DECONV_200
  before the supervisor killed it at the 210 s hard limit, 3.5× the budget mincon and fmincon
  were held to. In evaluation terms the SciPy baselines had the loosest budget, not the
  tightest.
* **The ordering is fragile to the frozen tolerance.** The oracle's gap is
  `(f − target) / max(1, |target|)`, and eight of the thirteen targets have `|f*| < 1`, so the
  1e-4 bar is effectively absolute. At 2e-4 mincon is 13/13 and the gap to fmincon-sqp and
  SLSQP disappears; at 5e-5 SLSQP falls to 11/13 and mincon is level with fmincon-sqp. A factor
  of two either way removes or inverts the ordering.

**No false certificate by mincon, on a small sample.** Across both tracks the mincon family had
three chances to claim success at a point the oracle rejects and took none;
fmincon-interior-point had three and took none either, so this run does not separate them. The
run's only false certificate is `scipy-trust-constr` on TCPORT_100, on both tracks: "`gtol`
termination condition is satisfied", `reported_success=true`, feasible, at an objective 1.2e-4
above the optimum of a problem whose minimum is unique because it is convex.

Certification is not attainment, and mincon is not clean here: on track A three of its twelve
successes carry no first-order certificate of either kind (LOGISTIC_CLEAN, PKFIT_CLEAN,
PKFIT_NOISY2). The harness's "KKT (supplied)" column understates every mincon build — it passes
no multipliers for the seven bound-only problems, so that certificate is never attempted, while
fmincon is scored with multipliers on all thirteen. On the solver-independent measure, the
oracle's own multiplier recovery, mincon leads: 8/13 on track A and 12/13 on track C.

## 2. Per problem, track A (attained, objective + constraint evaluations)

| problem | n | m | mincon | mincon-ip | mincon-sqp | fmincon-ip | fmincon-sqp | slsqp | trust-constr |
|---|---:|---:|---|---|---|---|---|---|---|
| CORRUGATED_BULKHEAD | 4 | 6 | Y 188 | Y 240 | Y 184 | Y 276 | Y 244 | Y 141 | Y 710 |
| DECONV_200 | 200 | 0 | n 16885 | n 14272 | Y 14472 | n 3015 | Y 20100 | Y 13066 | n (killed) |
| DECONV_60 | 60 | 0 | Y 1552 | Y 8015 | Y 7955 | n 3050 | Y 6039 | Y 2562 | Y 19520 |
| I_BEAM | 4 | 2 | Y 246 | Y 370 | Y 242 | Y 404 | Y 242 | Y 231 | Y 4680 |
| LOGISTIC_CLEAN | 3 | 0 | Y 123 | Y 106 | Y 119 | Y 102 | Y 98 | Y 51 | Y 120 |
| LOGISTIC_NOISY2 | 3 | 0 | Y 169 | Y 137 | Y 165 | Y 123 | Y 98 | Y 51 | Y 368 |
| NCBOXQP_12 | 12 | 0 | Y 201 | Y 313 | Y 104 | Y 379 | Y 117 | Y 104 | Y 455 |
| NCBOXQP_8 | 8 | 0 | Y 114 | Y 160 | Y 63 | Y 171 | Y 72 | Y 63 | Y 405 |
| NOISYQP_20 | 20 | 0 | Y 565 | Y 538 | Y 561 | Y 1092 | Y 560 | Y 300 | Y 63819 |
| PKFIT_CLEAN | 4 | 1 | Y 334 | Y 268 | Y 330 | Y 332 | Y 308 | Y 251 | Y 1330 |
| PKFIT_NOISY2 | 4 | 1 | Y 334 | Y 268 | Y 330 | Y 352 | Y 300 | Y 251 | Y 850 |
| TCPORT_100 | 100 | 2 | Y 5882 | Y 5874 | Y 3864 | n 6068 | Y 3654 | Y 3336 | n 7474 |
| TCPORT_20 | 20 | 2 | Y 702 | Y 932 | Y 694 | Y 1512 | Y 644 | Y 633 | Y 1470 |

Ten of the thirteen problems are attained by all seven solvers on track A and eleven on track C.
Outside the deconv and tcport families nothing separated anybody: pkfit, logistic, ncboxqp,
noisyqp and engineering3 — five of the seven families — are 9/9 for every solver on both tracks.

One number puts the budget asymmetry in proportion: over all thirteen track-A problems
fmincon-interior-point spent 12 404 objective evaluations in total, fewer than the 16 885 mincon
spent on DECONV_200 alone.

## 3. Per problem, track C (objective + constraint evaluations, then gradient + Jacobian calls)

| problem | n | m | mincon | mincon-ip | mincon-sqp | fmincon-ip | fmincon-sqp | slsqp | trust-constr |
|---|---:|---:|---|---|---|---|---|---|---|
| CORRUGATED_BULKHEAD | 4 | 6 | Y 53 + 30 | Y 49 + 44 | Y 49 + 30 | Y 92 + 92 | Y 100 + 100 | Y 28 + 28 | Y 142 + 142 |
| DECONV_200 | 200 | 0 | Y 13 + 205 | n 202 + 199 | Y 205 + 203 | Y 515 + 515 | Y 401 + 401 | Y 66 + 65 | Y 506 + 506 |
| DECONV_60 | 60 | 0 | Y 12 + 64 | Y 141 + 130 | Y 151 + 149 | Y 198 + 198 | Y 149 + 149 | Y 42 + 42 | Y 183 + 183 |
| I_BEAM | 4 | 2 | Y 57 + 46 | Y 75 + 70 | Y 53 + 46 | Y 92 + 92 | Y 50 + 50 | Y 46 + 46 | Y 1852 + 1852 |
| LOGISTIC_CLEAN | 3 | 0 | Y 49 + 24 | Y 32 + 24 | Y 45 + 24 | Y 28 + 28 | Y 41 + 41 | Y 18 + 11 | Y 23 + 23 |
| LOGISTIC_NOISY2 | 3 | 0 | Y 56 + 28 | Y 40 + 28 | Y 52 + 28 | Y 31 + 31 | Y 41 + 41 | Y 18 + 11 | Y 31 + 31 |
| NCBOXQP_12 | 12 | 0 | Y 17 + 21 | Y 26 + 23 | Y 10 + 8 | Y 30 + 30 | Y 9 + 9 | Y 8 + 8 | Y 35 + 35 |
| NCBOXQP_8 | 8 | 0 | Y 16 + 16 | Y 25 + 16 | Y 9 + 7 | Y 19 + 19 | Y 7 + 7 | Y 7 + 7 | Y 45 + 45 |
| NOISYQP_20 | 20 | 0 | Y 40 + 26 | Y 38 + 24 | Y 36 + 26 | Y 84 + 84 | Y 36 + 36 | Y 23 + 15 | Y 84 + 84 |
| PKFIT_CLEAN | 4 | 1 | Y 137 + 64 | Y 77 + 60 | Y 133 + 64 | Y 84 + 84 | Y 100 + 100 | Y 66 + 46 | Y 5992 + 5992 |
| PKFIT_NOISY2 | 4 | 1 | Y 137 + 64 | Y 81 + 60 | Y 133 + 64 | Y 88 + 88 | Y 92 + 92 | Y 66 + 46 | Y 5992 + 5992 |
| TCPORT_100 | 100 | 2 | Y 67 + 54 | Y 59 + 54 | Y 59 + 36 | Y 86 + 86 | Y 54 + 54 | Y 24 + 22 | n 78 + 78 |
| TCPORT_20 | 20 | 2 | Y 57 + 30 | Y 45 + 40 | Y 49 + 30 | Y 72 + 72 | Y 44 + 44 | Y 22 + 20 | Y 68 + 68 |

## 4. Evaluations, and what the metric does and does not count

The harness's cost metric is objective + constraint evaluations at the model boundary. Two
things distort it, in opposite directions, and both are disclosed here rather than resolved:

* **On track C it ignores derivative calls.** mincon's quadratic probe pays for its Hessian in
  gradient calls, so on DECONV_200 the metric credits mincon with 13 model calls for 218 actual
  ones — a 16.8× understatement. Adding gradient and Jacobian calls is only a fair correction
  **between the two Python solvers**: the MATLAB worker computes the gradient on every objective
  call without checking `nargout`, so every fmincon record is charged exactly 2× its
  objective count by construction, which measures the wrapper and not the solver.
* **On both tracks it applies a one-point cache** that charges nothing for an immediate
  re-request at the same point. The rule is applied to every worker, but only mincon benefits:
  over the thirteen track-A problems it suppresses 11.4 % of mincon's model crossings (30 821 →
  27 295), 10.2 % of mincon-ip's and 7.9 % of mincon-sqp's, against 0.6 % for SLSQP and 0.0 %
  for both fmincon arms and trust-constr — and 34 % on TCPORT_100 (8930 → 5882).

Geometric mean over the problems both solvers attain, 95 % family bootstrap.

| track | pair | attained | objective + constraint calls | including derivative calls | n common |
|---|---|---|---|---|---:|
| A | mincon / fmincon-ip | 12 / 10 | 0.75 [0.56, 0.99] | identical (no derivatives supplied) | 10 |
| A | mincon / fmincon-sqp | 12 / 13 | 1.08 [0.74, 1.39] | identical | 12 |
| A | mincon / SLSQP | 12 / 13 | **1.52 [1.13, 2.04]** | identical | 12 |
| A | mincon-sqp / fmincon-sqp | 13 / 13 | 1.03 [0.93, 1.18] | identical | 13 |
| C | mincon / fmincon-ip | 13 / 13 | 0.57 [0.21, 1.18] | not comparable (see above) | 13 |
| C | mincon / fmincon-sqp | 13 / 13 | 0.78 [0.29, 1.49] | not comparable | 13 |
| C | mincon / SLSQP | 13 / 13 | 1.55 [0.78, 2.46] | **1.85 [1.47, 2.27]** | 13 |
| C | mincon-sqp / fmincon-sqp | 13 / 13 | 1.02 [0.83, 1.20] | not comparable | 13 |

What survives: **against SciPy SLSQP mincon is measurably more expensive on both tracks**, and
the fuller count makes the track-C loss significant rather than inconclusive. Against fmincon
nothing is resolved except the track-A ratio to fmincon-interior-point, which is computed only
on the ten problems that solver finished inside its own default cap — every one of them with
n ≤ 20 — so it is a small-n, unequal-budget figure. The track-A ratio to fmincon-interior-point
also changes verdict across bootstrap seeds in 8 % of draws; the seed is hardcoded to 0.

## 5. Wall time (three repeats, median per problem)

| pair | geo-mean ratio [95 % family bootstrap] | median wall | model-callback share | excluded |
|---|---|---|---|---|
| mincon / fmincon-sqp | 0.28 [0.11, 0.78] | 4.1 ms vs 31.6 ms | 0.80 vs 0.43 | DECONV_200 |
| mincon / fmincon-interior-point | 0.05 [0.027, 0.107] | 3.5 ms vs 80.0 ms | 0.76 vs 0.07 | DECONV_200, DECONV_60, TCPORT_100 |

This set is in the cheap-model regime (median ~16 µs per model evaluation against round 4's
~287 µs), which is the regime that produced the older 30–100× figures, not round 4's
model-bound one. Four reasons not to read these as solver speed:

* 76–80 % of mincon's wall is inside the Python model callback, and the same SymPy-derived model
  costs up to 8.8× **more** per evaluation in its Python form than its MATLAB form at n ≥ 60
  while costing 5–7× **less** at n ≤ 8. Both geometric means sit on top of that confound.
* The MATLAB baselines' per-problem wall varies up to 19.3× across the three repeats
  (fmincon-sqp on LOGISTIC_NOISY2: 11.4 / 14.2 / 219.5 ms) against at most 1.30× for mincon.
  `timing.py`'s docstring promises to flag the first problem of a MATLAB worker for JIT warm-up
  and the code does not; TCPORT_20 was that first problem in two of three repeats for
  fmincon-interior-point, at 2.5× the time it recorded when it was not.
* The exclusion rule against fmincon-interior-point removes both of the problems where mincon is
  slow, leaving ten problems with a median n of 4. The 0.05× figure is a claim about n ≤ 20.
* mincon's wall excludes a model-build phase (25.2 s for DECONV_200, 7.6 s for TCPORT_100) that
  the MATLAB side never reports.

Where mincon loses on wall it is not the solver that is slow: on TCPORT_100 (4.67× fmincon-sqp)
and DECONV_60 (1.29×), mincon's own solver time is 49 ms against 131 ms and 6 ms against 133 ms.

## 6. Every failure in the run

Nine target misses out of 182 records, on three problems in two families.

* **mincon and mincon-ip, DECONV_200, track A** — the interior-point path stalls, and the clock
  ends it. Both return a feasible point (the problem has bounds only, so feasibility is free)
  at an objective 0.54 % and 0.56 % above the optimum, 1.8× and 1.9× the frozen bar. mincon's
  own notes say so: *"Budget exhausted with no measurable progress over the last 20 iterations
  … the solver is likely stuck."* The oracle's recovered stationarity is 1.514e-2 for both — and
  **1.514e-2 again for mincon-ip on track C**, where it stops voluntarily after 1.9 s of its
  60 s with exact derivatives and still misses by 1.18e-4. So this is an interior-point stall on
  an ill-conditioned bounded least-squares problem, not merely a resource exit. mincon's exit
  message, "Iteration or evaluation limit reached", is also wrong about which limit bound: 16 885
  of 100 000 evaluations and 78 of 2400 iterations were used, and the 60 s clock is what
  stopped it. There is no machine-readable field that distinguishes the three limits.
* **mincon's portfolio was beaten by its own SQP member on that problem.** The notes record that
  `ip-default` ran first (the n > 20 order), consumed the whole 60 s, and that `sqp`,
  `ip-cautious` and `ip-unscaled` were each "skipped: time budget exhausted by earlier members".
  Run standalone under the same budget, mincon-sqp attains it. The paired difference is +7.7 pp
  [+0.0, +23.1] on one problem and reverses on track C, so it is not a resolved result — but the
  mechanism is real and is filed as a follow-up.
* **fmincon-interior-point, track A, three misses** — its default 3000-evaluation cap, as above.
  Two of the three are far from the answer when it fires: DECONV_200 at 223 % relative error and
  DECONV_60 at 32 %.
* **scipy-trust-constr** — killed by the supervisor at 210.7 s on DECONV_200/A with no returned
  point at all, and the run's only false certificate on TCPORT_100 (both tracks).

**Nobody converged on DECONV_200 under finite differences.** No returned point passes the
oracle's first-order test; recovered stationarity runs from 1.33e-4 (fmincon-sqp) to 4.08e-2
(fmincon-interior-point). The three solvers scored as attaining it were all cut short by a
budget and happened to land inside the band: fmincon-sqp by 7.5e-5, mincon-sqp by 9.6e-5 and
SLSQP by 9.97e-5 — the last with 0.3 % of the tolerance to spare. Every track-A 13/13 in §1
rests on that.

## 7. What the round's own increment did

The quadratic-program probe fired on three of the twenty-six mincon auto records: DECONV_60 on
track A, and DECONV_60 and DECONV_200 on track C. **It changed no attainment outcome anywhere
in this run** — every problem it fired on was attained by most solvers regardless. Where it
fired it bought precision and iterations: DECONV_60/A in 2 iterations and 1552 evaluations
(1362 of them the structured Hessian build at half-bandwidth 27) against the quasi-Newton
path's 7955 and fmincon-sqp's 6039; DECONV_200/C certified at a gap of −5.0e-14 with recovered
stationarity 8.6e-11, three orders better than any other solver on that problem. It did not make
mincon cheapest even there: SLSQP attained DECONV_200/C in 0.83 s and 131 model calls against
mincon's 0.99 s and 218. On DECONV_200 track A the probe declined and left no note saying why —
the decline reason is emitted only under `MINCON_QP_DEBUG=1`. Reproduced outside the harness,
the reason is the probe's wall guard: no banded Hessian is found within two off-diagonal bands,
and the dense continuation is priced at about 19 900 further evaluations, which cannot fit in
half of a 60 s budget. Given 600 s the same solve builds the dense Hessian and certifies in 5
iterations and 6639 evaluations (42.7 s, gap 2.2e-9).

## 8. What this set is, and what the adversarial pass struck out

**Two of the thirteen targets are published roundings**, so they impose a floor on the
achievable gap: no solver can score better than 8.5e-6 on CORRUGATED_BULKHEAD or 1.5e-6 on
I_BEAM, spending 8.5 % and 1.5 % of the tolerance on the reference's last significant figure.
Worse, and recorded here as an error in the sealed set: **the reference *point* stored for
CORRUGATED_BULKHEAD, (57.692, 34.148, 57.555, 1.05), evaluates under the corpus's own model to
f = 6.846036, a gap of 4.6e-4 — 4.6× the tolerance the problem is scored at.** The generation
assertion allowed 5e-3 and so did not catch it. Nothing in this run is affected, because scoring
uses the target value and not the point, but the point as recorded is not accurate enough to be
used as a reference and should be recomputed before final5 is reused.

**DECONV is the friction problem `box_lsq` with only `n` and the seed changed** — the same
Gaussian kernel, row normalisation and block pattern — and `box_lsq` is the specific problem the
round's quadratic-probe and structured-build work was debugged against. That family carries most
of the track-A separation in this run. It is named here so a reader can discount it; it should
not have been the set's only discriminating family.

**NCBOXQP is well built but did not discriminate.** Re-enumeration confirms 6 local minima at
n = 8 and 139 at n = 12, with exactly one inside the scoring window, and a plain L-BFGS-B from
the same start lands in the second-best minimum at n = 8 — but all seven benchmark solvers found
the global optimum on both tracks. **TCPORT and NCBOXQP exist because a mincon feature exists,
and both charge mincon rather than crediting it** (the probe declines each, after paying for the
attempt), which is the right way round.

**The two "noisy" fits are not noisy problems**: 2 % noise is added to the data, so the objective
stays smooth, deterministic and closed-form. NOISYQP_20's 1e-7 perturbation is genuinely active
in the free block but is about three orders of magnitude inside the scoring tolerance. Neither
family tested what its name suggests.

Struck out by the adversarial pass, and therefore **not claimed**: that the derivative-inclusive
count shows mincon cheaper than fmincon on track C (it measures the MATLAB wrapper's
unconditional gradient, not the solver); that mincon's DECONV_200 miss is a mere resource exit
(it is a stall the solver itself diagnoses); that the probe won anything in this round (it
changed no outcome); that mincon's clean false-certificate record distinguishes it from
fmincon-interior-point here (both had three opportunities and took none).

After this run final5 is development material, like its predecessors.

## 9. One increment landed after this run

`quadratic_rows='values'` (commit `3aac1e5`, `bench/results/abl-i5-rows`) became the default
*after* the run above, and two of the sealed problems (TCPORT_20, TCPORT_100) are in the family it
touches, so the mincon arm was re-run at HEAD and the difference is reported in
`../s6v5-final5-rows/README.md`: **no attainment or status change on either track**
(+0.0 pp [+0.0, +0.0]), cost 1.00 [1.00, 1.01] on track A and 0.93 [0.79, 1.00] on track C, with
the option-off arm reproducing these records bit for bit. That is a disclosed **second look** at a
sealed set, so it is filed as a disclosure and not as a claim; the numbers in this file and in
`docs/17` items 19-21 stand as first measured, and a further claim needs a new sealed round.
