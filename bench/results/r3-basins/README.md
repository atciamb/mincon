# Basin study (plan §4 S-B), September 11, 2026

Question: when a solver returns a feasible point whose objective is not the
published minimum, is that a solver defect or a property of the problem?

Method (`bench/harness/second_order.py`): for every non-attained track-A
record of the development, validation and both held-out rounds with a
returned point (repeat 0; budget-exhausted records excluded), recover
sign-structured multipliers on the active set from the corpus' exact
derivatives, then run Lagrange–Newton iterations on that active set with the
exact Hessian of the Lagrangian (`newton_polish`, n ≤ 100), and classify the
polished point by the smallest eigenvalue of the projected Hessian on the null
space of the strongly active constraint gradients. "Moved" is the relative
∞-norm distance from the returned point to the polished one. Files:
`classification.jsonl` (raw classification of returned points),
`polished.jsonl` (before/after polishing).

## Results

| problem | solver(s) | returned f | published | moved | verdict |
|---|---|---|---|---|---|
| HS2 | fmincon-ip, SLSQP | 4.9412 | 0.0504 | 3e-8 | **strict local minimum** (x₂ at its bound, projected curvature 1191) |
| HS16 | mincon, fmincon-sqp, SLSQP | 3.9821 | 0.25 | 3e-6 | **strict local minimum** on the boundary of c₁ (curvature 346); fmincon-ip reaches the published one from the same start |
| HS20 | all five solvers | 40.1987 | 38.1987 | 2e-8 | **strict local minimum** (vertex, two active constraints in two variables); the published value is a different vertex |
| HS55 | mincon, fmincon-sqp | 6.6667 | 6.3333 | 3e-8 | **strict local minimum** (vertex of the polytope, nonconvex objective) |
| HS55 | fmincon-ip | 6.3353 | 6.3333 | — | **not a KKT point** (stationarity 3e-2 relative; multipliers 1.8e8 on rank-deficient equalities); Newton from it goes to a saddle (f = 6.806). fmincon reported "local minimum found". |
| HS108 | mincon, fmincon-ip | −0.67498 | −0.86603 | 3e-6 | **strict local minimum** (curvature 0.12–0.28) |
| CHAINROSEN_EQ_10/50, CHAINROSEN_BOX_10 | fmincon-sqp | 3.99–5.34 | 0 | ≤1e-5 | **strict local minima** of the chained Rosenbrock problems; mincon reaches f = 0 |
| HS33 | fmincon-sqp, SLSQP | −4.0 | −4.5858 | 1e-11 | **saddle point**: KKT holds, projected Hessian has eigenvalue −0.5 in the critical cone (one weakly active constraint). Both SQP codes stopped at a saddle; mincon (interior point) attains. |
| HS25 | all solvers | 32.835 | 0 | 2e-2 | **flat start**: gradient 1e-8 and Hessian ~1e-19 at x₀; every solver stops within three iterations. Degenerate stationary region, not a basin choice. |
| HS57 | fmincon-sqp, SLSQP | 0.03065 | 0.02846 | 1.2 | **asymptotically flat**: the polish drifts along x₂ → ∞ with the projected curvature → 0; a KKT point in the limit only (atlas cluster 5) |
| RANKLOSS_JAC | fmincon-ip, fmincon-sqp, SLSQP | 1.0279 | 1.0 | 0.17 | **accuracy at a degenerate constraint** (∇h = 0 at x*): stopped 0.17 away; polishing reaches the solution. mincon attains. |
| HS13 | fmincon-ip | 1.3550 | 1.0 | — | degenerate (MFCQ fails), stopped at the step tolerance; not a KKT point |
| UNITS | fmincon-sqp, SLSQP | 5.0 | 0.5 | 4e-22 | **scaling-degenerate KKT point**: exact stationarity, projected curvature 4e-12; the published minimum is 1.5e6 units away along x₂ |
| UNITS | mincon, fmincon-ip | 5.009, 5.107 | 0.5 | — | near the same point; the residual 1.5e4 in x₁ is the forward-difference error (f″ ≈ 2e12 at h = 1.5e-8). With `scaling = none` mincon reaches f = 0.50003 (attained) in 101 evaluations — a path effect, not a certificate. |
| BADSTART_DISC | mincon | 6.3751 | 0.04567 | 0.5 | **defect (D9)**: not a KKT point (exact gradient (87, −50), constraint slack 3.4e-3, reported multiplier 2165). Reproduced with the wheel: the gradient-based objective scale is 2.5e-10 (set from ‖∇f(x₀)‖ ≈ 4e11 at x₀ = (1000, −1000)), so the scaled KKT test accepts an unscaled stationarity residual of 87. With `scaling = none` mincon attains the published minimum (0.0456749, 208 evaluations). |
| BADSTART_DISC | fmincon-sqp | 1.2407 | 0.04567 | — | stopped at its default cap of `100·n` = 200 evaluations |
| QUADSPHERE2_300 | mincon | 0.038453 | 0.037773 | — | same mechanism as D9 in a milder form: objective scale 0.05 and finite-difference error (a_i up to 1e3) make the scaled test accept a 6.8e-4 relative stationarity residual; `scaling = none` gives f = 0.0377748 (attained) at 2.8× the evaluations |

Reference points: every published `ref_x` in the corpus that was checked
classifies as a strict local minimum or as a degenerate KKT point of the kind
described above; no target was found to be wrong.

## Answers

1. **Landing in a different basin is not a defect** when the returned point is
   a KKT point with positive projected curvature: HS2, HS16, HS20, HS55
   (vertex), HS108 and fmincon-sqp's chained-Rosenbrock points are all strict
   local minima. Which basin a local method reaches from a given start is a
   property of the path, and the published value is often just the best of
   several minima Hock and Schittkowski knew about. These are correctly
   reported as "not attained" by the protocol and should not be reported as
   failures of the solver; the honest thing a solver can say is "local
   minimum" — which is what every code, fmincon included, says.

2. **Two returned points were not KKT points and were reported as optimal:**
   mincon on BADSTART_DISC (defect D9, mechanism established, fix to be
   measured) and fmincon-ip on HS55 (fmincon's report; degenerate
   constraints). Both are false success claims, and only the oracle caught
   them. D9 also explains mincon's premature stop on QUADSPHERE2_300.

3. **SQP codes with a positive-definite quasi-Newton matrix can stop at a
   saddle** (HS33: fmincon-sqp and SLSQP). An SQP member for mincon needs
   this in its design: at termination with exact or finite-difference
   curvature available, a negative-curvature check on the critical cone, or
   at least a note that no second-order test was made.

4. **Flat and asymptotically flat problems (HS25, HS57, UNITS)** are
   tolerance phenomena: the first-order conditions hold to 1e-8 at points far
   from the published minimum. The remedy is a note, not a different answer:
   "converged at the starting point" (HS25: zero iterations with a tiny
   gradient), "iterates far from the start" (HS57, already implemented),
   "problem badly scaled" (UNITS: the derivative error estimate exceeds the
   gradient in some coordinate).

5. **Multi-start is not justified as a default** by this corpus: the six
   basin cases are 4 % of the problems, every solver shares most of them, and
   a restart policy would cost evaluations on the other 96 %. Offered as an
   option later if a user asks; not part of this phase.

## Follow-up recorded in the plan

* D9 (termination in scaled units only): fix and ablate before the SQP work;
  candidates are an unscaled relative stationarity guard and a drift-triggered
  rescale. Regression fixtures: BADSTART_DISC from (1000, −1000) must not
  report `Optimal` at f = 6.375; QUADSPHERE2_300 must attain.
* Atlas cluster 6 to be corrected: BADSTART_DISC moves from "legitimate local
  minimum" to D9; HS55 (fmincon-ip) recorded as a false claim by fmincon.
* SQP design: second-order check at termination (HS33).
