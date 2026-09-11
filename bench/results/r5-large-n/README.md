# R5: where the iterations go at n ≥ 100 (trace study, September 11, 2026)

Question (open item 1 after round 3): why do both mincon members need
150–500 iterations on the corpus' large problems where a Newton-type method
would need 10–50, and where `fmincon-ip` needs 45–92? Every record here is
candidate C6 (the round-3 build) run with `worker_python.py --trace` on the
benchmark laptop; the problems are development material (`final3` was
relabelled after round 3). Nothing was changed in the solver for this study.

## Records

| problem (n) | solver / track | iterations | `f_model` | exit | trace signature |
|---|---|---:|---:|---|---|
| MAXENT_200 (200) | mincon-ip A | 385 | 80 059 | Optimal | μ = 1.41e-4 held for 309 iterations (49–357), α < 0.1 in 200 of them |
| | mincon-ip C | 144 | 849 | StepTolerance (target attained) | μ = 2.55e-6 for 59 iterations, α < 0.1 in 55 |
| | mincon-sqp C | 391 | 1410 | Acceptable | ‖d‖ = 3.91e-6 for 339 consecutive iterations, α = 0.25 (250×) or 0.5 (130×); α = 1 accepted once |
| | mincon-sqp A | 380 | 77 678 | Optimal | ‖d‖ = 5.88e-5 for 289 iterations, α = 0.25 (290×) |
| QUADSPHERE_100 (100) | mincon-sqp C | 191 | 596 | Optimal | ‖d‖ = 1.24e-4 for 109 iterations, α = 0.5 (137×); first step α = 1 |
| | mincon-sqp A | 178 | 18 495 | Optimal | same |
| | mincon-ip A / C | 12 / 12 | 1414 / 13 | Optimal | first step cut to α = 0.042 → guarded diagonal scaling fires → Newton-like |
| MAXENT_50 (50) | mincon-sqp A | 59 | 3241 | Optimal | α = 0.25 (30×), 0.5 (17×) |
| | mincon-ip A | 77 | 4336 | Optimal | μ = 7.45e-5 for 47 iterations, α < 0.1 in 26 |
| ELLIPSOID2_200 (200) | mincon-ip C | 405 | 2172 | Optimal | μ at its floor from iteration 1; f swings 1343 → 7e-4 → 205 → 163 → 391 in the first 6 iterations; α < 0.1 in 79 |
| | mincon-ip A | 412 | 85 207 | Acceptable | same |
| | mincon-sqp C / A | 115 / 139 | 455 / 28 596 | Acceptable | penalty rises 8 → 121; α = 1 in 40 / 28 |
| CHAINROSEN_BOX_200 (200) | mincon-ip A | 496 | 100 150 | MaxReached (steady progress) | 448 full steps, f 190 → 90.6 linearly in the last 450 iterations |
| ELLIPSOID_500 (500) | mincon-ip A | 196 | 100 200 | MaxReached (steady progress) | α ≤ 1/16 for the last 100 iterations, violation 0.85 |

Reference points from round 3 (`../s6v3-final3`): on MAXENT_200 `fmincon-ip`
needs 92 iterations with exact gradients and about 45 with finite
differences and a raised cap (18 137 evaluations, `../r4-budget/fmincon-caps`);
`fmincon-sqp` 220 (C); SLSQP 125 (C). All of them are dense quasi-Newton codes
too, so the gap is not "dense BFGS at large n" as such.

## Mechanism

Every large problem in this corpus has a Hessian of the Lagrangian whose
scale is far from the unit initial quasi-Newton matrix **and changes by one
to two orders of magnitude along the path**:

* MAXENT: ∇²f = diag(1/xᵢ); the start is xᵢ = 0.5 (curvature 2), the
  solution xᵢ ≈ 0.005 (curvature 200–1000). The first full step covers that
  whole range, so even a perfectly scaled secant pair from it gives the
  *average* curvature (≈ 9), not the local one (≈ 200): a 20× underestimate.
* ELLIPSOID / ELLIPSOID2: ∇²L = 2I + λ·diag(2/rᵢ²) with λ growing from 0 to
  10–120 as the iterate reaches the ellipsoid.
* QUADSPHERE: ∇²f = diag(aᵢ), aᵢ ∈ [1, 1000], fixed — a pure scale error.
* CHAINROSEN: tridiagonal, curvature 2–1200 varying along the valley.

A rank-two BFGS update raises the model's curvature in one direction per
iteration, so from a matrix that underestimates by 20–1000× in every
direction the model needs of the order of n updates to catch up. During
those iterations the two globalizations turn the model error into short
steps in two different ways:

* **SQP member.** The QP step from an under-curved model overshoots; the
  line search cuts it; the adaptive step bound Δ (docs/20 §11.2) is
  reduced to twice the accepted step and afterwards grows only after a step
  taken *at the bound with α = 1*. With the model still wrong, α = 1 keeps
  failing the Armijo test and α = 0.5 or 0.25 is accepted: Δ never grows, the
  QP returns the same box-limited step every iteration (‖d‖ constant for
  100–340 iterations in the table) and the method degenerates into a
  box-limited gradient descent — linear convergence at a fixed step.
* **IP member.** At a fixed barrier parameter the filter line search rejects
  the overshooting step and accepts α ≈ 10⁻²–10⁻³; the barrier KKT error
  stays above κ_ε μ, so μ is not decreased (MAXENT_200 track A: 309
  iterations at μ = 1.41e-4). On ELLIPSOID2 the same model error produces
  wild early steps (f 1343 → 7e-4 → 205) and then hundreds of iterations of
  repair.

The one place where the C5 guarded diagonal scaling fires — QUADSPHERE_100
under the IP member, whose first step is cut to α = 0.04 — is the one place
where either member behaves like Newton's method (12 iterations): the
per-coordinate quotients yᵢ/sᵢ of the first pair *are* the diagonal Hessian
there. The SQP member's first step on the same problem is a full step, so
its guard never fires and it needs 191 iterations. The rule is right; its
trigger is too narrow: it looks only at the first step and only at whether
the line search cut it.

Finite differences add a second-order effect on the IP member (385 vs 144
iterations on MAXENT_200; forward-difference gradient error 1.8e-5 against
a barrier KKT threshold of 1.4e-3, so not a noise floor) that is not
resolved here; it disappears if the model error does, and is re-measured
after the increment.

## What follows (`docs/21_LARGE_N_CURVATURE_PLAN.md`)

An increment that watches the model's curvature error along every accepted
step, τ = sᵀy / sᵀBs, and when the model is off by more than an order of
magnitude rebuilds it from the per-coordinate curvature quotients (the C5
diagonal) provided that diagonal explains the pair better than the current
matrix does; and a step-bound growth rule for the SQP member that does not
require α = 1. Both are ablated on the whole corpus before either is kept.
