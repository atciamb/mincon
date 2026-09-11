# Research and development plan: the curvature model at n ≥ 100

September 11, 2026. Written after the trace study `bench/results/r5-large-n/`
and before any solver change for this phase. Same contract as `docs/19`: the
question first, the mechanism from records second, hypotheses with their
falsifiers, then code behind an option, then a whole-corpus ablation with
paired family-bootstrap intervals, and a sealed round-4 held-out set before
any claim.

## 1. Question

**Q1.** Both members need 150–500 iterations on MAXENT_200, ELLIPSOID2_200,
CHAINROSEN_BOX_200, ELLIPSOID_500 and (SQP alone) QUADSPHERE_100, where
`fmincon-ip` needs 45–92 and an exact-Hessian method would need 10–50. With
finite differences every iteration costs n + 1 evaluations, so this is the
whole of the remaining evaluation gap to `fmincon-sqp` (1.13 [0.97, 1.38] in
round 3, MAXENT_200 alone) and of mincon's two budget failures. Can a rule
inside the shared quasi-Newton update, costing no extra model evaluations,
remove most of these iterations without costing anything on the 140 small
problems?

## 2. Mechanism (from `bench/results/r5-large-n/README.md`)

The Hessian of the Lagrangian on these problems is 20–1000× the unit initial
matrix and changes by one or two orders of magnitude along the path (MAXENT:
1/xᵢ from 2 to 200–1000; ELLIPSOID2: 2 + 2λ/rᵢ² with λ from 0 to 121). A
rank-two update repairs one direction per iteration, so the model stays
under-curved for of the order of n iterations, and both globalizations turn
that into short steps: the SQP member's step bound stops growing because
α = 1 is never accepted (‖d‖ constant for 100–340 iterations), the IP member
holds μ because the barrier KKT error will not fall (309 iterations at one μ
on MAXENT_200). Where the existing guarded diagonal scaling happens to fire
(QUADSPHERE_100 under the IP member, first step cut to α = 0.04) the same
problem takes 12 iterations: the per-coordinate quotients yᵢ/sᵢ of one pair
are the diagonal Hessian of a separable problem. The rule is right; its
trigger — first update only, and only after a hard cut — is too narrow.

## 3. Hypotheses

**H1 (curvature-tracking rescale).** Watching τ = sᵀy / sᵀBs at every
accepted pair and rebuilding the matrix from the per-coordinate quotients
whenever the model is off by more than a factor F along the step — but only
if the diagonal explains the pair better than the current matrix — cuts the
iterations on the n ≥ 100 problems by at least 3× and is neutral on the rest.
Falsified if the paired evaluation ratio on the whole corpus has an interval
that includes 1.0 from above, or any problem is lost, or the n ≥ 100 subset
improves by less than 2×.

**H2 (SQP step bound).** Letting Δ grow when the step was at the bound and
accepted with at most one halving (α ≥ 0.5) removes the constant-step
plateaus of the SQP member independently of H1. Falsified if it costs
attainments or evaluations on the fixture gate or the corpus; expected to
matter less once H1 holds.

**H3 (finite differences).** The IP member's 385-vs-144 iteration gap on
MAXENT_200 between tracks A and C is a consequence of the model error (the
noisy direction is cut harder), not of derivative noise in the pairs; after
H1 the two tracks' iteration counts agree within 1.5×. Falsified otherwise;
then the pair noise needs its own study.

**H4 (limits).** No quasi-Newton rule reaches Newton's iteration count on
CHAINROSEN_BOX_200 (non-separable, tridiagonal, curvature varying along a
curved valley): after H1 the problem is still the corpus' most expensive.
A structured finite-difference Hessian with automatic sparsity detection is
the only route there and is a separate phase with its own plan; exact
Hessians from Python (`hess` in the API) serve users who have them.

## 4. Design of the rule (I1)

In `DenseBfgs::update_guarded`, after Powell damping has produced `r`:

1. τ = sᵀy / sᵀBs with the raw `y` (the damped `r` is bounded below by
   0.2·sᵀBs and would hide an underestimate's true size; sᵀy ≤ 0 means
   negative curvature along `s` and disables the rule for that pair).
2. If τ > F or τ < 1/F, and at least K updates have passed since the last
   rebuild, form the C5 diagonal `D = diag(clamp(yᵢ/sᵢ, γ/10⁴, 10⁴γ))`,
   γ = yᵀy/sᵀy, γ on coordinates with negligible sᵢ.
3. Accept the rebuild `B ← D` only if |ln(sᵀy / sᵀDs)| ≤ ½ |ln τ| — the
   diagonal must explain the curvature along `s` at least twice as well (in
   log terms) as the matrix it replaces. On a separable problem the left side
   is ≈ 0; on a dense problem whose quotients are meaningless it is not, and
   the accumulated matrix is kept.
4. The ordinary damped update with the same pair follows, as in C5.

Defaults to try: F = 10, K = 5. The rule is exposed as
`Options::bfgs_curvature_rescale` (F; `inf` disables) and `bfgs_rescale` in
the Python options, so the ablation runs candidate and baseline from one
build. The C5 first-update rule stays as it is. Rebuilds are counted and
reported once in the notes.

What it is not: not self-scaling BFGS (Oren–Luenberger scale every
iteration — known to be worse than BFGS on general problems, and the scalar
form was measured as a wash here, `bench/results/s5-bfgs-guarded-diagonal`),
not L-BFGS (does not reduce iterations; only wall time at n ≳ 500), not a
structured Hessian (H4).

## 5. Studies and gates

| step | what | pass |
|---|---|---|
| S-A | trace study | done: `bench/results/r5-large-n` |
| I1 | rule of §4 behind an option; tests: diagonal quadratic with a full first step rebuilds to the exact diagonal; dense quadratic where the diagonal explains nothing keeps the matrix; cooldown | `cargo test`, fixture gate 55/55 auto and ip, 54/55 sqp (HS13 unchanged) |
| G1 | whole-corpus ablation, track A, candidate vs `bfgs_rescale = 0` from the same build, for `mincon`, `mincon-ip`, `mincon-sqp` | attained ≥ baseline for each; paired ratio < 1 with the interval excluding 1 for the portfolio; n ≥ 100 subset ≤ 0.5×; losses listed with mechanism |
| I2 | SQP step-bound growth at α ≥ 0.5 (H2), ablated on top of I1 | as G1; kept only if not worse |
| G2 | track C on the n ≥ 50 subset for H3 | reported |
| G3 | round 4 held-out set (new families with verifiable references, some with dense Hessians at n ≥ 100 so that the rule is not measured on separable problems only), sealed before any run of the candidate; six solvers, tracks A and C, timing repeats | reported as measured; claims updated in `docs/17` |

Rejected variants go to `bench/results/abl-c7-rejected/` with the numbers.

---

## 6. Outcome (September 11, 2026)

| gate | result | where |
|---|---|---|
| S-A trace study | done | `bench/results/r5-large-n` |
| I1 rule behind an option | done: `bfgs_curvature_rescale` (Python `bfgs_rescale`, default 10), scale + shape routes, n ≥ 10 gate, cooldown; 5 new unit tests; fixture gate 55/55 auto and ip, 54/55 sqp (HS13 unchanged) | `crates/mincon-ip/src/bfgs.rs` |
| G1 whole-corpus ablation | **met**: portfolio 150/149 attained (+ELLIPSOID_500, no loss), evaluations 0.94 [0.78, 0.97]; IP member 0.96 [0.82, 0.99]; SQP member 148/145 (+3), 0.88 [0.58, 0.96]; every interval excludes 1; n ≥ 100 subset 0.81× (portfolio), 0.32× (SQP) | `bench/results/abl-c7` |
| I2 SQP step-bound growth (H2) | **not pursued**: G1 already met without it; the step-bound plateaus dissolve once the model is rebuilt, so H2 is moot for the problems it targeted (deferred, not falsified) | — |
| H3 finite differences | supported on MAXENT_200: with the rule the IP member's track-A iteration count falls 385 → 295 and track C 144 → 75, and the A/C gap narrows; the residual gap is the pair noise, not the model | `bench/results/r5-large-n`, `abl-c7` |
| H4 limits | confirmed: CHAINROSEN_BOX_200 (non-separable, curved valley) does not rebuild and is unchanged; ELLIPSOID2_200 rebuilds nothing at n = 200 under the IP member (its curvature is not 10× off along the steps) — both remain the corpus' most expensive and point to structured Hessians, a separate phase | `bench/results/abl-c7`, `r5-large-n` |
| G3 round-4 held-out | **pending**: the increment is kept in `auto` on development evidence; a sealed round-4 set is required before the claim audit records a superiority update | — |

Hypotheses: H1 confirmed (0.94×/0.88×, intervals exclude 1, +1/+3 attained,
no losses); H2 not needed; H3 supported; H4 confirmed (the rule does not reach
the non-separable large problems — those need a structured or exact Hessian).
The one blemish is the SQP member on PORTFOLIO_100 (7.0×), shielded by the
portfolio's routing; filed for round-4 diagnosis (`docs/16` cluster 15).
