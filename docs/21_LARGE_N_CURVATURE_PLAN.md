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
| G1 whole-corpus ablation | **met on attainment and on the whole-corpus ratio; not met on the n ≥ 100 subset criterion**: portfolio 150/149 attained (+ELLIPSOID_500, no loss), evaluations 0.94 [0.78, 0.97]; IP member 0.96 [0.82, 0.99]; SQP member 148/145 (+3), 0.88 [0.58, 0.96]; every interval excludes 1. The §5 pass condition also asked for the n ≥ 100 subset at ≤ 0.5×: over the problems both arms attain it is 0.81× for the portfolio and the IP member (ten problems; the newly attained ELLIPSOID_500 is outside that ratio) and 0.32× for the SQP member. The rule is kept on the attainment and whole-corpus evidence, with that criterion recorded as failed for two of three configurations rather than relaxed silently | `bench/results/abl-c7` |
| I2 SQP step-bound growth (H2) | **not pursued**: G1 already met without it; the step-bound plateaus dissolve once the model is rebuilt, so H2 is moot for the problems it targeted (deferred, not falsified) | — |
| H3 finite differences | **supported by G2** (run after the review, §7): MAXENT_200 under the IP member 59 iterations on track A vs 49 on track C (C6: 385 vs 144); over the 18 problems with n ≥ 50 the A/C iteration ratio has median 1.00 and maximum 1.54. An earlier version of this row quoted 385 → 295 (A) and 144 → 75 (C) from an intermediate build that is not in the repository; those figures were withdrawn before G2 was run | `abl-c7` (A), `g2-track-c` (C) |
| H4 limits | **partly confirmed**, and the first version of this row was wrong on the facts. The solver notes in `abl-c7` show ELLIPSOID2_200 rebuilt once under the IP member (412 → 424 iterations, 1.035×) and twice under the SQP member (139 → 64 iterations, 0.47× evaluations), and CHAINROSEN_BOX_200 rebuilt once under the SQP member (496 vs 494 iterations, budget exit either way; unchanged under the IP member). So the rule does reach one non-separable problem for one member; CHAINROSEN_BOX_200 stays the corpus' most expensive under both and ELLIPSOID2_200 under the IP member, and those still point to a structured or exact Hessian | `bench/results/abl-c7` (notes), `r5-large-n` |
| G3 round-4 held-out | **done, and the increment is rejected**: on the sealed final4 set (11 problems, 4 families, three with dense coupled Hessians at n ≥ 100) the rule costs 1.24× [0.99, 1.67] evaluations for the portfolio on track A and 1.13× [0.78, 1.82] on track C, 1.21× / 1.13× for the IP member, 1.14× [1.00, 1.54] / 1.07× for the SQP member, no attainment change; COVQP_120 3.6× and COVQP_300 3.4× worse by a first-update scale-route rebuild. Default reverted to off; opt-in kept | `bench/results/s6v4-final4` §2 |

**Round-4 verdict (September 12).** H1 is falsified on the sealed set for
every configuration (§7, G3 row above): what follows in this section is the
development-corpus record, kept as written. The rule is now opt-in.

Hypotheses, against the falsifiers written in §3: on the development corpus H1 is confirmed for the SQP
member (0.88×, interval excludes 1, +3 attained, n ≥ 100 subset 0.32×) and
**falsified by its own subset criterion for the portfolio and the IP member**
(0.94× / 0.96× with intervals excluding 1 and no losses, but the n ≥ 100
subset improves 1.2×, not the 2× the falsifier demanded); H2 not needed; H3
not tested; H4 partly confirmed (see the row). The rule is kept on the
whole-corpus evidence, which is a weaker claim than §3 set out to make. The
one un-shielded blemish is the SQP member on PORTFOLIO_100 (7.0× evaluations,
exit `Optimal` → `Acceptable`; the problem has no target, so it is outside the
attainment counts and the cost ratio, and the portfolio's record on it is
byte-identical rule on and off because it routes to the IP member).

### Deviations from §4, recorded

* §4 gated the rebuild on one test, |ln(sᵀy / sᵀDs)| ≤ ½ |ln τ| with D the
  quotient diagonal. That test is vacuous: sᵀDs = Σ (yᵢ/sᵢ) sᵢ² = sᵀy by
  construction wherever the quotient is used, so it fails only through the
  clamps and the negative-quotient fallback. The implementation replaced it
  with two tests, a *scale* route (quotients disagree with the model's
  diagonal in the same direction on ≥ 80 % of the pair's weight) and a *shape*
  route (the model's own diagonal explains the pair at least twice as well, in
  log terms, as the full matrix). The shape route validates diag(B) but the
  rebuild installs the quotients, a matrix the test never examined. That was
  first suspected as the PORTFOLIO_100 mechanism; the route counters added
  afterwards show PORTFOLIO_100's single rebuild comes through the *scale*
  route, and guarding the shape route was tried and rejected (§7).
* The n ≥ 10 gate was added after the un-gated whole-corpus run lost HS97 and
  HS98 (n = 6) and chosen so that the smallest winner (QUADSPHERE_10) stays
  in: a threshold fitted to the corpus, which is one more reason the round-4
  set must be sealed before the candidate runs on it.
* The un-gated run's records were regenerated in place instead of being kept
  in `bench/results/abl-c7-rejected/` as §5 required; only its summary (0.947×,
  HS97/HS98 lost) survives in the private worklog.
* Exit-status changes the ablation README did not list: HS114 under the IP
  member `Optimal` → `StepTolerance` (still attained, one rebuild); HS112 (SQP
  and portfolio) and DISPATCH_20 (IP) `Acceptable` → `Optimal`.

## 7. Next phase (C8): the scale route on coupled problems, then round 4

### 7.1 What was tried after the review (September 11, evening) and rejected

The review of C7 suspected the shape route (it validates diag(B) and installs
the quotients). Two things were done before any whole-corpus run:

* **Route counters.** `DenseBfgs::rebuild_routes` splits every rebuild into
  scale and shape-only, and both members report the split in the notes.
  Default behaviour is unchanged; the notes text is the only difference from
  the C7 records. Probe on the C7 wheel's problems
  (`bench/results/abl-c8-rejected/`): PORTFOLIO_100 under the SQP member
  rebuilds **once, by the scale route**; MAXENT_200 and ELLIPSOID2_200 under
  SQP once by each route; QUADSPHERE_100, ELLIPSOID_500, HS118 by scale only.
* **Shape guard** (in a shape-route rebuild keep the model's diagonal entry
  wherever the quotient disagrees with it by more than the factor): tried
  behind an option and **rejected on the probe** — PORTFOLIO_100 is
  byte-identical (its rebuild is not a shape rebuild) and ELLIPSOID2_200
  under the SQP member goes from 64 to 118 iterations (13 240 → 24 299
  objective evaluations), MAXENT_200 under the IP member from 59 to 96
  (12 382 → 19 953) and ELLIPSOID_500 under the IP member from 123 to 138,
  because the shape-route rebuilds there were installing quotients that *were*
  the right local curvature. The option was removed; the numbers are in
  `bench/results/abl-c8-rejected/README.md`.

So the open blemish is the scale route on a dense coupled QP: with B near the
unit matrix and a Hessian whose eigenvalues are all far above 1, the
quotients yᵢ/sᵢ = (Σs)ᵢ/sᵢ are large and positive on most of the pair's
weight (the 80 % test passes) yet are not the diagonal of anything.

### 7.2 Hypothesis and plan

**H5 (quotient stability).** On a separable problem the quotients from two
successive accepted pairs agree coordinate-wise (they are the diagonal
Hessian sampled at nearby points); on a coupled one they do not. Requiring,
for a scale-route rebuild after the first update, that the current quotients
agree with the previous pair's quotients within the factor on ≥ 80 % of the
weight keeps every separable rebuild (MAXENT, QUADSPHERE, ELLIPSOID,
DISPATCH, HS118) and refuses PORTFOLIO_100's. Costs one `Vec<f64>` of state
and O(n) per update. Falsified if it loses any C7 attainment or evaluation
win on the whole corpus, or if PORTFOLIO_100's rebuild happens on the first
update (where there is no previous pair) — the probe should record the
iteration of each rebuild before the rule is written.

| step | what | pass |
|---|---|---|
| S-B | record the update index and τ of every rebuild, and τ at the two updates after it, in the notes (`DenseBfgs::rebuild_log`, `post_rebuild_tau`) for the large problems under both members | the update of PORTFOLIO_100's rebuild is known — **done, see §7.3** |
| I5 | H5 behind an option | **not written: falsified by S-B** (§7.3) — every scale-route rebuild, PORTFOLIO_100's included, is at update 0, where there is no previous pair to compare against |
| I5' | rollback variant: undo a rebuild when τ at the next accepted update is again off by more than the factor | **not written: falsified by S-B** — after its rebuild PORTFOLIO_100's next two ratios are 0.21 and 1.3 (inside the factor) while MAXENT_200's are 18.7 and 8.5 and NNLS_SIMPLEX_30's 6.7 and 13.7 (the wins); the signal would roll back the wins and keep the blemish |
| G4 | whole-corpus ablation of whatever rule S-C produces, `abl-c8`, track A; rejected variants keep their records in `abl-c8-rejected/` | attained ≥ C7 for each; no C7 win lost by more than 1.1×; PORTFOLIO_100 under the SQP member back to `Optimal` within 2× its rule-off cost |
| S-C | trace study of PORTFOLIO_100 under the SQP member *after* its rebuild (α, ‖d‖, penalty, step bound per iteration): the model is not far off along the steps it takes (τ 0.2–1.3), so the 150 extra iterations are a direction-quality problem, not a curvature-scale one, and need their own mechanism before a rule is written | mechanism from records, then a hypothesis with a falsifier |
| I4 | `DenseBfgs::reset` also resets the rebuild cooldown | **not pursued**: no record shows a reset followed by a needed rebuild; a change without a mechanism from records is not ablated |
| G2 | track C on the n ≥ 50 subset (H3) | **done, H3 supported**: A/C iteration ratios median 1.00, max 1.54 (portfolio/IP) and 1.35 (SQP) on the 18 problems with n ≥ 50; MAXENT_200 59 vs 49 (IP) and 46 vs 39 (SQP) where C6 had 385 vs 144. CHAINROSEN_BOX_200 is attained on track C (915 iterations); PORTFOLIO_100 under SQP takes 175 iterations on both tracks, so its mechanism is not derivative noise. `bench/results/g2-track-c/README.md` |
| G3 | round-4 held-out set, sealed before any C7/C8 run on it (`bench/results/s6v4-final4/SEAL.md`: covqp, denselap, snl, obstacle; 11 problems), tracks A and C, six solvers, timing repeats, then rule on vs off | **done**: H1 falsified on held-out material (rule on costs 1.13–1.24× evaluations, table in §6); `docs/17` items 17 and 18 updated from it; the default is off |
| H4' | structured finite-difference Hessian with sparsity detection for CHAINROSEN_BOX_200 and ELLIPSOID2_200 under the IP member | separate plan (`docs/22`), after G3 |

### 7.3 S-B outcome (September 11, late evening)

Rebuild locations under the C7 default (update index counts accepted
quasi-Newton updates; τ = sᵀy / sᵀBs at that pair; "next" is τ at the two
accepted updates after the rebuild):

| problem | member | rebuilt at | next τ | iterations |
|---|---|---|---|---:|
| PORTFOLIO_100 | SQP | 0 [scale, τ 24] | 0.21, 1.3 | 175 (`Acceptable`) |
| QUADSPHERE_100 / _1000 / QUADSPHERE_10 | SQP | 0 [scale, τ 58–78] | 1.00 | 2 |
| QUADSPHERE2_300 | SQP | 0 [scale, τ 770] | 0.86, 0.60 | 16 |
| CHAINROSEN_EQ_200 | SQP | 0 [scale, τ 170] | 1.4, 0.9 | 43 |
| MAXENT_200 | SQP | 0 [scale, τ 28], 42 [shape, τ 11] | 18.7, 8.5, 1.0, 1.0 | 46 |
| MAXENT_200 | IP | 1 [scale, τ 11], 54 [shape, τ 17] | 2.5, 4.2, 1.0, 1.0 | 59 |
| NNLS_SIMPLEX_30 | SQP | 0 [scale, τ 76] | 6.7, 13.7 | 38 |
| ELLIPSOID2_200 | SQP | 46 [shape, τ 31], 61 [scale, τ 77] | 1.0, 1.0, 2.4, 5.1 | 64 |
| ELLIPSOID2_200 | IP | 27 [shape, τ 11] | 1.6, 1.8 | 424 |
| ELLIPSOID_500 | IP / SQP | 49 [shape, τ 16] / 10 [scale, τ 14] | 1.4, 2.0 / 1.5, 2.3 | 123 / 63 |
| HS118 | IP | 0 [scale, τ 2.4e-4], 11 [scale, τ 42] | 1.0, 1.0, 1.2e5, 0.13 | 19 |

Two things follow. First, the scale route fires almost always at the very
first update, on the wins and on the blemish alike, so no rule that needs a
previous pair (H5) can separate them, and the post-rebuild ratio does not
separate them either (I5'). Second, on PORTFOLIO_100 the rebuilt model is
*not* badly off along the steps the SQP member then takes; its 150 extra
iterations are therefore a step-direction problem on a dense QP with a
diagonal model, which is a different mechanism from the one C7 addressed
and needs its own trace study (S-C) before any rule is written. Nothing in
the solver changed as a result of S-B; the route counters and the rebuild
log stay in the notes.

The one row that is a design question of its own is HS118 under the IP
member: a second scale-route rebuild at update 11 (τ 42) is followed by a
ratio of 1.2e5 — the rebuilt diagonal was far too small along the next step
— and yet the run converges in 19 iterations (0.63× the rule-off cost).
Recorded, not acted on.
