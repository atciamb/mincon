# Round 5: robustness and frictionless use (plan, September 12, 2026)

Written before any increment of this round was coded, in the form of `docs/19` and `docs/21`:
question, mechanisms from records, hypotheses with falsifiers, increments behind options, gates.
The owner's priority for this round: the most robust solver with the least friction to use, in
Python, with fmincon's "it handles everything" feel and less ceremony than SciPy. Evaluations
and wall time are reported honestly on every run but do not gate increments; the honesty
rules of `docs/15` and `docs/17` stand unchanged (success means certified, usable stays
separate, nothing is dressed up, claims come only from a sealed held-out round).

## 1. Question

Given a problem written with fmincon's minimal inputs (an objective, a start, and whichever of
`A, b, Aeq, beq, lb, ub, nonlcon` it has), does `mincon.fmincon` / `mincon.minimize` return a
certified answer on the first try more often than fmincon and SciPy, never certify a wrong
point, tell the user what to do when it cannot, and let the user see and steer the run without
learning options? The baseline study S-D (`bench/results/s7-friction`) answers the first part
for fourteen realistic problems: mincon 12/14 first-try attainment against fmincon-interior-point
11, fmincon-sqp 11, SLSQP 10, trust-constr 11, with one false certificate of its own.

## 2. Mechanisms, from the records

| # | record | what happens | mechanism |
|---|---|---|---|
| M1 | `s7-friction` bad_scaling, mincon exit `Optimal` at f = 120.7 (reference 0.169), exact derivatives too | the interior-point member's start push moves x1 from 1e-6 to 1e-2 (push floor 1); the objective scale factor is taken there (gradient 2e10, factor 5e-9); scaled complementarity then lets the bound x1 >= 1e-9 count as active at distance 1.2e-5 with multiplier 8e6, which balances the gradient, so the scaled KKT test and the D9 relative-stationarity guard both pass | the scale factor is chosen where the solver put the iterate, not where the problem lives, and the D9 rescale fires only when relative stationarity is bad, which a balanced bound multiplier hides (defect **D11**, atlas cluster 19) |
| M2 | same problem, SQP member exit `LocallyInfeasible` at x0 (violation 4); fmincon-sqp does the same | the QP step satisfies the linearised row exactly (v_lin = 0); the unit initial quasi-Newton model predicts an objective change of 8e-12 against a true change of 16 (curvature 2e12 in x1); the merit line search rejects the step four times; the exit logic then reads a rejected step as "the elastic QP step vanished" | an infeasibility diagnosis is made without the linearisation being infeasible; a rejected step with v_lin = 0 is a curvature failure, not a stationary point of the violation (defect **D12**, cluster 20) |
| M3 | wrong_gradient: mincon exit 2 "Local minimum possible. Step size below tolerance", the other four solvers report success at wrong points | the user's gradient has a sign error in one component; every member's line search fails and the run stops with an honest but uninformative message | nothing compares the supplied gradient with the model; fmincon's `CheckGradients` is off by default and stops the run when on |
| M4 | box_lsq (n = 50 bound-constrained least squares): mincon 166 interior-point iterations, 8595 evaluations; SLSQP 63 iterations, 3213; round 4 OBSTACLE_500 198 iterations at 501 evaluations each, COVQP_120 65 SQP iterations vs SLSQP 12 | on a convex QP the barrier path and the damped quasi-Newton QP both converge linearly toward the active set; an active-set method with the true Hessian identifies it in a few dozen changes | the portfolio routes on n only (`SQP_FIRST_MAX_N = 20`); nothing detects that the objective is quadratic or that every row is linear, although the Jacobian probe already sees linear rows |
| M5 | `s7-friction` §3: iteration display prints after the solve; no callback; no `maxtime` stop from the user's side; no `resume`; the facade has no `method=`; `hess=` not exposed although both members take an exact Hessian | the engine runs without the GIL and has no per-iteration hook; the Python binding sets `Capabilities.hessian = false` | API surface, not solver behaviour |
| M6 | round 4 COVQP_300 stopped at the 60 s harness limit 3e-4 short with FD gradients costing 301 evaluations each | a second call starts from scratch: no warm start of x, multipliers and the quasi-Newton matrix | no resume path in the API |

## 3. Hypotheses and falsifiers

* **H1 (certificate integrity, D11).** Recomputing the objective scale factor whenever the
  unscaled gradient norm has drifted by more than 10x from where the factor was chosen, at any
  point where the scaled test would pass (dropping the `!rel_ok` condition of the D9 rule), and
  requiring complementarity in *unscaled* units for a bound to count as active, turns the
  bad_scaling exit into a non-certified one and does not change any attained run on the corpus.
  *Falsified if* the exit stays `Optimal` without attainment, or the whole-corpus ablation loses
  an attainment or costs more than 1.05x evaluations, or BADSTART_DISC / QUADSPHERE2_300 (the D9
  fixtures) regress.
* **H2 (infeasibility diagnosis, D12).** `LocallyInfeasible` requires the elastic QP's
  linearised violation to be positive at its minimiser (the linearisation itself infeasible);
  when the step is rejected with v_lin = 0 the SQP member instead rescales its curvature model
  from the rejected step (the ratio of actual to predicted objective change) and retries, and
  if that fails exits `StepTolerance` or `NumericalFailure` with the message saying the step was
  rejected. *Falsified if* any diagnostic problem (INFEASIBLE_*, cluster 17) loses its
  diagnosis, or the corpus ablation loses an attainment or costs more than 1.05x.
* **H3 (derivative check without ceremony).** When the user supplies a gradient or Jacobian, a
  directional check at x0 along a random direction (two extra model evaluations, independent of
  n) detects the wrong_gradient error at relative error > 1e-2 and never fires on the corpus
  track C (exact derivatives, 172 problems) or on the FD-noise problems. On a gross disagreement
  the solve stops with a message naming the component and the two values (a wrong gradient does
  not produce a right answer); on a mild one the note is kept and the solve continues.
  *Falsified if* any track-C run stops on a false alarm, or the check costs more than two
  evaluations per solve.
* **H4 (structure routing).** A start-up probe that (a) reads linear rows off the Jacobian probe
  and (b) tests the objective for quadratic behaviour with two secant pairs (consistent
  curvature along two directions) routes bound-constrained and linearly constrained convex QPs
  to the SQP member with the active-set QP and an exact (secant-built) Hessian, and cuts
  box_lsq, OBSTACLE_50/200 and COVQP_120 to at most 0.5x their evaluations with no attainment
  loss on the corpus. The S-C trace study (`docs/21` §7) runs first on box_lsq and COVQP_120 to
  say where the SQP member's 65 iterations go. *Falsified if* the probe misroutes any
  non-quadratic corpus problem (attainment loss) or the routed problems do not reach 0.5x.
* **H5 (streaming display, callback, stop).** An engine callback per iteration (GIL re-acquired
  only for the call) with `disp=True` printing live, `callback=` for SciPy parity and a
  return value that stops the run (`StoppedByUser`, fmincon's `OutputFcn`) costs less than 2 %
  wall on the cheapest fixtures and changes no record. *Falsified by* any record change or a
  measured overhead above 2 % on `run_testset`.
* **H6 (resume).** `mincon.resume(result)` (or `minimize(..., warm_start=result)`) restarting
  from x, multipliers and the quasi-Newton matrix finishes a run that was stopped by a budget in
  fewer evaluations than a cold start would need from that point plus the setup cost; a re-solve
  of an already converged result costs at most three iterations. *Falsified if* the resumed run
  needs more evaluations than the cold restart on any of the round-4 budget exits (COVQP_300,
  OBSTACLE_500 with a budget).
* **H7 (exact Hessians from Python).** `hess=` (Lagrangian Hessian callable) and the
  `Capabilities.hessian` flag reach both members; on final4 track C with exact Hessians the
  iteration counts fall to the Newton range (CHAINROSEN_BOX_200 below 100 iterations from 915).
  *Falsified if* a supplied Hessian changes an attained result to a non-attained one anywhere.
* **H8 (variable scaling, the only route that actually solves bad_scaling).** Scaling variables
  internally by max(|x0_i|, typical) when the components of x0 span more than 1e4 in magnitude
  (opt-in first, `scale_variables`) attains bad_scaling and does not change any corpus record
  where the span is below 1e4. *Falsified if* it loses any corpus attainment when turned on
  for the whole corpus, in which case it stays opt-in and the message for badly spanned starts
  suggests it.

## 4. Increments (each behind an option, each ablated on the whole corpus before the next)

| id | increment | option (default) | records that decide |
|---|---|---|---|
| I1 | D11: drift-triggered objective rescale without the `!rel_ok` condition; unscaled-unit complementarity for bound activity in the certificate | none: a certificate fix is not optional; the old behaviour is not kept | bad_scaling exit; `abl-i1` whole corpus; D9 fixtures |
| I2 | D12: no `LocallyInfeasible` with v_lin = 0; curvature rescale from a rejected step | none (correctness) | bad_scaling SQP exit; `abl-i2`; diagnostic problems |
| I3 | directional derivative check when derivatives are supplied | `check_derivatives = "auto"` (new default: two-evaluation directional check; `True` keeps the full check; `False` off) | wrong_gradient message; track C corpus no false alarm |
| I4 | streaming display, `callback=`, stop from the callback, `maxtime` honoured through it | `disp`, `callback` | `run_testset` timing; records unchanged |
| I5 | as planned: structure routing to the active-set QP for convex QPs; as built after S-C (7.7): a quadratic-program probe at the start that hands the SQP member the exact, constant Hessian | `quadratic_probe` (default on after `abl-i5`) | S-C traces; box_lsq, OBSTACLE, COVQP; `abl-i5` |
| I6 | `resume(result)` warm start | API only | round-4 budget exits re-run with a budget then resumed |
| I7 | `hess=` in Python; facade `method=` | API only | final4 track C with Hessians |
| I8 | variable scaling from x0 magnitudes | `scale_variables = False` (opt-in until ablated) | bad_scaling; `abl-i8` on the corpus with it on |

Order: I1, I2 (correctness before anything), I3, I4, I7 (API, cheap, no records change), I5
(the measured one), I6, I8. Every increment re-runs the friction audit (`bench/friction/run.py
--tag -cand`) and the whole corpus on track A (all 172 problems; final4 is development material
now) with the paired family bootstrap of `analyze.py`; rejected variants keep their records in
`bench/results/abl-iN-rejected/`.

## 5. Studies and gates

| gate | what | pass |
|---|---|---|
| S-D | friction baseline (done, `bench/results/s7-friction`) | 14 problems, references verified by the oracle, five solvers |
| S-C | trace study of the SQP member on box_lsq and COVQP_120 with exact derivatives: per iteration alpha, step norm, penalty, step bound, active-set size, QP iterations; the same for the IP member on box_lsq | a mechanism sentence for the 65 / 166 iterations before I5 is written |
| G1 | after each increment: `cargo test`, fixtures 55/55 auto and ip, 54/55 sqp (HS13), Python tests, friction re-run | mincon's friction attainment >= 12/14, false certificates 0, no record of an attained problem lost |
| G2 | whole-corpus ablation per increment (`abl-i1` ...), track A, paired family bootstrap | attained >= before; evaluations ratio interval not above 1.05 for correctness fixes; the increment's own falsifier not met |
| G3 | sealed round-5 held-out set: new realistic families (an ODE fit with a stiff model and one with 2 % noise; two more design problems with units, welded beam and speed reducer, re-derived; a portfolio with transaction costs; a bounded deconvolution at n = 200; a noisy simulator at 1e-7) generated with the corpus machinery (`heldout5.py`, split `final5`, targets v6, MATLAB equivalence, sealed before any candidate run), both tracks, six solvers plus trust-constr, timing repeats with the solver/model split explicit | the metric that matters first is attained on the first try with minimal inputs; then evaluations; then wall split into solver and model time. Claims (`docs/17`) are updated only from this run |

## 6. What is not in this round

Global optimisation (multistart stays local search); sparse user Jacobians; autodiff detection;
the structured-Hessian phase for CHAINROSEN_BOX_200 / ELLIPSOID2_200 (H4' of `docs/21`) unless
I7 makes it unnecessary; any change to the sealed-round protocol.

## 7. Outcome

Filled in as the increments land, in order.

### 7.1 I1, the certificate guard (D11), September 12

Mechanism confirmed by an env-gated dump of the interior-point termination quantities
(`MINCON_IP_DEBUG=1`, kept): at the false exit on bad_scaling the row `5 - x0 x1` has slack 7
and a user-unit multiplier of 13.75, the bound of x1 sits 1.2e-5 away with a multiplier of
8.2e6, and both products are below 1e-6 only in scaled units (row factor 1e-4, objective factor
5e-9). The guard is one line: complementarity must also hold in the user's units, `compl / d_f
<= tol_compl * max(1, |f|)`, added to the `rel_ok` condition of the D9 rule so that a failure
blocks the certificate and, when the gradient norm has drifted by more than 10x, rescales the
objective factor and continues. A first version that instead required zero multiplier weight on
every inactive row or bound (the oracle's active-set rule) was tried and dropped before any
ablation: it turned box_lsq's `Optimal` into `Acceptable` at +650 evaluations, because the
barrier legitimately leaves multipliers of 5e-3 on bounds 2e-5 away (products 1e-7).

| check | result |
|---|---|
| bad_scaling (friction) | exit `Acceptable` at f = 16.46 (`success = False`), two objective rescales logged; every other friction record unchanged to the evaluation |
| fixture `TORTURE_UNITS` (new, `Expect::NoFalseCertificate`: Optimal away from the optimum is a lie unless an independent finite-difference stationarity check passes) | auto 56/56, ip 56/56, sqp 55/56 (HS13) |
| `abl-i1` portfolio, track A, 169 problems vs the current default's records | attained 159/166 vs 158 (COVQP_300, a 60 s budget exit either way, landed inside the target this time); cost 1.00 [1.00, 1.00] on 158 common; the only two changed records are the two 60 s exits |
| `abl-i1` IP member | 156/166 both; cost 1.00 [1.00, 1.01]; changed records: the two 60 s exits, HS16 (+16 evaluations, unattained either way), HS17 (+18, attained either way) and **UNITS**, whose old exit `Optimal` at f = 5.009 against 0.5 (oracle: not KKT, relative stationarity 1.9e-3) was a false certificate already in the corpus and is now `Acceptable` |

H1 stands: the exit is no longer certified and no attained run changed. The problem itself is
still not solved by any member; that is H8's question.

### 7.2 I2, the infeasibility diagnosis (D12), September 12

Mechanism refined by `MINCON_SQP_DEBUG=1`: after the rejected first step the step bound
shrinks to a fraction of `max(1, |x_j|)`, which for a 1e-6 variable no longer admits the 4e-6
move the linearised row needs, so the elastic QP inside that box reports a positive linearised
violation and the exit logic calls it a stationary point of the violation. The fix asks the
linearisation without the step bound (`linearisation_can_improve`): only when the violation
cannot fall there is `LocallyInfeasible` a diagnosis; otherwise the bound is reset, the penalty
raised to what the merit needs (uncapped, since the step reaches the linearised feasible set)
and the iteration retried, at most three times, then `NumericalFailure` with a message saying
the step was rejected. On bad_scaling the SQP member then converges in 9 iterations to the
row-constrained point f = 16.44, the same first-order point fmincon-interior-point certifies
(oracle: KKT, relative stationarity 2e-12), which the friction table marks `k`, not `!`.
Fixtures: `TORTURE_INFEASIBLE` and the INFEASIBLE_* diagnostics unchanged (sqp 55/56, HS13
only). Whole-corpus ablation `abl-i2` (with the diagnostic problems included, 172): portfolio
158/158, IP 156/156, SQP 154/154 attained against the baseline, cost 1.00 with intervals of
width at most 0.01 in every arm, the only changed records being the 60 s budget exits (plus
I1's five in the IP arm); INFEASIBLE_LIN and INFEASIBLE_NL keep their diagnosis under every
arm (INFEASIBLE_NL now through the "cannot be reduced without the step bound" path);
UNBOUNDED_PAR under the SQP member alone is a budget exit exactly as before I2. H2 stands; the
refusal path fired on no corpus problem.

### 7.3 I3, the directional derivative check (H3), September 12

Two evaluations along one bounds-respecting direction at x0; the first version compared the
h-scaled quantities and the checker's absolute floor of 1 hid a 10x gradient error, fixed by
comparing directional derivatives per unit of the direction. wrong_gradient now stops before
iterating: "The supplied derivatives disagree with finite differences along a test direction
at x0 (relative error 3.8e-1) ... grad f [1]  4.0 vs -4.0" (11 evaluations: two for the
direction, the rest for the full check at x0 that names the component). Falsifier: the whole
corpus with exact derivatives (track C, 172 problems, the `spec` derivatives) through the check:
silent on all 172, no mild note, no false alarm. Cost on track C: two evaluations per solve.
`DerivativeCheck::Directional` is the default; `True`/`Full` keeps fmincon's semantics.

### 7.4 I4, the iteration callback (H5), September 12

`Options.callback` is called by both members after every iteration's record; `true` stops the
solve (`StoppedByUser`, note "Stopped by the user's callback at iteration N"). Python
`callback=` receives the trace row; `disp=True` streams the table through it. A no-op Python
callback on box_lsq (166 iterations): 33.4 ms against 35.9 ms without (noise). No record changes
by construction (the harness passes no callback).

### 7.5 I7, `hess=` and `method=` in Python (H7), September 12: falsified as built

Both members already consumed an exact Hessian; the binding now passes one (the Lagrangian
Hessian with the engine's multiplier convention; the facade converts to MATLAB's `HessianFcn`
groups). Measured with exact Hessians: a 2-variable quadratic converges in 1 iteration (5 with
BFGS), a bound-constrained one in 1 (IP: 5); but with nonlinear constraints the current handling
of an indefinite Lagrangian Hessian makes the solve *slower*: one equality, SQP 17 iterations
against 7 with BFGS (IP 7 either way); HS71, SQP 5 -> 370 iterations (`Acceptable`), IP 10 -> 56.
The SQP shifts the full-space Hessian by 1e-4 x 10^k until the QP is convex, which damps the
step whenever the reduced Hessian is fine but the full one is not; the IP's inertia correction
is IPOPT's Algorithm IC with its parameters, so its 56 iterations need a study of their own.
H7 is falsified for constrained problems as built; `hess=` stays in the API, documented as
experimental, and the regularisation of indefinite exact Hessians is the next study (S-E).
Traces for S-E (HS71, exact Hessian, eigenvalues of the Lagrangian Hessian at the solution
-2.67, 0.63, 1.06, 5.03; the active set leaves a one-dimensional null space on which the reduced
Hessian is positive): the SQP member's shift is 46.9 at every iteration, sticky across
iterations and grown by factors of 10 from 1e-4 times the diagonal scale, so the QP's Hessian is
the shift and every step is a scaled steepest-descent step (370 iterations, linear); the IP
member's inertia correction fires at every iteration with delta_w between 3 and 15 although the
KKT matrix should already have inertia (n, m) when the reduced Hessian is positive, which points
at the inertia test itself (56 iterations). The SQP part is a contained fix (I9 below); the IP
part needs the study. Reading `kkt.rs`: the loop first tries delta_w = 0 and demands a *certified* factorisation
with inertia (n, m); a factorisation with regularised pivots is never certified. The sparse
LDL^T has no 2x2 (Bunch-Kaufman) pivoting, so with an indefinite primal block (an exact
Lagrangian Hessian) a negative or tiny pivot met before the dual rows is regularised, the
certificate is lost, and delta_w is applied although the true inertia is (n, m); with a BFGS
(positive definite) primal block this never happens, which is why the quasi-Newton path is
unaffected. IPOPT relies on MA27/MA57's 2x2 pivots here. Candidate fixes for S-E: count
negative pivots in the primal block as legitimate for the inertia certificate when no pivot
is regularised, or a Bunch-Kaufman variant of the dense path for n + m < 500; falsifier HS71
with `hess=` under the IP member at most 10 iterations, and no record change on the corpus
(the corpus never supplies Hessians, so the change must be confined to the exact-Hessian path).

### 7.7 I9, the SQP member's exact-Hessian regularisation, September 12

`direction()` now regularises an indefinite exact Hessian afresh every iteration with
`delta (J'J + E_B)` (E_B the coordinates at a bound), which adds curvature along the constraint
normals only, and falls back to `delta I` when the reduced Hessian is itself indefinite; delta is
the smallest that makes the QP convex, found by a factor-of-10 search and four bisections, and is
reported as `delta_w`. HS71 with the exact Hessian: 370 -> 7 SQP iterations (`Optimal`, quasi-
Newton 5); the one-equality problem 17 -> 5 (quasi-Newton 7); the shift on HS71 is 11.2 at the
first iteration and 0.005 near the solution, where it was 46.9 throughout. Fixtures and the corpus
are unaffected (no exact Hessians there); gates 56/56, 56/56, 55/56, Python 34. The interior-point
member's exact-Hessian path (56 iterations) is unchanged and remains the S-E study.

**S-C conclusion, and what I5 should be.** With its exact (constant) Hessian, box_lsq takes 2
SQP iterations and 3 evaluations (153 with finite-difference gradients) instead of 139 and 140
(8595 through the facade), and 16 interior-point iterations instead of 166; COVQP_120 with exact
derivatives already takes 19 / 23 with the C7 rule off. The 3-5x iteration gap to SLSQP on
convex QPs is therefore the curvature model, not the QP step, and routing on size would not
close it. I5 becomes: detect a quadratic objective at the start (two secant pairs with
consistent curvature along two directions), build its Hessian by differencing the gradient
(n + 1 gradient evaluations, colouring when the Jacobian probe found structure), and run the
SQP member with that Hessian; falsifier as in H4 (no attainment loss on the corpus, box_lsq /
OBSTACLE / COVQP at most 0.5x). Not built this session.

### 7.6 I8, variable scaling from the start (H8), September 12 (ablation pending)

`ScaledNlp` (crate `mincon`, `scaled.rs`): the solver sees `x/d`, the model sees `x`, with the
chain rule on gradient, Jacobian columns, Hessian entries and Hessian-vector products; bound
multipliers are mapped back by `1/d`. `Options::scale_variables` is `Off` (default), `Auto`
(factors spanning at least 1e4) or `On`. Friction audit: `On` attains 13/14 (bad_scaling in 27
evaluations) but costs evaluations elsewhere (portfolio_risk 111 -> 247, pressure_vessel 57 ->
128, box_lsq 8595 -> 12677, infeasible_start_far 99 -> 62); `Auto` attains 13/14 with every
other record identical to the default's. Rust test `variable_scaling_solves_the_units_problem`.
Whole-corpus ablation (`abl-i8`): `Auto` 158/158 attained, cost 1.01 [1.00, 1.02], one changed
record besides the 60 s exits (HS117, attained, 5010 vs 612 evaluations: its start spans 1e4 and
the scaling by |x0| does not match the solution); `On` 151/158, -4.2 pp [-11.3, -2.1], cost
1.09 [1.00, 1.25], 112 records changed, QUADSPHERE_1000 / QUADSPHERE2_300 / POLYQP_100 / SNL_150
lost. **`On` falsified, `Auto` kept and made the default** (`VariableScaling::Auto`, Python
`'auto'`); `TORTURE_UNITS` is now an `Expect::Optimum` fixture (56/56, 56/56, 55/56). A start of
zeros carries no scale, so UNITS in the corpus stays unsolved; the message for such starts is a
later item.

### 7.8 S-E, the interior-point member's exact-Hessian path, September 12

Reproduced first (`MINCON_IP_DEBUG=1` and the trace of `hess=` on HS71 under
`method='interior-point'`): 56 iterations ending `Acceptable`, with `delta_w` between 2 and 16 at
every iteration from the second on while `mu` already sits at its floor, so every step is a damped
one and the convergence is linear; quasi-Newton takes 10. The mechanism is the one hypothesised
in 7.5, confirmed by a three-by-three unit test (`kkt.rs`,
`expected_pivot_signs_lose_the_certificate_on_an_indefinite_primal_block`): the `LDL^T` carries a
sign expectation per row (+1 primal, -1 dual) and perturbs a pivot of the other sign to
`+-1e-7`; with `W = diag(1, -1)` and one constraint fixing `x_2` the KKT matrix has inertia (2, 1)
but the negative primal pivot is perturbed, the factorisation belongs to a different matrix, the
certificate is lost, and `delta_w` is applied although nothing was wrong. With a positive-definite
quasi-Newton block a wrong-sign pivot can only be noise, which is why that path never showed it.

The fix is `Options::kkt_pivot_signs` (`PivotSigns::{Auto, Expected, Free}`, Python
`kkt_pivot_signs`): under `Free` every pivot keeps its sign and the inertia is counted, only a
numerically zero pivot is perturbed, and growth still voids the certificate; `Auto` (the default)
is `Free` when the member uses the model's exact Hessian and `Expected` with a quasi-Newton one.
No 2x2 pivoting was needed: with 1x1 pivots the count is Sylvester's law whenever the
factorisation did not break down, and a breakdown (a tiny pivot next to a large off-diagonal)
still ends in the `delta_w` loop exactly as before. Measured with the exact Hessian under the
interior-point member: HS71 56 -> **10** iterations, certified (`Optimal`), `delta_w` nonzero at
one iteration only; the one-equality problem 11 -> 9 (quasi-Newton 11); a 50-variable box-
constrained dense QP 57 -> 12 (SQP with the same Hessian: 1). Forcing `Expected` reproduces the 56.
Fixtures and Rust tests unchanged (184 tests, 56/56, 56/56, 55/56), Python 34. The falsifier
"at most 10 iterations" holds; the corpus falsifier is `abl-se` below.

**Ablation `abl-se`** (`bench/results/abl-se`, three arms: `mincon@kkt_pivot_signs=auto` as the
control, and `Free` forced onto the quasi-Newton path for the portfolio and the interior-point
member, which is the only way the option can touch a corpus record since the corpus supplies no
Hessians): the control arm changes the three records every run of this tree changes (the two
60 s exits and HS117 from the `scale_variables='auto'` default), 158/158 attained, cost 1.01
[1.00, 1.02]; `Free` forced onto the portfolio's quasi-Newton path changes exactly the same three,
so the sign expectations never decided a record; forced onto the interior-point member alone it
changes those plus I1's three (HS16, HS17, UNITS) and one of its own, RANKLOSS_JAC, a
step-tolerance exit that becomes `Optimal` at 215 against 213 evaluations (157/156 attained,
COVQP_300 a 60 s exit that landed inside the target this time; cost 1.02 [1.00, 1.02]). The
falsifier holds on both counts; `Auto` stays the default (`bench/results/abl-se/README.md`).

Both members are now Newton-fast with a supplied Hessian (SQP 7, interior point 10 on HS71,
against 5 and 10 with quasi-Newton), so `hess=` is no longer documented as experimental.

### 7.9 I5, the quadratic-program probe, September 12

Built as `docs/22` §7.7 concluded, not as the routing rule of §4: `crate mincon`, `quadratic.rs`,
`Options::quadratic_probe` (Python `quadratic_probe`). At the start the objective is sampled at
`x0 + k h d`, `k = 0..3`, along two random bounds-respecting lines (`h` a tenth of each
coordinate's magnitude, never past a bound at three steps); a quadratic has a zero third
difference and a linear constraint row a zero second difference, both tested at 1e-8 of the
variation along the line with a 1e-12 floor on the values' magnitude (a true quadratic sits at
1e-13; the friction problem noisy_simulator, a quadratic plus 1e-9 noise, sat at 1e-7 and was
first accepted, after which its Newton steps ended in a failed line search and an `Acceptable`
exit where the quasi-Newton member certifies, so the tolerance was tightened and only an
`Optimal` exit from the quadratic member now settles the portfolio). The first failing test ends
the probe, so a general problem pays three objective evaluations (four with `f(x0)`). When both
lines pass, the constant Hessian is built by differencing the gradient when the model supplies
one (`n` calls) or function values otherwise (`n (n + 3) / 2` calls, skipped above `n = 500`,
when the evaluation budget cannot pay for it, or when the probe's own evaluations predict the
build would take more than half of a wall-time budget) and checked at the six line points,
which did not build it; the SQP member then runs with it as an exact Lagrangian Hessian (the
constraints being linear) before the ordinary members, and falls through to them unless it
certifies. Only the objective's Hessian is built, so a quadratic constraint row (COVQP's risk
row, QUADSPHERE-style spheres) declines the probe by design: extending it to quadratic rows
would need their Hessians too.

Friction audit with the probe on (`s7-friction/mincon-fmincon-final5.jsonl`, `summary-final5.md`,
the final tree of this session at defaults),
13/14 as before with every record `Optimal`: **box_lsq 8595 -> 1485 evaluations (0.17x, 166 ->
2 iterations)**, linear_only 121 -> 78, infeasible_start_far 99 -> 116 (a projection onto the
simplex is a QP; the build costs more than the 10-variable solve saved), noisy_simulator declined
at 107 against 86 (the two lines pass, the build's 14 evaluations are spent, the model check or
the tightened line test declines), every nonlinear problem +4 evaluations. Corpus development
runs before the ablation: OBSTACLE_50 4246 -> 1485, OBSTACLE_200 51 656 -> 20 910,
QUADSPHERE2_30 1271 -> 595, POLYQP_100 certified (`Optimal` in 1 iteration, 9.5 s) where the
default stopped on the step tolerance (status 6, 18.3 s) at 5359 against 3648 evaluations;
losses in evaluations where the quasi-Newton path needed few iterations relative to `n`:
QUADSPHERE_100 1414 -> 5359, NNLS_SIMPLEX_120 7264 -> 7629, LQTRAJ_10 270 -> 593. The corpus
problem UNITS is detected as a QP and its Newton step lands on the row-constrained point
f = 5 that every solver reaches there, with a first-order certificate the oracle also grants
(the Hessian's condition number of 1e24 is beyond double precision; only scaled variables
solve it, and the start carries none, which is what the new note says, §7.10).

Whole-corpus ablation, first as built (`abl-i5-rejected`): **157/158, HS44 lost**, a nonconvex
QP with two local minima where the quasi-Newton path reaches the global one (f = -15) from the
zero start and the Newton path with the exact indefinite Hessian certifies the other (f = -13),
the basin failure the falsifier was written for. The fix is a restriction, not a tuning: the
Hessian is handed over only when it is positive semidefinite (Cholesky of `H + 1e-10 max|H_ii|
I`), since every KKT point of a convex QP is its global minimum. Re-ablated as `abl-i5`:
**158/158, no record lost**, cost 1.04 [0.90, 1.16] on 158 (a wash with a wide interval), 35
convex QPs detected and every one `Optimal`, HS44 declined and kept its global minimum,
diagnostics unchanged; gains of 0.23x to 0.79x on NNLS_SIMPLEX, OBSTACLE, QUADSPHERE_10,
QUADSPHERE2_30, HS118, POLYQP and the small HS quadratics, losses of 1.5x to 6.6x on LQTRAJ_50,
QUADSPHERE_100, QUADSPHERE2_300, MANY_INEQ and the two-variable problems (the build's
`n (n + 3) / 2` evaluations pay where the quasi-Newton path needed more than about `n / 2`
iterations). The falsifier holds and the robustness gains are real (POLYQP_100 certified,
box_lsq 0.17x), so **`quadratic_probe` is the default**; the evaluation trade-off is reported,
not tuned away. Follow-ups to measure: a build from a supplied gradient or a sparsity pattern
(OBSTACLE's tridiagonal Hessian), quadratic constraint rows.

### 7.10 A note for starts that carry no scale, September 12

`mincon_core::no_scale_hint`, called by both members after the first gradient: when the
objective gradient at the start spans a factor 1e6 or more across the variables and the start's
magnitudes cannot supply the scale factors (a coordinate is zero, or the nonzero magnitudes lie
within the 1e4 span that `scale_variables='auto'` leaves alone), the report's notes say so and
point at a start of typical magnitudes or `typical_x`. A note only. Fires on UNITS (gradient span
1e12 at the zero start). Measured on the whole corpus before any rule was written
(`abl-i5-rejected`, notes): 11 fires, UNITS and ten false alarms (HS14, HS22, HS32, HS53, HS60,
HS77, HS79, FLAT_BOUND, ROSEN_SPHERE_4, ROSEN_SPHERE_10) whose exact gradient at the start has
zero components that finite differences return as 1e-9 noise. With components below the
difference's rounding accuracy (`10 sqrt(eps) max(1, |f0|)`) counted as zero (`abl-i5-hint`):
3 fires, UNITS and the two ROSEN_SPHERE problems, where a forward difference's truncation error
at a zero coordinate (about 1e-6) reads as a 1e6 span. The committed rule keeps the floor and
asks a span of 1e8 from a finite-difference gradient (1e6 from an exact one), which by
construction keeps UNITS (1e12) and drops both ROSEN_SPHERE fires: one fire on the corpus.

### 7.11 I6, warm start, September 12

`Options::warm_start` (`WarmStart { lambda, z_l, z_u, mu, quasi_newton }`), Python
`warm_start=res` on `minimize` and `fmincon`, where `res` is a previous result of the same
problem: both members start from its multipliers (user units mapped into the member's scaled
units, the inverse of the report's map) and its quasi-Newton curvature model (`res.hess_approx`,
the dense `n x n` BFGS matrix every report now carries for `n <= 1000`, mapped through the
objective and variable scalings); the interior-point member also resumes at the barrier
parameter the trace ended at and clamps the bound multipliers to its neighbourhood. Measured
before writing any rule (`bench/friction/warm_start_study.py`, `bench/results/i6-warmstart`):
every friction problem solved at defaults, interrupted at a third and at two thirds of its
evaluations, then resumed from `res.x` warm and cold. **Multipliers alone changed nothing**: the
SQP member re-derives them from its QP every iteration, so warm and cold restarts spent
identical evaluations on all nine SQP-first problems and the interior-point one differed by
noise. With the curvature model handed over the picture is:

| problem | uninterrupted | cut at 2/3, warm total | cut at 2/3, cold total | cut at 1/3, warm | cut at 1/3, cold |
|---|---:|---:|---:|---:|---:|
| box_lsq (interior point first) | 8595 | 8391 | 11 561 | 8947 | 9459 |
| odefit | 96 | 100 | 147 | 125 | 125 |
| pressure_vessel | 57 | 68 | 95 | 68 | 77 |
| portfolio_risk | 111 | 127 | 171 | 120 | 120 |
| linear_only | 125 | 132 | 136 | 132 | 169 |
| hs71 | 32 | 37 | 42 | 37 | 37 |
| chainrosen20 | 2483 | 2504 | 2725 | 2793 | 2572 |
| with_args | 99 | 103 | 133 | 103 | 90 |
| infeasible_start_far | 99 | 177 | 273 | 110 | 55 |
| equality_circle | 26 | 30 | 30 | 30 | 30 |

Every restart attains the reference. Resumed late, the warm start beats the cold restart on
eight of ten problems (equal on the other two) and lands within 5 % of the uninterrupted run on
four, where the cold restart costs 1.1x to 1.5x more; resumed early, it is better on three,
equal on four and worse on three (chainrosen20, infeasible_start_far and with_args, where the
early curvature model is a worse start than the identity). No default changes: the warm start only runs when the user passes a previous result,
so no corpus ablation applies; the API is the increment. The round-4 budget exits (COVQP_300,
OBSTACLE_500) are wall-time exits on this machine and were not re-run.

### 7.12 I5's evaluation cost: the structured Hessian build, September 13

The owner called the probe's evaluation jumps a first-class open item (LQTRAJ_50 1855 -> 12 257,
QUADSPHERE_100 3030 -> 5770, QUADSPHERE2_300 27 092 -> 47 872, MANY_INEQ 36 -> 92 in `abl-i5`).
Measured before any rule was written: the baseline quasi-Newton iteration counts of the 35
detected QPs against `n` (`abl-c7` records). Every gain has `nit / n >= 0.48` (OBSTACLE_200 1.26,
OBSTACLE_50 1.54, NNLS_SIMPLEX_30 1.50, QUADSPHERE2_30 1.20, QUADSPHERE_10 2.20) and every loss
`nit / n <= 0.40` (LQTRAJ_50 0.05, QUADSPHERE_100 0.12, QUADSPHERE2_300 0.12, EQQP6 0.17,
LQTRAJ_10 0.23, PORTFOLIO_100 0.29, POLYQP_100 0.34, MANY_INEQ 0.40), which is the break-even
`n (n + 3) / 2` against `nit (n + 1)` of §7.9. The three candidates of the hand-off against those
numbers: (b) a size rule cannot separate them (OBSTACLE_200 at n = 200 is a 0.40x gain,
QUADSPHERE_100 at n = 100 a 3.8x loss, LQTRAJ_10 and QUADSPHERE2_30 are both n = 30); (c) the only
honest test of "few iterations needed" is the quasi-Newton path itself, and a capped run before
the build (the I6 hand-over would make it cheap) gives back part of every gain (OBSTACLE_200
20 910 -> about 31 000 at a cap of half the build) while the cap that protects QUADSPHERE2_300
(13 245 quasi-Newton evaluations against a 45 450 build) is too large to protect OBSTACLE, so
no cap satisfies the falsifier; (a) the problems' Hessians: QUADSPHERE and QUADSPHERE2 are
diagonal (`a_i x_i^2`), LQTRAJ is diagonal (`dt u_k^2` on the control block, zero elsewhere),
EQQP6 is the identity, MANY_INEQ's objective is linear (a zero Hessian), OBSTACLE's is
tridiagonal; PORTFOLIO, POLYQP and NNLS_SIMPLEX are dense. The losses are structure the dense
build ignores.

The increment is `Options::quadratic_build` (`QuadraticBuild::{Structured, Dense}`, Python
`quadratic_build`), the function-value build reordered by structure: the diagonal first (`2n`
evaluations, which also give the gradient), then one off-diagonal band at a time, the model
checked against the probe's six line points after each band and handed over at the first
structure that both reproduces them and passes the convexity check. A diagonal Hessian costs
`2n`, a tridiagonal one `3n - 1`, a dense one exactly the dense build's `n (n + 3) / 2` in a
different order, so on a dense QP nothing changes. The check after each band is the one the
dense build applied at the end, so a wrong early stop needs the off-band entries to cancel
along two random directions to 1e-8, and even then the SQP member's line search and first-order
certificate do not depend on the Hessian. One case the first version got wrong: box_lsq's
Gaussian kernel `K'K` decays away from the diagonal, a band model fits the line points at
half-bandwidth 23 while the truncated matrix fails the convexity check at 1e-10 (the kernel is
ill-conditioned), and the probe declined "not convex" (9576 evaluations, the quasi-Newton path);
the search now continues until the model both fits and is convex (box_lsq 1485 -> 1232). The
convexity check is a banded Cholesky (`O(n b^2)`), so a diagonal check at n = 5000 is trivial.
Limits: the structured build runs up to the gradient build's `n = 5000` (the diagonal phase
costs `2n`); when the dense continuation is unaffordable (n > 500, or outside the evaluation or
wall budget) the band search may spend `2n` more, four gradients' worth, before it declines, so
a dense QP at n = 1000 wastes 4n evaluations against 7 today; the corpus has no such problem.

Spot check on the problems that decided it (`quadratic_build=structured` against `abl-i5`,
model-boundary evaluations): QUADSPHERE_100 5359 -> **409**, LQTRAJ_50 11 933 -> **758**,
QUADSPHERE2_300 46 360 -> **1510**, LQTRAJ_10 593 -> 158, MANY_INEQ 50 -> 40, EQQP6 68 -> 53
(all below their quasi-Newton baselines except MANY_INEQ at 12 and EQQP6 at 34); OBSTACLE_50 1485
-> 309 and OBSTACLE_200 20 910 -> 1209 (the tridiagonal build); **OBSTACLE_500, a 100 000-evaluation
budget exit in every round-4 run, certified in 3009**; QUADSPHERE_1000 16 023 -> 4009 and
LQTRAJ_200 6605 -> 3008, both above the old dense limit; NNLS_SIMPLEX_120, POLYQP_100,
PORTFOLIO_100 unchanged (dense); COVQP_120 and HS44 still declined. Friction audit with the option:
13/14 as before, every record `Optimal`; box_lsq 1485 -> 1232, infeasible_start_far 116 -> 71 (a
diagonal Hessian, the simplex projection), noisy_simulator 107 -> 200 (the diagonal model fits the
noisy line points where the dense one misfit; the SQP member certifies with it in 12 iterations at
200 evaluations against the quasi-Newton path's 107), every other record identical.

**Ablation `abl-i5-build`** (`bench/results/abl-i5-build`, one arm `mincon@quadratic_build=structured`,
all 172 problems, scored against the recipe baseline and against the `abl-i5` arm it replaces):
**159/158 attained** either way (OBSTACLE_500 certified in 3009 evaluations where every earlier
run ended at the 100 000 budget), cost 0.93 [0.62, 1.04] against the quasi-Newton baseline and
**0.90 [0.65, 0.97] against the dense build**, 31 records changed against it and every one cheaper
or equal: OBSTACLE_200 20 910 -> 1209, QUADSPHERE2_300 47 872 -> 3022, QUADSPHERE_100 5770 -> 820,
LQTRAJ_50 12 257 -> 1082, QUADSPHERE_1000 34 048 -> 8020 and LQTRAJ_200 7873 -> 4232 (both above
the old dense limit, where the probe used to decline), HS118 223 -> 118, MANY_INEQ 92 -> 82; the
dense QPs (NNLS_SIMPLEX, POLYQP, PORTFOLIO) and the declined ones (HS44, COVQP) unchanged. The
four jumps the owner named now sit below their quasi-Newton baselines. The falsifier holds and
**`quadratic_build='structured'` is the default.** What remains of the trade-off is the dense QPs
the quasi-Newton path solved in few iterations (PORTFOLIO_100 1.71x, POLYQP_100 1.47x and
certified, NNLS_SIMPLEX_120 1.05x); a low-rank-plus-diagonal build (PORTFOLIO's Hessian is rank 3
plus a diagonal) is the next structure to measure, not a rule to write now.

### 7.13 I5's last extension: convex quadratic constraint rows, September 13

The probe built only the objective's Hessian, so any problem with a nonlinear constraint row was
declined even when the row was a plain quadratic (COVQP's risk row, ELLIPSOID's sphere, TCPORT's
variance row). Measured before writing the rule: across the corpus's quadratic-objective
problems with quadratic rows, the ones the quasi-Newton path found expensive all have a
**diagonal** row Hessian — ELLIPSOID_50 at 128 iterations, ELLIPSOID2_200 at 412, ELLIPSOID_500
at 196 — so the structured build of §7.12 costs `2n` there, not `n (n + 3) / 2`, and the row is
where the curvature lives (an ellipsoid projection has an identity objective Hessian).

`Options::quadratic_rows` (`QuadraticRows::{Off, Jacobian, Values}`, Python `quadratic_rows`).
A row is accepted only when it keeps the feasible set **convex**: a convex function bounded
above or a concave one bounded below, never an equality, never a ranged row. That is checked
twice — first by the sign of the row's second differences along the two probe lines, so an
equality or a wrong-sign row declines before anything is built, and again by a banded Cholesky
of the built Hessian on the side its bound needs. Every accepted row is built together, since
one constraint or Jacobian evaluation returns them all: `n + 1` Jacobian calls under `Jacobian`,
or the objective's own structured schedule under `Values` (`2n` for diagonal rows). The model's
Lagrangian Hessian is then the objective's times `sigma` plus each row's times its multiplier,
and `hessian_vector` sums the same way; the stored structure is the widest band found. The
restriction is the same argument as the convexity check of §7.9: with a convex objective over a
convex feasible set every KKT point is the global minimum, so the Newton path cannot end in a
basin the quasi-Newton path would not.

**Ablation `abl-i5-rows`** (`bench/results/abl-i5-rows`, all 185 problems, the same wheel with
the option off as the control, two arms on track A and three on track C): track A **172/171
attained** — ELLIPSOID_500 is gained, nothing is lost — at cost 0.99 [0.92, 1.01]; track C
175/175 at 0.91 [0.67, 1.00] on objective and constraint calls and 0.98 [0.85, 1.06] once
gradient and Jacobian calls are counted. The `Jacobian` and `Values` arms are identical on every
track-C record, since the corpus supplies Jacobians there. The gain is concentrated where the
measurement said it would be: ELLIPSOID_500's control ends the evaluation budget at an
*infeasible* point (f = 1995.49, violation 0.85) where the candidate returns the reference to
1.6e-14 in 26 iterations instead of 196; ELLIPSOID2_200 goes from a step-tolerance exit at
170 824 evaluations to `Optimal` at 9308, 0.05x and certified where it was not. The cost is
concentrated where it was predicted too: the problems whose *objective* Hessian is already dense
(COVQP_120 1.59x on track A, COVQP_300 143 -> 657 model calls on track C, TCPORT_100 121 -> 245)
pay for rows that buy them little. Diagnostics are preserved, and INFEASIBLE_NL on track C
improves from `-5` (numerical failure) to `-4` (the documented locally infeasible diagnosis).
**`quadratic_rows='values'` becomes the default.** The friction audit at that tree is 13/14 as
before with one record changed, `portfolio_risk` 248 model calls in 11 iterations -> 166 in 3;
the other thirteen are identical. A rule that skips the row build when the objective's Hessian
came back dense is the next thing to measure and is not written now.
