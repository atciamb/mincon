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
| I5 | structure routing to the active-set QP for convex QPs | `route = "auto"` / `"size"` (old) | S-C traces; box_lsq, OBSTACLE, COVQP; `abl-i5` |
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
