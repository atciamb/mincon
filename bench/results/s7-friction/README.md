# Friction audit S-D: realistic problems with a user's minimal inputs (September 12, 2026)

The baseline for round 5 (`docs/22_ROUND5_ROBUSTNESS_PLAN.md`). Fourteen problems written the
way a user writes them for MATLAB's `fmincon` (an objective, a start, and whichever of `A, b,
Aeq, beq, lb, ub, nonlcon` the problem has), given to five solvers on the first try with no
options, no tolerances and no derivatives, except `wrong_gradient`, whose point is the wrong
gradient the user supplied. Definitions, references and runners are in `bench/friction/`
(`problems.py`, `run.py`, `friction_problems.m`, `run_friction_matlab.m`, `summarize.py`);
`summary.md` and `scored.jsonl` here are the scored output, `*.jsonl` the raw records.

Every returned point is judged by the independent oracle (`bench/harness/oracle.py`):
attained means feasible to 1e-6 and objective within 1e-4 relative of a reference that is
independent of every solver under test (closed forms, published optima polished on their
active set, bounded-variable least squares, a KKT-verified convex solve). The MATLAB and Python
definitions agree at x0 to 3.6e-11 relative over all 28 MATLAB records. Solvers: mincon 0.1.0
(`mincon.fmincon`, wheel of the committed tree, default everything, so the portfolio and all
threads), SciPy 1.18.1 SLSQP and trust-constr (defaults), MATLAB R2025b `fmincon`
interior-point and sqp (`Display='off'` only, plus `SpecifyObjectiveGradient` on the one
problem where the user supplied a gradient). One run each, single machine; wall times are
one-shot and include MATLAB's first-call overhead, so they are not comparable across languages.

## 1. The problems

| problem | n | rows | what it is | MATLAB-minimal inputs | reference |
|---|---:|---:|---|---|---|
| chainrosen20 | 20 | 0 | the reviewer's script: chained Rosenbrock in a box, alternating start | fun, x0, lb, ub | f = 0 at x = 1 |
| odefit | 3 | 0 | (r, K, y0) of a logistic ODE fitted through a numerical integrator (RK45, rtol 1e-10) | fun, x0, lb, ub | clean data, f = 0 at the true parameters |
| portfolio_risk | 8 | 2 | Markowitz return with a variance cap as a nonlinear row, budget equality, position caps | fun, x0, Aeq, beq, lb, ub, nonlcon | KKT-verified convex solve (stationarity 2e-17) |
| pressure_vessel | 4 | 4 | the design problem in engineering units (a 1.3e6 volume row next to O(1) rows) | fun, x0, lb, ub, nonlcon | published continuous optimum 5885.3327736, polished |
| nan_region | 2 | 1 | objective NaN outside a disc; the constraint keeps the solution inside, probes may not be | fun, x0, nonlcon | closed form |
| hs71 | 4 | 2 | only nonlcon and bounds (the textbook fmincon example) | fun, x0, lb, ub, nonlcon | published, polished |
| linear_only | 6 | 10 | quadratic cost, resource rows and non-negativity as A rows, a budget equality | fun, x0, A, b, Aeq, beq | KKT-verified convex solve |
| with_args | 3 | 0 | exponential-decay fit with the data passed as extra arguments, one one-sided bound | fun(x, t, d), x0, lb | clean data, f = 0 |
| wrong_gradient | 4 | 0 | a supplied objective gradient with a sign error in one component | fun, x0, lb, ub, the gradient | closed form per coordinate |
| infeasible_start_far | 10 | 1 | projection onto the simplex from 1e3 (equality violated by 1e4) | fun, x0, Aeq, beq, lb | closed form |
| bad_scaling | 2 | 1 | a pressure (1e6) and an area (1e-6) coupled by x0 x1 >= 5; convex; Hessian condition 1e25 | fun, x0, lb, nonlcon | closed form on the active row |
| noisy_simulator | 4 | 1 | a smooth quadratic plus deterministic 1e-9 noise, a budget row and bounds | fun, x0, A, b, lb, ub | closed form (noiseless) |
| box_lsq | 50 | 0 | bounded least-squares deconvolution, dense Hessian, many active bounds | fun, x0, lb, ub | bounded-variable least squares (BVLS) |
| equality_circle | 3 | 3 | two nonlinear equalities and an inactive inequality, start infeasible on both | fun, x0, nonlcon | closed form |

## 2. First-try attainment (the metric that matters first)

Cell: `Y` attained or `n`, then model-boundary objective evaluations; `!` = a false
certificate (success reported at a point the oracle does not certify as first-order
stationary); `k` = success reported at a point the oracle does certify but which is not the
reference optimum; `?` = attained but not reported as success.

| problem | fmincon-interior-point | fmincon-sqp | **mincon** | scipy-slsqp | scipy-trust-constr |
|---|---|---|---|---|---|
| chainrosen20 | Y? 3014 | n 2014 | Y 2483 | n 2188 | Y 8001 |
| odefit | Y 160 | Y 135 | Y 96 | Y 104 | Y 376 |
| portfolio_risk | Y 250 | Y 111 | Y 111 | Y 99 | Y 153 |
| pressure_vessel | Y 63 | Y 32 | Y 57 | n 105 | Y 655 |
| nan_region | Y 32 | Y 18 | Y 22 | Y 31 | n! 21 |
| hs71 | Y 46 | Y 30 | Y 32 | Y 25 | Y 650 |
| linear_only | Y 142 | Y 92 | Y 121 | Y 61 | Y 175 |
| with_args | Y 86 | Y 77 | Y 99 | Y 61 | Y 100 |
| wrong_gradient | n! 45 | n! 49 | n 477 | n! 416 | n! 407 |
| infeasible_start_far | Y 199 | Y 33 | Y 99 | Y 33 | Y 462 |
| bad_scaling | nk 20 | n 11 | n! 64 | n 3 | nk 84 |
| noisy_simulator | Y 183 | Y 68 | Y 86 | Y 28 | Y 255 |
| box_lsq | n 3009 | Y? 5049 | Y 8595 | Y 3213 | Y 7803 |
| equality_circle | Y 44 | Y 24 | Y 26 | Y 20 | Y 44 |
| **attained** | **11/14** | **11/14** | **12/14** | **10/14** | **11/14** |
| false certificates (`!`) | 1 | 1 | 1 | 1 | 2 |
| certified stationary point that is not the optimum (`k`) | 1 | 0 | 0 | 0 | 1 |
| attained without reporting success (`?`) | 1 | 1 | 0 | 0 | 0 |

What the records say:

* **mincon attains the most on the first try (12/14)**, with one false certificate, like
  fmincon-interior-point, fmincon-sqp and SLSQP (trust-constr has two). It attains everything any
  other solver attains, plus chainrosen20 (fmincon-sqp and SLSQP stop at their default
  iteration or evaluation caps, fmincon-interior-point reaches the target but reports its cap)
  and box_lsq (fmincon-interior-point's 3000-evaluation default cap ends it 1.3e-3 short).
* **No solver attains `wrong_gradient` or `bad_scaling`.** Four of five certify a wrong point on
  `wrong_gradient`; mincon alone refuses (exit 2, `success = False`) but says "Local minimum
  possible. Step size below tolerance", which does not tell the user what is wrong. On
  `bad_scaling`, a two-variable convex problem, fmincon-interior-point and trust-constr stop
  at f = 16.4 against 0.169 and report success: the oracle certifies that point as first-order
  stationary (the gradient component along the 1e6-scale variable is 1e-7 relative to the
  other, invisible to any first-order tolerance), so it is the scaling pathology, not a false
  certificate. mincon reports success at f = 120.7, a point the oracle rejects, which is a
  false certificate; fmincon-sqp and SLSQP fail without claiming success (fmincon-sqp:
  "Converged to an infeasible point" on a feasible problem).
* **mincon's `bad_scaling` exit is a false certificate**, the one defect in this audit that
  belongs to this project. Mechanism, from the trace and the solver notes: the interior-point
  member pushes the start away from the bound x1 >= 1e-9 by 1e-2 (the push floor is 1, so a
  variable of size 1e-6 is moved four orders of magnitude); the gradient-based objective factor
  is then computed at that point, where the gradient is 2e10, giving a factor 5e-9; with the
  objective scaled by 5e-9 the barrier parameter and the complementarity tolerance are enormous
  in unscaled terms, and the run stops at x1 = 1.2e-5 with a bound multiplier of 8e6 that
  balances the gradient, so the scaled KKT test and the relative stationarity guard from round
  3 (D9) both pass. The exit is `Optimal` even with exact derivatives, scaling off gives
  `Acceptable` at another wrong point (f = 16.4), and the SQP member declares the problem
  locally infeasible at the start: its QP step satisfies the linearised row exactly, the unit
  quasi-Newton model predicts an objective change of 1e-11 where the true change is 16, the
  merit line search rejects the step, and the exit logic labels a rejected step "the elastic
  QP step vanished" (`docs/16` clusters 19 and 20, defects D11 and D12).
* **Message quality.** mincon's messages are the most specific when things go well and carry
  a `notes` list that explains scaling, sparsity and the portfolio; on the two user-error
  problems they are not more helpful than fmincon's. SLSQP's "Positive directional derivative
  for linesearch" (pressure_vessel, bad_scaling) and "Iteration limit reached" (chainrosen20,
  the default 100) leave the user to guess; trust-constr reports success on every problem it
  fails.
* **Evaluations.** On the twelve problems mincon attains, its model-boundary count is within
  1.3x of the best fmincon count on ten and above it on box_lsq (8595 vs fmincon-sqp 5049,
  which stopped at its cap 2.5e-5 short; SLSQP 3213) and with_args (99 vs 77). The
  50-variable bound-constrained QP takes 166 interior-point iterations; the same class of
  problem (OBSTACLE_*, COVQP_*) cost round 4 its evaluation comparison, and it is the routing
  item of `docs/22`.
* **Wall time** is below 0.05 s for every mincon solve except the ODE fit (0.42 s, all of it in
  the integrator). MATLAB's one-shot times (0.06 to 1 s) include first-call overhead and say
  nothing about the solver.

## 3. Input friction, by API

What the user must write beyond MATLAB's minimum, from the runner's translations:

| | `mincon.fmincon` | `scipy.optimize.minimize` (SLSQP) | `scipy.optimize.minimize` (trust-constr) | `fmincon` |
|---|---|---|---|---|
| inequality sign | `c <= 0`, as MATLAB | negate every `c` (`fun >= 0`) | `NonlinearConstraint(fun, -inf, 0)` | `c <= 0` |
| linear rows | `A, b, Aeq, beq` | write them as functions | `LinearConstraint(M, lo, hi)`, rows stacked | `A, b, Aeq, beq` |
| bounds | `lb, ub`, scalars or vectors | `Bounds(lb, ub)` with infinities | same | `lb, ub` |
| extra data | `args=` | `args=` on `minimize` and again in each constraint dict | same | captured by the handle |
| gradient | `jac=` (used when supplied) | `jac=` | `jac=` | `[f, g]` and `SpecifyObjectiveGradient` |
| algorithm | not selectable through the facade (a gap); `minimize(method=)` | `method=` | `method=` | `Algorithm` |
| iteration display | after the solve only (`disp=True`) | `callback=` | `verbose=` | streams (`Display='iter'`) |
| multipliers | `result.multipliers.eqlin` etc. | none | `v` on the result | `lambda` |
| default caps | none (a cap loses solutions, `docs/17` item 16) | 100 iterations (chainrosen20 stopped there) | 1000 | 400 iterations, 3000 / 100n evaluations |

## 4. Gaps this audit puts on the round-5 list

1. Correctness first: D11 (false `Optimal` on bad_scaling), D12 (false `LocallyInfeasible` on a
   feasible linearisation).
2. A wrong user gradient must be named as the likely cause when the solve stalls
   (a directional check costs two evaluations).
3. Bound-constrained and dense convex QPs should not take 166 interior-point iterations.
4. Streaming iteration display, a callback, and a warm-start resume do not exist.
5. The facade has no `method=`; `hess=` is not exposed although both members accept an exact
   Hessian.
6. The ODE-fit and noisy-simulator problems are attained by every solver here; harder
   instances (stiff models, larger noise) belong in the sealed round-5 set, not in this baseline.

## 5. After the first increments (candidate runs, same audit)

`mincon-fmincon-i1.jsonl` (I1, the certificate guard) and `mincon-fmincon-i3.jsonl` (I1 + I2 +
I3 + the API increments), summarised in `summary-i1.md` and `summary-i3.md`:

| candidate | attained | bad_scaling | wrong_gradient | every other record |
|---|---|---|---|---|
| I1 | 12/14 | `Acceptable`, `success = False`, f = 16.46 (two objective rescales; 115 evaluations) | unchanged | identical to the baseline to the evaluation |
| I1 + I2 + I3 | 12/14 | the SQP member no longer declares infeasibility and certifies the same first-order point fmincon-interior-point certifies (f = 16.44, `k`, 15 evaluations) | stops before iterating: "The supplied derivatives disagree with finite differences along a test direction at x0 (relative error 3.8e-1) ... grad f [1]  4.0 vs -4.0" (11 evaluations) | identical |
| final tree (I1-I4, I7-I9, `scale_variables='auto'` default), `mincon-fmincon-final.jsonl` | **13/14** | attained, `Optimal`, 27 evaluations: the start (1e6, 1e-6) reveals the units and the solve runs in scaled variables (`abl-i8`) | as above | identical (`-i8-auto` and `-final` runs) |

With the final tree mincon attains every problem in the audit except the one whose supplied
gradient is wrong, where it is the only solver that refuses to iterate and names the component.

## Files

`{solver}.jsonl` raw records (Python: `mincon-fmincon`, `scipy-slsqp`, `scipy-trust-constr`;
MATLAB: `fmincon-interior-point`, `fmincon-sqp`); `scored.jsonl` with oracle verdicts;
`summary.md` per-solver tables with messages; `python_env.json`; `matlab_run.log`. Re-run:
`python bench/friction/run.py --out bench/results/s7-friction`, then MATLAB
`run_friction_matlab('<this dir>')` from `bench/friction`, then `python bench/friction/summarize.py <this dir>`.
