# Round 4 held-out qualification (final4), September 12, 2026

Eleven problems in four families sealed before any candidate run (`SEAL.md`:
covqp 30/120/300, denselap 100/250, snl 24/60/150, obstacle 50/200/500; three
families with dense coupled Hessians at n ≥ 100). Six solvers, tracks A
(finite differences) and C (exact derivatives), defaults, single thread,
60 s / 100 000 evaluations, targets v5, benchmark laptop. Candidate: the C8
tree (C7 rule on at factor 10, route counters in the notes; numbers identical
to `abl-c7`), wheel `e821c101…`. Then the same three mincon configurations
with the rule off (`@bfgs_rescale=0`), which is the C6 behaviour, and a
timing run with three repeats. Everything below is as measured; the two
mincon runs that ended at the 60 s limit are counted as not attained where
the oracle says so.

## 1. Against fmincon and SLSQP

Attainment (feasible to 1e-6, objective within 1e-4 relative of the frozen
target) and paired evaluation ratios (geometric mean over problems both
attain, family bootstrap):

| track | solver | attained | vs fmincon-interior-point | vs fmincon-sqp | vs scipy-slsqp |
|---|---|---:|---|---|---|
| A | fmincon-interior-point | 5/11 | — | | |
| A | fmincon-sqp | 8/11 | 1.26 [0.44, 2.33] | — | |
| A | scipy-slsqp | 11/11 | 0.64 [0.39, 0.86] | 0.58 [0.34, 0.77] | — |
| A | **mincon** | **9/11** | **0.85 [0.47, 1.94]**, +4 attained | 1.02 [0.52, 2.13], +2 / −1 | 1.55 [0.70, 2.96], −2 |
| A | mincon-ip | 9/11 | 0.81 [0.47, 1.72] | 0.99 [0.52, 2.13] | 1.51 [0.70, 2.73] |
| A | mincon-sqp | 9/11 | 0.70 [0.36, 1.70] | 0.77 [0.39, 1.45] | 1.13 [0.49, 2.43] |
| C | fmincon-interior-point | 10/11 | — | | |
| C | fmincon-sqp | 10/11 | 0.83 [0.56, 1.23] | — | |
| C | scipy-slsqp | 11/11 | 0.45 | 0.37 [0.23, 0.58] | — |
| C | **mincon** | **11/11** | **0.53 [0.25, 1.10]**, +1 | 0.44 [0.14, 1.34], +1 | 1.16 [0.53, 2.30] |
| C | mincon-ip | 11/11 | 0.53 [0.25, 1.10] | 0.44 [0.14, 1.34] | 1.16 [0.53, 2.30] |
| C | mincon-sqp | 10/11 | 0.53 [0.26, 1.09] | 0.44 [0.14, 1.37] | 1.18 [0.60, 2.38] |

Per problem, track A (attained, exit status, total model evaluations):

| problem | mincon | mincon-sqp | fmincon-ip | fmincon-sqp | scipy-slsqp |
|---|---|---|---|---|---|
| COVQP_30 | Y 1 1 374 | Y 1 1 206 | Y 1 2 666 | Y 1 1 186 | Y 1 032 |
| COVQP_120 | Y 6 22 522 | Y 6 16 236 | n 0 6 064 | Y 1 5 356 | Y 4 371 |
| COVQP_300 | n 0 12 040 (60 s) | n 0 10 898 (60 s) | n 0 6 034 | Y 2 25 926 | Y 17 190 |
| DENSELAP_100 | Y 6 6 518 | Y 6 6 760 | Y 0 6 108 | Y 2 5 072 | Y 2 637 |
| DENSELAP_250 | Y 0 21 202 (60 s) | Y 6 15 668 | Y 0 6 024 | Y 2 39 450 | Y 5 035 |
| OBSTACLE_50 | Y 1 6 449 | Y 1 3 245 | n 0 3 026 | Y 2 3 251 | Y 2 242 |
| OBSTACLE_200 | Y 1 64 387 | Y 1 46 936 | n 0 3 055 | n 0 20 037 | Y 35 699 |
| OBSTACLE_500 | n 0 100 283 | n 0 100 238 | n 0 3 018 | n 0 50 067 | Y 251 232 |
| SNL_24 | Y 1 1 386 | Y 1 790 | Y 1 3 120 | Y 2 2 232 | Y 1 691 |
| SNL_60 | Y 1 3 028 | Y 1 3 036 | Y 0 6 030 | Y 2 10 916 | Y 8 283 |
| SNL_150 | Y 1 18 518 | Y 1 10 412 | n 0 6 166 | n 0 30 278 | Y 39 531 |

What the records say:

* **Reliability against fmincon holds on unseen problems.** With finite
  differences mincon attains 9 where fmincon-interior-point attains 5 (its
  default 3000-evaluation cap ends every n ≥ 100 problem) and fmincon-sqp 8;
  with exact derivatives 11 against 10 and 10. Evaluations against
  fmincon-interior-point are below 1 on both tracks with intervals that
  include 1 at n = 11.
* **SciPy SLSQP is the strongest solver on this set** and mincon is not
  ahead of it: 11/11 on both tracks, 0.64× fmincon-interior-point's
  evaluations on track A with an interval excluding 1, and mincon costs
  1.55× [0.70, 2.96] its evaluations. Two caveats belong next to that
  number: SLSQP has no evaluation budget the harness can enforce and
  attained OBSTACLE_500 at 251 232 evaluations, 2.5× the budget the other
  solvers were held to; and its exits are never certified (status 0, no
  multipliers), so the KKT columns are empty. On the dense convex QPs
  (covqp, obstacle) an active-set SQP with an exact QP subproblem is the
  right algorithm and SLSQP is one.
* **The two mincon misses on track A are budget exits, not wrong answers.**
  COVQP_300 hit the 60 s limit at 12 040 evaluations with a 3e-4 objective
  gap and 1e-5 violation (the model, a 45 000-term generated NumPy
  expression, costs 5 ms per evaluation and a finite-difference gradient
  needs 301 of them); OBSTACLE_500 ended the 100 000-evaluation budget at
  f = 1.55 against 0.59 after 198 iterations (501 evaluations each). With
  exact derivatives both are attained (OBSTACLE_500 in 1 631 iterations).
  DENSELAP_250 also hit 60 s but had already reached the target.
* **Wall time is not in mincon's favour here** (`../s6v4-final4-timing`,
  three repeats, median per problem): 1.39× [0.50, 4.73]
  fmincon-interior-point on the 5 problems both attain every time and
  1.87× [1.39, 2.34] fmincon-sqp on 7. The callback share of mincon's wall
  time is 0.97: on these dense generated models the Python evaluation
  dominates and MATLAB's JIT evaluates the same expressions faster, so the
  round-3 wall figure (0.077×) does not carry over to model-bound problems.
  Nothing in this run measures the solver's own time.

## 2. The C7 rule on unseen problems: rejected

The purpose of the round (`docs/21` G3). Same wheel, rule on (`abl-c7`
default, factor 10) against rule off (`@bfgs_rescale=0`, the C6 behaviour):

| track | configuration | attained on / off | evaluations on/off [95 % family bootstrap] | worst problems |
|---|---|---|---|---|
| A | portfolio | 9 / 9 | **1.24 [0.99, 1.67]** | COVQP_120 3.57×, OBSTACLE_50 1.52×, DENSELAP_250 1.36×; SNL_24 0.66× |
| A | IP member | 9 / 9 | 1.21 [0.96, 1.62] | COVQP_120 3.57×, OBSTACLE_50 1.52× |
| A | SQP member | 9 / 9 | 1.14 [1.00, 1.54] | COVQP_120 3.16×, OBSTACLE_500 1.51× |
| C | portfolio | 11 / 11 | **1.13 [0.78, 1.82]** | COVQP_120 3.57×, COVQP_300 3.42×, OBSTACLE_500 1.51×; SNL_24 0.44× |
| C | IP member | 11 / 11 | 1.13 [0.78, 1.82] | same |
| C | SQP member | 10 / 10 | 1.07 [0.77, 1.46] | COVQP_300 2.93×, COVQP_120 2.21×; OBSTACLE_200 0.76× |

The rule that cut the development corpus to 0.94× [0.78, 0.97] costs
evaluations on every configuration of the sealed set, wins one problem
(SNL_24) and loses on the dense QPs by 2–3.6×. The route counters say why:
on COVQP_120 and COVQP_300 the scale route fires at the first update
(the same mechanism as PORTFOLIO_100 in `abl-c7`) and rebuilds the model
to per-coordinate quotients of a dense covariance, which is not the
diagonal of anything; the damped updates then spend 60–90 iterations
undoing it. The development corpus is separable at large n (MAXENT,
QUADSPHERE, ELLIPSOID), which is exactly the case the rule was built on
and the case this set was designed not to be. H1 of `docs/21` is
**falsified on held-out material**; the whole-corpus win was a fit to the
corpus, as the fitted n ≥ 10 gate already suggested.

**Decision.** Per the protocol an increment is kept only if the held-out
round does not contradict it. The default of `bfgs_curvature_rescale` is
reverted to off (`f64::INFINITY`; Python `bfgs_rescale` default 0). The
rule stays available as an opt-in for separable problems, where its
development-corpus record stands (`abl-c7`), and the route counters and
rebuild log stay in the notes for whoever opts in. With the default off
the solver's behaviour is the round-3 C6 behaviour on every record checked
(`abl-c7` rule-off arm equals the C6 records; the fixture gate is
55/55 auto and ip, 54/55 sqp either way).

## 3. What round 4 says about the next phase

* The evaluations gap to SLSQP and to fmincon-sqp on dense convex QPs is
  the SQP member's QP subproblem and step acceptance, not the curvature
  model: with exact derivatives on COVQP_120 the SQP member takes 65
  iterations against SLSQP's 12 and fmincon-sqp's 21 even with the rule
  off (19). That is the S-C trace study of `docs/21` §7 with a new, scorable
  instance.
* The finite-difference cost on n ≥ 300 is what caps mincon at the time
  budget: the structured-Hessian phase (H4′) matters less than a cheaper
  gradient — sparsity-aware or user-supplied derivatives are the answer on
  these models, and the API takes them.
* Timing claims must say "solver time" or "wall time on this model" and
  which; the round-3 0.077× was on cheap models.

## Files

`{solver}.{A,C}.jsonl` raw records; `scored-{A,C}.jsonl` with oracle
verdicts (targets v5); `scored-{A,C}.analysis*.md/json` paired analyses
against each baseline; `*_bfgs_rescale-0.*` the rule-off arms with
`scored-*-off.jsonl` and `cmp-onoff-*.{mincon,mincon-ip,mincon-sqp}.analysis.*`
for the on/off comparison; `SEAL.md`; timing in `../s6v4-final4-timing`.
The run was interrupted twice by the machine's memory pressure (browsers)
and resumed by the supervisor from its per-problem records; no record was
produced twice. final4 is development material from here on.
