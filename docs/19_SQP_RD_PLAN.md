# Research and development plan: SQP, the evaluation budget, and the basin question

September 11, 2026. Written before any code was changed for this phase. The
plan is the contract for the phase: questions first, the mathematics and facts
second (`docs/20_SQP_MATHEMATICS.md`), measurements third, code last, and every
increment judged by the protocol of `docs/15_BENCHMARK_PROTOCOL_V2.md`.

The state this plan starts from is `docs/18_EVIDENCE_MATRIX.md` and
`docs/16_FAILURE_ATLAS.md`: candidate C4 is at least as reliable as
`fmincon` at defaults, uses about the same number of evaluations, and is an
order of magnitude faster in wall time. `fmincon-sqp` and SLSQP still use
0.6–0.8× the evaluations of any interior-point code on small dense problems,
and mincon has no SQP.

---

## 1. Questions

**Q1 (SQP).** Can a sequential quadratic programming member close the
small-dense-problem evaluation gap to `fmincon-sqp`/SLSQP without giving up
the robustness the interior-point member already has, and does adding it to
the portfolio raise the combined attainment rate above either member alone?

**Q2 (budget).** `fmincon` stops at 3000 function evaluations by default and
that cap is where most of its failures on this corpus come from. Does mincon
need a cap at all? If a budget beyond 3000 buys real solutions, how large a
budget, and how should the solver tell the user it is making no progress
instead of silently spending 100 000 evaluations?

**Q3 (basins).** On HS2, HS16, HS20, HS44, HS55 and BADSTART_DISC one solver or
another finishes at a KKT point that is not the published minimum. Is landing
in a different local minimum a defect in the solver or a property of the
problem? Which of these outcomes must be fixed, which are tolerable, and what
should the solver report when it happens?

**Q4 (stalls).** HS38, HS56, HS63 and ELLIPSOID2_20 cost mincon more
evaluations than `fmincon-ip` because the adaptive barrier stalls (atlas
clusters 8 and 13). Two primal-based fallback rules were tried and rejected as
neutral overall. Is there a criterion that fires earlier and helps on the
validation split, or does an SQP member simply make the question moot on
these small problems?

---

## 2. What the existing records already say

All numbers below are from the scored track-A records of the development,
validation and both held-out rounds (`s4-c2-dev`, `s4-c2-val`, `s6-final`,
`s6v2-final2`; 143 scorable problems, repeat 0). They are the input to the
studies of §4, not their conclusion.

### 2.1 The evaluation budget

| solver | attained | attained using > 3000 evaluations | > 15 000 | not attained |
|---|---|---|---|---|
| mincon (portfolio, 100 000 cap) | 133 / 143 | 10 | 4 | 10 |
| fmincon-ip (3000 cap) | 124 / 143 | 3 (all at the cap, 3005–3030) | 0 | 19 |
| fmincon-sqp (3000 cap) | 124 / 143 | 4 | 1 | 19 |
| SLSQP (uncapped) | 128 / 143 | 12 | 6 | 15 |

Ten of mincon's attainments needed more than 3000 evaluations, all of them
problems with n ≥ 50 solved with finite differences (CHAINROSEN_EQ_50/200,
CHAINROSEN_BOX_50, QUADSPHERE_100/1000, ELLIPSOID_50, CATENARY_40,
POLYQP_100, ELLIPSOID2_200, LQTRAJ_200). Four needed more than 15 000
(CHAINROSEN_EQ_200 61 707, QUADSPHERE_1000 100 293, ELLIPSOID2_200 83 063,
CATENARY_40 18 787). At 3000 evaluations mincon would attain 123 instead of
133; at 15 000 it would attain 129.

`fmincon-ip` hit its cap on 15 problems. On 10 of them mincon attained with a
larger budget; on 3 (CHAINROSEN_BOX_200, ELLIPSOID_500, PORTFOLIO_100) mincon
also failed, and on 2 (QUADSPHERE2_300, PORTFOLIO_100) `fmincon-sqp` or nobody
attained. Track C (exact derivatives) shows why: the same problems take
`fmincon-ip` 10–1800 gradient evaluations, so the cap is a finite-difference
artefact (n + 1 evaluations per gradient), not a difficulty of the problems.

The question mincon has to answer is different from fmincon's: with an
adaptive budget, how does it know the difference between CHAINROSEN_EQ_200
(305 iterations, then attained) and CHAINROSEN_BOX_200 (492 iterations,
100 020 evaluations, f = 128.9, not attained)?

### 2.2 The basins

mincon's ten non-attainments split into three groups.

* **Different KKT point, feasible, oracle confirms first-order conditions
  (or the recovered multipliers do):** HS55 (f = 6.667 vs 6.333; rank-deficient
  equality constraints), HS20 (f = 40.199 vs 38.199; every solver in the
  comparison lands here), HS108 (f = −0.675 vs −0.866; `fmincon-ip` lands at
  the same point), HS25 (f = 32.835 at the start; the gradient at x0 is
  numerically zero and `fmincon-ip` also stops after 0 iterations).
* **Different point where the oracle does not confirm first-order
  conditions:** HS16 (f = 3.982 vs 0.25), BADSTART_DISC (f = 6.375 vs 0.0457),
  UNITS (f = 5.009 vs 0.5; `fmincon-ip` 5.107). These need second-order and
  active-set diagnosis before they can be called "legitimate local minima" —
  the atlas currently asserts that on the basis of first-order checks with
  hand-computed multipliers, and for these three that assertion is not yet
  backed by the oracle.
* **Budget exhausted / large n:** CHAINROSEN_BOX_200, ELLIPSOID_500,
  QUADSPHERE2_300 (the last stopped `Optimal` at f = 0.03845 vs 0.03777,
  a 1.8e-3 relative gap: accuracy, not basin).

`fmincon-ip`'s non-attainments that are not the cap: HS2 (f = 4.94 vs 0.05,
basin), HS13 and RANKLOSS_JAC (degenerate: stopped at the tolerance), HS20,
HS55, HS108, HS25 (shared with mincon), UNITS.

### 2.3 Where SQP should pay

Track-A evaluations on problems every solver attains, geometric means over
the development split: `fmincon-sqp` 0.6–0.75× the interior-point codes, SLSQP
similar (`docs/18_EVIDENCE_MATRIX.md`). The mechanism, from the traces: an SQP
step needs one gradient per iteration and converges in fewer iterations on
problems with few active constraints because it does not have to drive a
barrier parameter to zero; an interior-point method spends its first
iterations centring and its last iterations shrinking `mu`.

Where SQP does not pay, also from the records: `fmincon-sqp` fails
CHAINROSEN_EQ_50/200, CHAINROSEN_BOX_50, CATENARY_40, ELLIPSOID2_200 under its
cap, attains QUADSPHERE2_300 where mincon does not, and its 400-iteration
default cap (`MaxIterations`) is what stops it on several large problems in
track C.

---

## 3. Hypotheses

**H1.** A damped-BFGS SQP with an exact-penalty or filter globalization,
second-order correction, and an elastic (always-feasible) QP subproblem will
attain at least the interior-point member's rate on the development split,
with 0.7–0.85× its evaluations on problems with n ≤ 20. Falsified if its
attainment on dev is lower than mincon-ip's or the paired evaluation ratio's
interval includes 1.0.

**H2.** The portfolio `auto` = {SQP first on small problems, IP first
otherwise, the other as fallback} attains strictly more than either member on
the validation split. Falsified if the union of attainments equals either
member's.

**H3** *(falsified September 11 — `bench/results/r4-budget/README.md`: every window rule that stops the two budget failures also stops four attained runs; policy is no default cap plus a progress verdict, and D10 fixed)*. An adaptive budget with a stall detector — no fixed cap, stop with a
"no progress" verdict when the best objective and the infeasibility have not
improved by a relative amount over a window scaled to n — reproduces every
attainment of the 100 000-evaluation runs while stopping the three budget
failures at least 5× earlier. Falsified if any current attainment is lost.

**H4** *(resolved September 11 — `bench/results/r3-basins/README.md`: HS2, HS16, HS20, HS55, HS108 are strict local minima; BADSTART_DISC is defect D9; UNITS is scaling-degenerate; HS33 is a saddle for the SQP codes)*. For HS16, BADSTART_DISC and UNITS the returned points satisfy
second-order sufficient conditions (or are degenerate KKT points the problem
admits), so they are legitimate local minima and not solver defects; for HS2
`fmincon-ip`'s point is likewise legitimate. Falsified if the corpus Hessian
of the Lagrangian shows a direction of negative curvature in the critical
cone, or the point is not a KKT point at all.

**H5.** A cheap multi-start from a small number of perturbations of x0 (only
when the first solve ends at a point whose objective is far above a lower
bound the problem gives, or never — this is the null policy) is not worth
default behaviour: it costs evaluations on every problem to help on a few.
The study decides whether to offer it as an option rather than a default.

**H6.** On HS38/HS56/HS63/ELLIPSOID2_20 the SQP member is cheaper than both
interior-point codes, so the barrier-stall question is resolved by routing
rather than by a new barrier rule. Falsified if SQP is not cheaper there.

---

## 4. Studies (before code)

**S-A. Mathematics document (`docs/20_SQP_MATHEMATICS.md`).** Derive and
cite, with enough detail to implement from: the local SQP step and its
equivalence to Newton on the KKT system; the QP subproblem with bounds kept as
bounds; the dual active-set method of Goldfarb and Idnani for strictly convex
QPs and what to do when `B` is only positive semidefinite; warm starting;
inconsistent linearizations and the elastic/ℓ1 relaxation (SNOPT's elastic
mode, `fmincon`'s "always feasible" reformulation); globalization by an ℓ1
merit function with Han–Powell/Powell penalty updates versus a filter, the
Maratos effect and the second-order correction; damped BFGS (Powell) and the
conditions for superlinear convergence; termination and multiplier scaling;
what MathWorks documents about `fmincon`'s `sqp` algorithm (differences from
`active-set`: strict feasibility with respect to bounds, non-finite retreat,
refactored linear algebra, the relaxed QP) and its defaults
(`MaxFunctionEvaluations = 100·n`, `MaxIterations = 400`). Every fact is
sourced; every design choice for mincon is recorded with the alternative it
was preferred to.

**S-B. Basin study (H4).** For each non-attained record with a valid point:
recompute the KKT residual with the corpus' exact derivatives; recover
multipliers; identify the active set; form the Hessian of the Lagrangian and
project it onto the null space of the active constraint Jacobian; report the
smallest eigenvalue. Classify each case: strict local minimum / degenerate
KKT point / saddle or not KKT (defect). Then trace the iterate path for the
defects to find where the basin is chosen. Output:
`bench/results/r3-basins/` and a table in the atlas with the policy for each
case.

**S-C. Budget study (H3).** From every mincon record with an iterate trace
(diagnostic track exists for the held-out rounds; re-run development with
`--trace` where needed), compute per-iteration best-objective and infeasibility
series; design a stall statistic; measure its earliest firing time on the
attained records (must never fire before attainment) and on the budget
failures (should fire early). Also run `fmincon-ip` with
`MaxFunctionEvaluations = 15 000` and `= 100 000` on the 15 problems where it
hit the cap, so the cap question is answered for the competitor as well as for
mincon (a diagnostic track, clearly labelled as non-default). Output:
`bench/results/r4-budget/`.

**S-D. QP subsolver design fixtures.** Before writing the solver, write the
fixtures that must pass: random strictly convex dense QPs checked against a
KKT solve on the true active set; a degenerate QP (redundant active
constraints); an infeasible QP; a QP whose solution has every bound active;
warm-start from a perturbed neighbour with a strictly lower iteration count.

---

## 5. Implementation plan (after the studies)

**I0. Correctness first (from the studies).** D9: unscaled relative
stationarity guard on termination and/or drift-triggered rescale of the
objective, ablated on the development split with BADSTART_DISC and
QUADSPHERE2_300 as fixtures. D10: bound-aware stationarity test in the
restoration phase so INFEASIBLE_NL ends `LocallyInfeasible`. Progress verdict
note at budget exits. `trace` exposed in the Python result (done).

**I1. `mincon-qp`.** Dual active-set method (Goldfarb–Idnani) on a dense
Cholesky factor of `B`, with bounds treated as constraints of the same form
(the subproblem is dense and small, n ≤ a few hundred; sparse is deferred).
Regularization when `B` is not positive definite (it always is with damped
BFGS; with an exact Hessian, shift by the inertia-correction rule already in
`mincon-ip`). Iteration cap and cycling guard. Reports infeasibility so the
elastic relaxation can be entered. Fixtures of S-D as tests.

**I2. `mincon-sqp`.** Damped BFGS on the Lagrangian (`mincon_ip::bfgs`
reused), the QP of I1 with elastic relaxation on inconsistency, globalization
chosen in S-A (the specification `docs/03_SPEC_SQP.md` argues for an ℓ1 merit
function to decorrelate from the interior-point filter; the mathematics
document decides after weighing the Maratos and penalty-parameter
difficulties), second-order correction, non-finite retreat, exact bound
satisfaction at every iterate, the same finite-difference machinery and
error-aware termination as `mincon-ip`, the same `SolveReport` fields and
notes. Fixture gate: every Hock–Schittkowski fixture the interior-point member
passes; HS13 and TORTURE_INFEASIBLE at least as good.

**I3. Portfolio routing.** `auto` chooses the first member by problem shape
(n, m, presence of equalities) with the other as fallback under the shared
budget; the choice is an ablation on the development split, frozen before
validation.

**I4. Budget policy.** Implement the stall detector of S-C as a termination
status (`NoProgress` with a note giving the window and the best point), keep
`max_evaluations` as a user cap with a default the study chooses (the null
hypothesis is "no cap, stall detector only").

**I5. Basin policy.** Whatever S-B decides: at minimum, a note when the
returned point is a KKT point at which a negative-curvature direction was
detected during the solve (the inertia correction already knows), and
documentation of the local-minimum semantics. Optional restarts only as an
option, unless the study shows a default that costs nothing on the rest of
the corpus.

---

## 6. Measurement and gates

Every increment goes through the frozen protocol: development split for
design and ablation, validation split for the go/no-go, then a fresh held-out
set (round 3, since every current problem has now been looked at) for the
qualification. Round 3 needs new families that are not in the corpus; they
are generated before the SQP member is tuned and stay unread until
qualification.

| gate | statistic | pass |
|---|---|---|
| G1 SQP member alone, dev | attained; paired evaluation ratio on jointly attained problems | ≥ mincon-ip attained; ratio < 1 with the interval excluding 1 on n ≤ 20 |
| G2 portfolio, validation | attained | > each member alone; no loss vs C4 |
| G3 budget policy, all records | attainments preserved; evaluations on budget failures | 100 %; ≤ 0.2× |
| G4 basins | classification table complete | every non-attainment classified; defects fixed or filed with mechanism |
| G5 round-3 held-out | attained, evaluations, wall time vs fmincon-ip and fmincon-sqp | reported as measured; claims updated in `docs/17_CLAIM_AUDIT.md` |

The claim being tested is the one in the mission: an honest advantage for
working researchers. A portfolio that attains more than fmincon at defaults,
in fewer evaluations than fmincon-ip on small problems and within the noise
of fmincon-sqp, at an order of magnitude less wall time, with a stall verdict
instead of a silent cap — that is the target. If the SQP member does not earn
its place on validation it is not shipped in `auto`.

---

## 7. Order of work and reporting

1. S-A mathematics document (report).
2. S-B basin study and S-C budget study, in parallel from existing records
   (report with tables).
3. S-D fixtures, then I1 QP subsolver (report: fixture results).
4. I2 SQP member; fixture gate; dev ablation (report: G1).
5. I3 routing + I4 budget policy + I5 basin policy; validation (report: G2, G3).
6. Round-3 held-out set generated, sealed, run; docs; commit; push (report: G5).

Each step is small enough to report on before the next begins.

---

## 8. Outcome (September 11, 2026)

| gate | result | where |
|---|---|---|
| G1 SQP member alone, dev | met: 134/143 vs 135/143 for the IP member; evaluations 0.78× [0.70, 0.86], 0.72× on n ≤ 5 | `bench/results/abl-sqp1` |
| G2 portfolio, validation | met on development material: 137/143 vs 135 and 134 for the members alone; 0.775× [0.71, 0.84] the previous candidate | `bench/results/abl-sqp2` |
| G3 budget policy | H3 falsified; policy = no default cap + progress verdict; D10 fixed (INFEASIBLE_NL 9784 → 105 evaluations); fmincon at 15k/100k attains 9/13 of its 17 cap-ended problems vs mincon's 14 | `bench/results/r4-budget` |
| G4 basins | every non-attainment classified; D9 found and fixed (+1 attained, 1.014× evaluations); HS33 saddle handled in SQP | `bench/results/r3-basins`, `abl-i0` |
| G5 round-3 held-out | 12/12 vs fmincon-ip 10/12 and fmincon-sqp 12/12; evaluations 0.81 [0.53, 1.05] vs ip, 1.13 [0.97, 1.38] vs sqp; wall 0.077 / 0.35; contract intervals not met at n = 12 | `bench/results/s6v3-final3`, `s6v3-timing` |

Hypotheses: H1 confirmed (0.78×, interval excludes 1); H2 confirmed on
development material; H3 falsified; H4 resolved (strict local minima except
BADSTART_DISC = D9); H5 confirmed (multi-start not a default); H6 partly:
the SQP member is cheaper than the IP member on HS63 (57 vs 303) and HS38
(174 vs 243), the stall question is resolved by routing for n ≤ 20.
