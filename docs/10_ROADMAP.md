# Roadmap

Milestones with **objective exit criteria**. A milestone is done when its gate
passes, measured, not when the code looks finished.

Order is by expected value, not by convenience. Reorder only with a reason
written down.

---

## Status board

| | Milestone | Gate | State |
|---|---|---|---|
| M0 | Working baseline | 52/54 testset, 0 lies | ✅ **done** |
| M1 | Feasibility restoration | `TORTURE_INFEASIBLE` → `LocallyInfeasible`; CUTEst +8pp | ⬜ |
| M2 | AD bridge + docs | JAX/PyTorch/CasADi examples; ≥10x fewer evals with `jac` | ⬜ |
| M3 | AMD ordering | fill ratio ≥3x better than RCM on `n>1000` | ⬜ |
| M4 | Limited-memory BFGS | `n = 10 000` solves in bounded memory | ⬜ |
| M5 | SQP + QP solver | portfolio rate > each member's | ⬜ |
| M6 | Full CUTEst run | published numbers vs SciPy and IPOPT | ⬜ |
| M7 | Batched Python evaluation | ≥5x fewer GIL crossings | ⬜ |
| M8 | Beat `fmincon` | ≥76% plug-and-play on constrained CUTEst | ⬜ |
| M9 | Warm starting, C ABI, adaptive `mu` | — | ⬜ |

---

## M0 — Working baseline ✅

**Done.** What exists:

* Sparse `LDL^T` with dynamic regularization, certified inertia and a growth
  guard; verified against an independent Jacobi eigensolver.
* Primal-dual interior point: filter line search, Algorithm IC, second-order
  corrections, fraction-to-boundary, bound-multiplier reset, scaled `E_mu`.
* Gradient-based scaling, bounds-aware finite differences, CPR coloring,
  sparsity detection, derivative checker, damped BFGS.
* Portfolio driver with deterministic ranking.
* Python wheel (`abi3-py39`) that installs and solves HS71 to published
  accuracy with no derivatives.
* 122 unit tests; 52/54 on `mincon-testset`; **zero false reports of success**.

**Known failures, both understood:** `HS13` (MFCQ fails at the solution) and
`TORTURE_INFEASIBLE` (refuses to lie, but reports `NumericalFailure` instead of
`Infeasible`). Both are M1.

---

## M1 — Feasibility restoration ★ the biggest gap

Spec: `docs/02_SPEC_INTERIOR_POINT.md` §8.

A line-search failure currently ends the solve. Restoration minimizes
infeasibility and re-enters, and it is also what turns "I could not converge"
into the *certificate* "this problem is locally infeasible".

Build in order:

1. **Algorithm R** first — plain Newton steps on the primal-dual system with
   fraction-to-boundary and no line search, accepted while `||F_mu||_1` falls
   by `kappa_F = 0.999`. Cheap and it rescues a large fraction of cases.
2. Then the full phase: minimize `rho*sum(p+n) + (zeta/2)||D_R(x - x_R)||^2`
   subject to `c(x) - p + n = 0`, reusing the same KKT machinery with the
   objective replaced.
3. Leave when the iterate is filter-acceptable **and**
   `theta <= 0.9 * theta_R`.

**Gate**
* `TORTURE_INFEASIBLE` reports `LocallyInfeasible`, not `NumericalFailure`.
* `HS13` reports `StepTolerance` or `Acceptable` at `f ≈ 1.0`.
* `mincon-testset` reaches **54/54**.
* CUTEst `constrained-small` success rate rises by **≥8 percentage points**.
* Still zero false successes.

---

## M2 — The AD bridge ★ highest value per hour

Spec: `docs/05_SPEC_DERIVATIVES.md` §1.1.

Not a research project — plumbing and documentation. Most Python users already
have exact derivatives available and do not know they can hand them over.

1. Accept a **sparse** Jacobian from Python (SciPy sparse), not just dense.
2. Accept `jac_sparsity=` to declare the pattern without computing values.
3. Worked, tested examples for JAX (`jax.grad`, `jax.jacrev`), PyTorch
   (`torch.func`), and CasADi.
4. Make `check_gradients` the first line of the troubleshooting docs.

**Gate**
* A 500-variable problem with a JAX gradient uses **≥10x fewer objective
  evaluations** than the finite-difference path, at equal or better accuracy.
* Every example runs in CI.

---

## M3 — AMD ordering

Spec: `docs/04_SPEC_LINEAR_ALGEBRA.md` §4. Wire in the `amd` crate (0.2.2,
BSD-3) behind `Ordering::Amd`.

**Gate**
* Fill ratio (`Symbolic::fill_ratio`) at least **3x better than RCM** on
  CUTEst problems with `n > 1000`.
* Factorization time strictly better on the same set.
* `all_orderings_agree_on_the_solution` still passes.

---

## M4 — Limited-memory BFGS

Compact Byrd–Nocedal–Schnabel form, so it enters the KKT matrix as a low-rank
update rather than forcing a matrix-free method. Dense BFGS caps usable `n` at
a couple of thousand — the same limit `fmincon`'s default has.

**Gate**
* A separable problem with `n = 10 000` solves in memory bounded by
  `O(n * history)`.
* No regression on `constrained-small` against dense BFGS.

---

## M5 — SQP and the QP solver

Spec: `docs/03_SPEC_SQP.md`. Dual active-set QP with warm starting; `l1` merit
function (**deliberately not a filter**, so the portfolio members fail
differently); mandatory second-order corrections; elastic always-feasible
subproblem.

**Gate** — all four:
1. Hock–Schittkowski rate at least equal to interior point's.
2. `HS13` and `TORTURE_INFEASIBLE` strictly better than interior point's.
3. Median evaluations on `constrained-small` below interior point's.
4. **The portfolio's combined rate strictly exceeds each member's alone.** If
   not, the members are too correlated and the design has failed.

---

## M6 — The full CUTEst run

`runner.py --set all --solvers all`, against SciPy SLSQP, SciPy trust-constr,
COBYLA, and IPOPT via `cyipopt` where available.

**Gate**
* All 1075 problems attempted; loads that fail are listed with reasons.
* Performance and data profiles published in the README.
* **Zero `lied` entries** for `mincon` in the summary table.
* Results reproducible from a single scripted command.

---

## M7 — Batched Python evaluation

Optional vectorized protocol: the model advertises that it takes `(k, n)` and
returns `k` values, so a whole coloring group crosses the GIL once.

**Gate**
* ≥5x fewer GIL crossings on a finite-differenced NumPy model.
* A scalar `lambda x: ...` model still works unchanged.

---

## M8 — Beat `fmincon`

**Gate**
* **≥76% success on the constrained CUTEst set under plug-and-play settings**
  (no derivatives, default options), which is `fmincon` interior-point's
  published 75.9% on the comparable set.
* Wall clock competitive or better on the problems both solve.
* An honest section in the README naming the problem classes where `fmincon`
  still wins.

That last bullet is not humility for its own sake. A benchmark with no losses
in it is a benchmark nobody believes.

---

## M9 — Everything else

Warm starting; C ABI (Julia, R, C++); adaptive barrier update; watchdog;
Ruiz equilibration; dense Bunch–Kaufman path for small problems; a true star
coloring; native forward-mode AD; the `fmincon`-shaped façade; least-squares
multiplier initialization.

---

## Rules for working through this

1. **Every milestone lands with its gate measured**, and the number goes in the
   README. A milestone whose gate was not run is not done.
2. **Never tune on the reported set.** Development on `constrained-small` and
   `mincon-testset`; hold the rest out. Anyone can win a performance profile by
   fitting to it.
3. **A test failure is a hypothesis, not a verdict.** Establish whether the
   test is wrong first. Two bugs found while building M0 were in the *test set*:
   `HS16` has a second local minimum the published value does not mention, and
   `TORTURE_DEGENERATE` was accidentally unbounded — the solver was right both
   times.
4. **Zero false successes is a release blocker**, at every milestone, forever.
5. **Keep the module docs true.** Every "not implemented" note in the source is
   load-bearing; delete it in the same commit that implements the thing.
