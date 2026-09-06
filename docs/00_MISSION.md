# Mission

Build the nonlinear constrained optimizer that a working scientist reaches for
first, and make it free.

```
pip install mincon
```

---

## The thesis, stated so it can be falsified

**Claim.** A solver can be built that beats MATLAB's `fmincon` on
*plug-and-play robustness* — the case where a user supplies only a function and
a starting point — while being free, embeddable and faster.

**Why it is plausible.** `fmincon`'s advantage is not algorithmic. Its
interior-point method is the KNITRO design from 1999–2006, published and
well understood. Its SQP is Han–Powell with damped BFGS, also textbook. What
`fmincon` has is two decades of *defensive engineering* and defaults that
work. Those are reproducible. Meanwhile it carries four fixable deficits:
scaling off by default, no automatic differentiation for function handles, no
sparsity exploitation for nonlinear Jacobians, and one algorithm per call.

**Why it is hard.** The gap between "implements the algorithm in the paper" and
"survives a real engineering model" is enormous and consists almost entirely of
cases nobody writes papers about: `NaN` from a diverged simulation, a model
undefined outside its box, constraints that are inconsistent by 1e-12, a
Jacobian that loses rank at the solution, multipliers that diverge because
LICQ fails. `fmincon` handles all of these. Most open-source solvers do not.

**How we will know.** `bench/` runs the full CUTEst set through S2MPJ with a
success criterion computed by the harness rather than reported by the solver.
The target is stated in `docs/10_ROADMAP.md` and the current standing is in
`bench/README.md`. If the numbers do not come, the thesis is wrong and the
README will say so.

---

## The four wins, in order of expected value

### 1. Defaults that are correct without being told

Every option we expose is a small failure. Every option a user *must* set to
get a solve is a large one.

* **Scaling on by default** (gradient-based, IPOPT's method). `fmincon` ships
  `ScaleProblem = false`. Measured: `TORTURE_SCALING`, with twelve orders of
  magnitude between objective and constraint gradients, solves in **one
  iteration** with scaling on.
* **Sparsity detected, not assumed absent.**
* **Iteration caps that scale with `n`** instead of a flat 400.
* **Two-tier termination** — converged and "acceptable" — reported distinctly,
  so a usable answer is not thrown away and a mediocre one is not dressed up.

### 2. An algorithm portfolio

Interior point and SQP fail on *different* problems. `fmincon` makes the user
choose and the documentation's advice is "try the other one". We race them and
return the best answer, deterministically ranked. On a multi-core machine it
costs wall clock, not time; on one core the members run in sequence with early
exit, so a problem the default handles costs exactly what it did before.

No single `fmincon` call can do this. It is the cheapest robustness multiplier
available and it is why `mincon-sqp` is milestone M5 rather than M9.

### 3. Sparse-native everything, including derivatives

Curtis–Powell–Reid coloring turns a finite-difference Jacobian from `n`
evaluations into a handful. Measured: a tridiagonal Jacobian compresses to **3
evaluations regardless of `n`**. `fmincon` does not do this for nonlinear
constraint Jacobians.

### 4. Being free, embeddable and installable

`pip install mincon`, no licence server, no MATLAB, no Fortran toolchain, no
HSL. This is the win that reaches people the others never will: the student,
the CI job, the reproduction script, the library that wants to depend on a
solver without dragging in a licence.

**This is also an architectural constraint, not a marketing line.** IPOPT's
default linear solver is HSL's `MA57`, which is not redistributable — which is
precisely why `pip install ipopt` does not exist. Every dependency in this
workspace is MIT/Apache/BSD, and `mincon-linalg` writes its own sparse `LDL^T`
rather than accept that constraint. See `docs/04_SPEC_LINEAR_ALGEBRA.md`.

---

## What we must not lose

These are `fmincon`'s real strengths. Matching them is a hard requirement.

1. **Bounds honoured at every iterate**, including finite-difference probes.
   Engineering models are full of quantities that must be positive.
2. **Non-finite tolerance.** `NaN`/`Inf` from the model shortens the step; it
   never ends the solve.
3. **Trustworthy exit flags.** `fmincon`'s `-2` means there really is no
   feasible point, and users have built two decades of habits on that. A solver
   that reports success on an infeasible problem has done something worse than
   fail: a user will build on that answer.
4. **Diagnostics.** When it does not work, the user must be able to tell why.

`mincon-testset`'s `torture` module exists to hold us to all four. Its pass
criterion for three problems is *reporting the right failure*.

---

## Non-goals

Saying no is what makes the rest possible.

* **Global optimization.** Multi-start is a wrapper someone else can write.
* **Mixed-integer.** Different field.
* **Modelling language.** casADi and JuMP exist and are good. Be a solver they
  can call.
* **Beating SLSQP on tiny dense problems by evaluation count.** SQP owns that
  regime; that is why we will have one, not why we should contort the
  interior-point method.
* **Every algorithm.** Two good ones beat five mediocre ones.

---

## The honesty rule

This project is measuring itself against a product with a twenty-year head
start. The temptation to flatter the numbers will be constant and it must be
refused, because the moment the benchmark is not trustworthy the whole project
is worthless — a solver nobody believes is worse than no solver.

Concretely:

* Success is decided by the harness, never by the solver's own report.
* The portfolio's cost is reported across **all** members, not just the winner.
* `Acceptable` is not counted as success.
* Known gaps are named in the README and in the module docs, not buried.
* When a test fails, first establish whether the test is wrong. Two of the
  bugs found while building this were in the *test set*: `HS16` has a second
  local minimum the published value does not mention, and `TORTURE_DEGENERATE`
  was accidentally unbounded. Both were found because the solver disagreed and
  the disagreement was investigated instead of suppressed.

If `fmincon` is better on some class of problems, the documentation says so and
names the class. That is how a serious tool behaves, and it is the only way
anyone will trust the cases where we win.
