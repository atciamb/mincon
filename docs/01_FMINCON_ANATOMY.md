# The anatomy of `fmincon`

*Know what you are actually up against.*

This document is the competitive analysis. It is deliberately generous to
`fmincon`, because the fastest way to lose to it is to underestimate it, and
because most of what makes it good is invisible in the algorithm description.

---

## 1. What `fmincon` actually is

`fmincon` is not one algorithm. It is four, plus a large amount of defensive
engineering, behind a single call.

```matlab
x = fmincon(fun, x0, A, b, Aeq, beq, lb, ub, nonlcon, options)
```

solving

```
min f(x)  s.t.  A x <= b,  Aeq x = beq,  c(x) <= 0,  ceq(x) = 0,  lb <= x <= ub
```

| Algorithm | Basis | Where it wins |
|---|---|---|
| `interior-point` **(default)** | Byrd–Hribar–Nocedal (1999) and Waltz–Morales–Nocedal–Orban (2006) — the KNITRO line | Large, sparse, many inequality constraints. The most robust general choice. |
| `sqp` | Nocedal & Wright ch. 18, Han, Powell, plus Spellucci and Tone for feasibility | Small to medium, expensive objectives, models that misbehave outside their box |
| `sqp-legacy` | as `sqp`, older linear algebra | compatibility |
| `active-set` | Gill–Murray–Wright dense active-set QP | small dense problems, historical |
| `trust-region-reflective` | Coleman–Li interior reflective Newton | **bounds only, or linear equalities only**; needs a user gradient |

### 1.1 The interior-point algorithm

Solves a sequence of barrier subproblems

```
min f(x) - mu * sum_i ln(s_i)   s.t.  h(x) = 0,  g(x) + s = 0,  s >= 0
```

and drives `mu -> 0`. Each iteration first attempts a **direct Newton step** on
the KKT system. If that fails — the matrix is not factorizable, or the step is
not a descent direction — it falls back to a **conjugate-gradient step inside a
trust region**. Progress is judged by the merit function

```
f_mu(x, s) + nu * || (h(x), g(x) + s) ||
```

with `nu` raised as needed.

This direct-step-with-CG-fallback structure is exactly KNITRO's, and it is
worth noticing what it buys: the direct step gives fast local convergence, and
the trust-region CG step gives a globally convergent fallback that works even
when the Hessian is indefinite and even matrix-free. It is a genuinely good
design.

Relevant options and defaults:

| Option | Default | Note |
|---|---|---|
| `HessianApproximation` | `'bfgs'` | **dense** BFGS, `O(n^2)`. `'lbfgs'` and `'finite-difference'` available |
| `SubproblemAlgorithm` | `'factorization'` | or `'cg'` |
| `BarrierParamUpdate` | `'monotone'` | or `'predictor-corrector'` |
| `HonorBounds` | `true` | every iterate inside the box |
| `ScaleProblem` | **`false`** | see §3.1 |
| `EnableFeasibilityMode` | `false` | an alternative route to feasibility |
| `InitBarrierParam` | `0.1` | |

### 1.2 The SQP algorithm

At each iterate, solve

```
min 1/2 d' H d + grad f(x)' d
s.t. linearized constraints
```

with `H` a **damped BFGS** approximation to the Hessian of the Lagrangian
(Powell's update, which keeps `H` positive definite even though the Lagrangian
Hessian is indefinite at the solution), then a line search on the Han–Powell
merit function with penalty update `r_i = max(lambda_i, (r_i + lambda_i)/2)`.

MathWorks lists four distinguishing features, and every one of them is a
*robustness* feature rather than a convergence-rate feature:

1. **Bounds are strictly satisfied at every iterate**, including during finite
   differencing.
2. **Tolerant of non-`double` results.** If the objective or constraints return
   `Inf`, `NaN`, or a complex number, it takes a smaller step and tries again.
3. **Reformulated QP subproblem** that is always feasible, so an inconsistent
   linearization does not stop the solve.
4. More efficient linear algebra than `active-set`.

Points 1 and 2 are the ones to internalize. They are not in any textbook's
convergence theory and they are a large part of why practitioners reach for
`fmincon`.

---

## 2. The defaults, in full

| Option | Default |
|---|---|
| `Algorithm` | `'interior-point'` |
| `OptimalityTolerance` | `1e-6` |
| `ConstraintTolerance` | `1e-6` |
| `StepTolerance` | `1e-6` (`1e-10` for `sqp`) |
| `MaxIterations` | `400` |
| `MaxFunctionEvaluations` | `100 * numberOfVariables` |
| `FiniteDifferenceType` | `'forward'` |
| `FiniteDifferenceStepSize` | `sqrt(eps)` forward, `eps^(1/3)` central |
| `TypicalX` | `ones(n,1)` |
| `SpecifyObjectiveGradient` | `false` |
| `SpecifyConstraintGradient` | `false` |
| `UseParallel` | `false` |
| `ScaleProblem` | `false` |
| `ObjectiveLimit` | `-1e20` |

Exit flags:

| Flag | Meaning |
|---|---|
| `1` | first-order optimality and constraints satisfied |
| `2` | step size below tolerance, constraints satisfied |
| `3` | objective change below tolerance (`trust-region-reflective`) |
| `4`, `5` | direction / directional-derivative criteria (`active-set`) |
| `0` | iteration or evaluation limit reached |
| `-1` | stopped by an output or plot function |
| `-2` | no feasible point found |
| `-3` | converged to an infeasible point (`interior-point`, `sqp`) |

The exit flags matter more than they look. `-2` on `fmincon` is *trustworthy*,
and users have built two decades of habits on that. Matching that
trustworthiness is a hard requirement, not a nice-to-have — see
`docs/11_PITFALLS.md`.

---

## 3. Where `fmincon` is genuinely strong

Do not plan to win these. Plan to match them.

**Plug-and-play robustness.** Hand it a MATLAB function and a starting point
and it works. The most directly comparable published figure is **75.9%**
convergence on the 30 constrained problems of Kronqvist et al.'s benchmark
under default settings, ahead of KNITRO's interior-point (74.6%) and SNOPT
(72.1%). No open-source solver has a comparable reputation for
just-working-without-being-told-anything.

**Bound honouring.** Every iterate and every finite-difference probe stays in
the box. Engineering models are full of quantities that must be positive, and
a solver that steps outside gets `NaN` and is blamed for it.

**Non-finite tolerance.** `Inf`/`NaN`/complex from the model shortens the step
instead of ending the solve.

**Diagnostics.** The iteration display, the first-order optimality measure, the
`output` struct, `CheckGradients`, and honest exit flags. Much of what a user
experiences as "it works" is really "when it doesn't, I can tell why".

**Integration.** It is in MATLAB, next to the person's data, their model, their
plots and their colleagues.

---

## 4. Where `fmincon` is weak

This is the attack surface. Each entry states what to build.

### 4.1 No scaling by default — *the biggest single opening*

`ScaleProblem` defaults to `false`. IPOPT's `nlp_scaling_method` defaults to
`gradient-based`. A model whose objective is in joules and whose constraints
are in millimetres has a KKT matrix with a condition number in the millions
before anything has gone wrong, and `fmincon` will grind on it.

> **Build:** gradient-based scaling on by default. `mincon-testset`'s
> `TORTURE_SCALING` — twelve orders of magnitude between objective and
> constraint gradients — currently solves in **one iteration** with scaling on.

### 4.2 No automatic differentiation for function handles

MATLAB's problem-based workflow supports AD for *optimization expressions*, but
`fmincon` called with a function handle finite-differences everything. Forward
differences give `sqrt(eps) ~ 1e-8` relative error in the gradient, which caps
the achievable optimality tolerance and costs `n` evaluations per gradient.

> **Build:** first-class analytic derivatives, a bridge to the user's AD
> framework (JAX, PyTorch, CasADi), and eventually native AD. See
> `docs/05_SPEC_DERIVATIVES.md`.

### 4.3 No sparsity exploitation for nonlinear constraint Jacobians

`fmincon` accepts `JacobPattern` and `HessPattern` for some algorithms, but it
does not *detect* structure, and it does not do Curtis–Powell–Reid coloring for
finite-difference constraint Jacobians. A tridiagonal Jacobian in 1000
variables costs it 1000 evaluations where three would do.

> **Build:** detection plus coloring, both already in `mincon-diff`. Measured:
> a tridiagonal Jacobian compresses from `n` evaluations to 3.

### 4.4 Dense BFGS by default

`O(n^2)` memory in the default interior-point configuration. `'lbfgs'` exists
but is not the default and does not integrate as well.

> **Build:** limited-memory BFGS in the compact Byrd–Nocedal–Schnabel
> representation, so it enters the KKT matrix as a low-rank update.

### 4.5 One algorithm per call

The user must choose, and the documentation's advice amounts to "try another
one if this fails". A user who guesses wrong concludes their problem is hard.

> **Build:** the portfolio. Race configurations that fail differently and
> return the best. `mincon::portfolio`. This is the single cheapest robustness
> multiplier available and no `fmincon` call can do it.

### 4.6 No warm starting

Re-solving a slightly perturbed problem starts over. Parametric studies,
sensitivity analysis and MPC all do exactly this.

### 4.7 It is proprietary

* It cannot go in CI, in a container, in a teaching notebook, in a paper's
  reproduction script, or inside someone else's library.
* A student without a licence cannot run your model.
* Results are not reproducible across MATLAB versions and the changes are not
  documented at the algorithmic level.

> This is not a technical weakness, and it is the largest one. `pip install
> mincon` reaches people `fmincon` never will, and being embeddable is a
> capability, not a licensing footnote.

---

## 5. The competitive landscape beyond `fmincon`

| Solver | Licence | Method | Strengths | Weaknesses |
|---|---|---|---|---|
| **IPOPT** | EPL | interior point, filter line search | Excellent, sparse, mature, the open-source reference | Needs HSL (not redistributable) or MUMPS; awkward to install; C++/Fortran |
| **KNITRO** | commercial | IP + active set, multi-algorithm | Very strong, has its own portfolio | Expensive |
| **SNOPT** | commercial | SQP, reduced Hessian | Superb for few degrees of freedom | Expensive, dense reduced Hessian |
| **WORHP** | free for academics | SQP | ESA-grade, sparse | Restrictive licence |
| **SciPy `SLSQP`** | BSD | dense SQP (Kraft, 1988 Fortran) | Fast on small problems, always available | Dense, no sparsity, weak on large or ill-conditioned problems, `n` limited |
| **SciPy `trust-constr`** | BSD | interior point / trust region | Handles sparsity, decent | Pure Python, slow, less robust than IPOPT |
| **NLopt** | LGPL/MIT | many | Wide algorithm choice | Constrained methods are the weak part |
| **Ceres** | BSD | Levenberg–Marquardt | Superb for least squares | Not a general NLP solver |
| **casADi** | LGPL | modelling + AD, wraps IPOPT | Best-in-class AD | It is a front end; the solver is still IPOPT |

**The gap in the market is precise:** IPOPT's algorithmic quality, with SciPy's
installation experience, and `fmincon`'s tolerance for models that misbehave.
Nothing occupies that spot.

### The HSL problem, and why it decides our architecture

IPOPT's default linear solver is `MA27`/`MA57` from HSL, which is not
redistributable under a permissive licence. That single fact is why `pip
install ipopt` is not a thing, why `cyipopt` builds are fragile, and why a
large share of people who would benefit from IPOPT use SciPy instead.

Avoiding that dependency is not a detail; it is the reason `mincon-linalg`
exists and writes its own sparse `LDL^T`. See
`docs/04_SPEC_LINEAR_ALGEBRA.md`.

---

## 6. Summary: the five things that decide this

1. **Defaults that work untuned.** Scaling, tolerances, algorithm choice,
   sparsity — all decided for the user, correctly.
2. **A portfolio.** Robustness bought with compute rather than with the user's
   judgement.
3. **Never lie.** Exit flags as trustworthy as `fmincon`'s. An infeasible
   problem reported as solved is worse than any amount of slowness.
4. **Survive bad models.** Bounds honoured, `NaN` tolerated, restricted domains
   respected.
5. **Be installable.** `pip install mincon`, no licence, no Fortran, no HSL.

Speed is sixth. It matters, and Rust gives it to us nearly for free, but nobody
switches solvers for speed on a problem the old solver already solved.

## Sources

- [Constrained Nonlinear Optimization Algorithms — MathWorks](https://www.mathworks.com/help/optim/ug/constrained-nonlinear-optimization-algorithms.html)
- [`fmincon` reference — MathWorks](https://www.mathworks.com/help/optim/ug/fmincon.html)
- [First-order optimality measure — MathWorks](https://www.mathworks.com/help/optim/ug/first-order-optimality-measure.html)
- [Kronqvist et al., *Nonlinear Programming Solvers for Unconstrained and Constrained Optimization Problems: a Benchmark Analysis*, arXiv:2204.05297](https://arxiv.org/pdf/2204.05297)
- [Wächter & Biegler, *On the implementation of an interior-point filter line-search algorithm*, Math. Prog. 106(1), 2006](https://link.springer.com/article/10.1007/s10107-004-0559-y)
- [IPOPT options reference](https://coin-or.github.io/Ipopt/OPTIONS.html)
