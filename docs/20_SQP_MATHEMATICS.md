# The mathematics of the SQP member

Written September 11, 2026 for the plan in `docs/19_SQP_RD_PLAN.md`, before
any implementation. Section 9 records the decisions this document commits
mincon to and the alternative each was preferred over. Section 10 lists the
sources; facts about `fmincon` are from the MathWorks documentation read on
this date (R2025b) and are marked (MW).

Notation. The problem is

```
min f(x)   s.t.  c_L <= c(x) <= c_U,   x_L <= x <= x_U,        x in R^n, c: R^n -> R^m
```

with `g = grad f`, `J = dc/dx` (m × n), Lagrangian
`L(x, λ, z) = f − λᵀc − zᵀx` in a sign convention where `λ_i ≥ 0` for an
active upper row, `λ_i ≤ 0` for an active lower row (a lower/upper pair on
the same row never being active together), and likewise `z` for bounds. The
KKT conditions are

```
g(x) − J(x)ᵀ λ − z = 0,      c_L <= c(x) <= c_U,   x_L <= x <= x_U,
λ_i (c_i − c_U,i) = 0 with λ_i >= 0,   λ_i (c_i − c_L,i) = 0 with λ_i <= 0,   (same for z),
```

which is the oracle's stationarity form `grad f + Jᵀlam − zL + zU` after a
sign change. mincon's own crates use the `l ≤ c ≤ u` form throughout, so this
document keeps it and never rewrites to `c(x) ≤ 0`.

---

## 1. The local SQP step is Newton's method on the KKT system

For equality constraints only, `c(x) = 0`, Newton's method on the KKT system
`∇L = 0, c = 0` from `(x_k, λ_k)` solves

```
[ W_k   −J_kᵀ ] [ d      ]   [ −g_k + J_kᵀ λ_k ]
[ J_k     0   ] [ λ_{k+1}] = [ −c_k            ]        W_k = ∇²_xx L(x_k, λ_k)   (1.1)
```

after the substitution `λ_{k+1} = λ_k + Δλ`. Exactly the same `(d, λ_{k+1})`
is the primal–dual solution of the equality-constrained QP

```
min_d  ½ dᵀ W_k d + g_kᵀ d      s.t.   J_k d + c_k = 0.                          (1.2)
```

That equivalence (Wilson 1963; Han 1977; Powell 1978) is the whole idea: the
QP is Newton's method written so that inequalities can be added. With
inequalities the QP becomes

```
min_d  ½ dᵀ B_k d + g_kᵀ d
s.t.   c_L − c_k <= J_k d <= c_U − c_k                                            (1.3)
       x_L − x_k <=   d   <= x_U − x_k
```

and Robinson's theorem (1974) gives the local result: if `x*` satisfies the
second-order sufficient conditions with strict complementarity and LICQ, then
for `(x_k, λ_k)` close to `(x*, λ*)` the QP (1.3) with `B_k = W_k` has a local
solution whose active set is the active set of the NLP, and the iterates
converge quadratically. So near the solution SQP identifies the active set in
finitely many steps and then behaves as Newton on the equality problem (1.2)
on that set.

Two consequences shape the design.

* The bounds in (1.3) are constraints of the QP, not a projection afterwards,
  so every iterate `x_k + d` is inside the box. That is what `fmincon`'s
  `sqp` documents as "takes every iterative step in the region constrained by
  bounds" (MW), and it is what makes a model that is undefined outside its
  box solvable at all.
* The multipliers come out of the QP for free and are the estimates used for
  the Hessian update and the termination test. There is no barrier parameter
  to drive to zero and no centrality to maintain; the price is that the
  active-set choice is combinatorial and made by the QP solver.

With inequalities, quadratic convergence needs `B_k → W_k` on the relevant
subspace; with a quasi-Newton `B_k` the rate is superlinear at best (§4).

---

## 2. The QP subproblem and its solver

### 2.1 What the SQP loop needs from the QP solver

Each major iteration solves a small dense QP (in this corpus `n ≤ 1000`,
usually `n ≤ 20`, `m ≤ n`) whose data changes slightly from the previous one,
whose Hessian `B_k` is positive definite (damped BFGS, §4) or made so
(exact Hessian shifted, §2.5), and whose linearized constraints may be
inconsistent (§3). The loop needs: the step `d`; multipliers `λ, z` with
correct signs; a definite verdict of infeasibility; and a cheap restart from
the previous active set, because consecutive subproblems share most of their
active set once the iterates settle.

### 2.2 Why a dual active-set method

An interior-point QP solver is polynomial and insensitive to degeneracy but
cannot use the previous active set and always needs 10–30 iterations to reach
1e-10, so the QP would cost more than the NLP step it serves. A primal
active-set method needs a feasible starting point (a phase 1, which is another
LP) and cannot cheaply detect infeasibility. A dual active-set method starts
from the unconstrained minimizer `d = −B⁻¹g`, which is always available when
`B` is positive definite, keeps dual feasibility (all working multipliers of
the right sign) at every step, adds violated constraints one at a time, and
terminates either at the optimum or with a certificate that the constraints
are inconsistent. It needs no phase 1, warm-starts from a given active set,
and on the small dense problems this member is for it typically finishes in a
number of iterations comparable to the number of active constraints. That is
the method of Goldfarb and Idnani (1983), used in R's `quadprog`, in
`eiquadprog`, and (in its Schittkowski form) in many SQP codes.

### 2.3 The Goldfarb–Idnani method

Write the QP in the form the method uses, with every general row and every
bound as a constraint `a_iᵀ d ≥ b_i` (a two-sided row `l ≤ a_iᵀd ≤ u` is two
constraints; equality rows `l = u` are treated first and kept in the working
set permanently),

```
min ½ dᵀ G d + hᵀ d    s.t.   a_iᵀ d ≥ b_i,  i = 1..p,      G symmetric positive definite.
```

*Invariant.* At every step the method holds a working set `A` (indices, `|A| =
q ≤ n`, with `N = [a_i]_{i∈A}` of full column rank) and the point `d` that
minimizes the objective subject to the working constraints as **equalities**,
with multipliers `u_A ≥ 0`. Such a pair `(d, A)` is dual feasible: it is the
solution of a relaxation of the QP, so its objective is a lower bound. The
method starts with `A = ∅`, `d = −G⁻¹h`.

*Step.* Pick a violated constraint `p` (`a_pᵀ d < b_p`), most violated by a
scaled measure. Compute

```
z = H a_p,                 H = G⁻¹ (I − N (Nᵀ G⁻¹ N)⁻¹ Nᵀ G⁻¹)        (the reduced inverse Hessian)
r = (Nᵀ G⁻¹ N)⁻¹ Nᵀ G⁻¹ a_p                                           (how the working multipliers respond)
```

`z` is the direction in which `d` moves when constraint `p` is tightened while
the working set stays satisfied as equalities; `r` is the rate at which the
working multipliers decrease as `u_p` grows. Then

```
t1 = min over i∈A with r_i > 0 of  u_i / r_i     (the first working multiplier to reach zero)
t2 = (b_p − a_pᵀ d) / (zᵀ a_p)                    if zᵀa_p > 0, else +∞   (the step that makes p active)
t  = min(t1, t2)
```

* If `t2 = ∞` and `t1 = ∞`: `a_p` is in the span of `N` (`z = 0`) and no
  working multiplier decreases along `r`, so constraint `p` cannot be
  satisfied without violating dual feasibility: **the QP is infeasible**, and
  the multipliers `(r, −1)` certify it (a Farkas direction).
* If `t = t2` (**full step**): `d ← d + t z`, `u ← (u − t r, t)`, add `p` to
  `A`. Constraint `p` is now active with multiplier `t ≥ 0`.
* If `t = t1 < t2` (**partial step**): `d ← d + t z`, `u ← u − t r`, drop the
  working constraint whose multiplier reached zero, and continue with the same
  `p`, recomputing `z, r` for the smaller working set.

Termination: no violated constraint remains, so `d` is primal feasible, dual
feasible and stationary on its working set — optimal. Each full step strictly
increases the dual objective, and the number of partial steps between full
steps is bounded by `q`, so the method terminates finitely when `G` is
positive definite (the dual objective is strictly concave in the working
multipliers, which excludes cycling except through exact ties; Goldfarb and
Idnani prove finite termination without an anti-cycling rule for strictly
convex `G`).

*Factorization.* The method never forms `H` or `(NᵀG⁻¹N)⁻¹`. With `G = L Lᵀ`
(Cholesky) it maintains `J = L⁻ᵀ Q` and `R` from the QR factorization
`L⁻¹ N = Q [R; 0]`; then with `e = Jᵀ a_p`, `z = J_2 e_2`, `r = R⁻¹ e_1`,
where the split is into the first `q` columns and the rest. Adding a
constraint appends a column to `R` and applies Givens rotations to `J`;
dropping one deletes a column and re-triangularizes with Givens rotations.
Each step is O(n²), and the whole thing is numerically stable in the sense of
Goldfarb–Idnani's title: the only inversion is of the Cholesky factor of `G`,
done once. When `N` becomes numerically rank deficient (a constraint parallel
to the working set), `|R_qq|` collapses; the guard is to treat
`zᵀa_p ≤ ε_z ‖a_p‖` as `z = 0` and to reject adding a constraint whose new
diagonal of `R` would be below a tolerance relative to the largest, in favour
of the partial-step branch (Powell 1985 discusses this dependence case).

*Warm start.* Given a candidate active set from the previous major iteration,
the method can add those constraints first as if they were violated (a
sequence of full steps at `t = t2` — if a candidate turns out not to be
binding, its multiplier goes negative and it is dropped). In practice near the
NLP solution the previous active set is the correct one and the warm-started
solve does `q` cheap additions and no drops. That is the mechanism by which
SQP's QP cost collapses to O(n²·q) as the iterates settle.

*Cost.* Cholesky of `B` O(n³) once per major iteration (or, with BFGS, an
O(n²) rank-two Cholesky update — not needed at this size), then O(n²) per QP
step.

### 2.4 Equalities, two-sided rows and bounds

Equality rows are added first, in a fixed order, and are never dropped
(their multipliers are free in sign; the `t1` test skips them). For a
two-sided row only one side can be violated at a time, and only one side can
ever be in the working set; adding the other side is refused (it would be
linearly dependent), so the pair is treated as one constraint with a side
flag. Bounds `x_L − x_k ≤ d ≤ x_U − x_k` are rows with `a = ±e_j`; nothing in
the method needs them to be special, but `Jᵀ e_j` is a column read, so they
are cheap. A fixed variable (`x_L = x_U`) is an equality row on `e_j`.

### 2.5 When `B` is not positive definite

Damped BFGS keeps `B` positive definite by construction (§4). An exact
Hessian of the Lagrangian may be indefinite; the QP then has no finite
solution unless the negative curvature lies outside the feasible cone, and the
dual method requires a Cholesky factor. The remedy is the one `mincon-ip`
already uses for inertia: `B ← B + δ I`, `δ` from a geometric sequence
starting at `1e-4·max(1, ‖B‖)` (or the previous successful `δ`/3), until the
Cholesky succeeds. A positive-definite `B` also guarantees `d` is a descent
direction for the merit function (§5). The regularized QP step is a
Levenberg–Marquardt-type compromise; it costs the quadratic rate on problems
with an indefinite Lagrangian Hessian, which quasi-Newton loses anyway.

### 2.6 Alternatives considered

* **Primal active set with a QR null-space step** — what MathWorks documents
  for `active-set`/`sqp` ("Z_k from the QR of the active constraints,
  `Z_kᵀ H Z_k p = −Z_kᵀ c`, step to the nearest constraint, drop the most
  negative multiplier") (MW). Needs a feasible starting point; when the
  current point is infeasible MathWorks solves an LP `min γ s.t. A_i x − γ ≤
  b_i` first (MW). A dual method avoids that LP.
* **Interior-point QP** — reuses `mincon-linalg`; no warm start; too slow for
  the small-QP regime. Kept as a possible fallback for large sparse
  subproblems, not for this phase.
* **ADMM (`osqp`)** — first order; not accurate enough in the tail of an SQP
  (the subproblem must be solved to high accuracy for superlinear
  convergence, and the multipliers feed the Hessian update).
* **Schittkowski's QL** — Goldfarb–Idnani with a Cholesky of `G` plus a
  regularization; essentially §2.3.

---

## 3. Inconsistent linearizations: the elastic (ℓ1) QP

### 3.1 The problem

The constraints of (1.3) are linearizations; even when the NLP is feasible
they can be inconsistent at `x_k` (the classic example is `x² ≥ 1`
linearized at `x = 0`: `0 ≥ 1`). The QP has no solution and a plain SQP stops.
`fmincon`'s `sqp` documents this as one of its four differences from
`active-set`: when the QP's constraints cannot be satisfied it "combines the
objective and constraint functions into a merit function" and "attempts to
minimize the merit function subject to relaxed constraints" (MW).

### 3.2 The relaxation

Fletcher's Sℓ1QP (1985) and SNOPT's elastic mode (Gill, Murray, Saunders
2005) both replace the constraints by an ℓ1 penalty on their violation:

```
min_{d,p,q}  ½ dᵀ B d + gᵀ d + ρ Σ_i (p_i + q_i)
s.t.         c_L − c_k <= J d + p − q <= c_U − c_k,   p, q >= 0,                   (3.1)
             x_L − x_k <=  d  <= x_U − x_k
```

This QP is always feasible (take `d = 0` and the slacks equal to the current
violation split by sign) and bounded below (B positive definite, ρ > 0). Its
objective is exactly the quadratic model of the ℓ1 merit function

```
φ_ρ(x) = f(x) + ρ · v(x),      v(x) = Σ_i dist(c_i(x), [c_L,i, c_U,i])         (3.2)
```

around `x_k`: `φ_ρ(x_k + d) ≈ f_k + gᵀd + ½dᵀBd + ρ Σ dist(c_k + Jd, [c_L, c_U])`,
with the slacks realizing the distances. Therefore:

* if the unrelaxed QP (1.3) is feasible and its multipliers satisfy `‖λ‖_∞ ≤ ρ`,
  its solution with `p = q = 0` is also the solution of (3.1) — nothing
  changes on the ordinary iterations;
* if (1.3) is infeasible, (3.1) gives the step that best reduces the model of
  the merit function, which is what we want the line search to use anyway;
* the multiplier of every row of (3.1) is bounded by `ρ` in magnitude
  (complementarity with the slack bounds), which is the SNOPT observation that
  the elastic problem also tames the unbounded multipliers of degenerate
  problems.

SNOPT starts `γ` (its ρ) at `γ_0‖g‖_∞` with `γ_0 = 10⁴`, increases it
geometrically when the elastic solution still has nonzero slacks and the
iterates stall, and declares the problem infeasible when γ reaches its cap
with violation remaining. Fletcher's method keeps a single penalty parameter
for both the subproblem and the merit function, which is the design chosen
here (§5.3) because it makes the QP step provably a descent direction for the
merit function that judges it.

### 3.3 Strict convexity of the elastic QP

The slacks have zero curvature, so (3.1) is not strictly convex in `(d, p, q)`
and the dual method of §2.3 does not apply directly. Two ways round it:

1. **Eliminate the slacks.** For fixed `d`, the optimal slacks are
   `p_i = max(0, c_L,i − c_k,i − (Jd)_i)`, `q_i = max(0, c_k,i + (Jd)_i − c_U,i)`,
   so (3.1) is `min_d ½dᵀBd + gᵀd + ρ Σ dist((c_k + Jd)_i, [c_L, c_U])` over
   the box — a strictly convex piecewise-quadratic. The dual method does not
   handle the kinks; a specialized active-set method on the kink structure
   does, but is more code.
2. **Regularize the slacks.** Add `½ ε (‖p‖² + ‖q‖²)` with `ε = 1e-8·ρ·max(1, ‖B‖)`
   — a strictly convex QP in `(d, p, q)` of size `n + 2m` with a diagonal
   block, solved by §2.3 unchanged. The optimal slacks become
   `p_i = max(0, (μ_i − ρ)/ε)`, exactly zero for every row whose multiplier
   does not exceed ρ, so consistent iterations are unaffected, and rows that
   must be relaxed take slack `O(1/ε)` times the multiplier excess — large
   enough to relax them fully at this `ε` unless the violation is astronomical,
   in which case the regularization slightly under-relaxes, harmlessly.

Route 2 is chosen. The elastic QP is entered only when the plain QP is
infeasible or its multipliers exceed ρ, so the extra `2m` variables cost
nothing on ordinary iterations; the Cholesky factor of the block-diagonal
Hessian `[B 0; 0 εI]` is the factor of `B` bordered by `√ε I`.

### 3.4 Local infeasibility

If the elastic solution has `Σ(p + q) > 0`, the linearization is inconsistent
at `x_k`. Two cases:

* the merit function still decreases along `d` (ρ large enough): take the
  step; the violation will fall and the linearization typically becomes
  consistent within a few iterations;
* the iterates converge to a point with `v(x) > 0` where `Jᵀ(∂v)` is in
  the normal cone — a stationary point of the infeasibility. Test: the
  feasibility-only QP (ρ → ∞ or `B` replaced by a scaled identity) produces a
  step with `‖d‖ ≤ ε_step` while `v > ε_feas`. Then report
  `LocallyInfeasible` with the same caveat the restoration phase of
  `mincon-ip` gives: first-order evidence of an infeasible stationary point,
  not a proof of infeasibility. Increase ρ at most to `ρ_max = 1e10·max(1, ‖g‖)`
  before concluding this; SNOPT's practice.

---

## 4. The Hessian approximation: damped BFGS on the Lagrangian

### 4.1 The update

With `s = x_{k+1} − x_k` and

```
y = ∇_x L(x_{k+1}, λ_{k+1}) − ∇_x L(x_k, λ_{k+1}) = (g_{k+1} − g_k) − (J_{k+1} − J_k)ᵀ λ_{k+1}     (4.1)
```

(the same, new multipliers on both sides, so that `y ≈ W s`; bound
multipliers cancel because bounds are linear) the BFGS update is

```
B_{k+1} = B_k − (B_k s sᵀ B_k)/(sᵀ B_k s) + (y yᵀ)/(sᵀ y).                             (4.2)
```

`sᵀy > 0` is required for `B_{k+1}` to stay positive definite, and for a
constrained problem there is no line search that guarantees it: the Hessian
of the Lagrangian is indefinite in general and only positive definite on the
tangent space of the active constraints. Powell's damping (1978) replaces `y`
by

```
r = θ y + (1 − θ) B_k s,      θ = 1                        if sᵀy ≥ 0.2 sᵀB_k s,
                               θ = 0.8 sᵀB_k s / (sᵀB_k s − sᵀy)   otherwise,          (4.3)
```

which gives `sᵀr = 0.2 sᵀBs > 0` in the damped case, so the update is always
defined and `B_{k+1}` stays positive definite. Nocedal and Wright call this
Procedure 18.2. MathWorks describes the same thing in words: when `qᵀs` is not
positive, "the most negative element of `q_k·s_k` is repeatedly halved"
and a vector `w v` is added (MW) — a componentwise variant of the same idea.
`mincon_ip::bfgs::update_guarded` already implements (4.3) with skips for
degenerate `s` and a guarded diagonal initial scaling; the SQP member reuses
it.

### 4.2 Why it converges superlinearly

Boggs, Tolle and Wang (1982) give the Dennis–Moré characterization for
constrained problems: the SQP iterates converge Q-superlinearly if and only if

```
‖ P_k (B_k − W*) (x_{k+1} − x_k) ‖ / ‖ x_{k+1} − x_k ‖  → 0,
```

where `P_k` projects onto the tangent space of the active constraints at
`x*`. The full-space BFGS update with damping does not achieve this in
general — Powell (1978) proved R-superlinear convergence under the assumption
that the damped update is eventually undamped — but in practice damped BFGS
is what every production SQP uses (`fmincon`, SNOPT's limited-memory variant,
NLPQLP, SLSQP), and the two-step limited-memory treatment in SNOPT exists to
cope with `sᵀy < 0` without damping (Gill, Murray, Saunders 2005, §2.9).
Things that provably help: use `λ_{k+1}` in (4.1) so that `y` measures the
curvature of the right Lagrangian; never update on a step that was rejected
or on a step of length below the step tolerance; reset to a scaled identity
when more than, say, `n` consecutive updates were damped (curvature model
hopeless).

### 4.3 Initial scaling

`B_0 = I` is the wrong scale on most problems; the guarded diagonal scaling of
`mincon-ip` (after the first accepted step, `γ = rᵀr/sᵀr`, diagonal `r_i/s_i`
clamped within `[γ/1e4, 1e4γ]`) is reused as is, since it earned its place
(`bench/results/s5-bfgs-guarded-diagonal`, 0.94× evaluations, +1 attainment).

### 4.4 Exact Hessians

When the model supplies `hess_lagrangian`, use it with the shift of §2.5.
Quadratic convergence then returns, and the bound constraints are handled by
the QP, not by the Hessian, so nothing else changes. With `B` exact, `y` is
not needed and the update is skipped.

---

## 5. Globalization: ℓ1 merit function, line search, second-order correction

### 5.1 Why a merit function at all

The QP step is Newton's step and, far from the solution, may increase the
objective, the violation, or both. Some function of the iterate must decide
whether a step is progress. Three families exist:

* **Penalty/merit functions** — a scalar `φ_ρ = f + ρ v` (ℓ1, §3.2) or the
  augmented Lagrangian `f − λᵀc + ½ρ‖c‖²` (SNOPT, NLPQLP). The ℓ1 merit is
  exact: for `ρ > ‖λ*‖_∞` a KKT point of the NLP is a local minimizer of
  `φ_ρ` (Han and Mangasarian 1979; Nocedal–Wright Theorem 17.3), so
  minimizing `φ_ρ` solves the NLP without driving ρ to infinity. Its
  difficulty is the choice of ρ: too small and the iterates converge to an
  infeasible minimizer of `φ_ρ`; too large and feasibility dominates,
  steps are short, and convergence slows (the "penalty trap"). Its
  non-smoothness is not a problem for a line search — the directional
  derivative exists (5.2).
* **Filters** (Fletcher and Leyffer 2002; the `mincon-ip` filter) — accept a
  step if it improves `f` or `v` by a margin relative to a set of previous
  pairs; no penalty parameter. Needs a feasibility-restoration phase when the
  QP is infeasible and switching conditions to keep superlinear convergence
  (Wächter–Biegler); more code paths.
* **Funnel/flexible penalty** (Curtis and Nocedal 2008; Gould and Toint 2010)
  — hybrids.

The specification (`docs/03_SPEC_SQP.md`) argued for a merit function so that
the two portfolio members fail differently; the elastic QP of §3 adds a second
reason: with a single ρ shared between the subproblem and the merit function,
the QP solution is *always* a descent direction of the merit function (5.2),
even when the linearization is inconsistent, which removes the
feasibility-restoration phase a filter would need. The remaining weakness —
ρ selection — is addressed with a model-based rule (5.3). Decision: ℓ1 merit
function with the elastic-QP-consistent penalty.

### 5.2 Descent

Let `(d, p, q)` solve (3.1) and let `m_ρ(d) = gᵀd + ½dᵀBd + ρ Σ dist((c_k + Jd)_i, [c_L,c_U])`
be the model reduction of `φ_ρ` from `x_k` (`m_ρ(0) = ρ v(x_k)`). Because
`dist(·, interval)` is convex and `c(x_k + αd) = c_k + αJd + O(α²)`, the
directional derivative of `φ_ρ` along `d` satisfies

```
D(φ_ρ; d) ≤ gᵀd + ρ ( Σ dist((c_k + Jd)_i) − v(x_k) )  =  m_ρ(d) − m_ρ(0) − ½dᵀBd  ≤  −½ dᵀBd  < 0     (5.1)
```

whenever `d ≠ 0`, since `d` minimizes the model so `m_ρ(d) ≤ m_ρ(0)`. This is
Lemma 18.2 of Nocedal–Wright extended to the elastic form (Fletcher 1985,
Theorem 14.3.1 in *Practical Methods*). The quantity
`Δm = m_ρ(0) − m_ρ(d) ≥ ½dᵀBd` is the predicted reduction used by the Armijo
test:

```
φ_ρ(x_k + α d) ≤ φ_ρ(x_k) − η α Δm,      η = 1e-4 (fmincon-like) … 0.1,          (5.2)
```

with backtracking `α ← α/2` (or a safeguarded quadratic interpolation) from
`α = 1`. Because `d` lies in the box, every trial point satisfies the bounds;
a non-finite `f` or `c` at a trial point is treated as a failed Armijo test
(step halved), which is `fmincon`'s "attempts to take a smaller step" (MW),
with a cap of 8 halvings before declaring the step failed.

### 5.3 The penalty parameter

Two requirements: `ρ > ‖λ‖_∞` at the solution (exactness) and `Δm` large
enough relative to the violation so that steps make real progress on
feasibility. Nocedal–Wright (18.36) fold both into one rule: choose ρ so that

```
Δm(ρ) ≥ τ · ρ · v(x_k),    τ ∈ (0, 1), e.g. 0.1,                                     (5.3)
```

i.e. the model predicts that at least a fraction τ of the current violation
penalty is removed. Expanding, with `v⁺ = Σ dist((c_k + Jd)_i)` the linearized
violation after the step (zero when the plain QP was consistent),

```
ρ ≥ ( gᵀd + ½ dᵀBd ) / ( (1 − τ) (v(x_k) − v⁺) )         when v(x_k) > v⁺.               (5.4)
```

If the plain QP is feasible, `v⁺ = 0` and (5.4) is Nocedal–Wright's rule
exactly. In elastic mode `d` depends on ρ, so (5.4) is applied and the QP
re-solved if ρ changed (at most twice per iteration; the "steering" of Byrd,
Nocedal and Waltz 2008 is this loop, with the feasibility-only QP as the
reference for `v⁺`). ρ is also raised to `max(ρ, 1.5‖λ‖_∞ + 1)` whenever the
QP multipliers exceed it. ρ never decreases within a solve except by the
Byrd–Nocedal–Waltz reset when it has been far above the multipliers for
several iterations (cap the decrease so `ρ ≥ 1.5‖λ‖_∞`), which counters the
penalty trap. Initial ρ: `max(1, ‖g_0‖_∞ / max(1, ‖J_0‖_∞))`, scale-aware, of
the same nature as MathWorks' `r_i = ‖∇f‖/‖∇g_i‖` (MW) but a single scalar
(per-row penalties buy little with the elastic form).

### 5.4 The Maratos effect and the second-order correction

Near the solution the unit step `d` can be rejected by any merit function
even though it converges quadratically: moving along the tangent of a curved
constraint increases the violation by `O(‖d‖²)` while the objective decreases
by only `O(‖d‖²)` too, and with ρ large the penalty wins (Maratos 1978;
Nocedal–Wright Example 15.4). Rejecting these steps destroys superlinear
convergence. The second-order correction (Fletcher 1982; Mayne and Polak
1982; Nocedal–Wright §15.6) fixes it: when the unit step fails the Armijo
test, evaluate `c(x_k + d)`, compute the minimum-norm correction

```
w = − J_kᵀ (J_k J_kᵀ)⁻¹ c̃,     c̃ = the violation of c(x_k + d) against [c_L, c_U] on the active rows,      (5.5)
```

and test `x_k + d + w` instead. `w = O(‖d‖²)` restores feasibility to second
order without changing the objective to first order. In practice (5.5) is
obtained by re-solving the QP with only its right-hand side changed:
`J d̂ ∈ [c_L − c(x_k + d) + J d, c_U − …]` — the same working set, one warm
solve. If the corrected step also fails, back off α on the uncorrected `d`.
This costs one extra constraint evaluation on the iterations where it fires
and it is what makes the ℓ1 merit function competitive with a filter in the
endgame. An alternative is the watchdog (Chamberlain et al. 1982) — accept a
few non-monotone steps and fall back if the merit does not recover — which
is more code and interacts badly with the FD-error-aware termination.

### 5.5 Line search failure

If no `α ≥ α_min = 1e-10·max(1, ‖x‖)/‖d‖` passes (5.2), three explanations are
possible and are checked in order: the derivatives are wrong or too inaccurate
(re-check the FD error estimate; escalate to central differences as
`mincon-ip` does); `B` is badly wrong (reset it to the scaled identity and
retry once); or the point is a stationary point of `φ_ρ` that is infeasible
(§3.4 → `LocallyInfeasible`) or a KKT point below the tolerance's reach
(→ `Acceptable`/`StepTolerance` by the same classification the interior-point
member uses after its restoration dead-ends).

---

## 6. Termination

Stationarity is judged by the same scaled KKT error as `mincon-ip` so that
the two members have the same success semantics:

```
E = max( ‖g − Jᵀλ − z‖_∞ / s_d,   v(x),   max_i |λ_i| · dist(c_i, active bound) / s_c ),
s_d = max(s_max, (‖λ‖_1 + ‖z‖_1)/(m + n_b)) / s_max,      s_max = 100,                    (6.1)
```

stop `Optimal` when `E ≤ tol` (default 1e-6, matching `fmincon`'s
`OptimalityTolerance`) with `v ≤ tol_feas`; `Acceptable` under the relaxed
tolerances after the error-aware finite-difference floor of `mincon-ip`
(`tol_eff = clamp(err_scaled, tol, acceptable)`) says the residual cannot be
reduced further with the current derivatives. `StepTolerance` when
`‖α d‖ ≤ 1e-10 (1 + ‖x‖)` at a feasible point (fmincon uses 1e-6 for `sqp`,
1e-10 for `interior-point` (MW); mincon keeps one value). Multipliers are
reported in the `l ≤ c ≤ u` sign convention of `SolveReport`. Termination
after a QP that returned `d = 0` with correct-sign multipliers is immediate:
that is a KKT point of the NLP up to the linearization's accuracy.

---

## 7. What `fmincon`'s `sqp` does, as documented

Verified against the MathWorks pages on the date above (MW):

* QP subproblem `min ½dᵀH_kd + ∇f(x_k)ᵀd` with linearized constraints; `H_k`
  updated by BFGS with `s_k = x_{k+1} − x_k`,
  `q_k = ∇f(x_{k+1}) + Σλ_i∇g_i(x_{k+1}) − (∇f(x_k) + Σλ_i∇g_i(x_k))`; damping
  by halving the most negative element of `q_k·s_k` and adding `w v`.
* Merit function `Ψ(x) = f(x) + Σ_{eq} r_i g_i(x) + Σ_{ineq} r_i max(0, g_i(x))`,
  `r_i` initialized as `‖∇f‖/‖∇g_i‖`, updated `r_i = max(λ_i, (r_i + λ_i)/2)`
  (Powell's rule), step `x + αd` from a line search on `Ψ`.
* QP solved by a primal active-set method with a QR-based null space; an LP
  `min γ s.t. A_i x − γ ≤ b_i` for an initial feasible point when needed.
* `sqp` vs `active-set`: every iterate and every finite-difference step
  respects the bounds; on `Inf`, `NaN` or complex values the algorithm
  "attempts to take a smaller step"; refactored linear algebra; when the
  constraints are not satisfied it minimizes the merit function subject to
  relaxed constraints, and when violations grow it "attempts to obtain
  feasibility using a second-order approximation to the constraints".
* `sqp-legacy` is "nearly identical" but slower and heavier.
* Defaults: `MaxFunctionEvaluations` 3000 for `interior-point`, `100·n` for
  every other algorithm; `MaxIterations` 1000 / 400; `OptimalityTolerance`
  1e-6; `StepTolerance` 1e-10 / 1e-6; `ConstraintTolerance` 1e-6; forward
  differences with step `√eps`, central `eps^(1/3)`; `HonorBounds` true;
  `ScaleProblem` false; `ObjectiveLimit` −1e20. Exit flags 1 (first-order
  optimality and constraint tolerances met), 2 (step tolerance and feasible),
  0 (iteration or evaluation limit), −1 (output function), −2 (no feasible
  point), −3 (below `ObjectiveLimit`).
* MathWorks' recommendation: "Use the interior-point algorithm first", then
  `sqp` "for speed on small-to-medium problems"; `sqp` cannot use sparse data
  or large-scale structure.

Consequence for the benchmark records: `fmincon-sqp` at defaults is capped at
`100·n` evaluations — 200 evaluations on a two-variable problem, 5000 at
n = 50, 20 000 at n = 200 — and 400 iterations. Both caps are visible in the
records (CHAINROSEN_BOX_50 stopped at 5014, CHAINROSEN_EQ_200 and
ELLIPSOID2_200 at 20 038 and 20 102, ELLIPSOID_500 at 400 iterations in track
C). The evaluation-cap study of the plan must count them as well.

---

## 8. Robustness checklist ("the typical worries", and how each is handled)

| worry | handling | source |
|---|---|---|
| Model undefined outside the box | steps and FD probes stay inside the bounds structurally (QP bounds; bounds-aware FD already in `mincon-diff`) | §1, MW |
| `NaN`/`Inf` from the model | treated as a failed Armijo test, step halved, ≤ 8 times; the best finite point is kept | §5.2, MW |
| Inconsistent linearization | elastic QP, always solvable; same ρ as the merit function so the step is a descent direction | §3, Fletcher 1985, SNOPT |
| Infeasible problem | stationary infeasibility test after ρ escalation → `LocallyInfeasible` with a note | §3.4 |
| Indefinite curvature | damped BFGS (never indefinite); exact Hessian shifted | §4.1, §2.5 |
| Wrong scale of `B_0` | guarded diagonal initial scaling | §4.3 |
| Curvature model hopeless (many damped updates) | reset to scaled identity | §4.2 |
| Maratos effect (unit steps rejected near the solution) | second-order correction with one warm QP re-solve | §5.4 |
| Penalty too small (converge to infeasible point) / too large (crawl) | model-based ρ rule (5.4), multiplier floor, guarded decrease | §5.3 |
| Degenerate constraints, unbounded multipliers | elastic QP bounds multipliers by ρ; dual method handles dependent rows by the partial-step branch | §2.3, §3.2 |
| Rank-deficient working set in the QP | Givens-based `R` monitored; dependent constraint refused, partial step taken | §2.3 |
| QP cycling | strictly convex → finite termination; iteration cap `10(n + p)` as a safety net → fall back to a fresh cold start, then to a regularized `B + δI` | §2.3 |
| Finite-difference noise limiting stationarity | error-aware termination and central-difference escalation shared with `mincon-ip` | §6 |
| Bad scaling of variables/constraints | the gradient-based scaling of `mincon-ip` applied to `f` and rows of `c` before the loop | §6 |
| Unbounded objective | the far-from-start rule of `mincon-ip` (`‖x‖ > 1e6 max(1, ‖x0‖)` with a 1e6× objective drop) | `mincon-ip` |
| Wrong user derivatives | `check_derivatives` option; the line-search failure path re-checks the FD error estimate | §5.5 |
| Silent budget exhaustion | the stall detector of the plan's study S-C, reported as a verdict with a note | plan §5 I4 |

---

## 9. Decisions

| decision | chosen | preferred over | reason |
|---|---|---|---|
| QP solver | Goldfarb–Idnani dual active set, dense Cholesky + Givens-updated `J, R` | primal active set with phase-1 LP; interior-point QP; ADMM | no phase 1, infeasibility certificate, warm start, finite termination, O(n²) per step |
| Bounds | QP constraints on `d` | projection | exact bound satisfaction at every iterate and FD probe |
| Inconsistency | elastic ℓ1 QP with regularized slacks, entered on demand | restoration phase; per-row penalties | always solvable; one ρ shared with the merit; no extra code path on normal iterations |
| Globalization | ℓ1 merit + Armijo backtracking + SOC | filter (as `mincon-ip`); augmented Lagrangian | descent guaranteed with the elastic QP; decorrelation from the IP member; exactness |
| Penalty rule | model-based (5.4) with multiplier floor and guarded decrease | Powell's `max(λ, (r+λ)/2)` | (5.4) ties progress on feasibility to the model; Powell's rule can undershoot |
| Hessian | damped BFGS (Powell), reused from `mincon-ip`, guarded scaling; exact Hessian if supplied | SR1, limited memory, undamped with skips | positive definiteness needed by the dual QP; proven in this code base |
| Multipliers in `y` | `λ_{k+1}` on both sides | `λ_k`/`λ_{k+1}` mixed | `y ≈ W(x, λ_{k+1}) s` |
| Termination | `mincon-ip`'s scaled KKT error, tolerances, error-aware floor | fmincon's first-order measure | identical success semantics across members |
| Failure reporting | same `ExitFlag` classification as the IP member's restoration dead-ends | new flags | the portfolio and the harness already understand them |

---

## 10. Sources

Nonlinear programming texts and papers:

* J. Nocedal, S. J. Wright, *Numerical Optimization*, 2nd ed., Springer 2006 —
  Ch. 15.4–15.6 (merit functions, Maratos effect, SOC), Ch. 17.2 (ℓ1 exact
  penalty, Theorem 17.3), Ch. 18 (SQP: Algorithm 18.3, Procedure 18.2 damped
  BFGS, rule (18.36), Sℓ1QP).
* R. Fletcher, *Practical Methods of Optimization*, 2nd ed., Wiley 1987 —
  Ch. 12.4 (SQP), 14.3 (ℓ1 exact penalty; Sℓ1QP), 14.5 (SOC).
* R. B. Wilson, *A simplicial algorithm for concave programming*, PhD thesis,
  Harvard 1963. S.-P. Han, "Superlinearly convergent variable metric
  algorithms for general nonlinear programming problems", Math. Programming
  11 (1976) 263–282, and "A globally convergent method for nonlinear
  programming", J. Optim. Theory Appl. 22 (1977) 297–309.
* M. J. D. Powell, "A fast algorithm for nonlinearly constrained
  optimization calculations", in *Numerical Analysis*, Lecture Notes in
  Math. 630, Springer 1978 (damped BFGS, penalty update); "The convergence of
  variable metric methods for nonlinearly constrained optimization
  calculations", in *Nonlinear Programming 3*, 1978 (R-superlinear
  convergence).
* S. M. Robinson, "Perturbed Kuhn–Tucker points and rates of convergence for
  a class of nonlinear-programming algorithms", Math. Programming 7 (1974)
  1–16 (local quadratic convergence of SQP).
* P. T. Boggs, J. W. Tolle, P. Wang, "On the local convergence of
  quasi-Newton methods for constrained optimization", SIAM J. Control Optim.
  20 (1982) 161–171 (Dennis–Moré characterization); P. T. Boggs, J. W. Tolle,
  "Sequential quadratic programming", Acta Numerica 4 (1995) 1–51.
* S.-P. Han, O. L. Mangasarian, "Exact penalty functions in nonlinear
  programming", Math. Programming 17 (1979) 251–269.
* N. Maratos, *Exact penalty function algorithms for finite dimensional and
  control optimization problems*, PhD thesis, Imperial College 1978.
  R. Fletcher, "Second order corrections for non-differentiable
  optimization", in *Numerical Analysis*, LNM 912, 1982. D. Q. Mayne,
  E. Polak, "A superlinearly convergent algorithm for constrained
  optimization problems", Math. Programming Study 16 (1982) 45–61.
  R. M. Chamberlain, M. J. D. Powell, C. Lemaréchal, H. C. Pedersen, "The
  watchdog technique for forcing convergence in algorithms for constrained
  optimization", Math. Programming Study 16 (1982) 1–17.
* R. Fletcher, "An ℓ1 penalty method for nonlinear constraints", in
  *Numerical Optimization 1984*, SIAM 1985, 26–40 (Sℓ1QP).
* P. E. Gill, W. Murray, M. A. Saunders, "SNOPT: An SQP algorithm for
  large-scale constrained optimization", SIAM Review 47 (2005) 99–131
  (elastic mode, γ₀ = 10⁴, augmented-Lagrangian merit, limited-memory
  quasi-Newton with the two-step curvature fix, termination test
  τ_P = 1e-6, τ_D = 2e-6).
* R. H. Byrd, J. Nocedal, R. A. Waltz, "Steering exact penalty methods for
  nonlinear programming", Optim. Methods Softw. 23 (2008) 197–213.
* R. Fletcher, S. Leyffer, "Nonlinear programming without a penalty
  function", Math. Programming 91 (2002) 239–269; A. Wächter, L. T. Biegler,
  "On the implementation of an interior-point filter line-search algorithm
  for large-scale nonlinear programming", Math. Programming 106 (2006)
  25–57. F. E. Curtis, J. Nocedal, "Flexible penalty functions for nonlinear
  constrained optimization", IMA J. Numer. Anal. 28 (2008) 749–769.

Quadratic programming:

* D. Goldfarb, A. Idnani, "A numerically stable dual method for solving
  strictly convex quadratic programs", Math. Programming 27 (1983) 1–33.
* M. J. D. Powell, "On the quadratic programming algorithm of Goldfarb and
  Idnani", Math. Programming Study 25 (1985) 46–61 (dependence handling;
  the ZQPCVX code).
* K. Schittkowski, "QL: a Fortran code for convex quadratic programming",
  University of Bayreuth report, 2005 (Goldfarb–Idnani with a Cholesky and a
  regularization for semidefinite `G`).
* R's `quadprog` (Turlach's `solve.QP`) and `eiquadprog` — reference
  implementations of the dual method with the `J, R` factor updates.

MathWorks (read September 11, 2026, Optimization Toolbox R2025b):

* "Constrained Nonlinear Optimization Algorithms" —
  https://www.mathworks.com/help/optim/ug/constrained-nonlinear-optimization-algorithms.html
  (SQP implementation, merit function, BFGS damping, QP active-set method,
  `sqp` vs `active-set` differences, `sqp-legacy`).
* `fmincon` reference — https://www.mathworks.com/help/optim/ug/fmincon.html
  (option defaults per algorithm, exit flags).
* "Choosing the Algorithm" —
  https://www.mathworks.com/help/optim/ug/choosing-the-algorithm.html.

---

## 11. As built (September 11, 2026) — what the corpus added to the textbook

The implementation in `crates/mincon-sqp/src/solver.rs` follows §§1–6 and §9.
Running it on the 149-problem corpus and the 55 fixtures forced six additions,
each recorded here with the problem that demanded it and the mechanism; none
of them changes the theory above, they close gaps the theory leaves open.

**11.1 Curvature rescale of the unit initial matrix (UNITS).** With
`B₀ = I` on a problem whose Hessian is 10⁸, the unit QP step is absurd and
every backtracking trial fails; escalating to central differences and
resetting `B` (both to the same unit matrix) cannot help. The rejected trials
themselves measure the curvature along `d`: with `q(α) = f(x + αd) − f(x) −
α gᵀd ≈ ½ α² dᵀHd`, the widest trial that still evaluated gives
`dᵀHd ≈ 2q/α²`, and `B ← (dᵀHd / dᵀd) I` once, at no extra evaluation. This is
the one-step analogue of the guarded diagonal scaling of §4.3, triggered by a
*failed* first line search instead of a cut one.

**11.2 Adaptive step bound on the QP (HS106).** A step 2700 long in
variables of size 5000 was accepted at α = 3·10⁻⁵ for 300 iterations, each
costing fifteen halvings. The QP gains a box `|d_j| ≤ Δ·max(1, |x_j|)` with
`Δ = ∞` until a line search fails (then `Δ ← ¼·α_widest·‖d‖_rel`) or cuts the
step below ¼ (then `Δ ← 2α‖d‖_rel`), and `Δ ← 4Δ` after a full step at the
bound (back to ∞ above 10³). It is SNOPT's major step limit with a trust-region
update instead of a fixed radius. Multipliers of bounds that came from Δ rather
than from the problem are zeroed before any KKT test, since they are not
multipliers of the NLP (this omission produced a false `Optimal` on HS13
before it was found).

**11.3 Penalty rule guard (HS13).** From a point with violation 5·10⁻¹² the
rule (5.4) produced ρ = 2.5·10⁹ and the next step traded objective for a
violation the merit could no longer see. The rule is applied only when
`v(x_k) > tol_feas` and the model reduction of the violation is at least
10⁻³ v, and ρ may grow by at most 100× per iteration. The multiplier floor
`ρ ≥ 1.5‖λ‖_∞` is unconditional.

**11.4 Repeated second-order corrections (HS46).** One correction left a
third-order violation `O(‖d‖³)` that the penalty term still outweighed; up to
four corrections are taken while the violation keeps falling by at least half
(IPOPT's `kappa_soc` rule). On HS46 this turned a 7000-evaluation crawl at
α = 2⁻⁹ into 229 evaluations.

**11.5 Zero-step re-test.** When the QP returns `d = 0` its multipliers are
the ones the termination test must see; the loop adopts them and re-tests
once before classifying the point (vertex problems went from `Acceptable` to
`Optimal` in one iteration).

**11.6 Second-order probe and saddle escape (HS33, HS25).** With a
positive-definite `B` the QP cannot see negative curvature, so SQP converges
happily to saddle points — `fmincon`'s `sqp` and SLSQP both stop at f = −4 on
HS33 (the basin study, `bench/results/r3-basins`). At a KKT candidate with a
null space of dimension `k ≤ 6` (strongly active gradients removed, Gram–
Schmidt), the Lagrangian `f + λᵀc` is probed by second differences along an
orthonormal basis (`k(k+1)/2` probes, two evaluations each, one-sided at a
weakly active bound), its `k × k` projected Hessian diagonalized (Jacobi),
and if the smallest eigenvalue is below −10⁻³ of the matrix scale the solve
leaves the saddle along that eigenvector (sign chosen feasible for the weakly
active constraints), each trial followed by one Newton restoration step onto
the strongly active constraints — moving along the tangent alone bends off
the active surface and the penalty swamps the second-order decrease, the
Maratos effect in another guise. HS33 now reaches −4.5858 and HS25 (gradient
10⁻⁸ at x₀, every other solver stops there) reaches its minimum. The probe
costs 2–42 evaluations once per candidate and is skipped for `k > 6` with a
note that second-order conditions were not verified.

**11.7 Degenerate multipliers.** If the multipliers at a point passing the
tolerances exceed 10⁸(1 + ‖g‖) in the user's units, no constraint
qualification holds there and the point is reported `Acceptable` with a
degeneracy note rather than as a certified KKT point (HS13: λ = 2·10¹⁰).

**11.8 Curvature-tracking rebuild at large n (C7, September 11).** The
damped BFGS matrix (shared with the interior-point member,
`crates/mincon-ip/src/bfgs.rs`) starts and stays 20–1000× under-curved on
problems whose Hessian of the Lagrangian grows one to two orders of magnitude
along the path (entropy terms `x log x`; a constraint multiplier climbing from
0 to 100). A rank-two update repairs one direction per iteration, so the SQP
member's step bound never grows (α = 1 keeps failing) and the method
degenerates into a box-limited gradient descent for hundreds of iterations
(`bench/results/r5-large-n`). When the curvature the model predicts along an
accepted step is more than 10× from the measured `sᵀy`, the matrix is rebuilt
from the per-coordinate quotients `yᵢ/sᵢ` — gated so it only fires when the
diagonal is demonstrably the better model (its scale is wrong, or its
off-diagonal is) and only for `n ≥ 10`. On QUADSPHERE_100 this turns 178
iterations into 2; the plan, mechanism and whole-corpus ablation are
`docs/21_LARGE_N_CURVATURE_PLAN.md` and `bench/results/abl-c7` (SQP member
0.88× [0.58, 0.96] evaluations, +3 attained, no loss). One un-shielded
regression for the SQP member alone: PORTFOLIO_100, a dense covariance QP,
where a single scale-route rebuild installs a diagonal that is not the
curvature of anything and costs 7× the evaluations (`docs/21` §7). **Round 4
(September 12) generalised that regression, not the win**: on the sealed
set the rule cost the SQP member 1.14× [1.00, 1.54] evaluations on track A
(COVQP_120 3.2×), so it is off by default and opt-in
(`bench/results/s6v4-final4` §2). The measured numbers below are the C7-on
development record.

**Measured** (`bench/results/abl-sqp1`, `abl-sqp2`): SQP alone 134/143 at
0.78× [0.70, 0.86] the interior-point member's evaluations; portfolio (SQP
first for n ≤ 20) 137/143 at 0.775× [0.71, 0.84] the previous candidate's.
With C7 the SQP member attains 148/158 on the current corpus at 0.88× its own
rule-off evaluations.
