# Specification: sequential quadratic programming

Not implemented. `crates/mincon-sqp` holds the interface; this is the plan.

---

## 1. Why build a second algorithm at all

Interior point is the better default and it is what `fmincon` defaults to. SQP
earns its place because the two fail on **different** problems:

| | interior point | SQP |
|---|---|---|
| Warm starting | poor — the barrier must be re-raised | excellent |
| Small, dense, few active constraints | fine | usually faster, often by 5x |
| Degeneracy / LICQ failure | multipliers blow up, `E_0` never converges | often still fine |
| Very large sparse | excellent | active-set combinatorics dominate |
| Infeasible start | needs restoration | handles naturally |
| Highly nonlinear constraints | good | can cycle on the active set |

The measured evidence is already in hand: on the four-problem smoke benchmark,
`mincon`'s interior point uses roughly **5x more objective evaluations per
portfolio member than SciPy's SLSQP** on small dense problems. That regime
belongs to SQP and no amount of tuning the barrier method will take it.

Combined with the portfolio, two members with partly independent failure sets
give a robustness number neither reaches alone. That is the whole argument.

---

## 2. The QP subproblem

```
min_d  1/2 d^T B_k d + grad f(x_k)^T d
s.t.   c_L - c(x_k) <= J_k d <= c_U - c(x_k)
       x_L - x_k    <=   d   <= x_U - x_k
```

The `d` bounds enforce **exact bound satisfaction at every iterate**
structurally, matching `fmincon`'s `sqp` and its `HonorBounds` behaviour. This
is not decoration: it is what lets a model undefined outside its box be solved
at all.

`B_k` is the damped-BFGS or exact Hessian of the Lagrangian, shared with
`mincon-ip` (`mincon_ip::bfgs`).

### 2.1 Always-feasible relaxation

Linearized constraints are frequently inconsistent even when the NLP is
perfectly well posed. `fmincon`'s `sqp` reformulates the subproblem so it is
always feasible, and MathWorks lists that as one of the four things making it
more robust than `active-set`.

Use the **elastic** form:

```
min_{d, p, n}  1/2 d^T B d + g^T d + rho * sum(p + n)
s.t.           c_L - c <= J d + p - n <= c_U - c,   p, n >= 0
               d bounds as above
```

`rho` starts at `1e3` and grows by `10x` whenever the solution has `p + n`
still nonzero, capped at `1e8`. Above the cap, declare local infeasibility.

---

## 3. Globalization

Use an **l1 merit function**, not a filter:

```
psi(x; nu) = f(x) + sum_i nu_i * violation_i(x)
```

with the Han–Powell penalty update `nu_i <- max(|lambda_i|, (nu_i + |lambda_i|)/2)`,
and a backtracking Armijo line search on `psi`.

> **This is a deliberate choice to differ from the interior-point member.**
> A filter here would be defensible on its own merits, but the portfolio's
> value comes from members that fail *differently*. Two filter methods
> correlate; a filter and a merit function do not. Design for decorrelation.

**Second-order correction is mandatory.** Without it the Maratos effect rejects
unit steps arbitrarily close to the solution and superlinear convergence is
lost. Same structure as the interior-point SOC: recompute the constraint
residual at the trial point, re-solve the QP with only the right-hand side
changed, reusing the factorization.

**Non-finite retreat.** `Inf`/`NaN`/complex from the model halves the step and
retries, up to 8 times. This is `fmincon`'s point 2 and it is worth more in
practice than any convergence theorem.

---

## 4. The QP solver (`crates/mincon-qp`)

Two families, and the choice is not obvious.

**Active set (dual or primal).** What `fmincon` and SNOPT use. Warm starts
beautifully — the whole point of SQP is that consecutive subproblems differ
slightly, so an active-set method resumes from the previous active set and
finishes in a handful of pivots. Worst-case exponential; can cycle on a
degenerate QP without anti-cycling rules (Bland's, or lexicographic).

**Interior point.** Polynomial, insensitive to the size of the active set, and
it reuses `mincon-linalg` directly since the KKT system has the same shape.
Warm starts badly, which forfeits SQP's main advantage.

**Recommendation: a dual active-set method with warm starting, plus an
interior-point fallback** when the active-set method exceeds an iteration
budget. That combination is what makes SQP worth having rather than a slower
copy of the interior-point NLP solver.

**Do not reach for `osqp-rust`.** It is Apache-2.0 and good, but ADMM is a
first-order method: excellent for moderate accuracy on large problems, and not
accurate enough for the tail of an SQP, where the subproblem must be solved
precisely for superlinear convergence. Reasonable as a fallback, wrong as the
primary.

---

## 5. SLQP, later

Sequential linear-quadratic programming: an LP step to *choose* the active set,
then an EQP step on that set. Better than SQP when the active set is large and
changes a lot, because an LP is cheaper than a QP. Only after SQP works.

---

## 6. Acceptance gate

SQP is "implemented" when all of these hold:

1. Hock–Schittkowski pass rate at least equal to interior point's.
2. **`HS13` and `TORTURE_INFEASIBLE` are strictly better** than interior point.
   These are the cases SQP should own — degeneracy and inconsistency — and if
   it does not win them the portfolio gains nothing.
3. Median objective evaluations on `constrained-small` **below** interior
   point's, since that regime is the reason to build it.
4. The portfolio's combined success rate strictly exceeds each member's alone.
   If it does not, the members are too correlated and the design has failed.

Point 4 is the real test. A second algorithm that fails on the same problems as
the first is wasted compute.
