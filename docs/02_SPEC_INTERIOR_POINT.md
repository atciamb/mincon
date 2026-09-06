# Specification: the primal-dual interior-point method

Implemented in `crates/mincon-ip`. This document is the reference; the code
follows it closely enough that Wächter & Biegler (2006) can be read alongside
either.

Constants below are the paper's, with IPOPT's option name in brackets where
they differ in spelling. **Do not change a constant without a benchmark run
showing the change is an improvement on the held-out set**; these values are
the product of a lot of other people's experiments.

---

## 1. Internal formulation

The canonical problem

```
min f(x)   s.t.   c_L <= c(x) <= c_U,   x_L <= x <= x_U
```

becomes equality-constrained by giving every **inequality** a slack:

```
v = (x, s),                nv = n + |I|,   I = { i : c_L_i < c_U_i }
c_hat_i(v) = c_i(x) - c_L_i            for i not in I   (equality rows)
c_hat_i(v) = c_i(x) - s_k(i)           for i in I
v_L = (x_L, c_L[I]),  v_U = (x_U, c_U[I])
```

**Equalities are not slacked.** A slack pinned by `c_L == c_U` would sit in a
degenerate barrier interval and drive `Sigma` to infinity. IPOPT makes the same
choice.

Barrier subproblem:

```
min phi_mu(v) = f(x) - mu * sum_{j: has lower} ln(v_j - v_L_j)
                     - mu * sum_{j: has upper} ln(v_U_j - v_j)
s.t. c_hat(v) = 0
```

---

## 2. The step

### 2.1 The linear system

```
[ W + Sigma + delta_w I      A     ] [ d_v      ]      [ grad phi_mu + A lambda ]
[        A^T            -delta_c I ] [ d_lambda ]  = - [ c_hat                  ]
```

* `W` is the Hessian of the Lagrangian in the `x` block, zero in the `s` block.
* `Sigma_j = z_L_j / (v_j - v_L_j) + z_U_j / (v_U_j - v_j)`.
* `A` is the `nv x m` transposed Jacobian, with `-1` in the slack row of each
  inequality.

Bound multipliers are recovered afterwards:

```
d_zL = mu (V - V_L)^-1 e - z_L - Sigma_L d_v
d_zU = mu (V_U - V)^-1 e - z_U + Sigma_U d_v
```

The matrix is assembled as an **upper triangle in CSC** and its structure is
fixed for the whole solve, so the symbolic factorization and every index map
are computed once (`KktSystem::new`) and each iteration is a value refill.
This is the largest constant-factor win available in an interior-point code.

### 2.2 Regularization — Algorithm IC

Raise `delta_w` until the step is usable.

```
IC-1  Try delta_w = 0, delta_c = 0.
IC-2  If the factorization is singular, delta_c <- delta_c_bar * mu^kappa_c.
IC-3  If delta_w_last == 0:  delta_w <- delta_w_0
      else:                  delta_w <- max(delta_w_min, kappa_w_minus * delta_w_last)
IC-4  Factor. If acceptable, remember delta_w_last <- delta_w and stop.
IC-5  delta_w <- kappa_w_plus_first * delta_w   (first perturbation of the run)
             or kappa_w_plus * delta_w          (otherwise);  go to IC-4.
IC-6  If delta_w > delta_w_max, enter feasibility restoration.
```

| Constant | Value |
|---|---|
| `delta_w_min` | `1e-20` |
| `delta_w_0` | `1e-4` |
| `delta_w_max` | `1e40` |
| `kappa_w_plus` | `8` |
| `kappa_w_plus_first` | `100` |
| `kappa_w_minus` | `1/3` |
| `delta_c_bar` | `1e-8` |
| `kappa_c` | `1/4` |

**Current implementation:** all three `RegularizationMode` values require
certified inertia `(nv,m,0)`. A failed ordering retries natural KKT order,
then positive diagonal congruence, before explicit regularization. The original
scaffold never invoked its advertised curvature fallback. See
[`12_RESTORATION_IMPLEMENTATION.md`](12_RESTORATION_IMPLEMENTATION.md).

**Planned distinction, not yet implemented:**

* `Inertia` — the factorization must certify inertia `(nv, m, 0)`.
* `InertiaFree` — non-singular, plus the Chiang–Zavala curvature test on the
  computed direction.
* `Hybrid` **(default)** — use the inertia when
  `Factorization::inertia_is_certified()`, else fall back to the curvature test.

The curvature test (Chiang & Zavala 2016, eq. 3.11):

```
d' (W + Sigma + delta_w I) d + max(-lambda_plus' c_hat, 0) >= alpha_d * ||d||^2
```

with `alpha_d = 1e-12` scaled by `mu`. The standalone helper still needs a
complete acceptance/retry loop and independent tests. Results in the cited
paper are not measurements of mincon.

---

## 3. The filter line search

`theta(v) = ||c_hat(v)||_1`, `phi = phi_mu(v)`.

Fraction-to-boundary with `tau = max(tau_min, 1 - mu)`, `tau_min = 0.99`:

```
alpha_max = max{ alpha in (0,1] : v + alpha d_v >= v_L + (1-tau)(v - v_L)
                                  and symmetric at the upper bounds }
alpha_z   = the same rule applied to (z_L, d_zL) and (z_U, d_zU)
```

Backtrack from `alpha_max`. A trial point is accepted when:

**Case I — the switching condition holds:**

```
grad phi' d_v < 0
and  alpha * (-grad phi' d_v)^s_phi > delta * theta^s_theta
and  theta <= theta_min
```

then Armijo alone decides: `phi(alpha) <= phi + eta_phi * alpha * grad phi' d_v`.
The filter is **not** augmented (this is what preserves fast local convergence).

**Case II — otherwise**, accept on sufficient decrease in either measure:

```
theta(alpha) <= (1 - gamma_theta) theta    or    phi(alpha) <= phi - gamma_phi theta
```

and augment the filter with
`{ theta >= (1-gamma_theta) theta_k, phi >= phi_k - gamma_phi theta_k }`.

| Constant | Value |
|---|---|
| `gamma_theta` | `1e-5` |
| `gamma_phi` | `1e-5` |
| `delta` | `1` |
| `s_theta` | `1.1` |
| `s_phi` | `2.3` |
| `eta_phi` | `1e-8` (the paper says `1e-4`; IPOPT ships `1e-8`) |
| `gamma_alpha` | `0.05` |
| `theta_max` | `1e4 * max(1, theta_0)` |
| `theta_min` | `1e-4 * max(1, theta_0)` |

Minimum step length before restoration (eq. 23):

```
alpha_min = gamma_alpha * min( gamma_theta,
                               gamma_phi * theta / (-grad phi' d),
                               delta * theta^s_theta / (-grad phi' d)^s_phi )
```

### 3.1 Second-order corrections

When the **first** trial step is rejected and `theta` got worse — the Maratos
signature — compute

```
c_soc = alpha_0 * c_hat(v) + c_hat(v + alpha_0 d_v)
```

and re-solve **with the same factorization**, changing only the right-hand
side. Up to `max_soc = 4` corrections; abort when
`theta_soc > kappa_soc * theta_old` with `kappa_soc = 0.99`.

Reusing the factorization is what makes SOC nearly free, and it is the reason
`KktSystem` exposes `rhs_mut` / `solve_scratch` separately from
`factor_with_correction`.

---

## 4. The barrier parameter

Monotone (Fiacco–McCormick). When `E_mu <= kappa_eps * mu`:

```
mu <- max( tol/10, min( kappa_mu * mu, mu^theta_mu ) )
```

with `kappa_eps = 10`, `kappa_mu = 0.2`, `theta_mu = 1.5`. Reset the filter on
each `mu` change.

**Adaptive mode is not implemented.** When it is: Mehrotra probing and the
quality-function oracle, with `adaptive_mu_globalization = obj-constr-filter`,
falling back to monotone after repeated rejections. IPOPT's defaults are
`mu_oracle = quality-function`, `mu_min = 1e-11`, `mu_max = 1e5`.

---

## 5. Termination

Scaled KKT error, `s_max = 100`:

```
E_mu = max( ||grad f + A lambda - z_L + z_U||_inf / s_d ,
            ||c_hat||_inf ,
            ||(V - V_L) Z_L e - mu e||_inf / s_c )

s_d = max(s_max, (||lambda||_1 + ||z||_1) / (m + n_bounds)) / s_max
s_c = max(s_max, ||z||_1 / n_bounds) / s_max
```

Converged when `E_0 <= optimality`, violation `<= feasibility` and
complementarity `<= complementarity`.

The scaling factors are not cosmetic. Without them a problem whose multipliers
are `1e6` can never satisfy an absolute gradient tolerance, and the solver
reports failure at a perfectly good solution. This is why `TORTURE_DEGENERATE`
is in the test set.

Other exits:

* **Acceptable** — relaxed tolerances met for `acceptable_iterations = 15`
  consecutive iterations. Reported as its own flag, never as success.
* **Step tolerance** — steps below `tol.step` for 8 consecutive iterations
  while feasible. This is the honest answer on a degenerate problem where the
  multipliers are unbounded and `E_0` can never come down.
* **Diverging iterates** — `||x||_inf` and the objective have both moved by
  `1e10` relative to the starting point while feasible. IPOPT uses an absolute
  `1e20` threshold on `||x||`; measured on `TORTURE_UNBOUNDED`, an iterate
  growing linearly reaches only `3e13` in 420 iterations, so the absolute test
  never fires and the user gets `MaxReached`, which tells them nothing. The
  relative test fires in 16 iterations.

---

## 6. Initialization

1. Push `x0` inside its bounds with `kappa_1 = kappa_2 = 1e-2`:
   `p_L = min(kappa_1 max(1, |x_L|), kappa_2 (x_U - x_L))`, likewise above.
2. Set slacks from `c(x0)`, pushed inside the constraint bounds the same way.
3. `z_L = z_U = 1`.
4. Relax all bounds by `bound_relax_factor` (`1e-10`) so the strict interior is
   never empty — this is what makes `TORTURE_PINNED` (`lb == ub` on every
   variable) solvable at all.

**Not implemented:** the least-squares initialization of `lambda`, clamped at
`lambda_max = 1e3`. Currently `lambda` starts at zero. Worth doing; it is one
extra factorization and IPOPT considers it worth the cost.

After every step, reset the bound multipliers with `kappa_Sigma = 1e10`:

```
z_L_j <- min( max( z_L_j, mu / (kappa_Sigma d_j) ), kappa_Sigma mu / d_j )
```

This one line prevents the multipliers and `Sigma` from drifting apart, and
leaving it out produces a solver that works on easy problems and diverges on
hard ones.

---

## 7. Hessian

| Mode | Status |
|---|---|
| `Exact` | Implemented. Model supplies the lower triangle; transposed once into the upper triangle used by the KKT assembler. |
| `DenseBfgs` | Implemented, with Powell damping (`theta` such that `s'y >= 0.2 s'Bs`). `O(n^2)`. |
| `LimitedMemoryBfgs` | **Not implemented.** Falls back to dense. Required before `n > ~2000` is in scope. Use the compact Byrd–Nocedal–Schnabel representation so it enters the KKT matrix as a low-rank update rather than forcing a matrix-free method. |
| `FiniteDifference` | **Not implemented.** Needs star coloring (`mincon_diff::coloring::star_coloring`), not distance-1. |

Powell damping is not optional. On a constrained problem `y` is the change in
the gradient of the *Lagrangian*, which is indefinite at the solution, so
`s'y` goes negative regularly and an undamped update destroys the
approximation.

---

## 8. Feasibility restoration

**Implemented locally; CUTEst qualification pending.** Soft restoration and a
reduced elastic phase now recover from line-search/factorization failures.
The precise elimination, Gauss-Newton and re-entry variant is specified in
[`12_RESTORATION_IMPLEMENTATION.md`](12_RESTORATION_IMPLEMENTATION.md).

The specification (Wächter–Biegler §3.3):

```
min_{x, p, n}  rho * sum(p + n) + (zeta/2) ||D_R (x - x_R)||^2
               - mu_bar sum ln(...)
s.t.           c(x) - p + n = 0,   p, n >= 0
```

with `x_R` the current iterate, `zeta = sqrt(mu_bar)`,
`D_R = diag(min(1, 1/|x_R|))`, `rho = 1000`, and
`mu_bar_0 = max(mu, ||c(x_k)||_inf)`.

Leave restoration when the iterate is acceptable to the filter **and**
`theta <= kappa_resto * theta_R` with `kappa_resto = 0.9`.

Before entering the full phase, try **Algorithm R**: plain Newton steps on the
primal-dual system with fraction-to-boundary and no line search, accepted while
`||F_mu||_1` decreases by a factor `kappa_F = 0.999`. Cheap, and it rescues a
large fraction of cases.

`TORTURE_INFEASIBLE` now produces `LocallyInfeasible`: stationary positive
violation at small restoration barrier parameter. This first-order local
diagnostic neither proves global infeasibility nor certifies a local minimum.

---

## 9. Also not implemented

* **The watchdog** (§3.4 Case II). After four consecutive iterations whose
  first trial step was rejected, accept a full step without the filter check
  and verify over the next iteration.
* **The filter-reset heuristic** (§3.4 Case I). `Filter::reset_and_tighten`
  exists but is not wired in.
* **Damping for singly-bounded variables**, `kappa_d = 1e-4`, which stops the
  iterate running away along an unbounded direction of the barrier problem.
* **Inequality-constrained warm starting.**

## Sources

- Wächter & Biegler, *On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming*, Math. Prog. 106(1):25–57, 2006. [PDF](https://optimization-online.org/wp-content/uploads/2004/03/836.pdf)
- Chiang & Zavala, *An inertia-free filter line-search algorithm for large-scale nonlinear programming*, Comput. Optim. Appl. 64, 2016. [PDF](https://www.mcs.anl.gov/papers/P5197-0914.pdf)
- [IPOPT options reference](https://coin-or.github.io/Ipopt/OPTIONS.html) — every default above cross-checked against it.
