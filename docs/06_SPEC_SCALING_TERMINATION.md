# Specification: scaling and termination

Two small subsystems that decide a surprising share of the benchmark.

---

## Part 1: Scaling

### Why this is the highest-leverage default

A model whose objective is in joules and whose constraints are in millimetres
has a KKT matrix with a condition number in the millions before anything has
gone wrong. Users do not think about units; they think about their problem.

`fmincon` ships `ScaleProblem = false`. IPOPT ships
`nlp_scaling_method = gradient-based`. **We take IPOPT's side, and it is the
most consequential single default in the project.**

Measured: `TORTURE_SCALING` — twelve orders of magnitude between the objective
and constraint gradients — solves in **one iteration** with scaling on.

### The method: gradient-based (implemented)

At the starting point, with `g_max = 100` (`nlp_scaling_max_gradient`):

```
d_f    = min(1, g_max / ||grad f(x0)||_inf)
d_c_i  = min(1, g_max / ||grad c_i(x0)||_inf)
```

Scale down only, never up. Scaling *up* a tiny gradient amplifies noise and
turns a flat direction into a spurious steep one; the asymmetry is deliberate
and IPOPT does the same.

The solve then runs on `d_f * f` and `d_c * c`, with constraint bounds scaled
to match. **Unscaling on the way out is where the bugs live:**

```
lambda_i = lambda_tilde_i * d_c_i / d_f
z        = z_tilde / d_f
```

Derivation: the scaled Lagrangian is `d_f f + sum_i lambda_tilde_i d_c_i c_i`;
dividing stationarity through by `d_f` gives the above. Getting this wrong
produces multipliers that look plausible and are wrong by a constant factor,
which no test catches unless it checks multipliers explicitly. Add one.

The reported objective and constraint violation are always **unscaled**. A user
who sees a different objective than their own function returns will not trust
anything else the solver says.

### Not implemented

* **Ruiz equilibration** (`ScalingMode::Equilibration`). Iterative row/column
  scaling of the KKT matrix, refreshed on a schedule. Stronger when the
  conditioning is structural rather than unit-driven, but costs a
  factorization's worth of work per refresh. Measure before shipping.
* **User-supplied scaling** (`ScalingMode::User`).
* **Re-scaling during the solve.** Currently scaling is computed once at `x0`.
  A problem whose conditioning changes en route is not helped. IPOPT does the
  same; there may be a real win here and there may not.

### The escape hatch is mandatory

Scaling is a transformation of the user's problem and there are models it
hurts. `ScalingMode::None` must always work, `Options::fmincon_compatible()`
sets it, and the portfolio carries an `ip-unscaled` member specifically as
insurance against our own most opinionated default.

---

## Part 2: Termination

### The scaled KKT error

```
E_mu = max( ||grad f + A lambda - z_L + z_U||_inf / s_d ,
            ||c_hat||_inf ,
            ||(V - V_L) Z_L e - mu e||_inf / s_c )

s_d = max(s_max, (||lambda||_1 + ||z||_1) / (m + n_bounds)) / s_max
s_c = max(s_max, ||z||_1 / n_bounds) / s_max          with s_max = 100
```

Multiplier scaling improves numerical interpretation but can conceal failed
stationarity when multipliers diverge, as HS13 demonstrates. Strict `Optimal`
also requires the dual residual **before** division by `s_d` to meet the
optimality tolerance, in the fixed scaled problem. This is conservative on
badly conditioned models; acceptable and step-tolerance exits remain separate.

### Two tiers, reported separately

| Tier | Condition | Flag |
|---|---|---|
| Converged | `E_0 <= 1e-8`, pre-`s_d` dual residual `<= 1e-8`, violation `<= 1e-6`, complementarity `<= 1e-6` | `Optimal` |
| Acceptable | relaxed (`1e-6` / `1e-4`) for 15 consecutive iterations | `Acceptable` |

A solver that only knows how to succeed or fail reports failure on problems
where it found a perfectly usable point. IPOPT's acceptable-point machinery is
a large part of why it looks robust in benchmarks.

**`Acceptable` is not counted as success** — `ExitFlag::is_success()` is false
for it, and the benchmark harness reports strict and relaxed rates separately.
Quietly loosening tolerances is how benchmark tables get gamed.

### Other exits

* **`StepTolerance`** — steps below `1e-12` for 8 consecutive iterations while
  feasible. The honest answer on a degenerate problem where `E_0` can never
  come down. `fmincon`'s exit flag 2.
* **`Unbounded`** — either `f` below `objective_limit`, or **diverging
  iterates**: `||x||_inf` and the objective both moved by `1e10` relative to
  the start while feasible.

  IPOPT uses an absolute `1e20` threshold on `||x||`. Measured on
  `TORTURE_UNBOUNDED`, an iterate growing linearly reaches only `3e13` in 420
  iterations, so the absolute test never fires and the user gets `MaxReached`,
  which tells them nothing. The relative test fires in 16 iterations. This is
  a heuristic and the note attached to the result says so.
* **`LocallyInfeasible`** — the reduced elastic phase reached stationary
  positive violation at small barrier parameter. It is a local diagnostic,
  not an infeasibility proof. See `12_RESTORATION_IMPLEMENTATION.md`.

### Default tolerances, and why they differ from `fmincon`

| | `fmincon` | `mincon` | reason |
|---|---|---|---|
| optimality | `1e-6` | `1e-8` | ours is a *scaled* KKT error, so it is not as tight as it looks; matches IPOPT |
| feasibility | `1e-6` | `1e-6` | users read violation directly and `1e-4` looks sloppy |
| step | `1e-6` | `1e-12` | `1e-6` stops too early on well-scaled problems |
| max iterations | `400` flat | `400 + 10n`, capped `10000` | a flat cap is far too few for large problems |

### Rule

**Never report a tighter status than the numbers support.** The exit flag is
a claim about the mathematics, and a user will build on it. `fmincon`'s `-2`
is trustworthy after twenty years; ours has to be trustworthy from the start,
because we do not have twenty years of goodwill to spend.
