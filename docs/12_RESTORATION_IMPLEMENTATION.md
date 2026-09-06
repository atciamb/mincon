# Restoration implementation contract

This refines `02` section 8 before implementation. The reference is Wachter and
Biegler, sections 3.3.1-3.3.2 of the [deposited paper](https://optimization-online.org/wp-content/uploads/2004/03/836.pdf).
It does not establish a measured CUTEst improvement.

## Soft restoration

After ordinary line-search failure, augment the current filter.
Try regular Newton directions with a **common** primal/dual fraction-to-boundary
step. Accept a trial only if the unscaled l1 norm of all primal-dual barrier
equations falls by `0.999`; use actual trial derivatives, including slack
stationarity and both bound complementarity blocks. Return to the main method
only when the trial passes the augmented filter. Otherwise repeat, charging
iterations and evaluations. A failed factorization skips this phase.

No barrier update occurs inside soft restoration. Rejected evaluations or
insufficient residual reduction lead to full restoration, not success. A user
abort propagates. Restoration must not reuse derivatives from a rejected point.

Before full restoration, allow one reset of an existing BFGS model to its
initial unit matrix and retry soft restoration. This addresses a curvature
model poisoned by tiny secant steps; it is tested on a quadratic with an
artificially inflated model. The actual nonlinear residual and filter still
decide acceptance. Exact Hessians are not reset and the retry shares the budget.

## Robust restoration through elimination of elastic variables

Use the existing scaled equality/slack variables `v` and residual `r=c_hat(v)`.
The elastic objective is `rho*sum(p+n) + sqrt(mu_R)/2*||D_R(v-v_R)||^2`,
with `r-p+n=0`, positive `p,n`, the existing bounds on `v`, and log barriers.
Keep `rho=1000`, `mu_R=max(mu,||r||_inf)`, and
`D_R[j]=1/max(1,abs(v_R[j]))`.

Implementation variant: minimize over each elastic pair analytically at every
trial rather than adding `2*m` primal variables. Let `a=mu_R/rho`,
`q=hypot(a,r_i)`. Then

```
p+n = a+q, p-n = r_i
lambda_R[i] = rho*r_i/(q+a)
T[i] = (d lambda_R[i]/d r_i)^(-1) = q*(q+a)/(rho*a)
```

Compute the smaller elastic using a cancellation-free formula, then the larger
as `abs(r_i)+smaller`. The reduced objective includes both elastic log terms;
its gradient is `J_hat^T*lambda_R` plus proximity and variable-bound barriers.

Use a positive Gauss-Newton curvature model, omitting constraint second
derivatives. Its augmented system is

```
[ diag(sqrt(mu_R)*D_R^2 + barrier_curvature)  J_hat^T ] [d_v] = [-gradient]
[ J_hat                                      -T   ] [d_l]   [    0    ]
```

This uses the existing KKT sparsity, a row-specific negative dual diagonal,
and certified inertia. A bound-respecting Armijo search on the reduced elastic
objective globalizes each step. This is an explicitly documented variant of
the full elastic phase, not a literal reproduction of the paper's p/n iterates.

Reduce `mu_R` using the existing monotone schedule when the restoration
gradient is small relative to `mu_R`. Proximity changes with `mu_R`.
Re-enter only after `theta<=0.9*theta_R` and acceptance by the augmented main
filter, with a finite original objective. Reset incompatible BFGS state and
reinitialize original-problem multipliers; do not return restoration multipliers
as if they solved the original objective.

## Stopping, budgets and evidence

### Prerequisite uncovered during implementation

Dynamic pivot repair can hide a singular original KKT matrix from Algorithm
IC. A repaired pivot must trigger explicit dual regularization when necessary.
The existing Hybrid path also accepted uncertified inertia without executing
its documented curvature check. Until a complete inertia-free step loop is
qualified, all modes conservatively require certified KKT inertia.

Before changing the mathematical matrix on a failed factorization, try a
positive diagonal congruence `K_scaled=S*K*S`. Scale primal rows from their
diagonal magnitude (row maximum when zero), then dual rows from their scaled Jacobian entries and
dual diagonal. Solve `K_scaled*y=S*b`, return `d=S*y`. Congruence preserves
inertia and this transformation preserves the Newton equations. It addresses
absolute pivot thresholds applied to differently scaled blocks, without changing
those thresholds. Refinement uses the scaled matrix; also return the residual
against the original assembled equations. This targeted prerequisite is moved
ahead of broader M9 equilibration because restoration requires trustworthy steps.

Before equilibration, a failed fill-reducing ordering may retry with primal
variables first (natural KKT order). RCM can place a zero equality diagonal
first even for `H=I` and a full-rank Jacobian. Increasing primal regularization
cannot fix that first pivot. Lazily build this fallback symbolic factorization
once and retain it. This trades potential sparse fill for a trustworthy solve;
large sparse performance still needs qualification. Test the unregularized
identity-Hessian equality system against its analytical solution.

HS13 also exposes a bound-relaxation defect: allowing violations of only 1e-10
changes its optimum by order 1e-3 because of the cubic constraint. With bound
honoring enabled, relax only fixed variable intervals internally; callbacks and
returned points use the exact fixed coordinate. Ordinary variable and slack
bounds keep their original endpoints. Initial scaling and sparsity probes must
also start inside the user box. Restoration never earns success by solving a
relaxed feasible set.

Strict success additionally requires the stationarity residual **before**
division by the multiplier-dependent `s_d` to satisfy the requested optimality
tolerance (in the solver's fixed scaled coordinates). Large multipliers can
otherwise make the scaled test arbitrarily small at a non-KKT limit. The
existing acceptable-iterate streak and step-stagnation rules remain available,
with their non-success flags. No new tolerance or relaxed success threshold is
introduced.

Positive violation plus small first-order restoration residual at small
`mu_R` supports **stationary infeasibility**, not a global infeasibility proof
or a second-order local-minimum certificate. `LocallyInfeasible` must explain
this limitation and must remain a non-success status. Near-feasible failed
restoration cannot produce that status. Tiny feasible steps may produce
`StepTolerance`, which also remains non-success.

Charge all restoration iterations, objective/constraint/derivative calls and
elapsed time to the parent solve. Record restoration trace rows and reasons.
Preserve a finite evaluated point on failure. Do not relax requested tolerances
or modify published constants to make the milestone pass.

The evaluator now counts individual finite-difference callbacks (including
failed batches), analytic derivative failures and sparsity-detection probes.
Final reporting reuses validated derivatives and makes no callback after a
budget/abort exit. Budgets are checked at iteration/backtracking boundaries:
a finite-difference batch can exceed an evaluation limit, and in-flight
callbacks cannot be interrupted. Hard per-callback limits remain future work.
Python's initial constraint-size probe remains outside Rust report counters.

Tests use analytical elastic optima, a hand-solvable augmented matrix,
contradictory affine constraints, and the known HS13 boundary solution. The
development regression is independently evaluated at returned points. The
CUTEst gate remains pending unless that dataset is available and the gate runs.
