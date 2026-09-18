# Changelog

Versions follow semantic versioning; before 1.0 a minor version may change
defaults. Every default below went behind an option first and was kept only
after an ablation on the whole development corpus (`docs/22`, `docs/23`). What
is measured against `fmincon` and SciPy, including where they win, is in
`README.md` and `docs/17_CLAIM_AUDIT.md`.

## 0.2.0 (September 18, 2026)

Experimental. `success` still means a certified point and nothing else; an
`Acceptable` or budget exit has `success=False` and `usable` says whether the
point can be used.

**Standing of this release.** The sealed comparisons in the README are rounds
1 to 5 and were run on earlier candidates. The defaults added since round 5
(`quadratic_bands`, `zero_step`, `saddle_step`, the budget accounting and the
`workers=` and `vectorized=` paths) are ablated on the 185-problem development
corpus and are not yet qualified on a held-out round. Round 6 (23 sealed
problems in 11 new families, `bench/results/s6v6-final6/SEAL.md`) measures
exactly this tree; its report follows in the repository and changes the README
whatever it finds.

### Added

* An SQP member (l1 merit function, elastic QP on a Goldfarb-Idnani dual
  active-set solver, damped BFGS, second-order corrections, a second-order
  check that walks off saddle points). The portfolio runs it first on problems
  with at most 20 variables; `method=` selects `'auto'`, `'interior-point'`
  or `'sqp'`.
* A quadratic-program probe, on by default (`quadratic_probe`): seven
  evaluations at the start test whether the objective is quadratic and the
  rows linear; if so the constant Hessian is built by structure
  (`quadratic_build='structured'`: the diagonal, then one band at a time) and
  the SQP member takes Newton steps with it. `quadratic_rows='values'` also
  accepts convex quadratic constraint rows. `quadratic_bands='decaying'` lets
  the band search go on while the fit error keeps falling, when the dense
  build would not fit the budget. `res.notes` says what was built or why the
  probe declined.
* `workers=` (finite-difference probes on worker processes, for expensive
  models) and `vectorized=True` (one batched NumPy call per gradient). The
  iterates are identical with and without them.
* `hess=` (exact Lagrangian Hessian, dense), `callback=` (one row per
  iteration, may stop the solve), `warm_start=` (a previous result, including
  its quasi-Newton model), `display` / `disp`.
* A supplied gradient or Jacobian is checked along one direction at `x0` (two
  extra evaluations): a gross disagreement raises and names the component, a
  mild one is recorded in `res.notes`. `check_derivatives='full'` tests every
  entry at three points; `check_gradients` does it on request.
* `res.limit`: at a budget exit (`status == 0`), which limit bound
  (evaluations, time or iterations).
* Constraint Jacobians from Python (`'jac'` in a constraint dict,
  `nonlcon_jac` for `fmincon`); linear rows given as `A`, `b`, `Aeq`, `beq`
  carry their exact Jacobian.
* macOS wheels (universal2) next to Windows x86_64 and Linux x86_64.

### Changed defaults

* `scale_variables='auto'`: the solve runs in variables divided by their
  starting magnitudes when those span a factor of 1e4 or more (`fmincon`'s
  `TypicalX`, done for you), and the answer is mapped back.
* `kkt_pivot_signs='auto'`: with a supplied Hessian the interior-point
  member's factorisation counts inertia instead of perturbing a pivot.
* `zero_step='decrease'` and `saddle_step='linearized'` in the SQP member: a
  step no line search can verify is re-tested with the QP's multipliers, and
  the step off a saddle is capped by the distance to the inactive rows.
* Termination matches the accuracy finite differences can support
  (`fd_error_aware`), instead of chasing a tolerance the derivatives cannot
  reach.
* A `maxiter` you set is a limit on the whole solve, shared across the
  portfolio's members like `maxfev` and `maxtime`; the default cap of
  `400 + 10 n` still applies to each member. The portfolio counts evaluations
  at the model boundary.

### Fixed

* A false `Optimal` from a stale objective scaling, and a stationary
  infeasible point that was not diagnosed as such (D9, D10 of
  `docs/16_FAILURE_ATLAS.md`).
* The certificate guard (D11): complementarity must also hold in the user's
  units before the interior-point member certifies, so a badly scaled problem
  exits `Acceptable` (`success=False`) where it used to claim `Optimal` away
  from the optimum.
* The unbounded-problem diagnosis, and multipliers of fixed variables.
* The Python docstring gave `quadratic_rows` the wrong default.

### Known limits, unchanged

A finite-difference gradient costs `n` model evaluations per iteration; dense
quasi-Newton models make problems beyond a few hundred variables expensive;
sparse Jacobians and `jac_sparsity` from Python and limited-memory curvature
are not implemented; budgets are checked between iterations, so they cannot
interrupt a running callback. On the sealed rounds so far SciPy SLSQP attains
as much as mincon at fewer evaluations (`README.md`). When several portfolio
members ran, `res.nfev` is smaller than the number of calls the model saw;
`res.notes` gives the total (`docs/17`, item 22). To be fixed in 0.2.1: the
tree is frozen while round 6 measures it.

## 0.1.0 (September 7, 2026)

First public release: a primal-dual interior-point method with feasibility
restoration, the `fmincon` and `minimize` entry points, bounds-aware finite
differences with sparsity detection and graph colouring, multistart.
