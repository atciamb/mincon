//! Solver options and their defaults.
//!
//! Gradient-based scaling, sparsity detection and the automatic portfolio
//! support use without user-supplied derivatives or algorithm settings.
//! Every option here is consumed by some part of the solver; an option that
//! did nothing, or silently did something else, is removed rather than kept
//! as a promise (September 18, 2026 re-audit of `docs/14`).

use std::fmt;
use std::sync::Arc;

use crate::{IterationRecord, EPS};

/// A per-iteration callback: receives the iteration record and returns
/// `true` to stop the solve (`fmincon`'s `OutputFcn` `stop`, SciPy's
/// `callback` returning `True`). Called after the record of every iteration,
/// including the last, by whichever algorithm is running; the portfolio
/// passes it to each member. A stop ends the solve with
/// `ExitFlag::StoppedByUser` and the current iterate.
#[derive(Clone)]
pub struct IterationCallback(Arc<dyn Fn(&IterationRecord) -> bool + Send + Sync>);

impl IterationCallback {
    /// Wrap a closure.
    pub fn new<F>(f: F) -> Self
    where
        F: Fn(&IterationRecord) -> bool + Send + Sync + 'static,
    {
        Self(Arc::new(f))
    }

    /// Invoke it. `true` means stop.
    #[must_use]
    pub fn call(&self, record: &IterationRecord) -> bool {
        (self.0)(record)
    }
}

impl fmt::Debug for IterationCallback {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("IterationCallback(..)")
    }
}

/// Which algorithm to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Algorithm {
    /// The portfolio: the SQP member first on problems with at most 20
    /// variables, the interior-point member first above that, then the
    /// cautious and unscaled interior-point configurations, run in sequence
    /// with shared budgets and early exit at the first usable answer (in
    /// parallel, at most `threads` at a time, when the model allows it).
    #[default]
    Auto,
    /// Primal-dual interior point with a filter line search.
    InteriorPoint,
    /// Sequential quadratic programming with an l1 merit function.
    Sqp,
}

/// How the barrier parameter is driven.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BarrierUpdate {
    /// Fiacco–McCormick: solve each barrier subproblem to `kappa * mu`, then
    /// `mu <- max(tol/10, min(kappa_mu * mu, mu^theta_mu))`.
    Monotone,
    /// Mehrotra-style probing / quality-function oracle chosen per iteration.
    /// Faster on well-behaved problems, occasionally erratic on hard ones.
    Adaptive,
    /// Start adaptive, fall back to monotone after repeated rejected steps.
    /// This is the default because it wins on both halves of the test set.
    #[default]
    AdaptiveThenMonotone,
}

/// Problem scaling strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScalingMode {
    /// No scaling. Use only to reproduce a `fmincon` run exactly.
    None,
    /// IPOPT's `gradient-based`: `d_f = min(1, g_max / ||grad f(x0)||_inf)` and
    /// likewise per constraint row. Cheap, one extra gradient, very effective.
    #[default]
    GradientBased,
}

/// How second derivatives are obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HessianMode {
    /// Exact Hessian of the Lagrangian from the model.
    Exact,
    /// Dense BFGS. `fmincon`'s interior-point default. `O(n^2)` memory; the
    /// practical ceiling is a few thousand variables.
    DenseBfgs,
    /// Exact if the model provides it, else dense BFGS.
    #[default]
    Auto,
}

/// Finite-difference flavour.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FdType {
    /// One extra evaluation per column; `O(sqrt(eps))` accurate.
    Forward,
    /// Two extra evaluations per column; `O(eps^(2/3))` accurate.
    Central,
    /// Forward until progress stalls, then central. `fmincon` makes the user
    /// choose; we switch automatically, which recovers most of the accuracy
    /// benefit for a fraction of the cost.
    #[default]
    Adaptive,
}

/// Whether the variables are scaled by their starting magnitudes (see
/// [`Options::scale_variables`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VariableScaling {
    /// Never.
    Off,
    /// Only when the scale factors `max(|x0_i|, typical_i)` span a factor of
    /// 1e4 or more, i.e. when the start says the units are mismatched.
    #[default]
    Auto,
    /// Always (when any factor differs from 1).
    On,
}

/// How user-supplied derivatives are checked before a solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DerivativeCheck {
    /// No check.
    Off,
    /// Two extra model evaluations along one bounds-respecting direction at
    /// `x0`, comparing the directional derivative of the objective and of
    /// every constraint row with the supplied gradient and Jacobian. The
    /// default: it costs the same for `n = 2` and `n = 2000`, and a wrong
    /// gradient is the most common reason a solve "does not work". A gross
    /// disagreement (relative error above 1e-2) stops the solve with the
    /// offending components named by the full check; a mild one is recorded
    /// in the report's notes and the solve continues. Skipped silently when
    /// the model supplies no derivatives.
    #[default]
    Directional,
    /// `fmincon`'s `CheckGradients`: every entry at several points; a failed
    /// or inconclusive check stops the solve.
    Full,
}

/// What the interior-point member's `LDL^T` does with a pivot whose sign
/// differs from the block it belongs to (see [`Options::kkt_pivot_signs`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PivotSigns {
    /// [`PivotSigns::Free`] when the member uses the model's exact Hessian,
    /// [`PivotSigns::Expected`] with a quasi-Newton one.
    #[default]
    Auto,
    /// Every primal pivot must be positive and every dual pivot negative; a
    /// pivot of the other sign is perturbed to the expected sign, which makes
    /// the factorisation belong to a nearby matrix and loses the inertia
    /// certificate. Right for a positive-definite (quasi-Newton) primal block,
    /// where a wrong-sign pivot can only be numerical noise.
    Expected,
    /// Pivots keep their sign and the inertia is counted (Sylvester's law);
    /// only pivots that are numerically zero are perturbed. An indefinite
    /// exact Hessian whose reduced Hessian is positive then certifies the
    /// inertia `(n, m)` without any `delta_w`, where `Expected` would perturb
    /// its negative primal pivots and force the inertia correction at every
    /// iteration (HS71 with the exact Hessian: 56 iterations against 10).
    Free,
}

/// How the quadratic-program probe builds the objective's constant Hessian
/// when it has function values only (see [`Options::quadratic_build`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QuadraticBuild {
    /// The diagonal first (`2n` evaluations, which also give the gradient),
    /// then one off-diagonal band at a time, the model checked against the
    /// probe's six line points after each; the build stops at the first
    /// structure that reproduces them. A diagonal Hessian costs `2n`, a
    /// tridiagonal one `3n - 1`, a dense one exactly what [`Self::Dense`]
    /// costs, in a different order. Above the dense size limit, or when the
    /// dense build would exceed the budget, the band search may spend `2n`
    /// evaluations more before it declines.
    #[default]
    Structured,
    /// Every pair at once, `n (n + 3) / 2` evaluations, up to `n = 500`.
    Dense,
}

/// Whether the quadratic-program probe also accepts quadratic constraint rows
/// (see [`Options::quadratic_rows`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QuadraticRows {
    /// Only linear rows: a row that is not linear along a probe line declines
    /// the probe (the behaviour of `abl-i5` and `abl-i5-build`).
    Off,
    /// Quadratic rows are accepted when the model supplies exact Jacobians:
    /// each row's constant Hessian comes from differencing the Jacobian,
    /// `n + 1` Jacobian calls for every row at once.
    Jacobian,
    /// As [`Self::Jacobian`], and without a Jacobian from constraint values:
    /// one constraint evaluation gives every row, so the build costs what the
    /// objective's does (`2n` evaluations for diagonal rows, `n (n + 3) / 2`
    /// for dense ones).
    #[default]
    Values,
}

/// When the SQP member adopts its QP's multipliers and re-runs the termination
/// test before trying the step (see [`Options::zero_step`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ZeroStep {
    /// When the step's norm is within the step tolerance (and the step is then
    /// the end of the solve either way). The behaviour before Phase D.
    Norm,
    /// As [`Self::Norm`], and once at a feasible point when the step's
    /// predicted decrease of the merit function is below the rounding noise of
    /// the merit function, `100 eps max(1, |merit|)`. No line search can verify
    /// such a step, and multipliers are otherwise updated only by an accepted
    /// step, so at a degenerate vertex the termination test can be looking at
    /// the previous step's multipliers, far from the QP's. If the test still
    /// fails with the QP's multipliers the step goes to the line search as
    /// under [`Self::Norm`]: the condition is never a reason to stop. The
    /// default since Phase D (`docs/22` section 7.18: no corpus record loses
    /// attainment or its certificate, sixteen save evaluations).
    #[default]
    Decrease,
}

/// The SQP member's first trial step when its second-order probe has found a
/// direction of negative curvature (see [`Options::saddle_step`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SaddleStep {
    /// `t = max |x|` (at least 1), halved until the merit function decreases.
    /// The behaviour before Phase D.
    Scale,
    /// As [`Self::Scale`], capped by a ratio test: 0.99 of the largest step
    /// that stays inside the linearization of every inactive constraint row
    /// (not the bounds: trial points are projected onto those). With many
    /// rows the uncapped first trials are
    /// all infeasible and each costs two model evaluations (`docs/22` section
    /// 7.17: 19 of the 104 evaluations of a 19-variable design with 1384 rows).
    /// The default since Phase D (`docs/22` section 7.18).
    #[default]
    Linearized,
}

/// A warm start (round 5 I6): the multipliers of a previous solve of the same
/// problem, in the user's units and the report's sign convention (`Solution`),
/// optionally with the barrier parameter the previous solve reached. Both
/// members start from them instead of zero (interior point: bound multipliers
/// clamped to the barrier's neighbourhood of `mu`, which is taken from `mu` or
/// the default), so a solve interrupted by a budget resumes from `x` and its
/// duals instead of re-estimating them. Lengths that do not match the problem
/// are ignored with a note.
#[derive(Debug, Clone, Default)]
pub struct WarmStart {
    /// Constraint multipliers, `m` entries.
    pub lambda: Vec<f64>,
    /// Lower-bound multipliers, `n` entries (zero where there is no bound).
    pub z_l: Vec<f64>,
    /// Upper-bound multipliers, `n` entries.
    pub z_u: Vec<f64>,
    /// Barrier parameter to resume the interior-point member at.
    pub mu: Option<f64>,
    /// The previous solve's quasi-Newton model (`SolveReport::quasi_newton`),
    /// dense row-major `n x n` in the user's units; both members start their
    /// BFGS model from it instead of the identity.
    pub quasi_newton: Option<Vec<f64>>,
}

/// How the KKT matrix is regularized and its inertia established.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RegularizationMode {
    /// Wächter–Biegler Algorithm IC: require inertia `(n, m, 0)` from the
    /// factorization, raising `delta_w` until it holds.
    Inertia,
    /// Reserved for a Chiang–Zavala curvature-based loop. Currently uses the
    /// certified-inertia path; the standalone curvature helper is not wired
    /// into the solver's acceptance/retry loop.
    InertiaFree,
    /// Currently requires certified inertia, like `Inertia`. A curvature
    /// fallback remains planned and must be independently qualified.
    #[default]
    Hybrid,
}

/// Termination tolerances.
///
/// Note the split between "converged" and "acceptable". A solver that only
/// knows how to succeed or fail reports failure on problems where it in fact
/// found a perfectly usable point; IPOPT's acceptable-point machinery is a
/// large part of why it looks robust in benchmarks, and `fmincon` has only a
/// weak analogue in its exit flag 2. We keep both and report which was met.
#[derive(Debug, Clone, Copy)]
pub struct Tolerances {
    /// Target on the scaled KKT error `E_0`. `fmincon`'s `OptimalityTolerance`.
    pub optimality: f64,
    /// Maximum allowed constraint violation. `fmincon`'s `ConstraintTolerance`.
    pub feasibility: f64,
    /// Maximum allowed complementarity residual.
    pub complementarity: f64,
    /// Relative step below which the solve stops. `fmincon`'s `StepTolerance`.
    pub step: f64,
    /// Relaxed optimality target for the "acceptable point" exit.
    pub acceptable_optimality: f64,
    /// Relaxed feasibility target for the "acceptable point" exit.
    pub acceptable_feasibility: f64,
    /// Number of consecutive acceptable iterations before stopping there.
    pub acceptable_iterations: usize,
    /// Below this objective value the solve stops and reports unboundedness.
    /// `fmincon`'s `ObjectiveLimit`.
    pub objective_limit: f64,
    /// Constraint violation above which, with no further progress possible,
    /// the problem is declared locally infeasible.
    pub infeasibility_declared: f64,
}

impl Default for Tolerances {
    fn default() -> Self {
        Self {
            // fmincon uses 1e-6 for both; IPOPT uses 1e-8 for the scaled KKT
            // error and 1e-4 for constraint violation. The optimality target is
            // 1e-6, like fmincon: with forward finite differences the
            // stationarity residual cannot reliably fall below ~1e-7, and a
            // 1e-8 target was measured (bench/results/s3-tolerance) to cost
            // 1.6x the evaluations on the development corpus for no gain in
            // independently verified accuracy, ending in "acceptable" exits
            // after 15 wasted iterations. Users with exact derivatives can
            // tighten it. Feasibility keeps fmincon's 1e-6 because users read
            // constraint violation directly.
            optimality: 1e-6,
            feasibility: 1e-6,
            complementarity: 1e-6,
            step: 1e-12,
            acceptable_optimality: 1e-4,
            acceptable_feasibility: 1e-4,
            acceptable_iterations: 15,
            objective_limit: -1e20,
            infeasibility_declared: 1e-8,
        }
    }
}

/// Everything that can be configured.
#[derive(Debug, Clone)]
pub struct Options {
    /// Which algorithm(s) to run.
    pub algorithm: Algorithm,
    /// Termination tolerances.
    pub tol: Tolerances,
    /// Iteration cap, or `None` for the default `400 + 10 n` (at most 10 000):
    /// `fmincon` uses a flat 400, which is far too few for large problems and
    /// generous for tiny ones. A cap the caller sets is a limit on the whole
    /// solve and the portfolio shares it across its members, like
    /// `max_evaluations` and `max_seconds`; the default applies to each member
    /// separately, as a safety net rather than a budget.
    pub max_iterations: Option<usize>,
    /// Objective-evaluation cap, or `None` for unlimited.
    pub max_evaluations: Option<u64>,
    /// Wall-clock cap in seconds, or `None`.
    pub max_seconds: Option<f64>,
    /// Record a per-iteration trace in the report. Cheap; on by default because
    /// "why did it stop there" is the most common user question.
    pub record_trace: bool,
    /// Called after every iteration with its record; returning `true` stops the
    /// solve (see [`IterationCallback`]).
    pub callback: Option<IterationCallback>,

    /// Scaling strategy.
    pub scaling: ScalingMode,
    /// Gradient magnitude above which scaling kicks in. IPOPT's
    /// `nlp_scaling_max_gradient`.
    pub scaling_max_gradient: f64,

    /// Second-derivative strategy.
    pub hessian: HessianMode,
    /// Replace the unit initial quasi-Newton matrix by a diagonal built from
    /// the first curvature pair when the first line search had to cut its step
    /// hard (a sign the unit matrix misjudged the problem's scale). A scalar
    /// rescale, guarded or not, was measured as a wash; the guarded diagonal
    /// form was measured at 0.94× evaluations with one more problem attained.
    pub bfgs_guarded_scaling: bool,
    /// Rebuild the quasi-Newton matrix from the per-coordinate curvature
    /// quotients of the latest pair whenever the model's curvature along the
    /// accepted step is off by more than this factor in either direction and
    /// the quotients say so consistently (see `DenseBfgs::set_curvature_rescale`
    /// in `mincon-ip`). The trace study `bench/results/r5-large-n` found both
    /// members spending hundreds of iterations repairing one direction per
    /// update on problems whose curvature grows by two orders of magnitude
    /// along the path. `f64::INFINITY` disables the rule, and is the
    /// default: the rule (candidate C7, factor 10) cut evaluations on the
    /// development corpus but cost 1.24× [0.99, 1.67] on the sealed round-4
    /// set, whose dense coupled problems it rebuilds to a wrong diagonal
    /// (`bench/results/s6v4-final4`). It stays available as an opt-in for
    /// separable problems.
    pub bfgs_curvature_rescale: f64,

    /// Scale the variables internally by `max(|x0_i|, typical_x_i)` (1 where
    /// both are zero) before solving, so that a start like (1e6, 1e-6) is
    /// seen by the solver as (1, 1): finite-difference steps, the start push,
    /// the step bound and the quasi-Newton model all assume variables of
    /// order one. The solution and the bound multipliers are mapped back.
    /// Default [`VariableScaling::Auto`]: on the 169-problem corpus it fires on
    /// one problem (HS117, attained either way at 8x the evaluations) and
    /// changes nothing else (`bench/results/abl-i8`), while it is the only
    /// thing that solves a unit-mismatched start such as (1e6, 1e-6), where
    /// every first-order certificate is blind (`docs/22` I8). `fmincon`'s
    /// `TypicalX` is the manual version.
    pub scale_variables: VariableScaling,
    /// Probe for a quadratic program at the start (round 5 I5): when the
    /// objective is quadratic along two random lines and every constraint row
    /// is linear along them, the constant Hessian is built by differencing
    /// and the SQP member runs with it as an exact Hessian before the
    /// ordinary portfolio. A general nonlinear problem pays three objective
    /// evaluations for the test. Default on since `bench/results/abl-i5`: no
    /// corpus attainment changed, cost 1.04 [0.90, 1.16] (0.2x to 0.8x on
    /// iteration-heavy QPs, 1.5x to 6.6x on dense ones the quasi-Newton path
    /// solved in few iterations), POLYQP_100 certified, box_lsq 0.17x.
    pub quadratic_probe: bool,
    /// How the probe builds the Hessian from function values: by structure
    /// (diagonal, then bands, stopping at the first that fits the line
    /// points) or dense at once. Default [`QuadraticBuild::Structured`]
    /// since `bench/results/abl-i5-build`: the QPs the dense build made
    /// 1.5x to 8x dearer (QUADSPHERE, LQTRAJ, MANY_INEQ) have diagonal or
    /// zero Hessians, OBSTACLE's is tridiagonal, and a dense Hessian costs
    /// the same either way.
    pub quadratic_build: QuadraticBuild,
    /// Whether the probe also accepts quadratic constraint rows. A row is
    /// accepted only when it keeps the feasible set convex (a convex function
    /// bounded above or a concave one bounded below, never an equality or a
    /// ranged row, checked first by the sign of its curvature along the probe
    /// lines and then by a Cholesky of its built Hessian); every row's
    /// constant Hessian is then added to the Lagrangian's with its
    /// multiplier. Default [`QuadraticRows::Values`] since
    /// `bench/results/abl-i5-rows`: one more problem attained on track A
    /// (a 500-variable ellipsoid projection whose control run ended the
    /// evaluation budget at an infeasible point), one more certified
    /// (ELLIPSOID2_200 at 0.05x the evaluations), nothing lost, cost
    /// 0.99 [0.92, 1.01] with finite differences and 0.98 [0.85, 1.06] with
    /// exact derivatives counted in full.
    pub quadratic_rows: QuadraticRows,
    /// The first trial step off a saddle point found by the SQP member's
    /// second-order probe. Default [`SaddleStep::Linearized`].
    pub saddle_step: SaddleStep,
    /// When the SQP member adopts its QP's multipliers and re-tests before
    /// trying the step. Default [`ZeroStep::Decrease`].
    pub zero_step: ZeroStep,
    /// Finite-difference flavour.
    pub fd_type: FdType,
    /// Relative finite-difference step, or `None` to use `sqrt(eps)` forward /
    /// `eps^(1/3)` central, matching `fmincon`.
    pub fd_step: Option<f64>,
    /// Keep every finite-difference probe inside the variable bounds by
    /// flipping the step direction near a bound. `fmincon`'s `sqp` and
    /// `interior-point` do this; it matters enormously for models that are
    /// undefined outside their box, and it is off in most open-source solvers.
    pub fd_respect_bounds: bool,
    /// Use graph coloring to batch finite-difference columns when structure is
    /// known or detected.
    pub fd_coloring: bool,
    /// Probe for Jacobian/Hessian sparsity when the model declares none.
    pub detect_sparsity: bool,
    /// With finite-difference derivatives, estimate their error near
    /// convergence (a few extra evaluations, at most every ten iterations)
    /// and stop when the KKT residual is below that error rather than chasing
    /// an optimality tolerance the derivatives cannot support; switch to
    /// central differences first when forward ones are too inaccurate.
    pub fd_error_aware: bool,

    /// Keep every iterate strictly inside the variable bounds.
    /// `fmincon`'s `HonorBounds`, default `true` there and here.
    pub honor_bounds: bool,
    /// Relax bounds by this relative amount before starting, so the strict
    /// interior is non-empty even when `lb == ub`. IPOPT's
    /// `bound_relax_factor`.
    pub bound_relax_factor: f64,

    /// Barrier parameter strategy.
    pub barrier_update: BarrierUpdate,
    /// Initial barrier parameter.
    pub mu_init: f64,
    /// Fraction-to-boundary floor.
    pub tau_min: f64,

    /// Multipliers to start from (see [`WarmStart`]).
    pub warm_start: Option<WarmStart>,
    /// KKT regularization strategy.
    pub regularization: RegularizationMode,
    /// What the `LDL^T` does with a pivot whose sign differs from its block's.
    /// Default [`PivotSigns::Auto`]: counted with an exact Hessian, expected
    /// with a quasi-Newton one (round 5 S-E, `bench/results/abl-se`).
    pub kkt_pivot_signs: PivotSigns,
    /// Steps of iterative refinement on each KKT solve. One is nearly free and
    /// buys a lot on ill-conditioned systems.
    pub refinement_steps: usize,

    /// Maximum second-order corrections per line search. IPOPT's `max_soc`.
    pub max_soc: usize,

    /// Worker threads, or `None` for "all available".
    pub threads: Option<usize>,
    /// Check user-supplied derivatives against finite differences at the start
    /// (see [`DerivativeCheck`]).
    pub check_derivatives: DerivativeCheck,
    /// Seed for every stochastic decision (multi-start perturbations, tie
    /// breaks). Fixing this makes a run bit-reproducible, which `fmincon`
    /// does not promise and which CI needs.
    pub seed: u64,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            algorithm: Algorithm::Auto,
            tol: Tolerances::default(),
            max_iterations: None,
            max_evaluations: None,
            max_seconds: None,
            record_trace: true,
            callback: None,

            scaling: ScalingMode::GradientBased,
            scaling_max_gradient: 100.0,

            hessian: HessianMode::Auto,
            bfgs_guarded_scaling: true,
            bfgs_curvature_rescale: f64::INFINITY,

            scale_variables: VariableScaling::Auto,
            quadratic_probe: true,
            quadratic_build: QuadraticBuild::Structured,
            quadratic_rows: QuadraticRows::Values,
            saddle_step: SaddleStep::Linearized,
            zero_step: ZeroStep::Decrease,
            fd_type: FdType::Adaptive,
            fd_step: None,
            fd_respect_bounds: true,
            fd_coloring: true,
            detect_sparsity: true,
            fd_error_aware: true,

            honor_bounds: true,
            bound_relax_factor: 1e-10,

            barrier_update: BarrierUpdate::AdaptiveThenMonotone,
            mu_init: 0.1,
            tau_min: 0.99,

            warm_start: None,
            regularization: RegularizationMode::Hybrid,
            kkt_pivot_signs: PivotSigns::Auto,
            refinement_steps: 1,

            max_soc: 4,

            threads: None,
            check_derivatives: DerivativeCheck::Directional,
            seed: 0x5EED_C0FF_EE00_0F0F,
        }
    }
}

impl Options {
    /// The default forward finite-difference relative step, `sqrt(eps)`.
    #[must_use]
    pub fn default_fd_step_forward() -> f64 {
        EPS.sqrt()
    }
    /// The default central finite-difference relative step, `eps^(1/3)`.
    #[must_use]
    pub fn default_fd_step_central() -> f64 {
        EPS.powf(1.0 / 3.0)
    }

    /// Options that reproduce `fmincon`'s interior-point defaults as closely as
    /// this solver can, for A/B comparison in the benchmark harness. Never the
    /// recommended configuration — it exists so we can measure how much of our
    /// win comes from better defaults versus better algorithms, which is a
    /// question a reviewer will ask and we should be able to answer.
    #[must_use]
    pub fn fmincon_compatible() -> Self {
        Self {
            algorithm: Algorithm::InteriorPoint,
            tol: Tolerances {
                optimality: 1e-6,
                feasibility: 1e-6,
                step: 1e-10,
                ..Tolerances::default()
            },
            max_iterations: Some(400),
            scaling: ScalingMode::None,
            hessian: HessianMode::DenseBfgs,
            fd_type: FdType::Forward,
            barrier_update: BarrierUpdate::Monotone,
            threads: Some(1),
            ..Self::default()
        }
    }

    /// Resolve `max_iterations` for a problem of size `n` when the caller left
    /// the default in place.
    #[must_use]
    pub fn effective_max_iterations(&self, n: usize) -> usize {
        self.max_iterations
            .unwrap_or_else(|| (400 + 10 * n).min(10_000))
    }

    /// Check for self-consistency.
    ///
    /// # Errors
    /// A description of the first inconsistency found.
    pub fn validate(&self) -> Result<(), String> {
        if self.tol.optimality <= 0.0 || self.tol.feasibility <= 0.0 {
            return Err("tolerances must be positive".into());
        }
        if self.tol.acceptable_optimality < self.tol.optimality {
            return Err("acceptable_optimality must not be tighter than optimality".into());
        }
        if !(0.0..1.0).contains(&self.tau_min) {
            return Err("tau_min must lie in [0, 1)".into());
        }
        if self.mu_init <= 0.0 {
            return Err("mu_init must be positive".into());
        }
        if self.bfgs_curvature_rescale.is_nan() || self.bfgs_curvature_rescale <= 1.0 {
            return Err(
                "bfgs_curvature_rescale must exceed 1 (use infinity to disable the rule)".into(),
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_validate() {
        Options::default().validate().unwrap();
        Options::fmincon_compatible().validate().unwrap();
    }

    #[test]
    fn iteration_cap_scales_with_n() {
        let o = Options::default();
        assert_eq!(o.effective_max_iterations(10), 500);
        assert_eq!(o.effective_max_iterations(100_000), 10_000);
        let o2 = Options {
            max_iterations: Some(42),
            ..Options::default()
        };
        assert_eq!(o2.effective_max_iterations(10), 42);
    }

    #[test]
    fn fmincon_profile_turns_off_our_advantages() {
        let o = Options::fmincon_compatible();
        assert_eq!(o.scaling, ScalingMode::None);
        assert_eq!(o.threads, Some(1));
        assert_eq!(o.algorithm, Algorithm::InteriorPoint);
    }
}
