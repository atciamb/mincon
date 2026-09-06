//! Solver options and their defaults.
//!
//! # Defaults are the product
//!
//! `fmincon`'s reputation rests on defaults that work without tuning. Any
//! option we expose is a small failure; any option a user *has* to set to get
//! a solve is a large one. Every default below is chosen so that the
//! plug-and-play path — a Python callable, no gradients, no scaling, no
//! options — is the strongest configuration we can offer.
//!
//! Three defaults differ deliberately from `fmincon` and are the core of the
//! competitive thesis (see `docs/01_FMINCON_ANATOMY.md`):
//!
//! 1. **Scaling is on by default.** `fmincon`'s `ScaleProblem` defaults to
//!    `false`. IPOPT's `nlp_scaling_method` defaults to `gradient-based`. A
//!    large share of `fmincon`'s real-world failures are badly scaled models
//!    that IPOPT walks through. We take IPOPT's side.
//! 2. **The portfolio is on by default.** When more than one worker thread is
//!    available we race interior-point against SQP and return the first
//!    success. A single `fmincon` call cannot do this. This is the cheapest
//!    available multiplier on the headline robustness number.
//! 3. **Sparsity is detected, not assumed absent.** `fmincon` needs
//!    `JacobPattern`/`HessPattern` to be told. We probe.

use crate::EPS;

/// Console output verbosity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Display {
    /// Nothing.
    #[default]
    None,
    /// One line at the end.
    Final,
    /// One line per iteration.
    Iter,
    /// Per-iteration plus regularization / line-search internals. For
    /// debugging the solver, not the model.
    Debug,
}

/// Which algorithm to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Algorithm {
    /// Race every applicable algorithm on separate threads, return the first
    /// that converges (ties broken by objective, then by constraint violation).
    /// Falls back to [`Algorithm::InteriorPoint`] on a single thread.
    #[default]
    Auto,
    /// Primal-dual interior point with a filter line search.
    InteriorPoint,
    /// Sequential quadratic programming with an l1 merit function.
    Sqp,
    /// Sequential linear-quadratic programming (LP step + EQP step). Best when
    /// the active set is large and changes a lot.
    Slqp,
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
    /// Ruiz equilibration on the KKT matrix, refreshed on a schedule. Stronger
    /// on problems whose conditioning is structural rather than unit-driven,
    /// but costs a factorization's worth of work per refresh.
    Equilibration,
    /// User-supplied factors.
    User,
}

/// How second derivatives are obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HessianMode {
    /// Exact Hessian of the Lagrangian from the model.
    Exact,
    /// Limited-memory BFGS on the Lagrangian, with Powell damping.
    LimitedMemoryBfgs,
    /// Dense BFGS. `fmincon`'s interior-point default. `O(n^2)` memory; only
    /// sensible for small `n`.
    DenseBfgs,
    /// Finite differences of the gradient, with graph coloring when a Hessian
    /// pattern is known.
    FiniteDifference,
    /// Exact if the model provides it, else limited-memory BFGS.
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

/// How the KKT matrix is regularized and its inertia established.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RegularizationMode {
    /// Wächter–Biegler Algorithm IC: require inertia `(n, m, 0)` from the
    /// factorization, raising `delta_w` until it holds.
    Inertia,
    /// Chiang–Zavala: no inertia query; accept `delta_w` when the computed
    /// direction passes a curvature test. Works with any linear solver,
    /// including iterative ones, and empirically needs 56–69% fewer
    /// regularizations.
    InertiaFree,
    /// Use the inertia when the factorization reports it reliably (all pivots
    /// bounded away from zero, so Sylvester's law applies), and fall back to
    /// the curvature test otherwise. Strictly more information than either.
    #[default]
    Hybrid,
}

/// Which linear solver backs the KKT systems.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LinearSolverKind {
    /// Dense Bunch–Kaufman `LBL^T`. Fastest below roughly `n + m < 500`.
    DenseLblt,
    /// Sparse `LDL^T` with AMD ordering and dynamic regularization.
    SparseLdlt,
    /// Choose by density and dimension.
    #[default]
    Auto,
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
            // error and 1e-4 for constraint violation. We take IPOPT's
            // optimality target (it is scaled, so it is not as tight as it
            // looks) and fmincon's tighter feasibility target, because users
            // read constraint violation directly and 1e-4 looks sloppy.
            optimality: 1e-8,
            feasibility: 1e-6,
            complementarity: 1e-6,
            step: 1e-12,
            acceptable_optimality: 1e-6,
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
    /// Iteration cap. `fmincon` uses a flat 400; that is far too few for large
    /// problems and generous for tiny ones, so we scale with `n`.
    pub max_iterations: usize,
    /// Objective-evaluation cap, or `None` for unlimited.
    pub max_evaluations: Option<u64>,
    /// Wall-clock cap in seconds, or `None`.
    pub max_seconds: Option<f64>,
    /// Console verbosity.
    pub display: Display,
    /// Record a per-iteration trace in the report. Cheap; on by default because
    /// "why did it stop there" is the most common user question.
    pub record_trace: bool,

    /// Scaling strategy.
    pub scaling: ScalingMode,
    /// Gradient magnitude above which scaling kicks in. IPOPT's
    /// `nlp_scaling_max_gradient`.
    pub scaling_max_gradient: f64,

    /// Second-derivative strategy.
    pub hessian: HessianMode,
    /// L-BFGS history length.
    pub lbfgs_history: usize,

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

    /// KKT regularization strategy.
    pub regularization: RegularizationMode,
    /// Linear solver backend.
    pub linear_solver: LinearSolverKind,
    /// Steps of iterative refinement on each KKT solve. One is nearly free and
    /// buys a lot on ill-conditioned systems.
    pub refinement_steps: usize,

    /// Maximum second-order corrections per line search. IPOPT's `max_soc`.
    pub max_soc: usize,
    /// Enable the watchdog (accept a non-monotone step, verify next iteration).
    pub watchdog: bool,
    /// Enable the feasibility restoration phase.
    pub restoration: bool,

    /// Worker threads, or `None` for "all available".
    pub threads: Option<usize>,
    /// Compare analytic derivatives against finite differences at the start.
    /// `fmincon`'s `CheckGradients`.
    pub check_derivatives: bool,
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
            max_iterations: 3000,
            max_evaluations: None,
            max_seconds: None,
            display: Display::None,
            record_trace: true,

            scaling: ScalingMode::GradientBased,
            scaling_max_gradient: 100.0,

            hessian: HessianMode::Auto,
            lbfgs_history: 10,

            fd_type: FdType::Adaptive,
            fd_step: None,
            fd_respect_bounds: true,
            fd_coloring: true,
            detect_sparsity: true,

            honor_bounds: true,
            bound_relax_factor: 1e-10,

            barrier_update: BarrierUpdate::AdaptiveThenMonotone,
            mu_init: 0.1,
            tau_min: 0.99,

            regularization: RegularizationMode::Hybrid,
            linear_solver: LinearSolverKind::Auto,
            refinement_steps: 1,

            max_soc: 4,
            watchdog: true,
            restoration: true,

            threads: None,
            check_derivatives: false,
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
            max_iterations: 400,
            scaling: ScalingMode::None,
            hessian: HessianMode::DenseBfgs,
            fd_type: FdType::Forward,
            barrier_update: BarrierUpdate::Monotone,
            threads: Some(1),
            ..Self::default()
        }
    }

    /// Resolve `max_iterations` for a problem of size `n` when the user left
    /// the default in place.
    #[must_use]
    pub fn effective_max_iterations(&self, n: usize) -> usize {
        if self.max_iterations == 3000 {
            (400 + 10 * n).min(10_000)
        } else {
            self.max_iterations
        }
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
        if self.lbfgs_history == 0 && matches!(self.hessian, HessianMode::LimitedMemoryBfgs) {
            return Err("lbfgs_history must be positive for limited-memory BFGS".into());
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
            max_iterations: 42,
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
