//! The primal-dual interior-point solver.
//!
//! Follows Wächter and Biegler (2006) closely enough that the paper is usable
//! as a reference while reading this file; deviations are called out in
//! comments and collected in `docs/02_SPEC_INTERIOR_POINT.md`.
//!
//! # Internal formulation
//!
//! The canonical problem `c_L <= c(x) <= c_U`, `x_L <= x <= x_U` is converted
//! to equality form by giving every **inequality** a slack:
//!
//! ```text
//!   v = (x, s)                      nv = n + (number of inequalities)
//!   c_hat_i(v) = c_i(x) - c_L_i               for an equality i
//!   c_hat_i(v) = c_i(x) - s_{k(i)}            for an inequality i
//!   v_L <= v <= v_U   with the slack bounds taken from (c_L, c_U)
//! ```
//!
//! Equalities are deliberately **not** slacked: a slack pinned by
//! `c_L == c_U` would sit in a degenerate barrier interval and drive `Sigma`
//! to infinity. This is the same choice IPOPT makes and it matters.
//!
//! # What is implemented and what is not
//!
//! Implemented: barrier subproblems with the monotone (Fiacco–McCormick)
//! update, the primal-dual KKT system with inertia correction, the
//! Fletcher–Leyffer filter line search with the switching condition,
//! second-order corrections, fraction-to-boundary, bound-multiplier resets,
//! the scaled `E_mu` termination test, gradient-based scaling, and damped BFGS.
//! Soft and reduced-elastic feasibility restoration are implemented; see
//! `docs/12_RESTORATION_IMPLEMENTATION.md` for the variant and its evidence.
//!
//! Current implementation limitations:
//!
//! 1. **Limited-memory BFGS.** Dense BFGS caps usable `n` at a couple of
//!    thousand.
//! 2. **Adaptive barrier update.** Only `Monotone` is wired up.
//! 3. **The watchdog and qualified inertia-free fallback.**

use std::time::Instant;

/// Growth factor in `||x||` and in the objective, relative to the starting
/// point, at which a still-feasible sequence is declared divergent.
const DIVERGING_GROWTH_FACTOR: f64 = 1e10;

/// Consecutive below-tolerance steps before reporting [`ExitFlag::StepTolerance`].
const STALL_ITERATIONS: usize = 8;

use mincon_core::{
    Algorithm, EvalError, ExitFlag, HessianMode, IterationRecord, Nlp, Options, ScalingMode,
    Solution, SolveError, SolveReport, Sparsity, Timings,
};
use mincon_diff::Evaluator;
use mincon_linalg::Ordering;

use crate::bfgs::DenseBfgs;
use crate::filter::{Acceptance, Filter, FilterParams};
use crate::kkt::{transpose_with_map, CorrectionParams, KktFailure, KktSystem, ValueMap};

mod restoration;

/// Barrier-loop constants (Wächter–Biegler Section 2.2).
#[derive(Debug, Clone, Copy)]
pub struct BarrierParams {
    /// `kappa_epsilon`: solve each subproblem to this multiple of `mu`.
    pub kappa_eps: f64,
    /// `kappa_mu`: linear decrease factor.
    pub kappa_mu: f64,
    /// `theta_mu`: superlinear decrease exponent.
    pub theta_mu: f64,
    /// `kappa_Sigma`: bound-multiplier reset width.
    pub kappa_sigma: f64,
    /// `s_max` in the `E_mu` scaling factors.
    pub s_max: f64,
    /// `kappa_1`, `kappa_2`: how far inside their bounds the initial point is pushed.
    pub bound_push: f64,
    /// Fraction of a two-sided interval usable by the initial push.
    pub bound_frac: f64,
    /// `lambda_max`: cap on the least-squares multiplier initialization.
    pub lambda_init_max: f64,
    /// `kappa_soc`: abort second-order corrections when `theta` stops improving.
    pub kappa_soc: f64,
}

impl Default for BarrierParams {
    fn default() -> Self {
        Self {
            kappa_eps: 10.0,
            kappa_mu: 0.2,
            theta_mu: 1.5,
            kappa_sigma: 1e10,
            s_max: 100.0,
            bound_push: 1e-2,
            bound_frac: 1e-2,
            lambda_init_max: 1e3,
            kappa_soc: 0.99,
        }
    }
}

/// Which Hessian representation is in use.
enum Hess {
    Exact {
        upper: Sparsity,
        map: ValueMap,
        lower_values: Vec<f64>,
        upper_values: Vec<f64>,
    },
    Bfgs(Box<DenseBfgs>),
}

impl Hess {
    fn pattern(&self) -> &Sparsity {
        match self {
            Hess::Exact { upper, .. } => upper,
            Hess::Bfgs(b) => b.pattern(),
        }
    }
}

/// Run the interior-point method.
///
/// # Errors
/// [`SolveError`] for problems that cannot be started at all. Everything else
/// is reported through [`SolveReport::exit_flag`].
pub fn solve<P: Nlp + ?Sized>(nlp: &P, opts: &Options) -> Result<SolveReport, SolveError> {
    opts.validate().map_err(SolveError::InvalidOptions)?;
    mincon_core::validate(nlp).map_err(SolveError::InvalidProblem)?;
    Solver::new(nlp, opts)?.run()
}

struct Solver<'a, P: Nlp + ?Sized> {
    eval: Evaluator<'a, P>,
    opts: Options,
    barrier: BarrierParams,
    correction: CorrectionParams,

    n: usize,
    m: usize,
    nv: usize,

    /// `slack_of[i]` is the primal index of constraint `i`'s slack, if any.
    slack_of: Vec<Option<usize>>,
    /// For equalities, the value `c_i` must take.
    eq_target: Vec<f64>,

    v_l: Vec<f64>,
    v_u: Vec<f64>,
    has_l: Vec<bool>,
    has_u: Vec<bool>,

    /// Objective scaling factor.
    d_f: f64,
    /// Per-constraint scaling factors.
    d_c: Vec<f64>,

    kkt: KktSystem,
    hess: Hess,
    jac_map: ValueMap,
    jac_values: Vec<f64>,
    jac_t_values: Vec<f64>,

    notes: Vec<String>,
    start: Instant,
    /// Unscaled objective at the starting point, for the divergence diagnosis.
    f0_user: f64,
}

/// Everything evaluated at a point.
#[derive(Clone)]
struct Point {
    v: Vec<f64>,
    f: f64,
    c: Vec<f64>,
    c_hat: Vec<f64>,
    theta: f64,
    phi: f64,
}

impl<'a, P: Nlp + ?Sized> Solver<'a, P> {
    fn new(nlp: &'a P, opts: &Options) -> Result<Self, SolveError> {
        let dims = nlp.dims();
        let (n, m) = (dims.n, dims.m);
        let eval = Evaluator::new(nlp, opts);
        let mut notes: Vec<String> = eval.setup_notes().to_vec();

        let (xl, xu) = nlp.x_bounds();
        let (cl, cu) = nlp.c_bounds();

        let mut slack_of = vec![None; m];
        let mut v_l = xl.to_vec();
        let mut v_u = xu.to_vec();
        let mut eq_target = vec![0.0; m];
        let mut nv = n;
        for i in 0..m {
            if (cu[i] - cl[i]).abs() <= 0.0 {
                eq_target[i] = cl[i];
            } else {
                slack_of[i] = Some(nv);
                v_l.push(cl[i]);
                v_u.push(cu[i]);
                nv += 1;
            }
        }

        let has_l: Vec<bool> = v_l.iter().map(|v| !mincon_core::is_free(*v)).collect();
        let has_u: Vec<bool> = v_u.iter().map(|v| !mincon_core::is_free(*v)).collect();

        // Relax bounds so the strict interior is never empty.
        let relax = opts.bound_relax_factor;
        for j in 0..nv {
            if opts.honor_bounds && v_l[j] != v_u[j] {
                continue;
            }
            if has_l[j] {
                v_l[j] -= relax * v_l[j].abs().max(1.0);
            }
            if has_u[j] {
                v_u[j] += relax * v_u[j].abs().max(1.0);
            }
        }

        let hess = match (opts.hessian, eval.has_exact_hessian()) {
            (HessianMode::Exact, false) => {
                return Err(SolveError::InvalidOptions(
                    "HessianMode::Exact requested but the model does not provide one".into(),
                ))
            }
            (HessianMode::Exact | HessianMode::Auto, true) => {
                let lower = nlp
                    .hessian_structure()
                    .expect("has_exact_hessian implies a structure")
                    .clone();
                let (upper, map) = transpose_with_map(&lower);
                let nnz = lower.nnz();
                notes.push(format!(
                    "Using the model's exact Hessian of the Lagrangian ({nnz} stored entries)."
                ));
                Hess::Exact {
                    upper,
                    map,
                    lower_values: vec![0.0; nnz],
                    upper_values: vec![0.0; nnz],
                }
            }
            (HessianMode::DenseBfgs | HessianMode::Auto | HessianMode::LimitedMemoryBfgs, _) => {
                if matches!(opts.hessian, HessianMode::LimitedMemoryBfgs) {
                    notes.push(
                        "Limited-memory BFGS is not implemented yet; using dense BFGS.".into(),
                    );
                }
                Hess::Bfgs(Box::new(DenseBfgs::new(n)))
            }
            (HessianMode::FiniteDifference, _) => {
                notes.push(
                    "Finite-difference Hessians are not implemented yet; using dense BFGS.".into(),
                );
                Hess::Bfgs(Box::new(DenseBfgs::new(n)))
            }
        };

        // Transposed Jacobian, embedded into the nv-row primal space.
        let jac_pattern = eval.jacobian_pattern().clone();
        let (jt, jac_map) = transpose_with_map(&jac_pattern);
        let jt_nv = Sparsity::new(nv, m, jt.col_ptr().to_vec(), jt.row_idx().to_vec())
            .map_err(SolveError::Internal)?;

        let ordering = Ordering::default();
        let kkt = KktSystem::new(nv, n, m, hess.pattern(), &jt_nv, &slack_of, ordering)
            .map_err(SolveError::Internal)?;

        Ok(Self {
            n,
            m,
            nv,
            slack_of,
            eq_target,
            v_l,
            v_u,
            has_l,
            has_u,
            d_f: 1.0,
            d_c: vec![1.0; m],
            kkt,
            hess,
            jac_map,
            jac_values: vec![0.0; jac_pattern.nnz()],
            jac_t_values: vec![0.0; jt.nnz()],
            eval,
            opts: opts.clone(),
            barrier: BarrierParams::default(),
            correction: CorrectionParams::default(),
            notes,
            start: Instant::now(),
            f0_user: f64::NAN,
        })
    }

    // ---- evaluation helpers (all in the *scaled* problem) ----

    fn f_at(&self, x: &[f64]) -> Result<f64, EvalError> {
        Ok(self.d_f * self.eval.f(x)?)
    }

    fn c_at(&self, x: &[f64], out: &mut [f64]) -> Result<(), EvalError> {
        self.eval.c(x, out)?;
        for i in 0..self.m {
            out[i] *= self.d_c[i];
        }
        Ok(())
    }

    fn c_hat(&self, v: &[f64], c: &[f64], out: &mut [f64]) {
        for i in 0..self.m {
            out[i] = match self.slack_of[i] {
                Some(k) => c[i] - v[k],
                None => c[i] - self.d_c[i] * self.eq_target[i],
            };
        }
    }

    fn barrier_term(&self, v: &[f64], mu: f64) -> f64 {
        let mut acc = 0.0;
        for j in 0..self.nv {
            if self.has_l[j] {
                let d = v[j] - self.v_l[j];
                if d <= 0.0 {
                    return f64::INFINITY;
                }
                acc -= mu * d.ln();
            }
            if self.has_u[j] {
                let d = self.v_u[j] - v[j];
                if d <= 0.0 {
                    return f64::INFINITY;
                }
                acc -= mu * d.ln();
            }
        }
        acc
    }

    fn evaluate(&self, v: &[f64], mu: f64) -> Result<Point, EvalError> {
        let mut bounded = v.to_vec();
        self.project_variables(&mut bounded);
        let v = bounded.as_slice();
        let x = &v[..self.n];
        let f = self.f_at(x)?;
        let mut c = vec![0.0; self.m];
        self.c_at(x, &mut c)?;
        let mut c_hat = vec![0.0; self.m];
        self.c_hat(v, &c, &mut c_hat);
        let theta = c_hat.iter().map(|v| v.abs()).sum::<f64>();
        let phi = f + self.barrier_term(v, mu);
        Ok(Point {
            v: v.to_vec(),
            f,
            c,
            c_hat,
            theta,
            phi,
        })
    }

    /// Unscaled maximum constraint violation, which is what the user sees.
    fn user_violation(&self, v: &[f64], c_scaled: &[f64]) -> f64 {
        let (cl, cu) = self.eval.nlp().c_bounds();
        let (xl, xu) = self.eval.nlp().x_bounds();
        let mut worst = 0.0_f64;
        for i in 0..self.m {
            let ci = c_scaled[i] / self.d_c[i];
            worst = worst.max(cl[i] - ci).max(ci - cu[i]);
        }
        for j in 0..self.n {
            worst = worst.max(xl[j] - v[j]).max(v[j] - xu[j]);
        }
        worst.max(0.0)
    }

    // ---- setup ----

    fn project_variables(&self, v: &mut [f64]) {
        if self.opts.honor_bounds {
            let (xl, xu) = self.eval.nlp().x_bounds();
            for j in 0..self.n {
                v[j] = v[j].clamp(xl[j], xu[j]);
            }
        }
    }

    fn compute_scaling(&mut self, x: &[f64]) -> Result<(), SolveError> {
        if !matches!(self.opts.scaling, ScalingMode::GradientBased) {
            return Ok(());
        }
        let f0 = self.eval.f(x).map_err(SolveError::InitialPoint)?;
        let mut g = vec![0.0; self.n];
        if self.eval.grad(x, f0, &mut g).is_err() {
            self.notes
                .push("Scaling skipped: the gradient could not be evaluated at x0.".into());
            return Ok(());
        }
        let gmax = self.opts.scaling_max_gradient;
        let gnorm = g.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        self.d_f = if gnorm > gmax { gmax / gnorm } else { 1.0 };

        if self.m > 0 {
            let mut c0 = vec![0.0; self.m];
            if self.eval.c(x, &mut c0).is_ok() {
                let mut jv = vec![0.0; self.eval.jacobian_pattern().nnz()];
                if self.eval.jac(x, &c0, &mut jv).is_ok() {
                    let p = self.eval.jacobian_pattern();
                    let mut rowmax = vec![0.0_f64; self.m];
                    for j in 0..self.n {
                        for pos in p.col_ptr()[j]..p.col_ptr()[j + 1] {
                            let i = p.row_idx()[pos];
                            rowmax[i] = rowmax[i].max(jv[pos].abs());
                        }
                    }
                    for i in 0..self.m {
                        self.d_c[i] = if rowmax[i] > gmax {
                            gmax / rowmax[i]
                        } else {
                            1.0
                        };
                    }
                }
            }
        }
        let scaled_f = (self.d_f - 1.0).abs() > 1e-12;
        for i in 0..self.m {
            if let Some(k) = self.slack_of[i] {
                if self.has_l[k] {
                    self.v_l[k] *= self.d_c[i];
                }
                if self.has_u[k] {
                    self.v_u[k] *= self.d_c[i];
                }
            }
        }
        let scaled_c = self.d_c.iter().any(|d| (d - 1.0).abs() > 1e-12);
        if scaled_f || scaled_c {
            let cmin = self.d_c.iter().copied().fold(f64::INFINITY, f64::min);
            self.notes.push(format!(
                "Gradient-based scaling applied: objective factor {:.3e}, smallest constraint factor {:.3e}. \
                 (fmincon leaves ScaleProblem off by default, so this is a deliberate divergence.)",
                self.d_f,
                if self.m == 0 { 1.0 } else { cmin }
            ));
        }
        Ok(())
    }

    fn initial_point(&self) -> Result<Vec<f64>, SolveError> {
        let x0 = self.eval.nlp().x0();
        let mut v = vec![0.0; self.nv];
        let k1 = self.barrier.bound_push;
        let k2 = self.barrier.bound_frac;
        for j in 0..self.n {
            v[j] = push_inside(
                x0[j],
                self.v_l[j],
                self.v_u[j],
                self.has_l[j],
                self.has_u[j],
                k1,
                k2,
            );
        }
        self.project_variables(&mut v);
        if self.m > 0 {
            let mut c = vec![0.0; self.m];
            self.c_at(&v[..self.n], &mut c)
                .map_err(SolveError::InitialPoint)?;
            for i in 0..self.m {
                if let Some(k) = self.slack_of[i] {
                    v[k] = push_inside(
                        c[i],
                        self.v_l[k],
                        self.v_u[k],
                        self.has_l[k],
                        self.has_u[k],
                        k1,
                        k2,
                    );
                }
            }
        }
        Ok(v)
    }

    // ---- derivative assembly ----

    fn refresh_jacobian(&mut self, x: &[f64], c: &[f64]) -> Result<(), EvalError> {
        if self.m == 0 {
            return Ok(());
        }
        // The evaluator works in unscaled space; undo the row scaling of `c`
        // before handing it over, then scale the Jacobian rows.
        let mut c_unscaled = vec![0.0; self.m];
        for i in 0..self.m {
            c_unscaled[i] = c[i] / self.d_c[i];
        }
        self.eval.jac(x, &c_unscaled, &mut self.jac_values)?;
        let p = self.eval.jacobian_pattern();
        for j in 0..self.n {
            for pos in p.col_ptr()[j]..p.col_ptr()[j + 1] {
                let i = p.row_idx()[pos];
                self.jac_values[pos] *= self.d_c[i];
            }
        }
        let values = std::mem::take(&mut self.jac_values);
        self.jac_map.scatter(&values, &mut self.jac_t_values);
        self.jac_values = values;
        Ok(())
    }

    fn hessian_values(&mut self, x: &[f64], lambda: &[f64]) -> Result<&[f64], EvalError> {
        match &mut self.hess {
            Hess::Exact {
                map,
                lower_values,
                upper_values,
                ..
            } => {
                // The scaled Lagrangian is d_f * f + sum_i lambda_i d_c_i c_i,
                // so sigma = d_f and the multipliers carry the row scaling.
                let scaled: Vec<f64> = lambda.iter().zip(&self.d_c).map(|(l, d)| l * d).collect();
                self.eval.hess(x, self.d_f, &scaled, lower_values)?;
                map.scatter(lower_values, upper_values);
                Ok(upper_values)
            }
            Hess::Bfgs(b) => Ok(b.upper_values()),
        }
    }

    /// Gradient of the scaled objective with respect to `v` (zero in the slack
    /// block), plus the barrier terms, giving `grad phi_mu`.
    fn grad_phi(&self, grad_f: &[f64], v: &[f64], mu: f64, out: &mut [f64]) {
        out.fill(0.0);
        out[..self.n].copy_from_slice(grad_f);
        for j in 0..self.nv {
            if self.has_l[j] {
                out[j] -= mu / (v[j] - self.v_l[j]);
            }
            if self.has_u[j] {
                out[j] += mu / (self.v_u[j] - v[j]);
            }
        }
    }

    /// `out <- A * lambda` where `A` is the transposed Jacobian including the
    /// slack `-1` entries.
    fn a_times(&self, lambda: &[f64], out: &mut [f64]) {
        out.fill(0.0);
        let p = self.eval.jacobian_pattern();
        for j in 0..self.n {
            let mut acc = 0.0;
            for pos in p.col_ptr()[j]..p.col_ptr()[j + 1] {
                acc += self.jac_values[pos] * lambda[p.row_idx()[pos]];
            }
            out[j] = acc;
        }
        for i in 0..self.m {
            if let Some(k) = self.slack_of[i] {
                out[k] -= lambda[i];
            }
        }
    }

    // ---- the main loop ----

    #[allow(clippy::too_many_lines)]
    fn run(mut self) -> Result<SolveReport, SolveError> {
        let (n, m, nv) = (self.n, self.m, self.nv);
        let max_iter = self.opts.effective_max_iterations(n);

        let initial_unscaled = self.initial_point()?;
        self.compute_scaling(&initial_unscaled[..n])?;
        let mut v = self.initial_point()?;

        let mut mu = self.opts.mu_init;
        let mut point = self.evaluate(&v, mu).map_err(SolveError::InitialPoint)?;

        let mut lambda = vec![0.0; m];
        let mut z_l = vec![0.0; nv];
        let mut z_u = vec![0.0; nv];
        for j in 0..nv {
            z_l[j] = if self.has_l[j] { 1.0 } else { 0.0 };
            z_u[j] = if self.has_u[j] { 1.0 } else { 0.0 };
        }

        let mut grad_f = vec![0.0; n];
        self.eval
            .grad(&v[..n], point.f / self.d_f, &mut grad_f)
            .map_err(SolveError::InitialPoint)?;
        for g in &mut grad_f {
            *g *= self.d_f;
        }
        self.refresh_jacobian(&v[..n], &point.c)
            .map_err(SolveError::InitialPoint)?;

        let mut filter = Filter::new(point.theta, FilterParams::default());
        let mut trace: Vec<IterationRecord> = Vec::new();
        let mut sigma = vec![0.0; nv];
        let mut grad_phi = vec![0.0; nv];
        let mut a_lambda = vec![0.0; nv];
        let mut d_v = vec![0.0; nv];
        let mut d_lambda = vec![0.0; m];
        let mut d_zl = vec![0.0; nv];
        let mut d_zu = vec![0.0; nv];
        let mut prev_x = v[..n].to_vec();
        let mut prev_lag_grad = vec![0.0; n];

        let x0_norm = v[..n].iter().fold(0.0_f64, |a, x| a.max(x.abs()));
        let f0_value = point.f / self.d_f;
        self.f0_user = f0_value;
        let mut stalled = 0usize;
        let mut exit = ExitFlag::MaxReached;
        let mut acceptable_streak = 0usize;
        let mut adaptive_mode = matches!(
            self.opts.barrier_update,
            mincon_core::BarrierUpdate::Adaptive | mincon_core::BarrierUpdate::AdaptiveThenMonotone
        );
        let mut adaptive_stall = 0usize;
        let mut best_e0 = f64::INFINITY;
        // Error-aware termination for finite-difference derivatives: the
        // stationarity target cannot be below the derivative error, so once the
        // iterate is close the error is estimated on the four steepest coordinates
        // (a few extra evaluations, at most every ten iterations) and the target is raised to it, capped at
        // the acceptable tolerance; forward differences that are too inaccurate
        // escalate to central before that cap is used.
        let approximate = self.opts.fd_error_aware
            && (self.eval.gradient_is_approximate() || self.eval.jacobian_is_approximate());
        let mut tol_eff = self.opts.tol.optimality;
        let mut error_checked_at: Option<usize> = None;
        let mut near_convergence_iters = 0usize;
        let mut iterations = 0usize;
        let mut restoration_work = 0usize;
        let mut last_delta_w = 0.0;
        let mut last_delta_c = 0.0;
        let mut last_soc = 0usize;
        let mut last_alpha = 0.0;
        let mut last_step_norm = 0.0;

        for iter in 0..=max_iter {
            iterations = iter + restoration_work;

            // --- termination ---
            let (e0, e_mu, compl) = self.optimality(&point, &grad_f, &lambda, &z_l, &z_u, mu);
            let violation = self.user_violation(&point.v, &point.c);
            if self.opts.record_trace {
                trace.push(IterationRecord {
                    iter: iterations,
                    f_count: mincon_core::EvalCounters::get(&self.eval.counters().f),
                    f: point.f / self.d_f,
                    constraint_violation: violation,
                    optimality: e0,
                    step_norm: last_step_norm,
                    alpha: last_alpha,
                    mu,
                    delta_w: last_delta_w,
                    delta_c: last_delta_c,
                    in_restoration: false,
                    soc_count: last_soc,
                });
            }

            if e0 <= 100.0 * self.opts.tol.optimality
                && violation <= self.opts.tol.acceptable_feasibility
            {
                near_convergence_iters += 1;
            } else {
                near_convergence_iters = 0;
            }
            // Only worth its evaluations when the iterate has lingered near the
            // end: a solve that finishes in the next step or two pays nothing.
            if approximate
                && near_convergence_iters >= 3
                && e0 > self.opts.tol.optimality
                && error_checked_at.is_none_or(|k| iter >= k + 10)
            {
                error_checked_at = Some(iter);
                let mut c_unscaled = point.c.clone();
                for i in 0..m {
                    c_unscaled[i] /= self.d_c[i];
                }
                let g_unscaled: Vec<f64> = grad_f.iter().map(|g| g / self.d_f).collect();
                let mut j_unscaled = self.jac_values.clone();
                {
                    let p = self.eval.jacobian_pattern();
                    for j in 0..n {
                        for pos in p.col_ptr()[j]..p.col_ptr()[j + 1] {
                            j_unscaled[pos] /= self.d_c[p.row_idx()[pos]];
                        }
                    }
                }
                if let Ok((g_err, j_err)) = self.eval.derivative_error_estimate(
                    &point.v[..n],
                    point.f / self.d_f,
                    &c_unscaled,
                    &g_unscaled,
                    &j_unscaled,
                    4,
                ) {
                    let d_c_max = self.d_c.iter().copied().fold(0.0_f64, f64::max);
                    let err_scaled = (g_err * self.d_f).max(j_err * d_c_max);
                    if err_scaled > self.opts.tol.acceptable_optimality
                        && !self.eval.uses_central_differences()
                        && matches!(self.opts.fd_type, mincon_core::FdType::Adaptive)
                    {
                        self.eval.escalate_accuracy();
                        self.notes.push(format!(
                            "Estimated forward-difference derivative error {err_scaled:.2e} (scaled) exceeds the \
                             acceptable optimality tolerance; switched to central differences at iteration {iter}."
                        ));
                        error_checked_at = Some(iter.saturating_sub(5)); // allow a re-estimate soon
                        self.eval
                            .grad(&point.v[..n], point.f / self.d_f, &mut grad_f)
                            .map_err(|e| {
                                SolveError::Internal(format!("gradient after escalation: {e}"))
                            })?;
                        for g in &mut grad_f {
                            *g *= self.d_f;
                        }
                        self.refresh_jacobian(&point.v[..n], &point.c)
                            .map_err(|e| {
                                SolveError::Internal(format!("Jacobian after escalation: {e}"))
                            })?;
                        continue;
                    }
                    tol_eff = err_scaled.clamp(
                        self.opts.tol.optimality,
                        self.opts.tol.acceptable_optimality,
                    );
                }
            }
            if e0 <= tol_eff
                && violation <= self.opts.tol.feasibility
                && compl <= self.opts.tol.complementarity
                && self.stationarity_inf(&grad_f, &lambda, &z_l, &z_u) <= tol_eff
            {
                if tol_eff > self.opts.tol.optimality {
                    self.notes.push(format!(
                        "Converged to the accuracy of the finite-difference derivatives: scaled KKT error \
                         {e0:.2e} is below the estimated derivative error {tol_eff:.2e}, which is above the \
                         requested optimality tolerance {:.1e}. Supply analytic derivatives for a tighter certificate.",
                        self.opts.tol.optimality
                    ));
                }
                exit = ExitFlag::Optimal;
                break;
            }
            if e0 <= self.opts.tol.acceptable_optimality
                && violation <= self.opts.tol.acceptable_feasibility
            {
                acceptable_streak += 1;
                if acceptable_streak >= self.opts.tol.acceptable_iterations {
                    exit = ExitFlag::Acceptable;
                    break;
                }
            } else {
                acceptable_streak = 0;
            }
            if point.f / self.d_f <= self.opts.tol.objective_limit
                && violation <= self.opts.tol.feasibility
            {
                exit = ExitFlag::Unbounded;
                break;
            }
            // Diverging iterates. Certifying unboundedness is undecidable for a
            // local method, so this is a heuristic - but it must be a heuristic
            // that actually fires. IPOPT's `diverging_iterates_tol` is an
            // absolute 1e20 on ||x||, which an iterate growing linearly will
            // never reach inside any sane budget: measured on TORTURE_UNBOUNDED,
            // 420 iterations get to 3e13 and the solver reports MaxReached,
            // telling the user nothing. So the test is *relative growth* from
            // the starting point, on all three of: feasibility maintained,
            // ||x|| exploded, and the objective fell by the same order.
            if violation <= self.opts.tol.feasibility.max(1e-6) {
                let xnorm = point.v[..n].iter().fold(0.0_f64, |a, v| a.max(v.abs()));
                let f_now = point.f / self.d_f;
                let grew = xnorm > DIVERGING_GROWTH_FACTOR * x0_norm.max(1.0);
                let fell = f_now < f0_value - DIVERGING_GROWTH_FACTOR * f0_value.abs().max(1.0);
                if grew && fell {
                    self.notes.push(format!(
                        "Iterates appear to be diverging: ||x||_inf grew from {x0_norm:.3e} to \
                         {xnorm:.3e} and the objective fell from {f0_value:.3e} to {f_now:.3e}, \
                         all while feasible. Reported as unbounded below. This is a growth \
                         heuristic, not a proof - a problem with a very distant minimum can \
                         trigger it, so check the returned point before trusting the flag."
                    ));
                    exit = ExitFlag::Unbounded;
                    break;
                }
            }
            // Step tolerance. fmincon's exit flag 2, and the honest answer on a
            // degenerate problem where the multipliers are unbounded so the KKT
            // residual can never come down: the iterate has stopped moving and
            // saying so beats burning the budget and reporting MaxReached.
            if iter > 0
                && last_step_norm < self.opts.tol.step
                && violation <= self.opts.tol.feasibility
            {
                stalled += 1;
                if stalled >= STALL_ITERATIONS {
                    self.notes.push(format!(
                        "Steps have been below the step tolerance ({:.1e}) for {STALL_ITERATIONS} \
                         consecutive iterations while feasible. The iterate has stopped moving; \
                         first-order optimality is {e0:.3e}, which may be unattainable if the \
                         constraint qualification fails at this point.",
                        self.opts.tol.step
                    ));
                    exit = ExitFlag::StepTolerance;
                    break;
                }
            } else {
                stalled = 0;
            }
            if let Some(limit) = self.opts.max_evaluations {
                if mincon_core::EvalCounters::get(&self.eval.counters().f) >= limit {
                    exit = ExitFlag::MaxReached;
                    break;
                }
            }
            if let Some(limit) = self.opts.max_seconds {
                if self.start.elapsed().as_secs_f64() >= limit {
                    exit = ExitFlag::MaxReached;
                    break;
                }
            }
            if iterations >= max_iter {
                exit = ExitFlag::MaxReached;
                break;
            }

            // --- barrier parameter ---
            let mu_min = self.opts.tol.optimality / 10.0;
            if adaptive_mode {
                // LOQO-style centrality rule (Vanderbei–Shanno; IPOPT's `mu_oracle loqo`):
                // mu = sigma * (average complementarity), sigma from how far the least
                // centred pair is from the average. Well-centred iterates drive mu down
                // fast; badly centred ones hold it. Bounded below by the termination
                // floor and above so a single iteration never re-inflates the barrier
                // beyond its starting value.
                let (avg, xi) = self.centrality(&point, &z_l, &z_u);
                if avg > 0.0 {
                    let sigma = 0.1 * (0.05 * (1.0 - xi) / xi.max(1e-12)).min(2.0).powi(3);
                    let mu_new = (sigma * avg).clamp(mu_min, self.opts.mu_init.max(mu));
                    if (mu_new - mu).abs() > 1e-3 * mu {
                        mu = mu_new;
                        filter = Filter::new(point.theta, FilterParams::default());
                        point.phi = point.f + self.barrier_term(&point.v, mu);
                    }
                }
                // Fall back to the monotone schedule when the adaptive iterates stop
                // making progress on the KKT error (a bounded, transparent safeguard).
                if e0 < best_e0 * 0.9 {
                    best_e0 = e0;
                    adaptive_stall = 0;
                } else {
                    adaptive_stall += 1;
                    if adaptive_stall >= 5 {
                        adaptive_mode = false;
                        self.notes.push(format!(
                            "Adaptive barrier update made no KKT progress for {adaptive_stall} iterations; \
                             switched to the monotone schedule at iteration {iter}."
                        ));
                    }
                }
            } else if e_mu <= self.barrier.kappa_eps * mu && mu > mu_min {
                mu = mu_min.max((self.barrier.kappa_mu * mu).min(mu.powf(self.barrier.theta_mu)));
                filter = Filter::new(point.theta, FilterParams::default());
                point.phi = point.f + self.barrier_term(&point.v, mu);
            }
            let tau = self.opts.tau_min.max(1.0 - mu);

            // --- barrier diagonal ---
            for j in 0..nv {
                let mut s = 0.0;
                if self.has_l[j] {
                    s += z_l[j] / (point.v[j] - self.v_l[j]).max(1e-300);
                }
                if self.has_u[j] {
                    s += z_u[j] / (self.v_u[j] - point.v[j]).max(1e-300);
                }
                sigma[j] = s;
            }

            // --- factor and solve ---
            let hess_ptr: Vec<f64> = {
                let vals = self
                    .hessian_values(&point.v[..n], &lambda)
                    .map_err(|e| SolveError::Internal(format!("hessian evaluation: {e}")))?;
                vals.to_vec()
            };
            let jt = self.jac_t_values.clone();
            let outcome = match self.kkt.factor_with_correction(
                &hess_ptr,
                &jt,
                &sigma,
                mu,
                self.opts.regularization,
                &self.correction,
            ) {
                Ok(o) => o,
                Err(KktFailure::RegularizationExhausted { .. }) => {
                    let saved_jac = self.jac_values.clone();
                    let saved_jac_t = self.jac_t_values.clone();
                    let recovered = self.restore(
                        &point,
                        &grad_f,
                        &lambda,
                        &z_l,
                        &z_u,
                        mu,
                        &mut filter,
                        max_iter.saturating_sub(iterations + 1),
                        iterations,
                        &mut trace,
                        false,
                    );
                    match recovered {
                        Ok(r) => {
                            restoration_work += r.steps;
                            iterations += r.steps;
                            point = r.point;
                            grad_f = r.grad;
                            lambda = r.lambda;
                            z_l = r.zl;
                            z_u = r.zu;
                            v.copy_from_slice(&point.v);
                            prev_x.copy_from_slice(&v[..n]);
                            if let Some(flag) = r.exit {
                                exit = flag;
                                break;
                            }
                            if let Hess::Bfgs(b) = &mut self.hess {
                                b.reset(1.0);
                            }
                            last_step_norm = f64::INFINITY;
                            continue;
                        }
                        Err(e) => {
                            self.jac_values = saved_jac;
                            self.jac_t_values = saved_jac_t;
                            exit = if matches!(e, EvalError::UserAbort) {
                                ExitFlag::StoppedByUser
                            } else {
                                ExitFlag::NumericalFailure
                            };
                            self.notes
                                .push(format!("Restoration evaluation failed: {e}"));
                            break;
                        }
                    }
                }
                Err(KktFailure::Linear(e)) => return Err(SolveError::LinearAlgebra(e.to_string())),
            };
            last_delta_w = outcome.delta_w;
            last_delta_c = outcome.delta_c;

            self.grad_phi(&grad_f, &point.v, mu, &mut grad_phi);
            self.a_times(&lambda, &mut a_lambda);
            {
                let rhs = self.kkt.rhs_mut();
                for j in 0..nv {
                    rhs[j] = -(grad_phi[j] + a_lambda[j]);
                }
                for i in 0..m {
                    rhs[nv + i] = -point.c_hat[i];
                }
            }
            self.kkt
                .solve_scratch(self.opts.refinement_steps)
                .map_err(|e| SolveError::LinearAlgebra(e.to_string()))?;
            d_v.copy_from_slice(&self.kkt.sol()[..nv]);
            d_lambda.copy_from_slice(&self.kkt.sol()[nv..]);

            for j in 0..nv {
                d_zl[j] = if self.has_l[j] {
                    let d = (point.v[j] - self.v_l[j]).max(1e-300);
                    mu / d - z_l[j] - (z_l[j] / d) * d_v[j]
                } else {
                    0.0
                };
                d_zu[j] = if self.has_u[j] {
                    let d = (self.v_u[j] - point.v[j]).max(1e-300);
                    mu / d - z_u[j] + (z_u[j] / d) * d_v[j]
                } else {
                    0.0
                };
            }

            // --- fraction to boundary ---
            let alpha_max = self.fraction_to_boundary(&point.v, &d_v, tau);
            let alpha_z = fraction_to_boundary_dual(&z_l, &d_zl, tau)
                .min(fraction_to_boundary_dual(&z_u, &d_zu, tau));

            let dphi: f64 = grad_phi.iter().zip(&d_v).map(|(g, d)| g * d).sum();
            let alpha_min = filter.min_step_size(dphi, point.theta);

            // --- filter line search with second-order corrections ---
            let mut alpha = alpha_max;
            let mut accepted: Option<(Point, Acceptance, f64, usize)> = None;

            for trial in 0..filter.params().max_backtracks {
                if alpha < alpha_min {
                    break;
                }
                let mut cand = vec![0.0; nv];
                for j in 0..nv {
                    cand[j] = point.v[j] + alpha * d_v[j];
                }
                match self.evaluate(&cand, mu) {
                    Ok(trial_point) => {
                        let acc = filter.evaluate(
                            point.theta,
                            point.phi,
                            trial_point.theta,
                            trial_point.phi,
                            alpha,
                            dphi,
                        );
                        if acc != Acceptance::Rejected {
                            accepted = Some((trial_point, acc, alpha, 0));
                            break;
                        }
                        // Second-order correction, only on the first trial and
                        // only when the step made feasibility worse - the
                        // Maratos-effect signature.
                        if trial == 0
                            && self.opts.max_soc > 0
                            && m > 0
                            && trial_point.theta >= point.theta
                        {
                            if let Some((soc_point, soc_acc, soc_alpha, used)) = self
                                .second_order_correction(
                                    &point, &d_v, alpha, mu, tau, dphi, &filter,
                                )
                            {
                                accepted = Some((soc_point, soc_acc, soc_alpha, used));
                                break;
                            }
                        }
                    }
                    Err(EvalError::UserAbort) => {
                        exit = ExitFlag::StoppedByUser;
                        break;
                    }
                    Err(_) => { /* model failure: retreat, exactly as fmincon's sqp does */ }
                }
                alpha *= filter.params().backtrack;
            }

            if exit == ExitFlag::StoppedByUser {
                break;
            }

            let Some((new_point, acceptance, alpha_taken, soc)) = accepted else {
                if self.eval.uses_central_differences() {
                    let saved_jac = self.jac_values.clone();
                    let saved_jac_t = self.jac_t_values.clone();
                    let recovered = self.restore(
                        &point,
                        &grad_f,
                        &lambda,
                        &z_l,
                        &z_u,
                        mu,
                        &mut filter,
                        max_iter.saturating_sub(iterations + 1),
                        iterations,
                        &mut trace,
                        true,
                    );
                    match recovered {
                        Ok(r) => {
                            restoration_work += r.steps;
                            iterations += r.steps;
                            point = r.point;
                            grad_f = r.grad;
                            lambda = r.lambda;
                            z_l = r.zl;
                            z_u = r.zu;
                            v.copy_from_slice(&point.v);
                            prev_x.copy_from_slice(&v[..n]);
                            if let Some(flag) = r.exit {
                                exit = flag;
                                break;
                            }
                            if let Hess::Bfgs(b) = &mut self.hess {
                                b.reset(1.0);
                            }
                            last_step_norm = f64::INFINITY;
                            continue;
                        }
                        Err(e) => {
                            self.jac_values = saved_jac;
                            self.jac_t_values = saved_jac_t;
                            exit = if matches!(e, EvalError::UserAbort) {
                                ExitFlag::StoppedByUser
                            } else {
                                ExitFlag::NumericalFailure
                            };
                            self.notes
                                .push(format!("Restoration evaluation failed: {e}"));
                            break;
                        }
                    }
                }
                if adaptive_mode
                    && matches!(
                        self.opts.barrier_update,
                        mincon_core::BarrierUpdate::AdaptiveThenMonotone
                    )
                {
                    // A rejected line search under the adaptive schedule: give the
                    // monotone schedule a turn before sharpening the derivatives.
                    adaptive_mode = false;
                    self.notes.push(format!(
                        "Line search failed under the adaptive barrier update; switched to the \
                         monotone schedule at iteration {iter}."
                    ));
                    continue;
                }
                // Before giving up, sharpen the derivatives: a failing line
                // search is most often finite-difference noise, not geometry.
                self.eval.escalate_accuracy();
                self.notes.push(
                    "Line search failed; switching to central finite differences and retrying."
                        .into(),
                );
                self.eval
                    .grad(&point.v[..n], point.f / self.d_f, &mut grad_f)
                    .map_err(|e| SolveError::Internal(format!("gradient after escalation: {e}")))?;
                for g in &mut grad_f {
                    *g *= self.d_f;
                }
                self.refresh_jacobian(&point.v[..n], &point.c)
                    .map_err(|e| SolveError::Internal(format!("Jacobian after escalation: {e}")))?;
                continue;
            };

            last_alpha = alpha_taken;
            last_soc = soc;
            last_step_norm = point
                .v
                .iter()
                .zip(&new_point.v)
                .fold(0.0_f64, |a, (old, new)| a.max((new - old).abs()));

            if acceptance == Acceptance::SufficientDecrease {
                filter.augment(point.theta, point.phi);
            }

            // --- accept the step ---
            for i in 0..m {
                lambda[i] += alpha_z * d_lambda[i];
            }
            for j in 0..nv {
                z_l[j] = (z_l[j] + alpha_z * d_zl[j]).max(0.0);
                z_u[j] = (z_u[j] + alpha_z * d_zu[j]).max(0.0);
            }
            self.reset_bound_multipliers(&new_point.v, mu, &mut z_l, &mut z_u);

            // --- quasi-Newton update ---
            if let Hess::Bfgs(_) = &self.hess {
                self.a_times(&lambda, &mut a_lambda);
                for j in 0..n {
                    prev_lag_grad[j] = grad_f[j] + a_lambda[j] - z_l[j] + z_u[j];
                }
            }

            v.copy_from_slice(&new_point.v);
            let f_unscaled = new_point.f / self.d_f;
            self.eval
                .grad(&v[..n], f_unscaled, &mut grad_f)
                .map_err(|e| SolveError::Internal(format!("gradient evaluation: {e}")))?;
            for g in &mut grad_f {
                *g *= self.d_f;
            }
            self.refresh_jacobian(&v[..n], &new_point.c)
                .map_err(|e| SolveError::Internal(format!("jacobian evaluation: {e}")))?;

            if let Hess::Bfgs(b) = &mut self.hess {
                let mut a_new = vec![0.0; nv];
                {
                    let p = self.eval.jacobian_pattern();
                    for j in 0..n {
                        let mut acc = 0.0;
                        for pos in p.col_ptr()[j]..p.col_ptr()[j + 1] {
                            acc += self.jac_values[pos] * lambda[p.row_idx()[pos]];
                        }
                        a_new[j] = acc;
                    }
                }
                let s: Vec<f64> = (0..n).map(|j| v[j] - prev_x[j]).collect();
                let y: Vec<f64> = (0..n)
                    .map(|j| (grad_f[j] + a_new[j] - z_l[j] + z_u[j]) - prev_lag_grad[j])
                    .collect();
                // Guarded initial scaling: only when the unit matrix has already
                // forced the line search to cut the very first step hard.
                let first_step_cut = self.opts.bfgs_guarded_scaling
                    && b.updates() == 0
                    && b.skipped() == 0
                    && alpha_taken < 0.125;
                b.update_guarded(&s, &y, first_step_cut);
            }
            prev_x.copy_from_slice(&v[..n]);
            point = new_point;
        }

        Ok(self.finish(exit, iterations, point, &grad_f, lambda, z_l, z_u, trace))
    }

    #[allow(clippy::too_many_arguments)] // the SOC needs the whole line-search state
    fn second_order_correction(
        &mut self,
        point: &Point,
        d_v: &[f64],
        alpha: f64,
        mu: f64,
        tau: f64,
        dphi: f64,
        filter: &Filter,
    ) -> Option<(Point, Acceptance, f64, usize)> {
        let (m, nv) = (self.m, self.nv);
        let mut theta_old = f64::INFINITY;
        let mut cand = vec![0.0; nv];
        for j in 0..nv {
            cand[j] = point.v[j] + alpha * d_v[j];
        }
        let trial = self.evaluate(&cand, mu).ok()?;
        let mut c_soc: Vec<f64> = (0..m)
            .map(|i| alpha * point.c_hat[i] + trial.c_hat[i])
            .collect();

        for k in 1..=self.opts.max_soc {
            // Reuse the existing factorization: the matrix has not changed,
            // only the right-hand side. This is what makes SOC nearly free.
            {
                let rhs = self.kkt.rhs_mut();
                for i in 0..m {
                    rhs[nv + i] = -c_soc[i];
                }
            }
            if self.kkt.solve_scratch(self.opts.refinement_steps).is_err() {
                return None;
            }
            let d_cor: Vec<f64> = self.kkt.sol()[..nv].to_vec();
            let alpha_soc = self.fraction_to_boundary(&point.v, &d_cor, tau);
            for j in 0..nv {
                cand[j] = point.v[j] + alpha_soc * d_cor[j];
            }
            let Ok(soc_point) = self.evaluate(&cand, mu) else {
                return None;
            };
            let acc = filter.evaluate(
                point.theta,
                point.phi,
                soc_point.theta,
                soc_point.phi,
                alpha_soc,
                dphi,
            );
            if acc != Acceptance::Rejected {
                return Some((soc_point, acc, alpha_soc, k));
            }
            if soc_point.theta > self.barrier.kappa_soc * theta_old {
                return None;
            }
            theta_old = soc_point.theta;
            for i in 0..m {
                c_soc[i] = alpha_soc * c_soc[i] + soc_point.c_hat[i];
            }
        }
        None
    }

    /// Average complementarity `d_j z_j` over the bounded coordinates and the
    /// centrality measure `xi = min(d_j z_j) / average` in `(0, 1]`.
    fn centrality(&self, point: &Point, z_l: &[f64], z_u: &[f64]) -> (f64, f64) {
        let mut sum = 0.0;
        let mut min = f64::INFINITY;
        let mut count = 0usize;
        for j in 0..self.nv {
            if self.has_l[j] {
                let p = (point.v[j] - self.v_l[j]) * z_l[j];
                sum += p;
                min = min.min(p);
                count += 1;
            }
            if self.has_u[j] {
                let p = (self.v_u[j] - point.v[j]) * z_u[j];
                sum += p;
                min = min.min(p);
                count += 1;
            }
        }
        if count == 0 || sum <= 0.0 {
            return (0.0, 1.0);
        }
        let avg = sum / count as f64;
        (avg, (min / avg).clamp(0.0, 1.0))
    }

    fn fraction_to_boundary(&self, v: &[f64], d: &[f64], tau: f64) -> f64 {
        let mut alpha = 1.0_f64;
        for j in 0..self.nv {
            if d[j] < 0.0 && self.has_l[j] {
                let room = v[j] - self.v_l[j];
                if room > 0.0 {
                    alpha = alpha.min(-tau * room / d[j]);
                }
            } else if d[j] > 0.0 && self.has_u[j] {
                let room = self.v_u[j] - v[j];
                if room > 0.0 {
                    alpha = alpha.min(tau * room / d[j]);
                }
            }
        }
        alpha.clamp(0.0, 1.0)
    }

    fn reset_bound_multipliers(&self, v: &[f64], mu: f64, z_l: &mut [f64], z_u: &mut [f64]) {
        let k = self.barrier.kappa_sigma;
        for j in 0..self.nv {
            if self.has_l[j] {
                let d = (v[j] - self.v_l[j]).max(1e-300);
                z_l[j] = z_l[j].min(k * mu / d).max(mu / (k * d));
            }
            if self.has_u[j] {
                let d = (self.v_u[j] - v[j]).max(1e-300);
                z_u[j] = z_u[j].min(k * mu / d).max(mu / (k * d));
            }
        }
    }

    /// Returns `(E_0, E_mu, worst complementarity)`, all in the scaled problem
    /// except the complementarity residual which is reported unscaled.
    fn optimality(
        &self,
        point: &Point,
        grad_f: &[f64],
        lambda: &[f64],
        z_l: &[f64],
        z_u: &[f64],
        mu: f64,
    ) -> (f64, f64, f64) {
        let (n, m, nv) = (self.n, self.m, self.nv);
        let mut a_lambda = vec![0.0; nv];
        self.a_times(lambda, &mut a_lambda);

        let mut dual = 0.0_f64;
        for j in 0..nv {
            let gf = if j < n { grad_f[j] } else { 0.0 };
            dual = dual.max((gf + a_lambda[j] - z_l[j] + z_u[j]).abs());
        }

        let primal = point.c_hat.iter().fold(0.0_f64, |a, c| a.max(c.abs()));

        let mut compl0 = 0.0_f64;
        let mut compl_mu = 0.0_f64;
        let mut nbounds = 0usize;
        let mut z_sum = 0.0;
        for j in 0..nv {
            if self.has_l[j] {
                let d = point.v[j] - self.v_l[j];
                compl0 = compl0.max((d * z_l[j]).abs());
                compl_mu = compl_mu.max((d * z_l[j] - mu).abs());
                nbounds += 1;
                z_sum += z_l[j].abs();
            }
            if self.has_u[j] {
                let d = self.v_u[j] - point.v[j];
                compl0 = compl0.max((d * z_u[j]).abs());
                compl_mu = compl_mu.max((d * z_u[j] - mu).abs());
                nbounds += 1;
                z_sum += z_u[j].abs();
            }
        }

        let s_max = self.barrier.s_max;
        let lam_sum: f64 = lambda.iter().map(|v| v.abs()).sum();
        let denom_d = (m + nbounds).max(1) as f64;
        let s_d = (s_max.max((lam_sum + z_sum) / denom_d)) / s_max;
        let s_c = (s_max.max(z_sum / nbounds.max(1) as f64)) / s_max;

        let e0 = (dual / s_d).max(primal).max(compl0 / s_c);
        let e_mu = (dual / s_d).max(primal).max(compl_mu / s_c);
        (e0, e_mu, compl0)
    }

    fn stationarity_inf(&self, grad: &[f64], lambda: &[f64], zl: &[f64], zu: &[f64]) -> f64 {
        let mut al = vec![0.0; self.nv];
        self.a_times(lambda, &mut al);
        let mut norm = 0.0_f64;
        for j in 0..self.nv {
            let r = if j < self.n { grad[j] } else { 0.0 };
            let r = r + al[j] - zl[j] + zu[j];
            if !r.is_finite() {
                return f64::INFINITY;
            }
            norm = norm.max(r.abs());
        }
        norm
    }

    /// True partial derivatives of `f` and every constraint row with respect to the fixed
    /// variables `fixed`, at the user's point `x`. Analytic when the model provides them,
    /// otherwise central probes off the pinned value (which the bound-honouring finite
    /// differences deliberately never take).
    fn fixed_variable_derivatives(
        &self,
        x: &[f64],
        c_user: &[f64],
        fixed: &[usize],
    ) -> Result<(Vec<f64>, Vec<Vec<f64>>), EvalError> {
        let nlp = self.eval.nlp();
        let caps = nlp.capabilities();
        let (n, m) = (self.n, self.m);
        let mut gf = vec![0.0; fixed.len()];
        let mut jcols = vec![vec![0.0; m]; fixed.len()];
        if caps.gradient {
            let mut g = vec![0.0; n];
            nlp.gradient(x, &mut g)?;
            for (k, &j) in fixed.iter().enumerate() {
                gf[k] = g[j];
            }
        }
        if m > 0 && caps.jacobian {
            if let Some(pat) = nlp.jacobian_structure() {
                let mut vals = vec![0.0; pat.nnz()];
                nlp.jacobian(x, &mut vals)?;
                for (k, &j) in fixed.iter().enumerate() {
                    for pos in pat.col_ptr()[j]..pat.col_ptr()[j + 1] {
                        jcols[k][pat.row_idx()[pos]] = vals[pos];
                    }
                }
            }
        }
        let need_f = !caps.gradient;
        let need_c = m > 0 && !(caps.jacobian && nlp.jacobian_structure().is_some());
        if need_f || need_c {
            let f0 = self.eval.f(x)?;
            let mut xp = x.to_vec();
            let mut cp = vec![0.0; m];
            let mut cm = vec![0.0; m];
            for (k, &j) in fixed.iter().enumerate() {
                let h = mincon_core::EPS.powf(1.0 / 3.0) * x[j].abs().max(1.0);
                xp[j] = x[j] + h;
                let hp = xp[j] - x[j];
                let fp = if need_f { self.eval.f(&xp)? } else { 0.0 };
                if need_c {
                    self.eval.c(&xp, &mut cp)?;
                }
                xp[j] = x[j] - h;
                let hm = x[j] - xp[j];
                let fm = if need_f { self.eval.f(&xp)? } else { 0.0 };
                if need_c {
                    self.eval.c(&xp, &mut cm)?;
                }
                xp[j] = x[j];
                if need_f {
                    gf[k] = (fp - fm) / (hp + hm);
                    if !gf[k].is_finite() {
                        return Err(EvalError::NonFinite(None));
                    }
                }
                if need_c {
                    for i in 0..m {
                        jcols[k][i] = (cp[i] - cm[i]) / (hp + hm);
                        if !jcols[k][i].is_finite() {
                            return Err(EvalError::NonFinite(None));
                        }
                    }
                }
            }
            let _ = (f0, c_user);
        }
        Ok((gf, jcols))
    }

    #[allow(clippy::too_many_arguments)]
    fn finish(
        self,
        mut exit: ExitFlag,
        iterations: usize,
        point: Point,
        grad_f: &[f64],
        lambda: Vec<f64>,
        z_l: Vec<f64>,
        z_u: Vec<f64>,
        trace: Vec<IterationRecord>,
    ) -> SolveReport {
        let n = self.n;
        let counters = self.eval.counters();
        // The accepted point already has a valid gradient and Jacobian.
        // Reporting must not call the model again after an abort or budget exit.
        let f_unscaled = point.f / self.d_f;
        let (optimality, _, compl) = self.optimality(&point, grad_f, &lambda, &z_l, &z_u, 0.0);
        let violation = self.user_violation(&point.v, &point.c);

        // Unscale the multipliers back into the user's problem.
        let lambda_user: Vec<f64> = lambda
            .iter()
            .zip(&self.d_c)
            .map(|(l, d)| l * d / self.d_f)
            .collect();
        let mut z_l_user: Vec<f64> = z_l[..n].iter().map(|z| z / self.d_f).collect();
        let mut z_u_user: Vec<f64> = z_u[..n].iter().map(|z| z / self.d_f).collect();
        let c_user: Vec<f64> = point.c.iter().zip(&self.d_c).map(|(c, d)| c / d).collect();

        // A fixed variable (x_L == x_U) lives in a relaxed interval of width ~2e-10, where
        // both barrier multipliers are huge and nearly equal; their difference carries no
        // information, and the finite-difference engine gives pinned variables a zero
        // derivative, so the solver never saw the true partial derivative. Reconstruct the
        // net bound multiplier from stationarity in the user's problem,
        // grad f + J^T lambda - z_L + z_U = 0, using analytic derivatives when the model has
        // them and otherwise two probes off the pinned value (only when a usable point is
        // being returned, so an aborted or budget-limited solve never calls the model again).
        let mut fixed_notes: Vec<String> = Vec::new();
        {
            let (xl, xu) = self.eval.nlp().x_bounds();
            let fixed: Vec<usize> = (0..n).filter(|&j| xl[j] == xu[j]).collect();
            if !fixed.is_empty() && exit.returned_usable_point() {
                let x_user = &point.v[..n];
                match self.fixed_variable_derivatives(x_user, &c_user, &fixed) {
                    Ok((gf, jcols)) => {
                        for (k, &j) in fixed.iter().enumerate() {
                            let jt_lambda: f64 =
                                jcols[k].iter().zip(&lambda_user).map(|(a, l)| a * l).sum();
                            let net = gf[k] + jt_lambda; // = z_L - z_U
                            if net >= 0.0 {
                                z_l_user[j] = net;
                                z_u_user[j] = 0.0;
                            } else {
                                z_l_user[j] = 0.0;
                                z_u_user[j] = -net;
                            }
                        }
                    }
                    Err(e) => {
                        for &j in &fixed {
                            z_l_user[j] = 0.0;
                            z_u_user[j] = 0.0;
                        }
                        fixed_notes.push(format!(
                            "Bound multipliers of the {} fixed variable(s) are reported as zero: the model \
                             derivative at the pinned value could not be obtained ({e}).",
                            fixed.len()
                        ));
                    }
                }
            }
        }

        let timings = Timings {
            total: self.start.elapsed(),
            model: self.eval.model_time(),
            ..Timings::default()
        };

        let mut notes = self.notes.clone();
        notes.extend(fixed_notes);
        {
            let x0n = self
                .eval
                .nlp()
                .x0()
                .iter()
                .fold(0.0_f64, |a, v| a.max(v.abs()));
            let xn = point.v[..n].iter().fold(0.0_f64, |a, v| a.max(v.abs()));
            let f0 = self.f0_user;
            let fell_far = f_unscaled < f0 - 1e6 * f0.abs().max(1.0);
            if xn > 1e6 * x0n.max(1.0) && exit.returned_usable_point() && fell_far {
                // The first-order conditions can be satisfied to tolerance at an
                // enormous point because the multipliers shrink with the iterates
                // (min -x1 s.t. x2 = x1^2 is the textbook case). A usable exit that
                // far away with an objective that has dropped by a factor 1e6 is an
                // unbounded diagnosis, not a solution.
                notes.push(format!(
                    "Reported as unbounded below: the iterate ran to ||x||_inf = {xn:.3e} (from \
                     {x0n:.3e}) while the objective fell from {f0:.3e} to {f_unscaled:.3e} and the \
                     first-order conditions still held to tolerance. This is a growth heuristic; \
                     add bounds if a finite solution is expected."
                ));
                exit = ExitFlag::Unbounded;
            } else if xn > 1e6 * x0n.max(1.0) && exit.returned_usable_point() {
                notes.push(format!(
                    "The returned point is very far from the start (||x||_inf = {xn:.3e} versus \
                     {x0n:.3e} at x0). The first-order conditions hold there, but an objective that \
                     flattens out along some direction has stationary points at infinity; check \
                     whether a bounded solution is what you wanted, and add bounds if so."
                ));
            }
        }
        if let Hess::Bfgs(b) = &self.hess {
            if b.skipped() > 0 {
                notes.push(format!(
                    "{} of {} BFGS updates were skipped for bad curvature; an exact Hessian would help this model.",
                    b.skipped(),
                    b.skipped() + b.updates()
                ));
            }
        }

        SolveReport {
            solution: Solution {
                x: point.v[..n].to_vec(),
                f: f_unscaled,
                c: c_user,
                lambda: lambda_user,
                z_l: z_l_user,
                z_u: z_u_user,
            },
            exit_flag: exit,
            algorithm: Algorithm::InteriorPoint,
            iterations,
            f_evals: mincon_core::EvalCounters::get(&counters.f),
            g_evals: mincon_core::EvalCounters::get(&counters.g),
            c_evals: mincon_core::EvalCounters::get(&counters.c),
            j_evals: mincon_core::EvalCounters::get(&counters.j),
            h_evals: mincon_core::EvalCounters::get(&counters.h),
            failed_evals: mincon_core::EvalCounters::get(&counters.failed),
            optimality,
            constraint_violation: violation,
            complementarity: compl,
            trace,
            timings,
            notes,
        }
    }
}

fn push_inside(x: f64, lo: f64, hi: f64, has_l: bool, has_u: bool, k1: f64, k2: f64) -> f64 {
    match (has_l, has_u) {
        (false, false) => x,
        (true, false) => x.max(lo + k1 * lo.abs().max(1.0)),
        (false, true) => x.min(hi - k1 * hi.abs().max(1.0)),
        (true, true) => {
            let width = hi - lo;
            let pl = (k1 * lo.abs().max(1.0)).min(k2 * width);
            let pu = (k1 * hi.abs().max(1.0)).min(k2 * width);
            x.clamp(lo + pl, hi - pu)
        }
    }
}

fn fraction_to_boundary_dual(z: &[f64], dz: &[f64], tau: f64) -> f64 {
    let mut alpha = 1.0_f64;
    for (zi, di) in z.iter().zip(dz) {
        if *di < 0.0 && *zi > 0.0 {
            alpha = alpha.min(-tau * zi / di);
        }
    }
    alpha.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_inside_respects_every_bound_configuration() {
        assert_eq!(push_inside(5.0, 0.0, 0.0, false, false, 0.01, 0.01), 5.0);
        let a = push_inside(0.0, 0.0, f64::INFINITY, true, false, 0.01, 0.01);
        assert!(a > 0.0);
        let b = push_inside(1.0, f64::NEG_INFINITY, 1.0, false, true, 0.01, 0.01);
        assert!(b < 1.0);
        let c = push_inside(0.0, 0.0, 1.0, true, true, 0.01, 0.01);
        assert!(c > 0.0 && c < 1.0);
        // A point already comfortably inside is left alone.
        assert_eq!(push_inside(0.5, 0.0, 1.0, true, true, 0.01, 0.01), 0.5);
    }

    #[test]
    fn dual_fraction_to_boundary_never_lets_z_go_negative() {
        let z = [1.0, 2.0];
        let dz = [-2.0, 0.5];
        let a = fraction_to_boundary_dual(&z, &dz, 0.99);
        for i in 0..2 {
            assert!(z[i] + a * dz[i] >= 0.0);
        }
        assert!(a > 0.0 && a <= 1.0);
    }

    /// A fixed variable's bound multipliers must satisfy stationarity in the user's problem.
    #[test]
    fn fixed_variable_multipliers_satisfy_stationarity() {
        use mincon_core::{Capabilities, NlpDims};
        // min (x0 - 1)^2 + x1^2   s.t.  x0 + x1 = 5,  x1 fixed at 3  ->  x0 = 2, lambda = -2,
        // stationarity at x1: 2*x1 + lambda - zL + zU = 0 -> zL - zU = 6 - 2 = 4.
        struct P {
            lb: Vec<f64>,
            ub: Vec<f64>,
            cl: Vec<f64>,
            cu: Vec<f64>,
            x0: Vec<f64>,
        }
        impl Nlp for P {
            fn dims(&self) -> NlpDims {
                NlpDims { n: 2, m: 1 }
            }
            fn x_bounds(&self) -> (&[f64], &[f64]) {
                (&self.lb, &self.ub)
            }
            fn c_bounds(&self) -> (&[f64], &[f64]) {
                (&self.cl, &self.cu)
            }
            fn x0(&self) -> &[f64] {
                &self.x0
            }
            fn capabilities(&self) -> Capabilities {
                Capabilities::none()
            }
            fn objective(&self, x: &[f64]) -> Result<f64, EvalError> {
                Ok((x[0] - 1.0).powi(2) + x[1] * x[1])
            }
            fn constraints(&self, x: &[f64], out: &mut [f64]) -> Result<(), EvalError> {
                out[0] = x[0] + x[1];
                Ok(())
            }
        }
        let p = P {
            lb: vec![-1e20, 3.0],
            ub: vec![1e20, 3.0],
            cl: vec![5.0],
            cu: vec![5.0],
            x0: vec![0.0, 3.0],
        };
        let r = solve(&p, &Options::default()).unwrap();
        assert!((r.solution.x[0] - 2.0).abs() < 1e-6, "{:?}", r.solution.x);
        let lam = r.solution.lambda[0];
        let g = [2.0 * (r.solution.x[0] - 1.0), 2.0 * r.solution.x[1]];
        for j in 0..2 {
            let res = g[j] + lam - r.solution.z_l[j] + r.solution.z_u[j];
            assert!(
                res.abs() < 1e-6,
                "stationarity residual {res} at variable {j}: z_l={} z_u={}",
                r.solution.z_l[j],
                r.solution.z_u[j]
            );
        }
        assert!(
            r.solution.z_l[1].min(r.solution.z_u[1]) == 0.0,
            "one side of a fixed variable must carry zero"
        );
    }

    /// min -x1 s.t. x2 = x1^2: the first-order conditions hold to tolerance at any
    /// far-away point because the multiplier shrinks with x1. That must be reported
    /// as unbounded, never as optimal.
    #[test]
    fn unbounded_along_a_parabola_is_not_reported_optimal() {
        use mincon_core::{Capabilities, NlpDims};
        struct P {
            lb: Vec<f64>,
            ub: Vec<f64>,
            cl: Vec<f64>,
            cu: Vec<f64>,
            x0: Vec<f64>,
        }
        impl Nlp for P {
            fn dims(&self) -> NlpDims {
                NlpDims { n: 2, m: 1 }
            }
            fn x_bounds(&self) -> (&[f64], &[f64]) {
                (&self.lb, &self.ub)
            }
            fn c_bounds(&self) -> (&[f64], &[f64]) {
                (&self.cl, &self.cu)
            }
            fn x0(&self) -> &[f64] {
                &self.x0
            }
            fn capabilities(&self) -> Capabilities {
                Capabilities::none()
            }
            fn objective(&self, x: &[f64]) -> Result<f64, EvalError> {
                Ok(-x[0])
            }
            fn constraints(&self, x: &[f64], out: &mut [f64]) -> Result<(), EvalError> {
                out[0] = x[1] - x[0] * x[0];
                Ok(())
            }
        }
        let p = P {
            lb: vec![-1e20; 2],
            ub: vec![1e20; 2],
            cl: vec![0.0],
            cu: vec![0.0],
            x0: vec![0.0, 0.0],
        };
        let r = solve(&p, &Options::default()).unwrap();
        assert!(
            !r.exit_flag.is_success(),
            "reported {:?} at x = {:?}",
            r.exit_flag,
            r.solution.x
        );
        assert!(
            matches!(r.exit_flag, ExitFlag::Unbounded | ExitFlag::MaxReached),
            "{:?}",
            r.exit_flag
        );
    }
}
