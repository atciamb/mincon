//! Sequential quadratic programming with an ℓ1 merit function.
//!
//! The mathematics and every design decision are in `docs/20_SQP_MATHEMATICS.md`;
//! section numbers below refer to it. In outline, each iteration:
//!
//! 1. solves the QP (1.3) — damped-BFGS or shifted exact Hessian, linearized
//!    constraints, bounds kept as bounds — with the dual active-set method of
//!    `mincon-qp`, warm-hinted by the previous active set;
//! 2. if the linearization is inconsistent, solves the elastic QP (3.1) instead,
//!    whose objective is the model of the ℓ1 merit function `f + rho * v`;
//! 3. raises the penalty `rho` by the model rule (5.4) and the multiplier floor;
//! 4. backtracks on the merit function from the unit step, trying a second-order
//!    correction (5.5) when the unit step is rejected, treating non-finite model
//!    values as rejected trials;
//! 5. updates the quasi-Newton matrix with the damped BFGS rule (4.3), using the
//!    new multipliers on both sides of the curvature pair.
//!
//! Termination uses the same scaled KKT error, tolerances, error-aware
//! finite-difference floor and unscaled-stationarity guard as the interior-point
//! member, so `Optimal` means the same thing from either.

use std::time::Instant;

use mincon_core::{
    Algorithm, EvalCounters, EvalError, ExitFlag, HessianMode, IterationRecord, Nlp, Options,
    ScalingMode, Solution, SolveError, SolveReport, Timings,
};
use mincon_diff::Evaluator;
use mincon_ip::DenseBfgs;
use mincon_qp::{solve_dense, Constraint, DenseQp, QpError, QpOptions};

/// Run SQP on `nlp`.
///
/// # Errors
/// [`SolveError`] for problems that cannot be started at all; everything else
/// is reported through [`SolveReport::exit_flag`].
pub fn solve<P: Nlp + ?Sized>(nlp: &P, opts: &Options) -> Result<SolveReport, SolveError> {
    opts.validate().map_err(SolveError::InvalidOptions)?;
    mincon_core::validate(nlp).map_err(SolveError::InvalidProblem)?;
    Sqp::new(nlp, opts)?.run()
}

/// Hessian representation.
enum Hess {
    Bfgs(Box<DenseBfgs>),
    /// Exact Hessian of the Lagrangian, dense row-major, refreshed each iteration.
    Exact {
        lower_values: Vec<f64>,
        dense: Vec<f64>,
        shift: f64,
    },
}

struct Sqp<'a, P: Nlp + ?Sized> {
    eval: Evaluator<'a, P>,
    opts: Options,
    n: usize,
    m: usize,
    /// Objective and row scale factors (gradient-based scaling; §6).
    d_f: f64,
    d_c: Vec<f64>,
    /// Scaled constraint bounds.
    cl: Vec<f64>,
    cu: Vec<f64>,
    /// Variable bounds (never scaled).
    xl: Vec<f64>,
    xu: Vec<f64>,
    hess: Hess,
    /// Jacobian values in the evaluator's sparsity order (unscaled).
    jac_values: Vec<f64>,
    /// Relative step bound on the QP: `|d_j| <= step_bound * max(1, |x_j|)`.
    /// Infinite until a line search has had to cut a step hard, then adapted
    /// like a trust region (SNOPT's major step limit is the fixed-radius form).
    step_bound: f64,
    notes: Vec<String>,
    start: Instant,
}

/// Everything known at an accepted iterate, in scaled units.
struct Point {
    x: Vec<f64>,
    f: f64,
    c: Vec<f64>,
    g: Vec<f64>,
    /// Dense scaled Jacobian, `m * n` row-major.
    j: Vec<f64>,
}

/// Outcome of one QP solve as the SQP loop uses it.
struct Direction {
    d: Vec<f64>,
    lambda: Vec<f64>,
    z_l: Vec<f64>,
    z_u: Vec<f64>,
    active: Vec<Constraint>,
    /// Linearized ℓ1 violation after the step (zero when the plain QP was consistent).
    v_lin: f64,
    /// `g^T d + 1/2 d^T B d`.
    model_obj: f64,
    elastic: bool,
}

const SOC_MAX: usize = 4;
const MAX_BACKTRACKS: usize = 25;
const ARMIJO: f64 = 1e-4;
const PENALTY_FRACTION: f64 = 0.1; // tau in (5.3)

impl<'a, P: Nlp + ?Sized> Sqp<'a, P> {
    fn new(nlp: &'a P, opts: &Options) -> Result<Self, SolveError> {
        let dims = nlp.dims();
        let (n, m) = (dims.n, dims.m);
        let eval = Evaluator::new(nlp, opts);
        let mut notes = eval.setup_notes().to_vec();
        let (xl, xu) = nlp.x_bounds();
        let (cl, cu) = nlp.c_bounds();
        let hess = match (opts.hessian, eval.has_exact_hessian()) {
            (HessianMode::Exact, false) => {
                return Err(SolveError::InvalidOptions(
                    "HessianMode::Exact requested but the model does not provide one".into(),
                ))
            }
            (HessianMode::Exact | HessianMode::Auto, true) => {
                let nnz = nlp
                    .hessian_structure()
                    .expect("has_exact_hessian implies a structure")
                    .nnz();
                notes.push(format!(
                    "SQP: using the model's exact Hessian of the Lagrangian ({nnz} stored entries), shifted when indefinite."
                ));
                Hess::Exact {
                    lower_values: vec![0.0; nnz],
                    dense: vec![0.0; n * n],
                    shift: 0.0,
                }
            }
            _ => Hess::Bfgs(Box::new(DenseBfgs::with_curvature_rescale(
                n,
                opts.bfgs_curvature_rescale,
            ))),
        };
        let jac_nnz = eval.jacobian_pattern().nnz();
        Ok(Self {
            eval,
            opts: opts.clone(),
            n,
            m,
            d_f: 1.0,
            d_c: vec![1.0; m],
            cl: cl.to_vec(),
            cu: cu.to_vec(),
            xl: xl.to_vec(),
            xu: xu.to_vec(),
            hess,
            jac_values: vec![0.0; jac_nnz],
            step_bound: f64::INFINITY,
            notes,
            start: Instant::now(),
        })
    }

    // ---- evaluation (scaled) -------------------------------------------------

    fn project(&self, x: &mut [f64]) {
        for j in 0..self.n {
            if x[j] < self.xl[j] {
                x[j] = self.xl[j];
            }
            if x[j] > self.xu[j] {
                x[j] = self.xu[j];
            }
        }
    }

    /// Objective and constraints at `x`, scaled. Non-finite values are errors.
    fn values(&self, x: &[f64]) -> Result<(f64, Vec<f64>), EvalError> {
        let f = self.eval.f(x)? * self.d_f;
        let mut c = vec![0.0; self.m];
        self.eval.c(x, &mut c)?;
        for i in 0..self.m {
            c[i] *= self.d_c[i];
        }
        Ok((f, c))
    }

    /// Gradient and dense scaled Jacobian at `x` (given unscaled-consistent
    /// `f`, `c` in scaled units for the finite differences' base values).
    fn derivatives(
        &mut self,
        x: &[f64],
        f: f64,
        c: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>), EvalError> {
        let (n, m) = (self.n, self.m);
        let mut g = vec![0.0; n];
        self.eval.grad(x, f / self.d_f, &mut g)?;
        for gj in &mut g {
            *gj *= self.d_f;
        }
        let mut j = vec![0.0; m * n];
        if m > 0 {
            let c_user: Vec<f64> = (0..m).map(|i| c[i] / self.d_c[i]).collect();
            self.eval.jac(x, &c_user, &mut self.jac_values)?;
            let p = self.eval.jacobian_pattern();
            for col in 0..n {
                for pos in p.col_ptr()[col]..p.col_ptr()[col + 1] {
                    let row = p.row_idx()[pos];
                    j[row * n + col] = self.jac_values[pos] * self.d_c[row];
                }
            }
        }
        Ok((g, j))
    }

    fn point(&mut self, x: Vec<f64>) -> Result<Point, EvalError> {
        let (f, c) = self.values(&x)?;
        let (g, j) = self.derivatives(&x, f, &c)?;
        Ok(Point { x, f, c, g, j })
    }

    /// ℓ1 violation of the scaled rows (bounds are always satisfied).
    fn violation_l1(&self, c: &[f64]) -> f64 {
        (0..self.m)
            .map(|i| (self.cl[i] - c[i]).max(0.0) + (c[i] - self.cu[i]).max(0.0))
            .sum()
    }

    /// Maximum violation in the user's units.
    fn user_violation(&self, x: &[f64], c: &[f64]) -> f64 {
        let (cl, cu) = self.eval.nlp().c_bounds();
        let mut worst = 0.0_f64;
        for i in 0..self.m {
            let ci = c[i] / self.d_c[i];
            worst = worst.max(cl[i] - ci).max(ci - cu[i]);
        }
        for j in 0..self.n {
            worst = worst.max(self.xl[j] - x[j]).max(x[j] - self.xu[j]);
        }
        worst.max(0.0)
    }

    fn merit(&self, f: f64, c: &[f64], rho: f64) -> f64 {
        f + rho * self.violation_l1(c)
    }

    // ---- scaling (§6; same rule as mincon-ip) ---------------------------------

    fn compute_scaling(&mut self, p: &Point) {
        if !matches!(self.opts.scaling, ScalingMode::GradientBased) {
            return;
        }
        let gmax = self.opts.scaling_max_gradient;
        let gnorm = p.g.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        self.d_f = if gnorm > gmax { gmax / gnorm } else { 1.0 };
        for i in 0..self.m {
            let rowmax = (0..self.n).fold(0.0_f64, |a, j| a.max(p.j[i * self.n + j].abs()));
            self.d_c[i] = if rowmax > gmax { gmax / rowmax } else { 1.0 };
        }
        for i in 0..self.m {
            self.cl[i] *= self.d_c[i];
            self.cu[i] *= self.d_c[i];
        }
        let scaled_f = (self.d_f - 1.0).abs() > 1e-12;
        let cmin = self.d_c.iter().copied().fold(1.0_f64, f64::min);
        if scaled_f || cmin < 1.0 - 1e-12 {
            self.notes.push(format!(
                "Gradient-based scaling applied: objective factor {:.3e}, smallest constraint factor {cmin:.3e}.",
                self.d_f
            ));
        }
    }

    /// Apply a change of the objective scale by the factor `k` to everything
    /// that carries it (D9 rescale; the Lagrangian and its curvature model
    /// scale with the objective).
    fn rescale_objective(
        &mut self,
        k: f64,
        p: &mut Point,
        lambda: &mut [f64],
        z_l: &mut [f64],
        z_u: &mut [f64],
        rho: &mut f64,
    ) {
        self.d_f *= k;
        p.f *= k;
        for g in &mut p.g {
            *g *= k;
        }
        for l in lambda.iter_mut() {
            *l *= k;
        }
        for z in z_l.iter_mut().chain(z_u.iter_mut()) {
            *z *= k;
        }
        *rho *= k;
        if let Hess::Bfgs(b) = &mut self.hess {
            b.scale(k);
        }
    }

    // ---- KKT error (§6) --------------------------------------------------------

    /// `(e0, complementarity, stationarity_rel)` in the interior-point member's
    /// scaling: the dual residual is divided by `s_d = max(s_max, mean |multiplier|) / s_max`.
    fn kkt(&self, p: &Point, lambda: &[f64], z_l: &[f64], z_u: &[f64]) -> (f64, f64, f64) {
        let (n, m) = (self.n, self.m);
        let mut jt_lam = vec![0.0; n];
        for i in 0..m {
            let li = lambda[i];
            if li != 0.0 {
                for j in 0..n {
                    jt_lam[j] += p.j[i * n + j] * li;
                }
            }
        }
        let (mut dual, mut g_norm, mut a_norm, mut z_norm) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
        for j in 0..n {
            dual = dual.max((p.g[j] + jt_lam[j] - z_l[j] + z_u[j]).abs());
            g_norm = g_norm.max(p.g[j].abs());
            a_norm = a_norm.max(jt_lam[j].abs());
            z_norm = z_norm.max(z_l[j].abs()).max(z_u[j].abs());
        }
        let primal = (0..m).fold(0.0_f64, |a, i| {
            a.max(self.cl[i] - p.c[i]).max(p.c[i] - self.cu[i])
        });
        let mut compl = 0.0_f64;
        let mut nb = 0usize;
        let mut zsum = 0.0;
        for j in 0..n {
            if self.xl[j].is_finite() && self.xl[j] > -mincon_core::INF_BOUND {
                compl = compl.max(((p.x[j] - self.xl[j]) * z_l[j]).abs());
                nb += 1;
                zsum += z_l[j].abs();
            }
            if self.xu[j].is_finite() && self.xu[j] < mincon_core::INF_BOUND {
                compl = compl.max(((self.xu[j] - p.x[j]) * z_u[j]).abs());
                nb += 1;
                zsum += z_u[j].abs();
            }
        }
        for i in 0..m {
            if self.cl[i] != self.cu[i] {
                let slack = if lambda[i] >= 0.0 {
                    (self.cu[i] - p.c[i]).max(0.0)
                } else {
                    (p.c[i] - self.cl[i]).max(0.0)
                };
                let slack = if slack.is_finite() { slack } else { 0.0 };
                compl = compl.max((lambda[i] * slack).abs());
            }
        }
        let s_max = 100.0_f64;
        let lam_sum: f64 = lambda.iter().map(|v| v.abs()).sum::<f64>() + zsum;
        let denom = (m + nb).max(1) as f64;
        let s_d = s_max.max(lam_sum / denom) / s_max;
        let e0 = (dual / s_d).max(primal).max(compl / s_d);
        let rel = dual / (self.d_f + g_norm + a_norm + z_norm);
        (e0, compl, rel)
    }

    // ---- the Hessian for the QP --------------------------------------------------

    fn hessian_dense(&mut self, p: &Point, lambda: &[f64]) -> Result<Vec<f64>, EvalError> {
        let n = self.n;
        match &mut self.hess {
            Hess::Bfgs(b) => Ok(b.dense()),
            Hess::Exact {
                lower_values,
                dense,
                shift,
            } => {
                // The model's Hessian is of sigma * f + sum lambda_i c_i in user
                // units; the scaled Lagrangian is d_f f + sum (lambda_i d_c_i) c_i.
                let lam_user: Vec<f64> = (0..self.m).map(|i| lambda[i] * self.d_c[i]).collect();
                self.eval.hess(&p.x, self.d_f, &lam_user, lower_values)?;
                let pat = self
                    .eval
                    .nlp()
                    .hessian_structure()
                    .expect("exact Hessian has a structure");
                dense.fill(0.0);
                for col in 0..n {
                    for pos in pat.col_ptr()[col]..pat.col_ptr()[col + 1] {
                        let row = pat.row_idx()[pos];
                        dense[row * n + col] = lower_values[pos];
                        dense[col * n + row] = lower_values[pos];
                    }
                }
                let mut h = dense.clone();
                for j in 0..n {
                    h[j * n + j] += *shift;
                }
                Ok(h)
            }
        }
    }

    // ---- the QP step (§2, §3) ------------------------------------------------------

    /// Solve the plain QP; on inconsistency the elastic QP with penalty `rho`.
    /// `h` is shifted and the solve retried while the QP reports an indefinite Hessian.
    fn direction(
        &mut self,
        p: &Point,
        h: &mut Vec<f64>,
        rho: f64,
        hint: &[Constraint],
    ) -> Result<Direction, String> {
        let (n, m) = (self.n, self.m);
        let a_l: Vec<f64> = (0..m).map(|i| self.cl[i] - p.c[i]).collect();
        let a_u: Vec<f64> = (0..m).map(|i| self.cu[i] - p.c[i]).collect();
        let x_l: Vec<f64> = (0..n)
            .map(|j| (self.xl[j] - p.x[j]).max(-self.step_bound * p.x[j].abs().max(1.0)))
            .collect();
        let x_u: Vec<f64> = (0..n)
            .map(|j| (self.xu[j] - p.x[j]).min(self.step_bound * p.x[j].abs().max(1.0)))
            .collect();
        let qopts = QpOptions::default();
        let mut shifts = 0;
        loop {
            let qp = DenseQp {
                n,
                m,
                h,
                g: &p.g,
                a: &p.j,
                a_l: &a_l,
                a_u: &a_u,
                x_l: &x_l,
                x_u: &x_u,
            };
            match solve_dense(&qp, hint, &qopts) {
                Ok(mut s) => {
                    let model_obj = s.objective;
                    self.drop_step_bound_multipliers(p, &mut s.z_l, &mut s.z_u);
                    return Ok(Direction {
                        d: s.x,
                        lambda: s.lambda,
                        z_l: s.z_l,
                        z_u: s.z_u,
                        active: s.active,
                        v_lin: 0.0,
                        model_obj,
                        elastic: false,
                    });
                }
                Err(QpError::NotPositiveDefinite(_)) => {
                    shifts += 1;
                    if shifts > 30 {
                        return Err("QP Hessian could not be made positive definite".into());
                    }
                    let scale = (0..n)
                        .fold(0.0_f64, |a, j| a.max(h[j * n + j].abs()))
                        .max(1.0);
                    let delta = 1e-4 * scale * 10f64.powi(shifts - 1);
                    for j in 0..n {
                        h[j * n + j] += delta;
                    }
                    if let Hess::Exact { shift, .. } = &mut self.hess {
                        *shift += delta;
                    }
                }
                Err(QpError::Infeasible(_)) => break,
                Err(QpError::IterationLimit(k)) => {
                    return Err(format!("QP active-set iteration limit ({k})"));
                }
                Err(QpError::InvalidData(msg)) => return Err(format!("QP data: {msg}")),
            }
        }
        // ---- elastic QP (3.1) with regularized slacks (§3.3) ----
        let ne = n + 2 * m;
        let hscale = (0..n)
            .fold(0.0_f64, |a, j| a.max(h[j * n + j].abs()))
            .max(1.0);
        let eps = 1e-6 * hscale;
        let mut he = vec![0.0; ne * ne];
        for i in 0..n {
            for j in 0..n {
                he[i * ne + j] = h[i * n + j];
            }
        }
        for k in n..ne {
            he[k * ne + k] = eps;
        }
        let mut ge = vec![0.0; ne];
        ge[..n].copy_from_slice(&p.g);
        for k in n..ne {
            ge[k] = rho;
        }
        // rows: J d + p - q  in [a_l, a_u]
        let mut ae = vec![0.0; m * ne];
        for i in 0..m {
            for j in 0..n {
                ae[i * ne + j] = p.j[i * n + j];
            }
            ae[i * ne + n + i] = 1.0;
            ae[i * ne + n + m + i] = -1.0;
        }
        let mut xle = vec![0.0; ne];
        let mut xue = vec![f64::INFINITY; ne];
        xle[..n].copy_from_slice(&x_l);
        xue[..n].copy_from_slice(&x_u);
        let qp = DenseQp {
            n: ne,
            m,
            h: &he,
            g: &ge,
            a: &ae,
            a_l: &a_l,
            a_u: &a_u,
            x_l: &xle,
            x_u: &xue,
        };
        match solve_dense(&qp, &[], &qopts) {
            Ok(mut s) => {
                let d = s.x[..n].to_vec();
                let v_lin: f64 = s.x[n..].iter().sum();
                let (mut zl, mut zu) = (s.z_l[..n].to_vec(), s.z_u[..n].to_vec());
                self.drop_step_bound_multipliers(p, &mut zl, &mut zu);
                s.z_l[..n].copy_from_slice(&zl);
                s.z_u[..n].copy_from_slice(&zu);
                // model objective without the penalty part
                let mut hd = vec![0.0; n];
                for i in 0..n {
                    hd[i] = (0..n).map(|j| h[i * n + j] * d[j]).sum();
                }
                let model_obj = dot(&p.g, &d) + 0.5 * dot(&d, &hd);
                Ok(Direction {
                    d,
                    lambda: s.lambda,
                    z_l: s.z_l[..n].to_vec(),
                    z_u: s.z_u[..n].to_vec(),
                    active: s
                        .active
                        .into_iter()
                        .filter(|c| !matches!(c, Constraint::BoundLower(j) | Constraint::BoundUpper(j) if *j >= n))
                        .collect(),
                    v_lin,
                    model_obj,
                    elastic: true,
                })
            }
            Err(e) => Err(format!("elastic QP failed: {e}")),
        }
    }

    /// Multipliers of QP bounds that came from the step bound rather than the
    /// problem's own bounds are not KKT multipliers: zero them.
    fn drop_step_bound_multipliers(&self, p: &Point, z_l: &mut [f64], z_u: &mut [f64]) {
        if !self.step_bound.is_finite() {
            return;
        }
        for j in 0..self.n {
            let r = self.step_bound * p.x[j].abs().max(1.0);
            if self.xl[j] - p.x[j] < -r {
                z_l[j] = 0.0;
            }
            if self.xu[j] - p.x[j] > r {
                z_u[j] = 0.0;
            }
        }
    }

    // ---- second-order check at a KKT candidate (basin study: HS33) --------------------

    /// Probe the curvature of the Lagrangian on the null space of the strongly
    /// active constraint gradients with second differences of `f + lambda^T c`
    /// (two evaluations per probe, `k(k+1)/2` probes for a `k`-dimensional null
    /// space; skipped when `k > 6`). Returns a descent direction of negative
    /// curvature that is feasible for the weakly active constraints, if one is
    /// found; `None` when the check passed or could not be made (the second
    /// element says which).
    fn saddle_probe(
        &mut self,
        p: &Point,
        lambda: &[f64],
        z_l: &[f64],
        z_u: &[f64],
    ) -> (Option<Vec<f64>>, &'static str) {
        let (n, m) = (self.n, self.m);
        let g_scale = 1.0 + p.g.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        let strong_tol = 1e-6 * g_scale;
        let act_tol = 1e-6;
        // strongly active gradients and weakly active ones with their feasible sign
        let mut strong: Vec<Vec<f64>> = Vec::new();
        let mut weak: Vec<(Vec<f64>, f64)> = Vec::new(); // (gradient, required sign of gradient^T d)
        for i in 0..m {
            let row = p.j[i * n..(i + 1) * n].to_vec();
            let at_lo = (p.c[i] - self.cl[i]).abs() <= act_tol * (1.0 + self.cl[i].abs());
            let at_up = (p.c[i] - self.cu[i]).abs() <= act_tol * (1.0 + self.cu[i].abs());
            if self.cl[i] == self.cu[i] || lambda[i].abs() > strong_tol {
                if at_lo || at_up || self.cl[i] == self.cu[i] {
                    strong.push(row);
                }
            } else if at_lo && self.cl[i].is_finite() {
                weak.push((row, 1.0)); // c_i >= cl active: need grad^T d >= 0
            } else if at_up && self.cu[i].is_finite() {
                weak.push((row, -1.0));
            }
        }
        for j in 0..n {
            let mut e = vec![0.0; n];
            e[j] = 1.0;
            let at_lo = (p.x[j] - self.xl[j]).abs() <= act_tol * (1.0 + self.xl[j].abs());
            let at_up = (p.x[j] - self.xu[j]).abs() <= act_tol * (1.0 + self.xu[j].abs());
            if self.xl[j] == self.xu[j] || z_l[j] > strong_tol || z_u[j] > strong_tol {
                if at_lo || at_up {
                    strong.push(e);
                }
            } else if at_lo && self.xl[j].is_finite() {
                weak.push((e, 1.0));
            } else if at_up && self.xu[j].is_finite() {
                weak.push((e, -1.0));
            }
        }
        // orthonormal basis of range(A_strong^T), then of its complement, by Gram-Schmidt
        let mut basis: Vec<Vec<f64>> = Vec::new();
        for row in &strong {
            let mut v = row.clone();
            for b in &basis {
                let c = dot(&v, b);
                for (vi, bi) in v.iter_mut().zip(b) {
                    *vi -= c * bi;
                }
            }
            let nv = dot(&v, &v).sqrt();
            if nv > 1e-10 * (1.0 + dot(row, row).sqrt()) {
                for vi in &mut v {
                    *vi /= nv;
                }
                basis.push(v);
            }
        }
        let rank = basis.len();
        let k = n - rank;
        if k == 0 {
            return (None, "vertex: no null space");
        }
        if k > 6 {
            return (None, "null space too large to probe");
        }
        let mut z_basis: Vec<Vec<f64>> = Vec::new();
        for j in 0..n {
            if z_basis.len() == k {
                break;
            }
            let mut v = vec![0.0; n];
            v[j] = 1.0;
            for b in basis.iter().chain(&z_basis) {
                let c = dot(&v, b);
                for (vi, bi) in v.iter_mut().zip(b) {
                    *vi -= c * bi;
                }
            }
            let nv = dot(&v, &v).sqrt();
            if nv > 1e-8 {
                for vi in &mut v {
                    *vi /= nv;
                }
                z_basis.push(v);
            }
        }
        if z_basis.len() != k {
            return (None, "null-space basis incomplete");
        }
        let x_scale = p.x.iter().fold(0.0_f64, |a, v| a.max(v.abs())).max(1.0);
        let h = mincon_core::EPS.powf(0.25) * x_scale;
        let lag = |this: &Self, xt: &[f64]| -> Option<f64> {
            for j in 0..n {
                if xt[j] < this.xl[j] || xt[j] > this.xu[j] {
                    return None;
                }
            }
            let (f, c) = this.values(xt).ok()?;
            Some(f + dot(lambda, &c))
        };
        let l0 = p.f + dot(lambda, &p.c);
        // Central second difference when both probes stay in the box, otherwise a
        // one-sided one on whichever side is inside (weakly active bounds make one
        // side infeasible; curvature is even in the direction, so either works).
        let second = |this: &Self, v: &[f64]| -> Option<f64> {
            let at = |t: f64| -> Vec<f64> { (0..n).map(|j| p.x[j] + t * v[j]).collect() };
            let lp = lag(this, &at(h));
            let lm = lag(this, &at(-h));
            match (lp, lm) {
                (Some(lp), Some(lm)) => Some((lp - 2.0 * l0 + lm) / (h * h)),
                (Some(lp), None) => {
                    let l2 = lag(this, &at(2.0 * h))?;
                    Some((l2 - 2.0 * lp + l0) / (h * h))
                }
                (None, Some(lm)) => {
                    let l2 = lag(this, &at(-2.0 * h))?;
                    Some((l2 - 2.0 * lm + l0) / (h * h))
                }
                (None, None) => None,
            }
        };
        let mut mat = vec![0.0; k * k];
        for a in 0..k {
            let Some(daa) = second(self, &z_basis[a]) else {
                return (None, "probe left the box or the model failed");
            };
            mat[a * k + a] = daa;
        }
        for a in 0..k {
            for b in a + 1..k {
                let sum: Vec<f64> = (0..n).map(|j| z_basis[a][j] + z_basis[b][j]).collect();
                let Some(dab) = second(self, &sum) else {
                    return (None, "probe left the box or the model failed");
                };
                let off = 0.5 * (dab - mat[a * k + a] - mat[b * k + b]);
                mat[a * k + b] = off;
                mat[b * k + a] = off;
            }
        }
        let (eig, vecs) = jacobi_eigen(&mat, k);
        let mscale = mat
            .iter()
            .fold(0.0_f64, |a, v| a.max(v.abs()))
            .max(1e-8 * g_scale / x_scale);
        let (imin, &emin) = eig
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(b.1))
            .expect("k >= 1");
        // second differences carry truncation and rounding error; demand a clear margin
        if emin >= -1e-3 * mscale {
            return (None, "no negative curvature found");
        }
        let mut dir = vec![0.0; n];
        for a in 0..k {
            for j in 0..n {
                dir[j] += vecs[a * k + imin] * z_basis[a][j];
            }
        }
        let feasible = |sgn: f64| -> bool {
            weak.iter()
                .all(|(grad, need)| sgn * dot(grad, &dir) * need >= -1e-10)
        };
        if feasible(1.0) {
            (Some(dir), "negative curvature")
        } else if feasible(-1.0) {
            for v in &mut dir {
                *v = -*v;
            }
            (Some(dir), "negative curvature")
        } else {
            (
                None,
                "negative curvature only outside the cone of the weakly active constraints",
            )
        }
    }

    /// One Newton step onto the strongly active rows from `xt` (values `ct`),
    /// using the Jacobian at the base point `p`: `w = -J_a^T (J_a J_a^T)^{-1} r`.
    fn restore_active(
        &self,
        p: &Point,
        lambda: &[f64],
        xt: &[f64],
        ct: &[f64],
    ) -> Option<Vec<f64>> {
        let (n, m) = (self.n, self.m);
        let g_scale = 1.0 + p.g.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        let rows: Vec<usize> = (0..m)
            .filter(|&i| self.cl[i] == self.cu[i] || lambda[i].abs() > 1e-6 * g_scale)
            .collect();
        if rows.is_empty() {
            return None;
        }
        let q = rows.len();
        // Strongly active rows are pinned to the side their multiplier points at
        // (second-order theory moves along the active surface, not into its interior).
        let mut r = vec![0.0; q];
        for (k, &i) in rows.iter().enumerate() {
            let target = if self.cl[i] == self.cu[i] || lambda[i] < 0.0 {
                self.cl[i]
            } else {
                self.cu[i]
            };
            r[k] = if target.is_finite() {
                ct[i] - target
            } else {
                0.0
            };
        }
        if r.iter().all(|v| *v == 0.0) {
            return None;
        }
        let mut jjt = vec![0.0; q * q];
        for a in 0..q {
            for b in 0..q {
                jjt[a * q + b] = (0..n)
                    .map(|j| p.j[rows[a] * n + j] * p.j[rows[b] * n + j])
                    .sum();
            }
        }
        let y = solve_small(&mut jjt, &mut r, q)?;
        let mut x = xt.to_vec();
        for (k, &i) in rows.iter().enumerate() {
            for j in 0..n {
                x[j] -= p.j[i * n + j] * y[k];
            }
        }
        self.project(&mut x);
        Some(x)
    }

    // ---- reporting ------------------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    fn finish(
        mut self,
        exit: ExitFlag,
        iterations: usize,
        p: &Point,
        lambda: &[f64],
        z_l: &[f64],
        z_u: &[f64],
        trace: Vec<IterationRecord>,
    ) -> SolveReport {
        let (n, m) = (self.n, self.m);
        if let Hess::Bfgs(b) = &self.hess {
            if b.rebuilds() > 0 {
                self.notes.push(format!(
                    "The quasi-Newton model was rebuilt from per-coordinate curvature quotients {} time(s): its curvature along an accepted step was off by more than {:.0}x.",
                    b.rebuilds(),
                    self.opts.bfgs_curvature_rescale
                ));
            }
        }
        let (e0, compl, _) = self.kkt(p, lambda, z_l, z_u);
        let violation = self.user_violation(&p.x, &p.c);
        let f_user = p.f / self.d_f;
        let c_user: Vec<f64> = (0..m).map(|i| p.c[i] / self.d_c[i]).collect();
        let lambda_user: Vec<f64> = (0..m).map(|i| lambda[i] * self.d_c[i] / self.d_f).collect();
        let mut zl_user: Vec<f64> = z_l.iter().map(|z| z / self.d_f).collect();
        let mut zu_user: Vec<f64> = z_u.iter().map(|z| z / self.d_f).collect();
        // Fixed variables: the finite differences never step off a pinned value, so
        // their gradient entries are zero; reconstruct the multiplier from the true
        // partial derivatives (two probes per fixed variable) as the IP member does.
        let fixed: Vec<usize> = (0..n).filter(|&j| self.xl[j] == self.xu[j]).collect();
        if !fixed.is_empty() {
            if let Ok((gf, jcols)) = self.fixed_variable_derivatives(&p.x, &c_user, &fixed) {
                for (k, &j) in fixed.iter().enumerate() {
                    let mut r = gf[k];
                    for i in 0..m {
                        r += jcols[k][i] * lambda_user[i];
                    }
                    // g + J^T lambda - z_l + z_u = 0
                    if r >= 0.0 {
                        zl_user[j] = r;
                        zu_user[j] = 0.0;
                    } else {
                        zl_user[j] = 0.0;
                        zu_user[j] = -r;
                    }
                }
            }
        }
        let counters = self.eval.counters();
        let timings = Timings {
            total: self.start.elapsed(),
            model: self.eval.model_time(),
            ..Timings::default()
        };
        SolveReport {
            solution: Solution {
                x: p.x.clone(),
                f: f_user,
                c: c_user,
                lambda: lambda_user,
                z_l: zl_user,
                z_u: zu_user,
            },
            exit_flag: exit,
            algorithm: Algorithm::Sqp,
            iterations,
            f_evals: EvalCounters::get(&counters.f),
            g_evals: EvalCounters::get(&counters.g),
            c_evals: EvalCounters::get(&counters.c),
            j_evals: EvalCounters::get(&counters.j),
            h_evals: EvalCounters::get(&counters.h),
            failed_evals: EvalCounters::get(&counters.failed),
            optimality: e0,
            constraint_violation: violation,
            complementarity: compl,
            trace,
            timings,
            notes: std::mem::take(&mut self.notes),
        }
    }

    fn fixed_variable_derivatives(
        &self,
        x: &[f64],
        _c_user: &[f64],
        fixed: &[usize],
    ) -> Result<(Vec<f64>, Vec<Vec<f64>>), EvalError> {
        let (n, m) = (self.n, self.m);
        let nlp = self.eval.nlp();
        let caps = nlp.capabilities();
        let mut gf = vec![0.0; fixed.len()];
        let mut jcols = vec![vec![0.0; m]; fixed.len()];
        if caps.gradient {
            let mut g = vec![0.0; n];
            nlp.gradient(x, &mut g)?;
            for (k, &j) in fixed.iter().enumerate() {
                gf[k] = g[j];
            }
        }
        let jac_exact = m > 0 && caps.jacobian && nlp.jacobian_structure().is_some();
        if jac_exact {
            let pat = nlp.jacobian_structure().expect("checked");
            let mut vals = vec![0.0; pat.nnz()];
            nlp.jacobian(x, &mut vals)?;
            for (k, &j) in fixed.iter().enumerate() {
                for pos in pat.col_ptr()[j]..pat.col_ptr()[j + 1] {
                    jcols[k][pat.row_idx()[pos]] = vals[pos];
                }
            }
        }
        let need_f = !caps.gradient;
        let need_c = m > 0 && !jac_exact;
        if need_f || need_c {
            let mut xp = x.to_vec();
            let mut cp = vec![0.0; m];
            let mut cm = vec![0.0; m];
            for (k, &j) in fixed.iter().enumerate() {
                let h = mincon_core::EPS.powf(1.0 / 3.0) * x[j].abs().max(1.0);
                xp[j] = x[j] + h;
                let fp = if need_f { self.eval.f(&xp)? } else { 0.0 };
                if need_c {
                    self.eval.c(&xp, &mut cp)?;
                }
                xp[j] = x[j] - h;
                let fm = if need_f { self.eval.f(&xp)? } else { 0.0 };
                if need_c {
                    self.eval.c(&xp, &mut cm)?;
                }
                xp[j] = x[j];
                if need_f {
                    gf[k] = (fp - fm) / (2.0 * h);
                }
                if need_c {
                    for i in 0..m {
                        jcols[k][i] = (cp[i] - cm[i]) / (2.0 * h);
                    }
                }
            }
        }
        Ok((gf, jcols))
    }

    // ---- main loop --------------------------------------------------------------------

    #[allow(clippy::too_many_lines)]
    fn run(mut self) -> Result<SolveReport, SolveError> {
        let (n, m) = (self.n, self.m);
        let max_iter = self.opts.effective_max_iterations(n);
        let tol = self.opts.tol;

        // starting point inside the box
        let mut x0 = self.eval.nlp().x0().to_vec();
        self.project(&mut x0);
        let mut p = self.point(x0).map_err(SolveError::InitialPoint)?;
        self.compute_scaling(&p);
        if (self.d_f - 1.0).abs() > 1e-12 || self.d_c.iter().any(|d| (d - 1.0).abs() > 1e-12) {
            // re-express the initial point in scaled units
            let x = p.x.clone();
            p = self.point(x).map_err(SolveError::InitialPoint)?;
        }
        let f0_user = p.f / self.d_f;
        let x0_norm = p.x.iter().fold(0.0_f64, |a, v| a.max(v.abs()));

        let mut lambda = vec![0.0; m];
        let mut z_l = vec![0.0; n];
        let mut z_u = vec![0.0; n];
        let mut hint: Vec<Constraint> = Vec::new();
        // penalty (5.3): scale-aware start
        let g0 = p.g.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        let j0 = p.j.iter().fold(0.0_f64, |a, v| a.max(v.abs())).max(1.0);
        let mut rho = (g0 / j0).max(1.0);

        let approximate = self.opts.fd_error_aware
            && (self.eval.gradient_is_approximate() || self.eval.jacobian_is_approximate());
        let mut tol_eff = tol.optimality;
        let mut error_checked_at: Option<usize> = None;
        let mut near_convergence = 0usize;
        let mut acceptable_streak = 0usize;
        let mut guard_blocked = 0usize;
        let mut rescales = 0usize;
        let mut bfgs_resets = 0usize;
        let mut escalated = false;
        let mut escapes = 0usize;
        let mut curvature_rescaled = false;
        let mut zero_step_retest = false;
        let mut bound_shrinks = 0usize;
        let mut progress: Vec<(f64, f64)> = Vec::new();
        let mut trace: Vec<IterationRecord> = Vec::new();
        let mut last_alpha = 0.0;
        let mut last_step = 0.0;
        let mut last_soc = 0usize;
        let mut exit = ExitFlag::MaxReached;
        let mut iterations = 0usize;

        for iter in 0..=max_iter {
            iterations = iter;
            let (e0, compl, stat_rel) = self.kkt(&p, &lambda, &z_l, &z_u);
            let violation = self.user_violation(&p.x, &p.c);
            if self.opts.record_trace {
                trace.push(IterationRecord {
                    iter,
                    f_count: EvalCounters::get(&self.eval.counters().f),
                    f: p.f / self.d_f,
                    constraint_violation: violation,
                    optimality: e0,
                    step_norm: last_step,
                    alpha: last_alpha,
                    mu: 0.0,
                    delta_w: match &self.hess {
                        Hess::Exact { shift, .. } => *shift,
                        Hess::Bfgs(_) => 0.0,
                    },
                    delta_c: rho,
                    in_restoration: false,
                    soc_count: last_soc,
                });
            }

            // ---- error-aware finite-difference floor (as mincon-ip) ----
            if e0 <= 100.0 * tol.optimality && violation <= tol.acceptable_feasibility {
                near_convergence += 1;
            } else {
                near_convergence = 0;
            }
            if approximate
                && near_convergence >= 3
                && e0 > tol.optimality
                && error_checked_at.is_none_or(|k| iter >= k + 10)
            {
                error_checked_at = Some(iter);
                let c_user: Vec<f64> = (0..m).map(|i| p.c[i] / self.d_c[i]).collect();
                let g_user: Vec<f64> = p.g.iter().map(|g| g / self.d_f).collect();
                if let Ok((g_err, j_err)) = self.eval.derivative_error_estimate(
                    &p.x,
                    p.f / self.d_f,
                    &c_user,
                    &g_user,
                    &self.jac_values,
                    4,
                ) {
                    let d_c_max = self.d_c.iter().copied().fold(0.0_f64, f64::max);
                    let err_scaled = (g_err * self.d_f).max(j_err * d_c_max);
                    if err_scaled > tol.acceptable_optimality
                        && !self.eval.uses_central_differences()
                        && matches!(self.opts.fd_type, mincon_core::FdType::Adaptive)
                    {
                        self.eval.escalate_accuracy();
                        escalated = true;
                        self.notes.push(format!(
                            "Estimated forward-difference derivative error {err_scaled:.2e} (scaled) exceeds the acceptable optimality tolerance; switched to central differences at iteration {iter}."
                        ));
                        error_checked_at = Some(iter.saturating_sub(5));
                        let (g, j) = self.derivatives(&p.x, p.f, &p.c).map_err(|e| {
                            SolveError::Internal(format!("derivatives after escalation: {e}"))
                        })?;
                        p.g = g;
                        p.j = j;
                        continue;
                    }
                    tol_eff = err_scaled.clamp(tol.optimality, tol.acceptable_optimality);
                }
            }

            // ---- termination ----
            let scaled_pass = e0 <= tol_eff
                && violation <= tol.feasibility
                && compl <= tol.complementarity.max(tol_eff);
            let rel_ok = stat_rel <= tol.acceptable_optimality;
            let acceptable_level =
                e0 <= tol.acceptable_optimality && violation <= tol.acceptable_feasibility;
            if (scaled_pass || acceptable_level) && !rel_ok && rescales < 3 {
                let g_unscaled = p.g.iter().fold(0.0_f64, |a, g| a.max(g.abs())) / self.d_f;
                let gmax = self.opts.scaling_max_gradient;
                let d_f_new = if g_unscaled > gmax {
                    gmax / g_unscaled
                } else {
                    1.0
                };
                let k = d_f_new / self.d_f;
                if k.is_finite() && k > 10.0 {
                    rescales += 1;
                    self.notes.push(format!(
                        "Objective rescaled at iteration {iter}: the gradient norm fell from {:.2e} (where the scale factor {:.2e} was chosen) to {g_unscaled:.2e}; new factor {d_f_new:.2e}.",
                        gmax / self.d_f,
                        self.d_f
                    ));
                    self.rescale_objective(k, &mut p, &mut lambda, &mut z_l, &mut z_u, &mut rho);
                    acceptable_streak = 0;
                    guard_blocked = 0;
                    continue;
                }
            }
            if scaled_pass && !rel_ok {
                guard_blocked += 1;
                if guard_blocked >= tol.acceptable_iterations {
                    self.notes.push(format!(
                        "Stopped with the scaled KKT error {e0:.2e} below tolerance but the stationarity relative to the gradient terms at {stat_rel:.1e} (very large multipliers or a degenerate active set)."
                    ));
                    exit = ExitFlag::Acceptable;
                    break;
                }
            } else {
                guard_blocked = 0;
            }
            if scaled_pass && rel_ok {
                // Unbounded multipliers: no constraint qualification holds here, so
                // the point is optimal to tolerance but not a certified KKT point.
                let g_user = p.g.iter().fold(0.0_f64, |a, v| a.max(v.abs())) / self.d_f;
                let mult_user = (0..m)
                    .map(|i| (lambda[i] * self.d_c[i] / self.d_f).abs())
                    .chain(z_l.iter().chain(z_u.iter()).map(|z| (z / self.d_f).abs()))
                    .fold(0.0_f64, f64::max);
                if mult_user > 1e8 * (1.0 + g_user) {
                    self.notes.push(format!(
                        "Stopped at a point satisfying the tolerances with multipliers of size {mult_user:.2e} (gradient {g_user:.2e}): the active constraints are degenerate here (no bounded multipliers), so the point is reported as acceptable rather than as a certified KKT point."
                    ));
                    exit = ExitFlag::Acceptable;
                    break;
                }
                // Second-order check on small null spaces (SQP with a positive
                // definite quasi-Newton matrix cannot see a saddle by itself).
                if escapes < 3 {
                    let (dir_opt, why) = self.saddle_probe(&p, &lambda, &z_l, &z_u);
                    if let Some(dir) = dir_opt {
                        // move off the saddle along the negative-curvature direction
                        // Along the tangent the active constraints bend away at second
                        // order and the penalty term would swamp the Lagrangian decrease
                        // (the Maratos effect again), so each trial is followed by one
                        // Newton restoration step onto the strongly active constraints.
                        let phi0 = self.merit(p.f, &p.c, rho);
                        let mut t = x_scale_of(&p.x);
                        let mut moved: Option<(Vec<f64>, f64, Vec<f64>)> = None;
                        for _ in 0..20 {
                            let mut xt: Vec<f64> = (0..n).map(|j| p.x[j] + t * dir[j]).collect();
                            self.project(&mut xt);
                            if let Ok((ft, ct)) = self.values(&xt) {
                                let (xr, fr, cr) = match self.restore_active(&p, &lambda, &xt, &ct)
                                {
                                    Some(xr) => match self.values(&xr) {
                                        Ok((fr, cr)) => (xr, fr, cr),
                                        Err(_) => (xt, ft, ct),
                                    },
                                    None => (xt, ft, ct),
                                };
                                if self.merit(fr, &cr, rho) < phi0 - 1e-12 * phi0.abs().max(1.0) {
                                    moved = Some((xr, fr, cr));
                                    break;
                                }
                            }
                            t *= 0.5;
                        }
                        if let Some((xt, ft, ct)) = moved {
                            escapes += 1;
                            self.notes.push(format!(
                                "Iteration {iter}: the first-order conditions held but a direction of negative curvature of the Lagrangian was found (second-difference probe); left the saddle point along it (step {t:.2e}) and continued."
                            ));
                            match self.derivatives(&xt, ft, &ct) {
                                Ok((g, j)) => {
                                    p = Point {
                                        x: xt,
                                        f: ft,
                                        c: ct,
                                        g,
                                        j,
                                    };
                                    if let Hess::Bfgs(b) = &mut self.hess {
                                        b.reset(1.0);
                                    }
                                    acceptable_streak = 0;
                                    near_convergence = 0;
                                    continue;
                                }
                                Err(e) => {
                                    self.notes.push(format!("Derivative evaluation failed after leaving the saddle: {e}"));
                                }
                            }
                        }
                    } else if why != "no negative curvature found" && why != "vertex: no null space"
                    {
                        self.notes
                            .push(format!("Second-order conditions not verified: {why}."));
                    }
                }
                if tol_eff > tol.optimality {
                    self.notes.push(format!(
                        "Converged to the accuracy of the finite-difference derivatives: scaled KKT error {e0:.2e} is below the estimated derivative error {tol_eff:.2e}, which is above the requested optimality tolerance {:.1e}. Supply analytic derivatives for a tighter certificate.",
                        tol.optimality
                    ));
                }
                exit = ExitFlag::Optimal;
                break;
            }
            if acceptable_level {
                acceptable_streak += 1;
                if acceptable_streak >= tol.acceptable_iterations {
                    exit = ExitFlag::Acceptable;
                    break;
                }
            } else {
                acceptable_streak = 0;
            }
            if p.f / self.d_f <= tol.objective_limit && violation <= tol.feasibility {
                exit = ExitFlag::Unbounded;
                break;
            }
            // diverging iterates (the interior-point member's rule)
            let x_norm = p.x.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
            if x_norm > 1e6 * x0_norm.max(1.0)
                && p.f / self.d_f < f0_user - 1e6 * f0_user.abs().max(1.0)
            {
                self.notes.push(format!(
                    "Iterates moved far from the start (|x| = {x_norm:.2e}) while the objective fell by more than 1e6 times its initial size; the problem looks unbounded."
                ));
                exit = ExitFlag::Unbounded;
                break;
            }
            progress.push((p.f / self.d_f, violation));
            let budget_exit = self
                .opts
                .max_evaluations
                .is_some_and(|limit| EvalCounters::get(&self.eval.counters().f) >= limit)
                || self
                    .opts
                    .max_seconds
                    .is_some_and(|limit| self.start.elapsed().as_secs_f64() >= limit)
                || iter >= max_iter;
            if budget_exit {
                exit = ExitFlag::MaxReached;
                self.notes.push(progress_verdict(&progress));
                break;
            }

            // ---- QP step ----
            let mut h = match self.hessian_dense(&p, &lambda) {
                Ok(h) => h,
                Err(e) => {
                    self.notes.push(format!(
                        "Hessian evaluation failed: {e}; falling back to BFGS."
                    ));
                    self.hess = Hess::Bfgs(Box::new(DenseBfgs::with_curvature_rescale(
                        n,
                        self.opts.bfgs_curvature_rescale,
                    )));
                    match self.hessian_dense(&p, &lambda) {
                        Ok(h) => h,
                        Err(e) => return Err(SolveError::Internal(format!("Hessian: {e}"))),
                    }
                }
            };
            let mut dir = match self.direction(&p, &mut h, rho, &hint) {
                Ok(d) => d,
                Err(msg) => {
                    self.notes
                        .push(format!("QP subproblem failed at iteration {iter}: {msg}"));
                    exit = ExitFlag::NumericalFailure;
                    break;
                }
            };
            let v_now = self.violation_l1(&p.c);
            // penalty: multiplier floor, then the model rule (5.4)
            let lam_max = dir
                .lambda
                .iter()
                .chain(&dir.z_l)
                .chain(&dir.z_u)
                .fold(0.0_f64, |a, v| a.max(v.abs()));
            if lam_max > rho {
                rho = (1.5 * lam_max).max(rho);
                if dir.elastic {
                    // the elastic solution depends on rho: re-solve once
                    if let Ok(d2) = self.direction(&p, &mut h, rho, &hint) {
                        dir = d2;
                    }
                }
            }
            // The model rule only makes sense while the violation is significant:
            // from a nearly feasible point a tiny "reduction" would send rho to
            // infinity (the penalty trap), so it is skipped there and the increase
            // is capped to a factor 100 per iteration.
            let reduction = v_now - dir.v_lin;
            if v_now > tol.feasibility && reduction > 1e-3 * v_now && dir.model_obj > 0.0 {
                let need = dir.model_obj / ((1.0 - PENALTY_FRACTION) * reduction);
                if need > rho {
                    rho = (1.5 * need).min(100.0 * rho);
                    if dir.elastic {
                        if let Ok(d2) = self.direction(&p, &mut h, rho, &hint) {
                            dir = d2;
                        }
                    }
                }
            }
            // predicted merit decrease (5.1): >= 1/2 d^T B d
            let delta_m = -dir.model_obj + rho * (v_now - dir.v_lin);
            let d_norm = dir.d.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
            let x_scale = p.x.iter().fold(0.0_f64, |a, v| a.max(v.abs())).max(1.0);

            if d_norm <= tol.step * x_scale {
                // The QP says "stay": a KKT point of the linearization. Its multipliers
                // are the ones the termination test should see, so adopt them and
                // re-run the test once before classifying.
                if !zero_step_retest {
                    zero_step_retest = true;
                    lambda = dir.lambda;
                    z_l = dir.z_l;
                    z_u = dir.z_u;
                    hint = dir.active;
                    continue;
                }
                if violation <= tol.feasibility {
                    // stationarity could still be above the tolerance for FD reasons
                    lambda = dir.lambda.clone();
                    z_l = dir.z_l.clone();
                    z_u = dir.z_u.clone();
                    let (e0b, _, relb) = self.kkt(&p, &lambda, &z_l, &z_u);
                    exit = if e0b <= tol.acceptable_optimality
                        && relb <= 10.0 * tol.acceptable_optimality
                    {
                        ExitFlag::Acceptable
                    } else {
                        ExitFlag::StepTolerance
                    };
                    self.notes.push(format!(
                        "The QP step vanished at a feasible point (scaled KKT error {e0b:.2e}); reported as {exit:?}."
                    ));
                } else if dir.elastic {
                    self.notes.push(format!(
                        "The elastic QP step vanished with constraint violation {violation:.3e}: a stationary point of the constraint violation; this is a local diagnostic, not a proof of infeasibility."
                    ));
                    exit = ExitFlag::LocallyInfeasible;
                } else {
                    exit = ExitFlag::NumericalFailure;
                }
                break;
            }

            // ---- line search on the merit function (5.2) ----
            let phi0 = self.merit(p.f, &p.c, rho);
            let mut alpha = 1.0_f64;
            let mut accepted: Option<(Vec<f64>, f64, Vec<f64>)> = None;
            let mut soc_used = 0usize;
            let mut trials = 0usize;
            let mut soc_tried = false;
            // largest step with a finite objective, for the curvature rescale below
            let mut widest_finite: Option<(f64, f64)> = None; // (alpha, f)
            while trials < MAX_BACKTRACKS {
                trials += 1;
                let mut xt: Vec<f64> = (0..n).map(|j| p.x[j] + alpha * dir.d[j]).collect();
                self.project(&mut xt);
                let trial = self.values(&xt);
                let ok = match &trial {
                    Ok((ft, ct)) => self.merit(*ft, ct, rho) <= phi0 - ARMIJO * alpha * delta_m,
                    Err(_) => false,
                };
                if std::env::var_os("MINCON_SQP_DEBUG").is_some() {
                    if let Ok((ft, ct)) = &trial {
                        eprintln!(
                            "it {iter} trial alpha={alpha:.3e} f {:.9e}->{:.9e} v {:.3e}->{:.3e} phi {:.9e}->{:.9e} need<= {:.9e} delta_m={delta_m:.3e} model_obj={:.3e} v_lin={:.3e} rho={rho:.3e} elastic={} ok={ok}",
                            p.f, ft, self.violation_l1(&p.c), self.violation_l1(ct), phi0, self.merit(*ft, ct, rho), phi0 - ARMIJO * alpha * delta_m, dir.model_obj, dir.v_lin, dir.elastic
                        );
                    } else {
                        eprintln!("it {iter} trial alpha={alpha:.3e} non-finite");
                    }
                }
                if let Ok((ft, _)) = &trial {
                    if widest_finite.is_none() {
                        widest_finite = Some((alpha, *ft));
                    }
                }
                if ok {
                    let (ft, ct) = trial.expect("checked");
                    accepted = Some((xt, ft, ct));
                    break;
                }
                // second-order corrections on a rejected unit step (5.5): re-linearize
                // the rows at the trial point and re-solve with the same working set;
                // repeat while the violation keeps falling (IPOPT's kappa_soc rule).
                if alpha == 1.0 && !soc_tried && !dir.elastic && m > 0 {
                    soc_tried = true;
                    if let Ok((_, ct0)) = &trial {
                        let mut ct = ct0.clone();
                        let mut v_prev = self.violation_l1(&ct);
                        let mut d_cur = dir.d.clone();
                        for _soc in 0..SOC_MAX {
                            // c(x + d_cur) - J d_cur in place of c(x)
                            let mut jd = vec![0.0; m];
                            for i in 0..m {
                                jd[i] = (0..n).map(|j| p.j[i * n + j] * d_cur[j]).sum();
                            }
                            let c_shift: Vec<f64> = (0..m).map(|i| ct[i] - jd[i]).collect();
                            let p_shift = Point {
                                x: p.x.clone(),
                                f: p.f,
                                c: c_shift,
                                g: p.g.clone(),
                                j: p.j.clone(),
                            };
                            let Ok(corr) = self.direction(&p_shift, &mut h, rho, &dir.active)
                            else {
                                break;
                            };
                            if corr.elastic {
                                break;
                            }
                            let mut xc: Vec<f64> = (0..n).map(|j| p.x[j] + corr.d[j]).collect();
                            self.project(&mut xc);
                            let Ok((fc, cc)) = self.values(&xc) else {
                                break;
                            };
                            let v_c = self.violation_l1(&cc);
                            if std::env::var_os("MINCON_SQP_DEBUG").is_some() {
                                eprintln!(
                                    "it {iter} SOC trial f {:.9e}->{:.9e} v {:.3e}->{:.3e} phi {:.9e}->{:.9e} need<= {:.9e}",
                                    p.f,
                                    fc,
                                    self.violation_l1(&p.c),
                                    v_c,
                                    phi0,
                                    self.merit(fc, &cc, rho),
                                    phi0 - ARMIJO * delta_m
                                );
                            }
                            soc_used += 1;
                            if self.merit(fc, &cc, rho) <= phi0 - ARMIJO * delta_m {
                                // keep the QP multipliers of the original step
                                accepted = Some((xc, fc, cc));
                                break;
                            }
                            if v_c > 0.5 * v_prev {
                                break; // the correction is not converging: give up on it
                            }
                            v_prev = v_c;
                            ct = cc;
                            d_cur = corr.d;
                        }
                        if accepted.is_some() {
                            break;
                        }
                    }
                }
                alpha *= 0.5;
                // A step this short is not progress, whatever the merit says: below
                // the step tolerance the Armijo test is satisfied by rounding noise.
                if alpha * d_norm < tol.step.max(1e-10) * x_scale {
                    break;
                }
            }
            last_soc = soc_used;

            let Some((xn, fn_, cn)) = accepted else {
                // ---- line-search failure (5.5) ----
                // First response: the step was too long for the linearization to be
                // any good; bound the next QP step to a fraction of the widest trial
                // that still evaluated (at most 4 shrinks in a row).
                if bound_shrinks < 4 && !dir.elastic || (dir.elastic && bound_shrinks < 2) {
                    let rel = (0..n)
                        .map(|j| dir.d[j].abs() / p.x[j].abs().max(1.0))
                        .fold(0.0_f64, f64::max);
                    let a = widest_finite.map_or(1.0 / 32.0, |(a, _)| a);
                    let new_bound = (0.25 * a * rel).max(1e-8);
                    if new_bound < 0.5 * self.step_bound.min(rel) {
                        bound_shrinks += 1;
                        self.step_bound = new_bound;
                        continue;
                    }
                }
                // A unit quasi-Newton matrix that is off by orders of magnitude makes
                // every trial fail; the rejected trials themselves measure the
                // curvature along d, so rescale B from them once and retry (no
                // extra evaluations).
                if let Hess::Bfgs(b) = &mut self.hess {
                    if b.updates() == 0 && !curvature_rescaled {
                        if let Some((a, ft)) = widest_finite {
                            let gd = dot(&p.g, &dir.d);
                            let dd = dot(&dir.d, &dir.d);
                            let curv = 2.0 * (ft - p.f - a * gd) / (a * a * dd);
                            if curv.is_finite() && curv > 1.0 {
                                curvature_rescaled = true;
                                b.reset(curv);
                                self.notes.push(format!(
                                    "The unit initial quasi-Newton matrix rejected every trial step at iteration {iter}; rescaled it to {curv:.3e} from the curvature measured along the step and retrying."
                                ));
                                continue;
                            }
                        }
                    }
                }
                if approximate
                    && !self.eval.uses_central_differences()
                    && !escalated
                    && matches!(self.opts.fd_type, mincon_core::FdType::Adaptive)
                {
                    self.eval.escalate_accuracy();
                    escalated = true;
                    self.notes.push(format!(
                        "Line search failed at iteration {iter}; switched to central finite differences and retrying."
                    ));
                    let (g, j) = self.derivatives(&p.x, p.f, &p.c).map_err(|e| {
                        SolveError::Internal(format!("derivatives after escalation: {e}"))
                    })?;
                    p.g = g;
                    p.j = j;
                    continue;
                }
                if let Hess::Bfgs(b) = &mut self.hess {
                    if bfgs_resets < 2 {
                        bfgs_resets += 1;
                        b.reset(1.0);
                        self.notes.push(format!(
                            "Line search failed at iteration {iter}; reset the BFGS matrix and retrying."
                        ));
                        continue;
                    }
                }
                exit = if violation <= tol.feasibility {
                    if e0 <= tol.acceptable_optimality {
                        self.notes.push(format!(
                            "The line search could not make further progress at a feasible point with scaled KKT error {e0:.3e}; reported as acceptable."
                        ));
                        ExitFlag::Acceptable
                    } else {
                        self.notes.push(format!(
                            "The line search could not make further progress at a feasible point; scaled KKT error {e0:.3e} is above the acceptable tolerance, so first-order optimality is unverified."
                        ));
                        ExitFlag::StepTolerance
                    }
                } else if dir.elastic {
                    self.notes.push(format!(
                        "The line search could not reduce the ℓ1 merit function from an infeasible point (violation {violation:.3e}) along the elastic QP step; this is a local diagnostic, not a proof of infeasibility."
                    ));
                    ExitFlag::LocallyInfeasible
                } else {
                    self.notes.push(format!(
                        "The line search failed at an infeasible point (violation {violation:.3e})."
                    ));
                    ExitFlag::NumericalFailure
                };
                break;
            };

            // ---- step bound adaptation (trust-region flavour) ----
            {
                let rel = (0..n)
                    .map(|j| dir.d[j].abs() / p.x[j].abs().max(1.0))
                    .fold(0.0_f64, f64::max);
                if alpha < 0.25 {
                    self.step_bound = (2.0 * alpha * rel).max(1e-8).min(self.step_bound);
                } else if alpha >= 1.0 && self.step_bound.is_finite() {
                    let at_bound = rel >= 0.9 * self.step_bound;
                    self.step_bound = if at_bound {
                        4.0 * self.step_bound
                    } else {
                        self.step_bound
                    };
                    if self.step_bound > 1e3 {
                        self.step_bound = f64::INFINITY;
                    }
                }
                bound_shrinks = 0;
            }

            // ---- accept: derivatives at the new point, BFGS update (4.1)-(4.3) ----
            let (gn, jn) = match self.derivatives(&xn, fn_, &cn) {
                Ok(v) => v,
                Err(e) => {
                    self.notes.push(format!(
                        "Derivative evaluation failed at the accepted point: {e}"
                    ));
                    exit = ExitFlag::NumericalFailure;
                    break;
                }
            };
            let s: Vec<f64> = (0..n).map(|j| xn[j] - p.x[j]).collect();
            last_step = s.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
            last_alpha = alpha;
            if let Hess::Bfgs(b) = &mut self.hess {
                // y = grad_x L(x+, lam+) - grad_x L(x, lam+), L = f + lam^T c
                let mut y = vec![0.0; n];
                for j in 0..n {
                    let mut new = gn[j];
                    let mut old = p.g[j];
                    for i in 0..m {
                        new += jn[i * n + j] * dir.lambda[i];
                        old += p.j[i * n + j] * dir.lambda[i];
                    }
                    y[j] = new - old;
                }
                let first_step_cut = self.opts.bfgs_guarded_scaling
                    && b.updates() == 0
                    && b.skipped() == 0
                    && alpha < 0.125;
                b.update_guarded(&s, &y, first_step_cut);
            }
            zero_step_retest = false;
            lambda = dir.lambda;
            z_l = dir.z_l;
            z_u = dir.z_u;
            hint = dir.active;
            p = Point {
                x: xn,
                f: fn_,
                c: cn,
                g: gn,
                j: jn,
            };
        }
        Ok(self.finish(exit, iterations, &p, &lambda, &z_l, &z_u, trace))
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Gaussian elimination with partial pivoting for a small dense system.
fn solve_small(a: &mut [f64], b: &mut [f64], n: usize) -> Option<Vec<f64>> {
    for col in 0..n {
        let mut piv = col;
        for r in col + 1..n {
            if a[r * n + col].abs() > a[piv * n + col].abs() {
                piv = r;
            }
        }
        if a[piv * n + col].abs() < 1e-14 {
            return None;
        }
        if piv != col {
            for j in 0..n {
                a.swap(col * n + j, piv * n + j);
            }
            b.swap(col, piv);
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let f = a[r * n + col] / a[col * n + col];
            if f != 0.0 {
                for j in col..n {
                    a[r * n + j] -= f * a[col * n + j];
                }
                b[r] -= f * b[col];
            }
        }
    }
    Some((0..n).map(|i| b[i] / a[i * n + i]).collect())
}

fn x_scale_of(x: &[f64]) -> f64 {
    x.iter().fold(0.0_f64, |a, v| a.max(v.abs())).max(1.0)
}

/// The note attached to a budget exit (same rule as the interior-point member).
fn progress_verdict(hist: &[(f64, f64)]) -> String {
    const W: usize = 20;
    let Some(&(f_now, v_now)) = hist.last() else {
        return "Budget exhausted before the first iteration.".into();
    };
    if hist.len() < 3 {
        return "Budget exhausted within the first iterations; no progress verdict possible."
            .into();
    }
    let k = hist.len().saturating_sub(1 + W.min(hist.len() - 2));
    let (f_then, v_then) = hist[k];
    let w = hist.len() - 1 - k;
    let df_rel = (f_then - f_now) / f_then.abs().max(1.0);
    let v_ratio = if v_then > 0.0 { v_now / v_then } else { 1.0 };
    let steady = df_rel > 1e-3 || (v_then > 1e-6 && v_ratio < 0.9);
    if steady {
        format!(
            "Budget exhausted while still making steady progress: over the last {w} iterations the objective moved from {f_then:.6e} to {f_now:.6e} and the constraint violation from {v_then:.2e} to {v_now:.2e}. A larger budget, analytic derivatives, or a better-scaled start would likely finish this solve."
        )
    } else {
        format!(
            "Budget exhausted with no measurable progress over the last {w} iterations (objective {f_then:.6e} -> {f_now:.6e}, violation {v_then:.2e} -> {v_now:.2e}): the solver is likely stuck. The returned point is the last accepted iterate."
        )
    }
}

/// Eigenvalues and eigenvectors (columns of the returned row-major matrix) of a
/// small symmetric matrix by cyclic Jacobi rotations.
fn jacobi_eigen(a: &[f64], k: usize) -> (Vec<f64>, Vec<f64>) {
    let mut m = a.to_vec();
    let mut v = vec![0.0; k * k];
    for i in 0..k {
        v[i * k + i] = 1.0;
    }
    for _sweep in 0..100 {
        let mut off = 0.0;
        for p in 0..k {
            for q in p + 1..k {
                off += m[p * k + q] * m[p * k + q];
            }
        }
        if off < 1e-30 {
            break;
        }
        for p in 0..k {
            for q in p + 1..k {
                let apq = m[p * k + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let theta = (m[q * k + q] - m[p * k + p]) / (2.0 * apq);
                let t = if theta == 0.0 {
                    1.0
                } else {
                    theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for r in 0..k {
                    let mrp = m[r * k + p];
                    let mrq = m[r * k + q];
                    m[r * k + p] = c * mrp - s * mrq;
                    m[r * k + q] = s * mrp + c * mrq;
                }
                for cidx in 0..k {
                    let mpc = m[p * k + cidx];
                    let mqc = m[q * k + cidx];
                    m[p * k + cidx] = c * mpc - s * mqc;
                    m[q * k + cidx] = s * mpc + c * mqc;
                }
                for r in 0..k {
                    let vrp = v[r * k + p];
                    let vrq = v[r * k + q];
                    v[r * k + p] = c * vrp - s * vrq;
                    v[r * k + q] = s * vrp + c * vrq;
                }
            }
        }
    }
    ((0..k).map(|i| m[i * k + i]).collect(), v)
}
