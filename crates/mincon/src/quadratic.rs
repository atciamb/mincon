//! I5 (`docs/22`): recognise a quadratic program at the start and hand the SQP
//! member its exact, constant Hessian.
//!
//! A bound- or linearly-constrained quadratic is the problem class where a
//! quasi-Newton model is most wasteful: the curvature is constant and dense,
//! yet BFGS rebuilds it one pair per iteration (a 50-variable bounded
//! least-squares deconvolution takes 166 iterations and 8595 evaluations
//! through the facade; with its exact Hessian the SQP member takes 2). The
//! probe spends a handful of evaluations to find out whether the objective is
//! quadratic and every constraint row linear, and only then pays for the
//! Hessian, which is exact for a quadratic whatever the difference step.
//!
//! # The test
//!
//! Along two random bounds-respecting lines through the (projected) start the
//! objective is sampled at `x0 + k h d`, `k = 0..3`; a quadratic has a zero
//! third difference (`f3 - 3 f2 + 3 f1 - f0`) up to rounding, a linear
//! constraint row a zero second difference and a quadratic one a zero third
//! difference. The first failing test ends the probe, so a general nonlinear
//! problem pays three objective evaluations. A quadratic row is accepted only
//! when it keeps the feasible set convex (a convex function bounded above or
//! a concave one bounded below; never an equality), its Hessian is built by
//! the same differencing of the constraint vector (every quadratic row at
//! once), and the model's Lagrangian Hessian sums them with the multipliers.
//! When both lines pass, the Hessian is built by differencing (the gradient
//! when the model supplies one, `n` calls; function values otherwise,
//! `n (n + 3) / 2` calls for a dense one, so that build is skipped when it
//! would exceed the evaluation budget or [`FD_BUILD_MAX_N`]) and the
//! quadratic model is checked at the six line points, which were not used to
//! build it. A model that does not fit is discarded and the ordinary
//! portfolio runs.
//!
//! With function values the build is ordered by structure
//! (`QuadraticBuild::Structured`, the default): the diagonal first (`2n`
//! evaluations, which also give the gradient), then one off-diagonal band at
//! a time, the line-point check and the convexity check repeated after each
//! band, and the build stops at the first structure that passes both. A
//! diagonal Hessian costs `2n`, a tridiagonal one `3n - 1`, a dense one
//! exactly the dense build (`bench/results/abl-i5-build`).
//!
//! A near-quadratic that passes the tolerance gets a constant Hessian that is
//! wrong by the same small amount; the SQP member's line search and its
//! first-order certificate do not depend on the Hessian, so the cost of that
//! is a few more iterations, never a wrong answer.

use mincon_core::{
    is_free, Capabilities, EvalError, Nlp, NlpDims, Options, QuadraticBuild, QuadraticRows,
    Sparsity,
};

/// Largest `n` for which the Hessian is built from function values alone.
/// `n (n + 3) / 2` evaluations: 125 750 at the cap.
pub const FD_BUILD_MAX_N: usize = 500;
/// Largest `n` for which the Hessian is built from a supplied gradient.
pub const GRADIENT_BUILD_MAX_N: usize = 5000;
/// Largest total dense storage, in entries, of the quadratic rows' Hessians
/// during their build (`rows x n x n`; 128 MB of doubles).
pub const ROW_BUILD_MAX_ENTRIES: usize = 16_000_000;
/// Relative tolerance of the third-difference and model-fit tests. A true
/// quadratic's third difference is rounding, about 1e-13 of the variation at
/// the probe's step; a model with 1e-9 simulator noise on a variation of
/// 1e-2 sits at 1e-7 and must be declined, since Newton steps on a noisy
/// objective end in a failed line search where quasi-Newton steps certify.
const REL_TOL: f64 = 1e-8;
/// Absolute floor of those tests, relative to the magnitude of the values.
const ABS_TOL: f64 = 1e-12;

/// A problem whose objective was found to be quadratic and whose constraints
/// are linear, carrying the constant Hessian as an exact one.
pub struct QuadraticModel<'a, P: Nlp + ?Sized> {
    inner: &'a P,
    n: usize,
    /// Lower triangle, column-major in the order of `structure`.
    hess_lower: Vec<f64>,
    /// The quadratic constraint rows' Hessians, same layout, by row index.
    rows: Vec<(usize, Vec<f64>)>,
    structure: Sparsity,
    /// Objective, constraint and gradient evaluations spent by the probe and
    /// the build.
    pub f_evals: u64,
    /// See `f_evals`.
    pub c_evals: u64,
    /// See `f_evals`.
    pub g_evals: u64,
    /// See `f_evals`.
    pub j_evals: u64,
    /// What was detected and what it cost, for the report's notes.
    pub note: String,
}

/// Why the probe declined; only reported under `MINCON_QP_DEBUG=1`.
enum Decline {
    NotConvex,
    NotQuadratic(usize),
    NotLinear(usize, usize),
    TooLarge(usize),
    Budget(u64),
    Failed(String),
    ModelMisfit(f64),
    Unstructured(usize),
    NotQuadraticRow(usize, usize),
    NonconvexRow(usize),
}

impl std::fmt::Display for Decline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotConvex => write!(f, "the quadratic is not convex"),
            Self::NotQuadratic(line) => {
                write!(f, "the objective is not quadratic along line {line}")
            }
            Self::NotLinear(line, row) => {
                write!(f, "constraint row {row} is not linear along line {line}")
            }
            Self::TooLarge(n) => write!(f, "n = {n} is outside the build limits"),
            Self::Budget(cost) => write!(
                f,
                "the build ({cost} evaluations) would exceed the evaluation or time budget"
            ),
            Self::Failed(e) => write!(f, "an evaluation failed: {e}"),
            Self::ModelMisfit(err) => {
                write!(f, "the built model misfits a line point by {err:.2e}")
            }
            Self::Unstructured(bands) => write!(
                f,
                "no diagonal or banded Hessian within {bands} off-diagonal bands, and the dense build is outside the size or budget limits"
            ),
            Self::NotQuadraticRow(line, row) => {
                write!(f, "constraint row {row} is neither linear nor quadratic along line {line}")
            }
            Self::NonconvexRow(row) => write!(
                f,
                "quadratic constraint row {row} does not keep the feasible set convex (an equality, a ranged row, or the wrong curvature for its bound)"
            ),
        }
    }
}

struct XorShift(u64);
impl XorShift {
    fn next(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Probe the problem; `Some` when it is a quadratic program whose Hessian
/// could be built and verified.
pub fn probe<'a, P: Nlp + ?Sized>(nlp: &'a P, opts: &Options) -> Option<QuadraticModel<'a, P>> {
    probe_counted(nlp, opts).0
}

/// [`probe`] returning the objective evaluations spent even when it declines
/// (three per line at most, plus the build when it was attempted), so the
/// caller can keep the count honest, and the reason for a decline so the
/// report can say why (it used to reach only `MINCON_QP_DEBUG=1` stderr).
pub fn probe_counted<'a, P: Nlp + ?Sized>(
    nlp: &'a P,
    opts: &Options,
) -> (Option<QuadraticModel<'a, P>>, u64, Option<String>) {
    let debug = std::env::var_os("MINCON_QP_DEBUG").is_some();
    match probe_inner(nlp, opts) {
        Ok(q) => {
            let spent = q.f_evals;
            (Some(q), spent, None)
        }
        Err((why, spent)) => {
            let reason = why.to_string();
            if debug {
                eprintln!("quadratic probe declined: {reason} after {spent} objective evaluations");
            }
            (None, spent, Some(reason))
        }
    }
}

#[allow(clippy::too_many_lines)]
fn probe_inner<'a, P: Nlp + ?Sized>(
    nlp: &'a P,
    opts: &Options,
) -> Result<QuadraticModel<'a, P>, (Decline, u64)> {
    let NlpDims { n, m } = nlp.dims();
    let caps = nlp.capabilities();
    let mut f_evals = 0u64;
    let mut c_evals = 0u64;
    let mut g_evals = 0u64;
    let mut j_evals = 0u64;
    // Quadratic rows: the Jacobian variant needs the model's Jacobian.
    let rows_mode = match opts.quadratic_rows {
        QuadraticRows::Jacobian if !caps.jacobian => QuadraticRows::Off,
        mode => mode,
    };
    let (cl, cu) = nlp.c_bounds();
    if n == 0 {
        return Err((Decline::TooLarge(0), 0));
    }
    let (lb, ub) = nlp.x_bounds();
    // The base point: the start projected into the bounds.
    let x0: Vec<f64> = nlp
        .x0()
        .iter()
        .zip(lb.iter().zip(ub))
        .map(|(v, (l, u))| {
            let mut x = *v;
            if !is_free(*l) {
                x = x.max(*l);
            }
            if !is_free(*u) {
                x = x.min(*u);
            }
            x
        })
        .collect();
    // Per-coordinate step and the side it moves to: the roomier side, at most a
    // tenth of the magnitude, and never past the bound at three steps.
    let mut h = vec![0.0; n];
    let mut side = vec![1.0; n];
    for j in 0..n {
        let room_up = if is_free(ub[j]) {
            f64::INFINITY
        } else {
            ub[j] - x0[j]
        };
        let room_dn = if is_free(lb[j]) {
            f64::INFINITY
        } else {
            x0[j] - lb[j]
        };
        let (room, s) = if room_up >= room_dn {
            (room_up, 1.0)
        } else {
            (room_dn, -1.0)
        };
        let step = (0.1 * x0[j].abs().max(1.0)).min(room / 3.0);
        h[j] = if step > 0.0 && step.is_finite() {
            step
        } else {
            0.0
        };
        side[j] = s;
    }
    if h.iter().all(|v| *v == 0.0) {
        return Err((Decline::TooLarge(n), 0));
    }
    let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
    let mut directions = Vec::with_capacity(2);
    for _ in 0..2 {
        let d: Vec<f64> = (0..n)
            .map(|j| side[j] * h[j] * (0.5 + 0.5 * rng.next()))
            .collect();
        directions.push(d);
    }

    let eval_f = |x: &[f64], count: &mut u64| -> Result<f64, (Decline, u64)> {
        *count += 1;
        match nlp.objective(x) {
            Ok(v) if v.is_finite() => Ok(v),
            Ok(_) => Err((Decline::Failed("non-finite objective".into()), *count)),
            Err(e) => Err((Decline::Failed(e.to_string()), *count)),
        }
    };
    let probe_clock = std::time::Instant::now();
    let f0 = eval_f(&x0, &mut f_evals)?;
    let mut c0 = vec![0.0; m];
    if m > 0 {
        c_evals += 1;
        nlp.constraints(&x0, &mut c0)
            .map_err(|e| (Decline::Failed(e.to_string()), f_evals))?;
    }

    // Six line points, kept for the model check, with the constraint values
    // at each (a quadratic row's model is checked against them too).
    let mut line_points: Vec<(Vec<f64>, f64)> = Vec::with_capacity(6);
    let mut line_c: Vec<Vec<f64>> = Vec::with_capacity(6);
    // Per row: found quadratic (not linear) along at least one line.
    let mut row_quadratic = vec![false; m];
    for (line, d) in directions.iter().enumerate() {
        let mut fs = [f0, 0.0, 0.0, 0.0];
        let mut xs: Vec<Vec<f64>> = Vec::with_capacity(3);
        for k in 1..=3 {
            let kf = k as f64;
            let x: Vec<f64> = x0.iter().zip(d).map(|(a, b)| a + kf * b).collect();
            fs[k] = eval_f(&x, &mut f_evals)?;
            xs.push(x);
        }
        let d1 = fs[1] - fs[0];
        let d2a = fs[2] - 2.0 * fs[1] + fs[0];
        let d2b = fs[3] - 2.0 * fs[2] + fs[1];
        let d3 = fs[3] - 3.0 * fs[2] + 3.0 * fs[1] - fs[0];
        let scale = fs.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        let variation = d1.abs().max(d2a.abs()).max(d2b.abs());
        if d3.abs() > REL_TOL * variation + ABS_TOL * scale {
            return Err((Decline::NotQuadratic(line), f_evals));
        }
        if m > 0 {
            let mut cs: Vec<Vec<f64>> = Vec::with_capacity(3);
            for x in &xs {
                let mut c = vec![0.0; m];
                c_evals += 1;
                nlp.constraints(x, &mut c)
                    .map_err(|e| (Decline::Failed(e.to_string()), f_evals))?;
                if c.iter().any(|v| !v.is_finite()) {
                    return Err((Decline::Failed("non-finite constraint".into()), f_evals));
                }
                cs.push(c);
            }
            for i in 0..m {
                let (c1, c2, c3) = (cs[0][i], cs[1][i], cs[2][i]);
                let d1 = c1 - c0[i];
                let d2a = c2 - 2.0 * c1 + c0[i];
                let d2b = c3 - 2.0 * c2 + c1;
                let d3 = c3 - 3.0 * c2 + 3.0 * c1 - c0[i];
                let scale = c0[i].abs().max(c1.abs()).max(c2.abs()).max(c3.abs());
                let variation = d1.abs().max((c2 - c1).abs()).max((c3 - c2).abs());
                let tol = REL_TOL * variation + ABS_TOL * scale;
                if d2a.abs() > tol || d2b.abs() > tol {
                    if rows_mode == QuadraticRows::Off {
                        return Err((Decline::NotLinear(line, i), f_evals));
                    }
                    // Not linear: a quadratic row has a zero third difference.
                    let variation = d1.abs().max(d2a.abs()).max(d2b.abs());
                    let tol = REL_TOL * variation + ABS_TOL * scale;
                    if d3.abs() > tol {
                        return Err((Decline::NotQuadraticRow(line, i), f_evals));
                    }
                    // The feasible set stays convex only for a convex row
                    // bounded above or a concave one bounded below; the second
                    // differences are the row's curvature along the line, so
                    // an equality, a ranged row or the wrong sign declines
                    // before anything is built.
                    let sign = match (is_free(cl[i]), is_free(cu[i])) {
                        (true, false) => 1.0,
                        (false, true) => -1.0,
                        _ => return Err((Decline::NonconvexRow(i), f_evals)),
                    };
                    if sign * d2a < -tol || sign * d2b < -tol {
                        return Err((Decline::NonconvexRow(i), f_evals));
                    }
                    row_quadratic[i] = true;
                }
            }
            line_c.extend(cs);
        }
        for (x, f) in xs.into_iter().zip(fs[1..].iter()) {
            line_points.push((x, *f));
        }
    }
    // A quadratic row is handed over only when it keeps the feasible set
    // convex: a convex function bounded above, or a concave one bounded below
    // (`sign` says which side its Hessian must be definite on). An equality
    // or a ranged quadratic row makes a nonconvex set, where the Newton path
    // could end in another basin than the quasi-Newton path (HS44's lesson).
    let mut qrows: Vec<usize> = Vec::new();
    let mut qrow_sign: Vec<f64> = Vec::new();
    for i in 0..m {
        if row_quadratic[i] {
            match (is_free(cl[i]), is_free(cu[i])) {
                (true, false) => {
                    qrows.push(i);
                    qrow_sign.push(1.0);
                }
                (false, true) => {
                    qrows.push(i);
                    qrow_sign.push(-1.0);
                }
                _ => return Err((Decline::NonconvexRow(i), f_evals)),
            }
        }
    }
    let nq = qrows.len();
    if nq.saturating_mul(n).saturating_mul(n) > ROW_BUILD_MAX_ENTRIES {
        return Err((Decline::TooLarge(n), f_evals));
    }
    let rows_by_jacobian = nq > 0 && caps.jacobian;

    // Build the Hessians and the gradients at x0.
    let build_by_gradient = caps.gradient;
    let structured = opts.quadratic_build == QuadraticBuild::Structured;
    // What the objective's first phase costs: `n` gradient calls, the
    // diagonal alone (`2n` function values, which also give the gradient)
    // under the structured build, or the whole dense build.
    let build_cost: u64 = if build_by_gradient {
        0
    } else if structured {
        2 * n as u64
    } else {
        (n * (n + 3) / 2) as u64
    };
    let size_limit = if build_by_gradient || structured {
        GRADIENT_BUILD_MAX_N
    } else {
        FD_BUILD_MAX_N
    };
    if n > size_limit {
        return Err((Decline::TooLarge(n), f_evals));
    }
    if let Some(limit) = opts.max_evaluations {
        if f_evals + build_cost + 8 > limit {
            return Err((Decline::Budget(build_cost), f_evals));
        }
    }
    // The build must leave at least half of a wall-time budget to the solve;
    // the probe's own evaluations say what one costs. The rows' first phase
    // costs `n` Jacobian or `2n` constraint evaluations more.
    if let Some(limit) = opts.max_seconds {
        let per_eval = probe_clock.elapsed().as_secs_f64() / f_evals as f64;
        let row_cost: u64 = match (nq > 0, rows_by_jacobian) {
            (false, _) => 0,
            (true, true) => n as u64,
            (true, false) => 2 * n as u64,
        };
        let predicted = per_eval
            * (build_cost.max(if build_by_gradient { n as u64 } else { 0 }) + row_cost) as f64;
        if predicted > 0.5 * limit {
            return Err((Decline::Budget(build_cost + row_cost), f_evals));
        }
    }
    let step = |j: usize| side[j] * h[j].max(1e-8 * x0[j].abs().max(1.0));
    // The six line points relative to x0, for the model check.
    let dxs: Vec<Vec<f64>> = line_points
        .iter()
        .map(|(x, _)| x.iter().zip(&x0).map(|(a, b)| a - b).collect())
        .collect();
    let wall_deadline = opts.max_seconds.map(|l| 0.5 * l);

    // The objective.
    let mut hess = vec![0.0; n * n]; // full, row-major H[i * n + j]
    let mut g0 = vec![0.0; n];
    // The structure the build ended with: the half-bandwidth it stopped at
    // (0 diagonal), or `None` for dense.
    let mut bandwidth: Option<usize> = None;
    if build_by_gradient {
        let mut grad = |x: &[f64], out: &mut [f64]| -> Result<(), (Decline, u64)> {
            g_evals += 1;
            nlp.gradient(x, out)
                .map_err(|e| (Decline::Failed(e.to_string()), f_evals))?;
            if out.iter().any(|v| !v.is_finite()) {
                return Err((Decline::Failed("non-finite gradient".into()), f_evals));
            }
            Ok(())
        };
        grad(&x0, &mut g0)?;
        let mut gj = vec![0.0; n];
        let mut x = x0.clone();
        for j in 0..n {
            let hj = step(j);
            x[j] = x0[j] + hj;
            grad(&x, &mut gj)?;
            x[j] = x0[j];
            for i in 0..n {
                hess[i * n + j] = (gj[i] - g0[i]) / hj;
            }
        }
        // Symmetrise: a quadratic's gradient differences are exactly symmetric
        // up to rounding.
        for i in 0..n {
            for j in 0..i {
                let v = 0.5 * (hess[i * n + j] + hess[j * n + i]);
                hess[i * n + j] = v;
                hess[j * n + i] = v;
            }
        }
    } else {
        let line_f: Vec<Vec<f64>> = line_points.iter().map(|(_, f)| vec![*f]).collect();
        let built = fd_build(
            |x| match nlp.objective(x) {
                Ok(v) if v.is_finite() => Ok(vec![v]),
                Ok(_) => Err(Decline::Failed("non-finite objective".into())),
                Err(e) => Err(Decline::Failed(e.to_string())),
            },
            &FdBuildSpec {
                n,
                x0: &x0,
                f0: &[f0],
                step: &step,
                dxs: &dxs,
                line_vals: &line_f,
                signs: &[1.0],
                structured,
                eval_limit: opts.max_evaluations.map(|l| l.saturating_sub(f_evals + 8)),
                clock: &probe_clock,
                wall_deadline,
                evals_before: f_evals,
            },
        );
        let built = match built {
            Ok(b) => b,
            Err((why, spent)) => return Err((why, f_evals + spent)),
        };
        f_evals += built.evals;
        hess = built.hess.into_iter().next().unwrap_or_default();
        g0 = built.g0.into_iter().next().unwrap_or_default();
        bandwidth = built.bandwidth;
    }

    // Only a convex quadratic is handed over: every KKT point of a convex QP is
    // its global minimum, so the Newton path cannot end in a different basin
    // from the quasi-Newton one (HS44, a nonconvex QP with several local
    // minima, did exactly that in the first ablation, `abl-i5-rejected`).
    if !is_positive_semidefinite(&hess, n, bandwidth.unwrap_or(n - 1), 1.0) {
        return Err((Decline::NotConvex, f_evals));
    }

    // The model must reproduce the six line points, which did not build it
    // (the structured build checked this after every band; the check is
    // repeated on the final matrix for the record).
    let q: Vec<f64> = dxs
        .iter()
        .map(|dx| f0 + quadratic_increment(&g0, &hess, n, dx))
        .collect();
    let mut worst = match model_fit(&line_points, &q, f0) {
        Ok(worst) => worst,
        Err(err) => return Err((Decline::ModelMisfit(err), f_evals)),
    };

    // The quadratic rows, all at once from the constraint vector (an
    // evaluation gives every row), by the same structured differencing; a
    // row's Hessian must be definite on the side its bound needs.
    let mut row_hess: Vec<Vec<f64>> = Vec::new();
    let mut row_bandwidth: Option<usize> = Some(0);
    let mut row_build_evals: u64 = 0;
    if rows_by_jacobian {
        let built = match jacobian_row_build(nlp, &x0, &step, &qrows, &mut j_evals) {
            Ok(b) => b,
            Err(why) => return Err((why, f_evals)),
        };
        // The rows' models must reproduce the constraint values at the line
        // points, which did not build them (a wrong Jacobian fails here).
        for (k, &i) in qrows.iter().enumerate() {
            let q: Vec<f64> = dxs
                .iter()
                .map(|dx| c0[i] + quadratic_increment(&built.g0[k], &built.hess[k], n, dx))
                .collect();
            let points: Vec<(Vec<f64>, f64)> = line_c.iter().map(|c| (Vec::new(), c[i])).collect();
            match model_fit(&points, &q, c0[i]) {
                Ok(w) => worst = worst.max(w),
                Err(err) => return Err((Decline::ModelMisfit(err), f_evals)),
            }
        }
        row_bandwidth = built.bandwidth;
        for (k, hk) in built.hess.iter().enumerate() {
            let bw = built.bandwidth.unwrap_or(n - 1);
            if !is_positive_semidefinite(hk, n, bw, qrow_sign[k]) {
                return Err((Decline::NonconvexRow(qrows[k]), f_evals));
            }
        }
        row_hess = built.hess;
    } else if nq > 0 && rows_mode == QuadraticRows::Values {
        let c0q: Vec<f64> = qrows.iter().map(|&i| c0[i]).collect();
        let line_q: Vec<Vec<f64>> = line_c
            .iter()
            .map(|c| qrows.iter().map(|&i| c[i]).collect())
            .collect();
        let built = fd_build(
            |x| {
                let mut c = vec![0.0; m];
                nlp.constraints(x, &mut c)
                    .map_err(|e| Decline::Failed(e.to_string()))?;
                if c.iter().any(|v| !v.is_finite()) {
                    return Err(Decline::Failed("non-finite constraint".into()));
                }
                Ok(qrows.iter().map(|&i| c[i]).collect())
            },
            &FdBuildSpec {
                n,
                x0: &x0,
                f0: &c0q,
                step: &step,
                dxs: &dxs,
                line_vals: &line_q,
                signs: &qrow_sign,
                structured,
                eval_limit: None,
                clock: &probe_clock,
                wall_deadline,
                evals_before: c_evals,
            },
        );
        let built = match built {
            Ok(b) => b,
            Err((why, _spent)) => return Err((why, f_evals)),
        };
        c_evals += built.evals;
        row_build_evals = built.evals;
        row_bandwidth = built.bandwidth;
        worst = worst.max(built.worst);
        for (k, hk) in built.hess.iter().enumerate() {
            let bw = built.bandwidth.unwrap_or(n - 1);
            if !is_positive_semidefinite(hk, n, bw, qrow_sign[k]) {
                return Err((Decline::NonconvexRow(qrows[k]), f_evals));
            }
        }
        row_hess = built.hess;
    } else if nq > 0 {
        // Unreachable: `rows_mode` is `Off` only when no row was accepted.
        return Err((Decline::NotLinear(0, qrows[0]), f_evals));
    }

    // Lower triangle, column-major, within the widest structure found.
    let half_band = match (bandwidth, row_bandwidth) {
        (Some(a), Some(b)) => a.max(b),
        _ => n - 1,
    };
    let mut triplets = Vec::with_capacity(n * (half_band + 1));
    for j in 0..n {
        for i in j..n.min(j + half_band + 1) {
            triplets.push((i, j));
        }
    }
    let structure =
        Sparsity::from_triplets(n, n, &triplets).map_err(|e| (Decline::Failed(e), f_evals))?;
    let lower = |dense: &[f64]| -> Vec<f64> {
        let mut out = Vec::with_capacity(structure.nnz());
        for j in 0..n {
            for &i in structure.col(j) {
                out.push(dense[i * n + j]);
            }
        }
        out
    };
    let hess_lower = lower(&hess);
    let rows: Vec<(usize, Vec<f64>)> = qrows
        .iter()
        .zip(&row_hess)
        .map(|(&i, hk)| (i, lower(hk)))
        .collect();
    let shape = |b: Option<usize>| match b {
        Some(0) => "diagonal".to_string(),
        Some(b) if b + 1 < n => format!("half-bandwidth {b}"),
        _ => "dense".to_string(),
    };
    let how = if build_by_gradient {
        format!("{g_evals} gradient evaluations, dense")
    } else {
        format!(
            "{} objective evaluations, {}",
            f_evals - 7,
            shape(bandwidth)
        )
    };
    let rows_text = if nq > 0 {
        format!(
            " and {nq} quadratic constraint row{} (from {}, {})",
            if nq == 1 { "" } else { "s" },
            if rows_by_jacobian {
                format!("{j_evals} Jacobian evaluations")
            } else {
                format!("{row_build_evals} constraint evaluations")
            },
            shape(row_bandwidth)
        )
    } else {
        String::new()
    };
    let note = if nq > 0 {
        format!(
            "Quadratic objective, linear and convex quadratic constraint rows detected at the start (7 evaluations along two lines); the constant Hessians were built by differencing ({how}){rows_text} and verified at the line points (worst fit {worst:.1e} of the tolerance), and the SQP member ran with them."
        )
    } else {
        format!(
            "Quadratic objective and linear constraints detected at the start (7 evaluations along two lines); its constant Hessian was built by differencing ({how}) and verified at the line points (worst fit {worst:.1e} of the tolerance), and the SQP member ran with it."
        )
    };
    Ok(QuadraticModel {
        inner: nlp,
        n,
        hess_lower,
        rows,
        structure,
        f_evals,
        c_evals,
        g_evals,
        j_evals,
        note,
    })
}

/// The quadratic rows' Hessians from differencing the model's Jacobian along
/// each coordinate (`n + 1` calls for every row at once), symmetrised; the
/// half-bandwidth is that of the exact nonzeros (a constant Jacobian entry
/// differences to exactly zero).
fn jacobian_row_build<P: Nlp + ?Sized>(
    nlp: &P,
    x0: &[f64],
    step: &dyn Fn(usize) -> f64,
    qrows: &[usize],
    j_evals: &mut u64,
) -> Result<Built, Decline> {
    let NlpDims { n, m } = nlp.dims();
    let dense;
    let pat: &Sparsity = if let Some(p) = nlp.jacobian_structure() {
        p
    } else {
        dense = Sparsity::dense(m, n);
        &dense
    };
    let nnz = pat.nnz();
    let mut slot = vec![usize::MAX; m];
    for (k, &i) in qrows.iter().enumerate() {
        slot[i] = k;
    }
    let mut jac = |x: &[f64], out: &mut [f64]| -> Result<(), Decline> {
        *j_evals += 1;
        nlp.jacobian(x, out)
            .map_err(|e| Decline::Failed(e.to_string()))?;
        if out.iter().any(|v| !v.is_finite()) {
            return Err(Decline::Failed("non-finite Jacobian".into()));
        }
        Ok(())
    };
    let nq = qrows.len();
    let mut j0 = vec![0.0; nnz];
    jac(x0, &mut j0)?;
    let mut g0 = vec![vec![0.0; n]; nq];
    for col in 0..n {
        let base = pat.col_ptr()[col];
        for (p, &row) in pat.col(col).iter().enumerate() {
            if slot[row] != usize::MAX {
                g0[slot[row]][col] = j0[base + p];
            }
        }
    }
    let mut hess = vec![vec![0.0; n * n]; nq];
    let mut jj = vec![0.0; nnz];
    let mut x = x0.to_vec();
    for j in 0..n {
        let hj = step(j);
        x[j] = x0[j] + hj;
        jac(&x, &mut jj)?;
        x[j] = x0[j];
        for col in 0..n {
            let base = pat.col_ptr()[col];
            for (p, &row) in pat.col(col).iter().enumerate() {
                if slot[row] != usize::MAX {
                    hess[slot[row]][col * n + j] = (jj[base + p] - j0[base + p]) / hj;
                }
            }
        }
    }
    let mut band = 0usize;
    for hk in &mut hess {
        for i in 0..n {
            for j in 0..i {
                let v = 0.5 * (hk[i * n + j] + hk[j * n + i]);
                hk[i * n + j] = v;
                hk[j * n + i] = v;
                if v != 0.0 {
                    band = band.max(i - j);
                }
            }
        }
    }
    Ok(Built {
        hess,
        g0,
        bandwidth: Some(band),
        worst: 0.0,
        evals: *j_evals,
    })
}

/// `g' dx + 0.5 dx' H dx` for a dense row-major `H`.
fn quadratic_increment(g: &[f64], hess: &[f64], n: usize, dx: &[f64]) -> f64 {
    let mut v = 0.0;
    for i in 0..n {
        v += g[i] * dx[i];
        let mut hv = 0.0;
        for j in 0..n {
            hv += hess[i * n + j] * dx[j];
        }
        v += 0.5 * dx[i] * hv;
    }
    v
}

/// Inputs of a function-value build of `k` quadratic functions sampled
/// together (the objective alone, or every quadratic constraint row).
struct FdBuildSpec<'a> {
    n: usize,
    x0: &'a [f64],
    /// The `k` values at `x0`.
    f0: &'a [f64],
    step: &'a dyn Fn(usize) -> f64,
    /// The line points relative to `x0`.
    dxs: &'a [Vec<f64>],
    /// The `k` sampled values at each line point.
    line_vals: &'a [Vec<f64>],
    /// Per function, the side its Hessian must be definite on: `1.0`
    /// positive semidefinite, `-1.0` negative semidefinite.
    signs: &'a [f64],
    structured: bool,
    /// Evaluations this build may spend, when they are budgeted.
    eval_limit: Option<u64>,
    clock: &'a std::time::Instant,
    /// Seconds on `clock` after which no further band is started.
    wall_deadline: Option<f64>,
    /// Evaluations of this kind the probe spent before the build, so that
    /// `clock` over the running total prices one.
    evals_before: u64,
}

/// What a function-value build produced: per function the dense Hessian and
/// the gradient at `x0`, the half-bandwidth the build stopped at (`None`
/// dense), the worst line-point fit and the evaluations spent.
struct Built {
    hess: Vec<Vec<f64>>,
    g0: Vec<Vec<f64>>,
    bandwidth: Option<usize>,
    worst: f64,
    evals: u64,
}

/// The function-value build: the diagonal first (`2n` evaluations, which also
/// give the gradients), then, under the structured build, one off-diagonal
/// band at a time with every function's model checked against the line
/// points and its convexity after each band, so the build stops at the first
/// structure that passes both and a dense Hessian costs exactly the dense
/// build. A band model that reproduces the line points to 1e-8 can still
/// fail the convexity check at 1e-10 when the true Hessian is
/// ill-conditioned with a decaying tail (box_lsq's Gaussian kernel: fitted at
/// half-bandwidth 23, convex only with every band), which is why the search
/// goes on until both pass. On a failure the error carries the evaluations
/// spent.
#[allow(clippy::too_many_lines)]
fn fd_build<E>(mut eval: E, spec: &FdBuildSpec<'_>) -> Result<Built, (Decline, u64)>
where
    E: FnMut(&[f64]) -> Result<Vec<f64>, Decline>,
{
    let n = spec.n;
    let k = spec.f0.len();
    let mut evals = 0u64;
    let mut hess = vec![vec![0.0; n * n]; k];
    let mut g0 = vec![vec![0.0; n]; k];
    let mut x = spec.x0.to_vec();
    let mut fj = vec![vec![0.0; n]; k];
    let mut ev = |x: &[f64], evals: &mut u64| -> Result<Vec<f64>, (Decline, u64)> {
        *evals += 1;
        eval(x).map_err(|why| (why, *evals))
    };
    for j in 0..n {
        let hj = (spec.step)(j);
        x[j] = spec.x0[j] + hj;
        let v1 = ev(&x, &mut evals)?;
        x[j] = spec.x0[j] + 2.0 * hj;
        let v2 = ev(&x, &mut evals)?;
        x[j] = spec.x0[j];
        for f in 0..k {
            fj[f][j] = v1[f];
            let hjj = (v2[f] - 2.0 * v1[f] + spec.f0[f]) / (hj * hj);
            hess[f][j * n + j] = hjj;
            g0[f][j] = (v1[f] - spec.f0[f]) / hj - 0.5 * hj * hjj;
        }
    }
    // The models' values at the line points, accumulated band by band.
    let mut q: Vec<Vec<f64>> = (0..k)
        .map(|f| {
            spec.dxs
                .iter()
                .map(|dx| {
                    let mut v = spec.f0[f];
                    for i in 0..n {
                        v += g0[f][i] * dx[i] + 0.5 * hess[f][i * n + i] * dx[i] * dx[i];
                    }
                    v
                })
                .collect()
        })
        .collect();
    let points_of = |f: usize| -> Vec<(Vec<f64>, f64)> {
        spec.line_vals
            .iter()
            .map(|vals| (Vec::new(), vals[f]))
            .collect()
    };
    let points: Vec<Vec<(Vec<f64>, f64)>> = (0..k).map(points_of).collect();
    let check = |hess: &[Vec<f64>], q: &[Vec<f64>], band: usize| -> (Result<f64, f64>, bool) {
        let mut worst = 0.0_f64;
        let mut fit: Result<f64, f64> = Ok(0.0);
        let mut convex = true;
        for f in 0..k {
            match model_fit(&points[f], &q[f], spec.f0[f]) {
                Ok(w) => worst = worst.max(w),
                Err(e) => {
                    fit = Err(e);
                    break;
                }
            }
        }
        if fit.is_ok() {
            fit = Ok(worst);
            convex = (0..k).all(|f| is_positive_semidefinite(&hess[f], n, band, spec.signs[f]));
        }
        (fit, convex)
    };
    // One pair (i, j): its mixed second differences, one per function.
    let mut pair = |i: usize, j: usize, x: &mut Vec<f64>, evals: &mut u64| {
        let (hi, hj) = ((spec.step)(i), (spec.step)(j));
        x[i] = spec.x0[i] + hi;
        x[j] = spec.x0[j] + hj;
        let vij = ev(x, evals);
        x[i] = spec.x0[i];
        x[j] = spec.x0[j];
        vij.map(|v| {
            (0..k)
                .map(|f| (v[f] - fj[f][i] - fj[f][j] + spec.f0[f]) / (hi * hj))
                .collect::<Vec<f64>>()
        })
    };
    let mut bandwidth: Option<usize> = None;
    let worst;
    if spec.structured {
        let mut band = 0usize;
        let (mut fit, mut convex) = check(&hess, &q, band);
        if !(fit.is_ok() && convex) {
            // Whether the dense build is affordable; when it is not, the band
            // search may spend as much again as the diagonal did (`2n`, four
            // gradients' worth) before it declines.
            let dense_rest = (n * (n - 1) / 2) as u64;
            let per_eval =
                spec.clock.elapsed().as_secs_f64() / (spec.evals_before + evals).max(1) as f64;
            let dense_ok = n <= FD_BUILD_MAX_N
                && spec.eval_limit.is_none_or(|l| evals + dense_rest <= l)
                && spec
                    .wall_deadline
                    .is_none_or(|l| per_eval * dense_rest as f64 <= l);
            let band_budget: u64 = if dense_ok { dense_rest } else { 2 * n as u64 };
            let mut band_spent: u64 = 0;
            while !(fit.is_ok() && convex) {
                band += 1;
                let cost = (n - band) as u64;
                if band >= n || band_spent + cost > band_budget {
                    return Err((
                        if fit.is_ok() {
                            Decline::NotConvex
                        } else {
                            Decline::Unstructured(band - 1)
                        },
                        evals,
                    ));
                }
                if let Some(limit) = spec.eval_limit {
                    if evals + cost > limit {
                        return Err((Decline::Budget(cost), evals));
                    }
                }
                if let Some(limit) = spec.wall_deadline {
                    if spec.clock.elapsed().as_secs_f64() > limit {
                        return Err((Decline::Budget(cost), evals));
                    }
                }
                for i in 0..n - band {
                    let j = i + band;
                    let v = pair(i, j, &mut x, &mut evals)?;
                    for f in 0..k {
                        hess[f][i * n + j] = v[f];
                        hess[f][j * n + i] = v[f];
                        for (qp, dx) in q[f].iter_mut().zip(spec.dxs) {
                            *qp += v[f] * dx[i] * dx[j];
                        }
                    }
                }
                band_spent += cost;
                (fit, convex) = check(&hess, &q, band);
            }
        }
        bandwidth = Some(band);
        worst = fit.unwrap_or(0.0);
    } else {
        for j in 0..n {
            for i in 0..j {
                let v = pair(i, j, &mut x, &mut evals)?;
                for f in 0..k {
                    hess[f][i * n + j] = v[f];
                    hess[f][j * n + i] = v[f];
                    for (qp, dx) in q[f].iter_mut().zip(spec.dxs) {
                        *qp += v[f] * dx[i] * dx[j];
                    }
                }
            }
        }
        let (fit, _) = check(&hess, &q, n - 1);
        worst = match fit {
            Ok(w) => w,
            Err(e) => return Err((Decline::ModelMisfit(e), evals)),
        };
    }
    Ok(Built {
        hess,
        g0,
        bandwidth,
        worst,
        evals,
    })
}

/// Whether the model's values `q` at the line points reproduce the sampled
/// objective values: `Ok(worst fit as a fraction of the tolerance)`, or
/// `Err(the first error that exceeds it)`.
fn model_fit(line_points: &[(Vec<f64>, f64)], q: &[f64], f0: f64) -> Result<f64, f64> {
    let mut worst = 0.0_f64;
    for ((_, f), q) in line_points.iter().zip(q) {
        let var = (f - f0).abs().max((q - f0).abs());
        let tol = REL_TOL * var + ABS_TOL * f0.abs().max(f.abs());
        let err = (f - q).abs();
        worst = worst.max(if tol > 0.0 { err / tol } else { 0.0 });
        if err > tol {
            return Err(err);
        }
    }
    Ok(worst)
}

/// Cholesky of `H + delta I` with `delta = 1e-10 max(1, max |H_ii|)`: succeeds
/// exactly when the smallest eigenvalue is above `-delta`, i.e. the quadratic is
/// convex up to rounding (a singular positive semidefinite Hessian, as in a
/// rank-deficient least-squares fit, passes).
///
/// `half_band` is the matrix's half-bandwidth (`n - 1` for a dense one): the
/// factor has the same band, so the check costs `O(n b^2)` and its storage
/// `n (b + 1)`, which keeps a diagonal check at `n = 5000` trivial.
///
/// `sign` is `1.0` for positive semidefinite and `-1.0` to test `-H`
/// instead (a concave constraint row bounded below).
fn is_positive_semidefinite(hess: &[f64], n: usize, half_band: usize, sign: f64) -> bool {
    let b = half_band.min(n - 1);
    let w = b + 1;
    let mut max_diag = 1.0_f64;
    for i in 0..n {
        max_diag = max_diag.max(hess[i * n + i].abs());
    }
    let delta = 1e-10 * max_diag;
    // `l[i * w + (i - k)]` holds `L_ik` for `i - b <= k <= i`.
    let mut l = vec![0.0; n * w];
    for j in 0..n {
        let mut d = sign * hess[j * n + j] + delta;
        for k in j.saturating_sub(b)..j {
            let v = l[j * w + (j - k)];
            d -= v * v;
        }
        if d <= 0.0 || !d.is_finite() {
            return false;
        }
        let ljj = d.sqrt();
        l[j * w] = ljj;
        for i in j + 1..n.min(j + w) {
            let mut s = sign * hess[i * n + j];
            for k in i.saturating_sub(b)..j {
                s -= l[i * w + (i - k)] * l[j * w + (j - k)];
            }
            l[i * w + (i - j)] = s / ljj;
        }
    }
    true
}

impl<P: Nlp + ?Sized> Nlp for QuadraticModel<'_, P> {
    fn dims(&self) -> NlpDims {
        self.inner.dims()
    }
    fn x_bounds(&self) -> (&[f64], &[f64]) {
        self.inner.x_bounds()
    }
    fn c_bounds(&self) -> (&[f64], &[f64]) {
        self.inner.c_bounds()
    }
    fn x0(&self) -> &[f64] {
        self.inner.x0()
    }
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            hessian: true,
            hessian_vector: true,
            ..self.inner.capabilities()
        }
    }
    fn objective(&self, x: &[f64]) -> Result<f64, EvalError> {
        self.inner.objective(x)
    }
    fn constraints(&self, x: &[f64], out: &mut [f64]) -> Result<(), EvalError> {
        self.inner.constraints(x, out)
    }
    fn objective_batch(&self, xs: &[f64]) -> Vec<Result<f64, EvalError>> {
        self.inner.objective_batch(xs)
    }
    fn constraints_batch(&self, xs: &[f64], out: &mut [f64]) -> Vec<Result<(), EvalError>> {
        self.inner.constraints_batch(xs, out)
    }
    fn gradient(&self, x: &[f64], out: &mut [f64]) -> Result<(), EvalError> {
        self.inner.gradient(x, out)
    }
    fn jacobian_structure(&self) -> Option<&Sparsity> {
        self.inner.jacobian_structure()
    }
    fn jacobian(&self, x: &[f64], out: &mut [f64]) -> Result<(), EvalError> {
        self.inner.jacobian(x, out)
    }
    fn hessian_structure(&self) -> Option<&Sparsity> {
        Some(&self.structure)
    }
    fn hessian_lagrangian(
        &self,
        _x: &[f64],
        sigma: f64,
        lambda: &[f64],
        out: &mut [f64],
    ) -> Result<(), EvalError> {
        // Every Hessian is constant: sigma times the objective's plus each
        // quadratic row's times its multiplier (linear rows contribute nothing).
        for (o, h) in out.iter_mut().zip(&self.hess_lower) {
            *o = sigma * h;
        }
        for (row, hr) in &self.rows {
            let l = lambda.get(*row).copied().unwrap_or(0.0);
            if l != 0.0 {
                for (o, h) in out.iter_mut().zip(hr) {
                    *o += l * h;
                }
            }
        }
        Ok(())
    }
    fn hessian_vector(
        &self,
        _x: &[f64],
        sigma: f64,
        lambda: &[f64],
        v: &[f64],
        out: &mut [f64],
    ) -> Result<(), EvalError> {
        out.iter_mut().for_each(|o| *o = 0.0);
        let n = self.n;
        let mut add = |weight: f64, lower: &[f64]| {
            for j in 0..n {
                let col = self.structure.col(j);
                let base = self.structure.col_ptr()[j];
                for (k, &i) in col.iter().enumerate() {
                    let hij = weight * lower[base + k];
                    out[i] += hij * v[j];
                    if i != j {
                        out[j] += hij * v[i];
                    }
                }
            }
        };
        add(sigma, &self.hess_lower);
        for (row, hr) in &self.rows {
            let l = lambda.get(*row).copied().unwrap_or(0.0);
            if l != 0.0 {
                add(l, hr);
            }
        }
        Ok(())
    }
    fn typical_x(&self) -> Option<&[f64]> {
        self.inner.typical_x()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Problem;

    fn dense_qp(n: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
        // H = A'A + I with a fixed pseudo-random A, c random.
        let mut rng = XorShift(12345);
        let a: Vec<Vec<f64>> = (0..n + 5)
            .map(|_| (0..n).map(|_| 2.0 * rng.next() - 1.0).collect())
            .collect();
        let mut h = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                h[i][j] = a.iter().map(|r| r[i] * r[j]).sum::<f64>() + f64::from(u8::from(i == j));
            }
        }
        let c: Vec<f64> = (0..n).map(|_| 2.0 * rng.next() - 1.0).collect();
        (h, c)
    }

    #[test]
    fn a_bounded_quadratic_is_recognised_and_its_hessian_is_exact() {
        let n = 6;
        let (h, c) = dense_qp(n);
        let (h2, c2) = (h.clone(), c.clone());
        let p = Problem::new(n, move |x| {
            let mut v = 0.0;
            for i in 0..n {
                v += c2[i] * x[i];
                for j in 0..n {
                    v += 0.5 * x[i] * h2[i][j] * x[j];
                }
            }
            v
        })
        .start_at(&[0.3, -0.2, 0.0, 0.5, 0.1, -0.4])
        .lower_bounds(&[-1.0; 6])
        .upper_bounds(&[1.0; 6]);
        let opts = Options::default();
        let q = probe(&p, &opts).expect("a quadratic must be recognised");
        assert_eq!(q.g_evals, 0);
        assert_eq!(q.f_evals, 7 + (n * (n + 3) / 2) as u64);
        let mut out = vec![0.0; n * (n + 1) / 2];
        q.hessian_lagrangian(&[0.0; 6], 1.0, &[], &mut out).unwrap();
        let s = q.hessian_structure().unwrap();
        for j in 0..n {
            for (k, &i) in s.col(j).iter().enumerate() {
                let got = out[s.col_ptr()[j] + k];
                assert!(
                    (got - h[i][j]).abs() < 1e-7 * h[i][j].abs().max(1.0),
                    "({i},{j}) {got} vs {}",
                    h[i][j]
                );
            }
        }
    }

    fn structured() -> Options {
        Options {
            quadratic_build: QuadraticBuild::Structured,
            ..Options::default()
        }
    }

    /// The structured build with quadratic constraint rows accepted from
    /// constraint values (the default since `abl-i5-rows`).
    fn with_rows() -> Options {
        Options {
            quadratic_rows: QuadraticRows::Values,
            ..structured()
        }
    }

    /// The projection onto an ellipsoid: a diagonal quadratic objective and a
    /// diagonal convex row bounded above.
    fn ellipsoid_projection(n: usize) -> Problem<'static> {
        let p_pt: Vec<f64> = (0..n).map(|i| 1.5 - 0.4 * (i % 5) as f64).collect();
        let r: Vec<f64> = (0..n).map(|i| 0.5 + 0.3 * (i % 6) as f64).collect();
        Problem::new(n, move |x| {
            x.iter().zip(&p_pt).map(|(a, b)| (a - b) * (a - b)).sum()
        })
        .start_at(&vec![0.0; n])
        .inequality(1, move |x, c| {
            c[0] = x
                .iter()
                .zip(&r)
                .map(|(a, b)| (a / b) * (a / b))
                .sum::<f64>()
                - 1.0;
        })
    }

    fn band_qp(n: usize, half_band: usize) -> Problem<'static> {
        // H_ii = 2 + i/n, H_{i,i+k} = 0.3 / k for k <= half_band: diagonally
        // dominant, so convex.
        Problem::new(n, move |x| {
            let mut v = 0.0;
            for i in 0..n {
                v += 0.5 * (2.0 + i as f64 / n as f64) * x[i] * x[i] - 0.7 * x[i];
                for k in 1..=half_band {
                    if i + k < n {
                        v += 0.3 / k as f64 * x[i] * x[i + k];
                    }
                }
            }
            v
        })
        .start_at(&vec![0.4; n])
        .lower_bounds(&vec![-1.0; n])
        .upper_bounds(&vec![1.0; n])
    }

    #[test]
    fn the_structured_build_stops_at_the_bandwidth_it_finds() {
        // Diagonal: 2n evaluations after the probe's 7; tridiagonal: 3n - 1;
        // pentadiagonal: 4n - 3. Every model is exact.
        for (half_band, expected) in [(0, 2 * 12), (1, 3 * 12 - 1), (2, 4 * 12 - 3)] {
            let p = band_qp(12, half_band);
            let q = probe(&p, &structured()).expect("a banded QP must be recognised");
            assert_eq!(q.f_evals, 7 + expected as u64, "half-bandwidth {half_band}");
            assert!(q.note.contains(if half_band == 0 {
                "diagonal"
            } else {
                "half-bandwidth"
            }));
            let s = q.hessian_structure().unwrap();
            assert_eq!(
                s.nnz(),
                12 * (half_band + 1) - half_band * (half_band + 1) / 2
            );
            let r = mincon_sqp::solve(
                &q,
                &Options {
                    algorithm: mincon_core::Algorithm::Sqp,
                    ..structured()
                },
            )
            .unwrap();
            assert!(r.exit_flag.is_success(), "{:?}", r.exit_flag);
            assert!(r.iterations <= 3, "iterations {}", r.iterations);
        }
    }

    #[test]
    fn a_dense_quadratic_costs_the_same_under_either_build() {
        let n = 6;
        let (h, c) = dense_qp(n);
        let p = Problem::new(n, move |x| {
            let mut v = 0.0;
            for i in 0..n {
                v += c[i] * x[i];
                for j in 0..n {
                    v += 0.5 * x[i] * h[i][j] * x[j];
                }
            }
            v
        })
        .start_at(&[0.3, -0.2, 0.0, 0.5, 0.1, -0.4]);
        let q = probe(&p, &structured()).expect("a dense QP must be recognised");
        assert_eq!(q.f_evals, 7 + (n * (n + 3) / 2) as u64);
        assert!(q.note.contains("dense"));
        assert_eq!(q.hessian_structure().unwrap().nnz(), n * (n + 1) / 2);
    }

    #[test]
    fn an_ill_conditioned_kernel_is_built_until_it_is_convex() {
        // box_lsq's shape: 0.5 |K x - y|^2 with a Gaussian kernel, whose
        // Hessian K'K decays away from the diagonal. A band model fits the
        // line points long before the truncated matrix passes the convexity
        // check; the structured build must go on and hand the model over.
        let n = 24;
        let k: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let row: Vec<f64> = (0..n)
                    .map(|j| (-0.5 * ((i as f64 - j as f64) / 3.0).powi(2)).exp())
                    .collect();
                let s: f64 = row.iter().sum();
                row.into_iter().map(|v| v / s).collect()
            })
            .collect();
        let y: Vec<f64> = (0..n)
            .map(|i| if (8..14).contains(&i) { 0.8 } else { 0.1 })
            .collect();
        let p = Problem::new(n, move |x| {
            let mut v = 0.0;
            for i in 0..n {
                let r: f64 = k[i].iter().zip(x).map(|(a, b)| a * b).sum::<f64>() - y[i];
                v += 0.5 * r * r;
            }
            v
        })
        .start_at(&vec![0.5; n])
        .lower_bounds(&vec![0.0; n])
        .upper_bounds(&vec![1.0; n]);
        let dense = probe(
            &p,
            &Options {
                quadratic_build: QuadraticBuild::Dense,
                ..Options::default()
            },
        )
        .expect("the dense build accepts it");
        let q = probe(&p, &structured()).expect("the structured build must accept it too");
        assert_eq!(q.f_evals, dense.f_evals, "{}", q.note);
        assert!(q.note.contains("dense"), "{}", q.note);
    }

    #[test]
    fn a_linear_objective_is_a_zero_hessian_built_from_the_diagonal_alone() {
        // MANY_INEQ's shape: the build finds a zero diagonal that fits.
        let p = Problem::new(5, |x| -x.iter().sum::<f64>())
            .start_at(&[0.0; 5])
            .lower_bounds(&[-1.0; 5])
            .upper_bounds(&[1.0; 5]);
        let q = probe(&p, &structured()).expect("a linear objective is a quadratic");
        assert_eq!(q.f_evals, 7 + 10);
        let r = mincon_sqp::solve(
            &q,
            &Options {
                algorithm: mincon_core::Algorithm::Sqp,
                ..structured()
            },
        )
        .unwrap();
        assert!(r.exit_flag.is_success(), "{:?}", r.exit_flag);
        assert!((r.solution.f + 5.0).abs() < 1e-8, "f {}", r.solution.f);
    }

    #[test]
    fn above_the_dense_limit_a_diagonal_builds_and_a_dense_one_declines_cheaply() {
        let n = 600;
        let diag = band_qp(n, 0);
        let q = probe(&diag, &structured()).expect("a diagonal QP at n = 600");
        assert_eq!(q.f_evals, 7 + 2 * n as u64);
        let dense_build = Options {
            quadratic_build: QuadraticBuild::Dense,
            ..Options::default()
        };
        assert!(
            probe(&diag, &dense_build).is_none(),
            "the dense build declines at n = 600"
        );
        // A dense coupling: the band search may spend 2n beyond the diagonal.
        let dense = Problem::new(n, move |x| {
            let s: f64 = x.iter().sum();
            x.iter().map(|v| v * v).sum::<f64>() + 0.001 * s * s
        })
        .start_at(&vec![0.4; n]);
        let (q, spent, _) = probe_counted(&dense, &structured());
        assert!(q.is_none());
        assert!(spent <= 7 + 4 * n as u64, "spent {spent}");
        assert!(spent > 7 + 2 * n as u64, "spent {spent}");
    }

    #[test]
    fn a_non_quadratic_objective_is_declined_after_three_evaluations() {
        let p = Problem::new(2, |x| {
            100.0 * (x[1] - x[0] * x[0]).powi(2) + (1.0 - x[0]).powi(2)
        })
        .start_at(&[-1.2, 1.0]);
        let (q, spent, _) = probe_counted(&p, &Options::default());
        assert!(q.is_none());
        assert_eq!(spent, 4);
    }

    #[test]
    fn a_nonconvex_quadratic_is_declined() {
        // A saddle: the Newton path could end in a basin the quasi-Newton path
        // does not (HS44 in abl-i5-rejected).
        let p = Problem::new(2, |x| x[0] * x[0] - x[1] * x[1] + 0.5 * x[0] * x[1])
            .start_at(&[0.3, 0.2])
            .lower_bounds(&[-1.0, -1.0])
            .upper_bounds(&[1.0, 1.0]);
        assert!(probe(&p, &Options::default()).is_none());
        // A singular but positive semidefinite one passes.
        let q = Problem::new(2, |x| (x[0] + x[1]) * (x[0] + x[1])).start_at(&[0.3, 0.2]);
        assert!(probe(&q, &Options::default()).is_some());
    }

    #[test]
    fn a_convex_quadratic_row_is_accepted_and_solved_with_newton_steps() {
        let n = 6;
        let r: Vec<f64> = (0..n).map(|i| 0.5 + 0.3 * (i % 6) as f64).collect();
        let p = ellipsoid_projection(n);
        // `Off` reproduces the behaviour before `abl-i5-rows`: any nonlinear
        // row declines the probe.
        assert!(probe(
            &p,
            &Options {
                quadratic_rows: QuadraticRows::Off,
                ..structured()
            }
        )
        .is_none());
        let q = probe(&p, &with_rows()).expect("a convex quadratically constrained QP");
        assert_eq!(q.f_evals, 7 + 2 * n as u64, "{}", q.note);
        // 1 at x0, 6 at the line points, 2n for the row's diagonal build.
        assert_eq!(q.c_evals, 7 + 2 * n as u64, "{}", q.note);
        assert!(q.note.contains("1 quadratic constraint row"), "{}", q.note);
        let opts = Options {
            algorithm: mincon_core::Algorithm::Sqp,
            ..with_rows()
        };
        let newton = mincon_sqp::solve(&q, &opts).unwrap();
        assert!(newton.exit_flag.is_success(), "{:?}", newton.exit_flag);
        let quasi = mincon_sqp::solve(&p, &opts).unwrap();
        assert!(quasi.exit_flag.is_success(), "{:?}", quasi.exit_flag);
        assert!(
            (newton.solution.f - quasi.solution.f).abs() < 1e-6 * quasi.solution.f.abs().max(1.0),
            "{} vs {}",
            newton.solution.f,
            quasi.solution.f
        );
        assert!(
            newton.iterations < quasi.iterations,
            "{} vs {}",
            newton.iterations,
            quasi.iterations
        );
        // The Lagrangian Hessian carries the row's curvature times its multiplier.
        let mut out = vec![0.0; q.hessian_structure().unwrap().nnz()];
        q.hessian_lagrangian(&[0.0; 6], 1.0, &[2.0], &mut out)
            .unwrap();
        let s = q.hessian_structure().unwrap();
        for j in 0..n {
            let diag = out[s.col_ptr()[j]];
            let expected = 2.0 + 2.0 * 2.0 / (r[j] * r[j]);
            assert!((diag - expected).abs() < 1e-6, "({j}) {diag} vs {expected}");
        }
    }

    #[test]
    fn quadratic_rows_that_do_not_keep_the_set_convex_are_declined() {
        // An equality sphere (the feasible set is not convex).
        let sphere = Problem::new(2, |x| x[0] * x[0] + 2.0 * x[1] * x[1])
            .start_at(&[0.5, 0.5])
            .equality(1, |x, c| c[0] = x[0] * x[0] + x[1] * x[1] - 1.0);
        assert!(probe(&sphere, &with_rows()).is_none());
        // A convex function bounded below: the outside of a disc.
        let outside = Problem::new(2, |x| x[0] * x[0] + 2.0 * x[1] * x[1])
            .start_at(&[1.5, 0.5])
            .inequality(1, |x, c| c[0] = 1.0 - x[0] * x[0] - x[1] * x[1]);
        assert!(probe(&outside, &with_rows()).is_none());
        // A saddle row bounded above: convex along one probe line and concave
        // along another, so neither the line test nor the Cholesky may pass it.
        let saddle = Problem::new(2, |x| x[0] * x[0] + 2.0 * x[1] * x[1])
            .start_at(&[0.3, 0.4])
            .inequality(1, |x, c| c[0] = x[0] * x[0] - x[1] * x[1] - 1.0);
        assert!(probe(&saddle, &with_rows()).is_none());
        // A cubic row is neither linear nor quadratic.
        let cubic = Problem::new(2, |x| x[0] * x[0] + 2.0 * x[1] * x[1])
            .start_at(&[0.5, 0.5])
            .inequality(1, |x, c| c[0] = x[0] * x[0] * x[0] + x[1] - 1.0);
        let (q, spent, _) = probe_counted(&cubic, &with_rows());
        assert!(q.is_none());
        assert_eq!(spent, 4);
    }

    #[test]
    fn a_quadratic_with_a_linear_row_is_accepted_and_the_sqp_member_takes_one_step() {
        let p = Problem::new(3, |x| {
            (x[0] - 1.0).powi(2) + 2.0 * (x[1] + 0.5).powi(2) + x[2] * x[2] + x[0] * x[2]
        })
        .start_at(&[2.0, 2.0, 2.0])
        .equality(1, |x, c| c[0] = x[0] + x[1] + x[2] - 1.0);
        let opts = Options::default();
        let q = probe(&p, &opts).expect("a QP must be recognised");
        let r = mincon_sqp::solve(
            &q,
            &Options {
                algorithm: mincon_core::Algorithm::Sqp,
                ..opts
            },
        )
        .unwrap();
        assert!(r.exit_flag.is_success(), "{:?}", r.exit_flag);
        assert!(r.iterations <= 3, "iterations {}", r.iterations);
    }
}
