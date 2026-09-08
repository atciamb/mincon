//! Bounds-aware finite differences.
//!
//! # Why the bounds handling is the important part
//!
//! Textbook finite differences perturb `x_i` by `+h`. Real models are often
//! undefined outside their box — a thickness that must be positive, a
//! concentration that must not exceed one, a simulation that diverges. A
//! solver that steps outside the box to build a derivative gets `NaN` back,
//! and the user experiences it as "the optimizer crashed on my problem".
//!
//! `fmincon`'s `sqp` and `interior-point` algorithms keep every finite
//! difference inside the bounds, and it is one of the least-discussed reasons
//! they feel robust on engineering models. Most open-source solvers do not. We
//! do, by default ([`FdConfig::respect_bounds`]), flipping the step direction
//! near a bound and shrinking it only if both directions are blocked.
//!
//! # Step sizes
//!
//! The relative step is `sqrt(eps) ~ 1.5e-8` forward and `eps^(1/3) ~ 6.1e-6`
//! central, matching `fmincon`, scaled by `max(|x_i|, typical_x_i)`. Two
//! details that matter more than they look:
//!
//! * The actual step is recomputed as `(x_i + h) - x_i` after rounding, so the
//!   divisor is the perturbation that really happened. Skipping this loses
//!   digits precisely where `x_i` is large.
//! * A variable pinned by `lb == ub` gets a zero step and a zero derivative
//!   rather than a division by zero.

use mincon_core::{Capabilities, EvalError, FdType, Nlp, Sparsity};
use rayon::prelude::*;

use crate::coloring::{distance1_coloring, Coloring};

/// Configuration for the finite-difference engine.
#[derive(Debug, Clone, Copy)]
pub struct FdConfig {
    /// Forward, central, or adaptive.
    pub fd_type: FdType,
    /// Relative step, or `None` for the type's default.
    pub rel_step: Option<f64>,
    /// Keep every probe inside the variable bounds.
    pub respect_bounds: bool,
    /// Use coloring to batch columns when a pattern is available.
    pub use_coloring: bool,
    /// Evaluate independent probes on a thread pool.
    pub parallel: bool,
}

impl Default for FdConfig {
    fn default() -> Self {
        Self {
            fd_type: FdType::Adaptive,
            rel_step: None,
            respect_bounds: true,
            use_coloring: true,
            parallel: false,
        }
    }
}

impl FdConfig {
    /// The relative step this configuration will use, resolving `Adaptive` to
    /// `forward` (the adaptive switch happens at the call site, which knows
    /// whether progress has stalled).
    #[must_use]
    pub fn relative_step(&self, central: bool) -> f64 {
        self.rel_step.unwrap_or(if central {
            f64::EPSILON.powf(1.0 / 3.0)
        } else {
            f64::EPSILON.sqrt()
        })
    }
}

/// A perturbation of one variable: where to move to and by how much, after
/// bounds clipping and floating-point rounding.
#[derive(Debug, Clone, Copy)]
pub struct Step {
    /// The perturbed value of the variable.
    pub x_plus: f64,
    /// The realized step `x_plus - x`. Zero when the variable cannot move.
    pub h: f64,
}

/// Compute a bounds-respecting forward step for one variable.
#[must_use]
pub fn forward_step(x: f64, typical: f64, rel: f64, lb: f64, ub: f64, respect: bool) -> Step {
    let scale = x.abs().max(typical.abs()).max(1.0);
    let mut h = rel * scale;
    if x < 0.0 {
        h = -h;
    }
    if respect {
        let room_up = ub - x;
        let room_dn = x - lb;
        if h > 0.0 && h > room_up {
            // Prefer flipping direction over shrinking: a full-size step in the
            // other direction is more accurate than a tiny one toward the bound.
            h = if room_dn >= h {
                -h
            } else if room_dn >= room_up {
                -room_dn
            } else {
                room_up
            };
        } else if h < 0.0 && (-h) > room_dn {
            h = if room_up >= -h {
                -h
            } else if room_up >= room_dn {
                room_up
            } else {
                -room_dn
            };
        }
        if !h.is_finite() {
            h = 0.0;
        }
    }
    // Recompute the step that will actually be taken after rounding.
    let x_plus = if respect {
        (x + h).clamp(lb, ub)
    } else {
        x + h
    };
    Step {
        x_plus,
        h: x_plus - x,
    }
}

/// Compute a bounds-respecting central pair. Returns `None` when the caller
/// must instead use inward probes because the box is too tight.
#[must_use]
pub fn central_step(
    x: f64,
    typical: f64,
    rel: f64,
    lb: f64,
    ub: f64,
    respect: bool,
) -> Option<f64> {
    let scale = x.abs().max(typical.abs()).max(1.0);
    let h = rel * scale;
    if respect && (x + h > ub || x - h < lb) {
        return None;
    }
    if !h.is_finite() || h == 0.0 {
        return None;
    }
    Some(h)
}

/// Finite-difference engine, holding the coloring and the scratch buffers so
/// that a solve does not allocate per iteration.
#[derive(Debug)]
pub struct FiniteDifferences {
    config: FdConfig,
    n: usize,
    m: usize,
    typical: Vec<f64>,
    coloring: Option<Coloring>,
    pattern: Option<Sparsity>,
    /// Set by the caller when the last iteration made poor progress; makes
    /// [`FdType::Adaptive`] switch to central differences.
    escalate: bool,
}

impl FiniteDifferences {
    /// Build an engine for a problem, computing the coloring once.
    #[must_use]
    pub fn new(
        config: FdConfig,
        n: usize,
        m: usize,
        typical: Option<&[f64]>,
        jac_pattern: Option<&Sparsity>,
    ) -> Self {
        let typical = typical.map_or_else(|| vec![1.0; n], <[f64]>::to_vec);
        let (coloring, pattern) = match (config.use_coloring, jac_pattern) {
            (true, Some(p)) => {
                let c = distance1_coloring(p);
                debug_assert!(crate::coloring::verify_coloring(p, &c).is_ok());
                (Some(c), Some(p.clone()))
            }
            (_, p) => (None, p.cloned()),
        };
        Self {
            config,
            n,
            m,
            typical,
            coloring,
            pattern,
            escalate: false,
        }
    }

    /// The configuration in use.
    #[must_use]
    pub fn config(&self) -> &FdConfig {
        &self.config
    }

    /// How many model evaluations one Jacobian will cost.
    #[must_use]
    pub fn jacobian_cost(&self) -> usize {
        let base = self.coloring.as_ref().map_or(self.n, Coloring::num_groups);
        if self.use_central() {
            2 * base
        } else {
            base
        }
    }

    /// Ask for central differences from now on. Called when the line search
    /// starts failing, which is the signature of derivative noise.
    pub fn escalate(&mut self) {
        self.escalate = true;
    }

    /// Whether central differences are currently in use.
    #[must_use]
    pub fn use_central(&self) -> bool {
        match self.config.fd_type {
            FdType::Forward => false,
            FdType::Central => true,
            FdType::Adaptive => self.escalate,
        }
    }

    /// Estimate the truncation error of the current finite-difference scheme on
    /// the coordinates `cols`, by re-differencing with twice the step
    /// (forward: one extra objective and constraint evaluation per coordinate;
    /// central: two). Returns `(max gradient disagreement, max Jacobian
    /// disagreement, evaluations)` in the model's own units; the disagreement
    /// `D(h) - D(2h)` is, to first order, the truncation error of `D(h)` for
    /// forward differences and three times it for central ones. Coordinates
    /// that cannot move (pinned, or no room for a doubled step) are skipped.
    ///
    /// # Errors
    /// Propagates model failures.
    #[allow(clippy::too_many_arguments)]
    pub fn error_estimate<P: Nlp + ?Sized>(
        &self,
        nlp: &P,
        x: &[f64],
        f0: f64,
        c0: &[f64],
        grad: &[f64],
        jac_dense_col: &dyn Fn(usize, &mut [f64]),
        cols: &[usize],
    ) -> Result<(f64, f64, u64), EvalError> {
        let (lb, ub) = nlp.x_bounds();
        let central = self.use_central();
        let rel = 2.0 * self.config.relative_step(central);
        let respect = self.config.respect_bounds;
        let m = c0.len();
        let mut g_err = 0.0_f64;
        let mut j_err = 0.0_f64;
        let mut evals = 0u64;
        let mut xp = x.to_vec();
        let mut cp = vec![0.0; m];
        let mut cm = vec![0.0; m];
        let mut jcol = vec![0.0; m];
        for &j in cols {
            if respect && lb[j] == ub[j] {
                continue;
            }
            let (d_f, d_c): (f64, Vec<f64>) = if central {
                let Some(h) = central_step(x[j], self.typical[j], rel, lb[j], ub[j], respect)
                else {
                    continue;
                };
                xp[j] = x[j] + h;
                let fp = nlp.objective(&xp)?;
                if m > 0 {
                    nlp.constraints(&xp, &mut cp)?;
                }
                xp[j] = x[j] - h;
                let fm = nlp.objective(&xp)?;
                if m > 0 {
                    nlp.constraints(&xp, &mut cm)?;
                }
                xp[j] = x[j];
                evals += 2;
                (
                    (fp - fm) / (2.0 * h),
                    (0..m).map(|i| (cp[i] - cm[i]) / (2.0 * h)).collect(),
                )
            } else {
                let st = forward_step(x[j], self.typical[j], rel, lb[j], ub[j], respect);
                if st.h == 0.0 {
                    continue;
                }
                xp[j] = st.x_plus;
                let fp = nlp.objective(&xp)?;
                if m > 0 {
                    nlp.constraints(&xp, &mut cp)?;
                }
                xp[j] = x[j];
                evals += 1;
                (
                    (fp - f0) / st.h,
                    (0..m).map(|i| (cp[i] - c0[i]) / st.h).collect(),
                )
            };
            if !d_f.is_finite() || d_c.iter().any(|v| !v.is_finite()) {
                return Err(EvalError::NonFinite(None));
            }
            g_err = g_err.max((d_f - grad[j]).abs());
            if m > 0 {
                jac_dense_col(j, &mut jcol);
                for i in 0..m {
                    j_err = j_err.max((d_c[i] - jcol[i]).abs());
                }
            }
        }
        Ok((g_err, j_err, evals))
    }

    /// Approximate the objective gradient at `x`, given `f0 = f(x)`.
    ///
    /// # Errors
    /// Propagates a model failure only when it happens at a point the engine
    /// cannot work around; a single failed probe is retried with a halved step
    /// before giving up.
    pub fn gradient<P: Nlp + ?Sized>(
        &self,
        nlp: &P,
        x: &[f64],
        f0: f64,
        out: &mut [f64],
    ) -> Result<u64, EvalError> {
        self.gradient_with_step(nlp, x, f0, out, 1.0)
    }

    /// As [`FiniteDifferences::gradient`] with the relative step multiplied by
    /// `step_factor`. Two evaluations with factors 1 and 2 give a Richardson-style
    /// estimate of the truncation error of the first.
    ///
    /// # Errors
    /// As [`FiniteDifferences::gradient`].
    pub fn gradient_with_step<P: Nlp + ?Sized>(
        &self,
        nlp: &P,
        x: &[f64],
        f0: f64,
        out: &mut [f64],
        step_factor: f64,
    ) -> Result<u64, EvalError> {
        let (lb, ub) = nlp.x_bounds();
        let central = self.use_central();
        let rel = self.config.relative_step(central) * step_factor;
        let respect = self.config.respect_bounds;
        let parallel = self.config.parallel && nlp.capabilities().parallel_safe;

        let one = |j: usize| -> Result<(f64, u64), EvalError> {
            if respect && lb[j] == ub[j] {
                return Ok((0.0, 0));
            }
            let mut xp = x.to_vec();
            let mut evals = 0u64;
            if central {
                if let Some(h) = central_step(x[j], self.typical[j], rel, lb[j], ub[j], respect) {
                    let fp =
                        eval_with_retreat(|v| nlp.objective(v), &mut xp, j, x[j], h, &mut evals);
                    if matches!(fp, Err(EvalError::UserAbort)) {
                        return Err(EvalError::UserAbort);
                    }
                    let fm =
                        eval_with_retreat(|v| nlp.objective(v), &mut xp, j, x[j], -h, &mut evals);
                    if matches!(fm, Err(EvalError::UserAbort)) {
                        return Err(EvalError::UserAbort);
                    }
                    return Ok((difference(f0, fp.ok(), fm.ok(), j)?, evals));
                }
            }
            let step = forward_step(x[j], self.typical[j], rel, lb[j], ub[j], respect);
            xp[j] = step.x_plus;
            let fp = eval_with_retreat(|v| nlp.objective(v), &mut xp, j, x[j], step.h, &mut evals)?;
            let other = if central {
                let probe = eval_with_retreat(
                    |v| nlp.objective(v),
                    &mut xp,
                    j,
                    x[j],
                    fp.1 * 0.5,
                    &mut evals,
                );
                if matches!(probe, Err(EvalError::UserAbort)) {
                    return Err(EvalError::UserAbort);
                }
                probe.ok()
            } else {
                None
            };
            Ok((difference(f0, Some(fp), other, j)?, evals))
        };

        if parallel {
            let results: Result<Vec<(f64, u64)>, EvalError> =
                (0..self.n).into_par_iter().map(one).collect();
            let results = results?;
            let mut total = 0;
            for (j, (g, e)) in results.into_iter().enumerate() {
                out[j] = g;
                total += e;
            }
            Ok(total)
        } else {
            let mut total = 0;
            for j in 0..self.n {
                let (g, e) = one(j)?;
                out[j] = g;
                total += e;
            }
            Ok(total)
        }
    }

    /// Approximate the constraint Jacobian at `x`, given `c0 = c(x)`.
    ///
    /// Writes values in the order of the pattern given at construction, or in
    /// dense column-major order when there is no pattern.
    ///
    /// # Errors
    /// Propagates an unrecoverable model failure.
    pub fn jacobian<P: Nlp + ?Sized>(
        &self,
        nlp: &P,
        x: &[f64],
        c0: &[f64],
        out: &mut [f64],
    ) -> Result<u64, EvalError> {
        self.jacobian_with_step(nlp, x, c0, out, 1.0)
    }

    /// As [`FiniteDifferences::jacobian`] with the relative step multiplied by
    /// `step_factor` (see [`FiniteDifferences::gradient_with_step`]).
    ///
    /// # Errors
    /// As [`FiniteDifferences::jacobian`].
    pub fn jacobian_with_step<P: Nlp + ?Sized>(
        &self,
        nlp: &P,
        x: &[f64],
        c0: &[f64],
        out: &mut [f64],
        step_factor: f64,
    ) -> Result<u64, EvalError> {
        if self.m == 0 {
            return Ok(0);
        }
        let (lb, ub) = nlp.x_bounds();
        let central = self.use_central();
        let rel = self.config.relative_step(central) * step_factor;
        let respect = self.config.respect_bounds;

        // Column groups: one per color if we have a coloring, else one per column.
        let groups: Vec<Vec<usize>> = match &self.coloring {
            Some(c) => c.groups().to_vec(),
            None => (0..self.n).map(|j| vec![j]).collect(),
        };

        let eval_group = |group: &[usize]| -> Result<GroupDerivative, EvalError> {
            let mut xp = x.to_vec();
            let mut h = Vec::with_capacity(group.len());
            let mut hneg = Vec::with_capacity(group.len());
            let mut inward = Vec::with_capacity(group.len());
            for &j in group {
                if respect && lb[j] == ub[j] {
                    h.push(0.0);
                    hneg.push(0.0);
                    inward.push(false);
                } else if let Some(hj) = central
                    .then(|| central_step(x[j], self.typical[j], rel, lb[j], ub[j], respect))
                    .flatten()
                {
                    h.push(hj);
                    hneg.push(-hj);
                    inward.push(false);
                } else {
                    let step = forward_step(x[j], self.typical[j], rel, lb[j], ub[j], respect);
                    if step.h == 0.0 || !step.h.is_finite() {
                        return Err(EvalError::NonFinite(Some(j)));
                    }
                    h.push(step.h);
                    hneg.push(if central { step.h * 0.5 } else { 0.0 });
                    inward.push(central);
                }
            }
            let mut evals = 0u64;
            let mut cp = vec![0.0; self.m];
            let mut cm = vec![0.0; self.m];
            let plus =
                eval_constraints_with_retreat(nlp, &mut xp, x, &mut h, group, &mut cp, &mut evals);
            if matches!(plus, Err(EvalError::UserAbort)) {
                return Err(EvalError::UserAbort);
            }
            // After a successful retreat, place inward probes halfway toward
            // the base so the two samples remain distinct. Opposite-sided
            // columns retain their independent nominal steps.
            if plus.is_ok() {
                for k in 0..group.len() {
                    if inward[k] {
                        hneg[k] = h[k] * 0.5;
                    }
                }
            }
            xp.copy_from_slice(x);
            let minus = eval_constraints_with_retreat(
                nlp, &mut xp, x, &mut hneg, group, &mut cm, &mut evals,
            );
            if matches!(minus, Err(EvalError::UserAbort)) {
                return Err(EvalError::UserAbort);
            }
            let mut entries = Vec::new();
            for (k, &j) in group.iter().enumerate() {
                let write = |i: usize| -> Result<f64, EvalError> {
                    if respect && lb[j] == ub[j] {
                        return Ok(0.0);
                    }
                    difference(
                        c0[i],
                        (plus.is_ok() && h[k] != 0.0).then_some((cp[i], h[k])),
                        (minus.is_ok() && hneg[k] != 0.0).then_some((cm[i], hneg[k])),
                        j,
                    )
                };
                match &self.pattern {
                    Some(p) => {
                        for pos in p.col_ptr()[j]..p.col_ptr()[j + 1] {
                            entries.push((pos, write(p.row_idx()[pos])?));
                        }
                    }
                    None => {
                        for i in 0..self.m {
                            entries.push((j * self.m + i, write(i)?));
                        }
                    }
                }
            }
            Ok(GroupDerivative { entries, evals })
        };

        let parallel = self.config.parallel && nlp.capabilities().parallel_safe;
        let per_group: Vec<GroupDerivative> = if parallel {
            groups
                .par_iter()
                .map(|g| eval_group(g))
                .collect::<Result<Vec<_>, _>>()?
        } else {
            groups
                .iter()
                .map(|g| eval_group(g))
                .collect::<Result<Vec<_>, _>>()?
        };

        let mut total = 0u64;
        for result in per_group {
            total += result.evals;
            for (pos, value) in result.entries {
                out[pos] = value;
            }
        }
        Ok(total)
    }
}

struct GroupDerivative {
    entries: Vec<(usize, f64)>,
    evals: u64,
}

/// Derivative of the interpolating quadratic at zero for arbitrary distinct
/// displacements. A surviving single probe gives a first-order fallback.
fn difference(
    base: f64,
    plus: Option<(f64, f64)>,
    minus: Option<(f64, f64)>,
    j: usize,
) -> Result<f64, EvalError> {
    let value = match (plus, minus) {
        (Some((fa, a)), Some((_, b))) if a == b => (fa - base) / a,
        (Some((fa, a)), Some((fb, b))) if a == -b => (fa - fb) / (a - b),
        (Some((fa, a)), Some((fb, b))) => {
            let sa = (fa - base) / a;
            let sb = (fb - base) / b;
            // Keep the direct secant as the leading term. When a and -b
            // differ only by rounding, the correction is correspondingly small.
            (fa - fb) / (a - b) - ((a + b) / (a - b)) * (sa - sb)
        }
        (Some((f, h)), None) | (None, Some((f, h))) => (f - base) / h,
        (None, None) => return Err(EvalError::NonFinite(Some(j))),
    };
    if value.is_finite() {
        Ok(value)
    } else {
        Err(EvalError::NonFinite(Some(j)))
    }
}

/// Evaluate `f` at `xp`, halving the step toward `x_base` if the model
/// refuses, and restoring `xp[j]` afterwards.
fn eval_with_retreat<F>(
    f: F,
    xp: &mut [f64],
    j: usize,
    x_base: f64,
    mut h: f64,
    evals: &mut u64,
) -> Result<(f64, f64), EvalError>
where
    F: Fn(&[f64]) -> Result<f64, EvalError>,
{
    const MAX_RETREAT: usize = 8;
    for _ in 0..=MAX_RETREAT {
        xp[j] = x_base + h;
        let actual = xp[j] - x_base;
        if actual == 0.0 || !actual.is_finite() || !xp[j].is_finite() {
            break;
        }
        *evals += 1;
        match f(xp) {
            Ok(v) if v.is_finite() => {
                xp[j] = x_base;
                return Ok((v, actual));
            }
            Ok(_) | Err(EvalError::NonFinite(_)) | Err(EvalError::OutOfDomain(_)) => {}
            Err(e @ EvalError::UserAbort) => {
                xp[j] = x_base;
                return Err(e);
            }
            Err(_) => {}
        }
        h *= 0.5;
    }
    xp[j] = x_base;
    Err(EvalError::NonFinite(Some(j)))
}

/// The group version: shrink every step in the group together so the columns
/// stay consistent with the divisors used to recover them.
fn eval_constraints_with_retreat<P: Nlp + ?Sized>(
    nlp: &P,
    xp: &mut [f64],
    x: &[f64],
    h: &mut [f64],
    group: &[usize],
    out: &mut [f64],
    evals: &mut u64,
) -> Result<(), EvalError> {
    const MAX_RETREAT: usize = 8;
    if h.iter().all(|&hj| hj == 0.0) {
        return Ok(());
    }
    for _ in 0..=MAX_RETREAT {
        // Keep nominal displacements until acceptance so rounding does not
        // compound across retreats. Zero entries denote unperturbed columns.
        for (k, &j) in group.iter().enumerate() {
            xp[j] = x[j] + h[k];
            let actual = xp[j] - x[j];
            if h[k] != 0.0 && (actual == 0.0 || !actual.is_finite() || !xp[j].is_finite()) {
                return Err(EvalError::NonFinite(Some(j)));
            }
        }
        *evals += 1;
        match nlp.constraints(xp, out) {
            Ok(()) if out.iter().all(|v| v.is_finite()) => {
                for (k, &j) in group.iter().enumerate() {
                    h[k] = xp[j] - x[j];
                }
                return Ok(());
            }
            Err(e @ EvalError::UserAbort) => return Err(e),
            _ => {}
        }
        for (k, hj) in h.iter_mut().enumerate() {
            let was_active = *hj != 0.0;
            *hj *= 0.5;
            if was_active && *hj == 0.0 {
                return Err(EvalError::NonFinite(Some(group[k])));
            }
        }
    }
    Err(EvalError::NonFinite(None))
}

/// The capabilities a model needs for the engine to be able to run in parallel.
#[must_use]
pub fn parallel_is_safe(caps: Capabilities) -> bool {
    caps.parallel_safe
}

#[cfg(test)]
mod tests {
    use super::*;
    use mincon_core::NlpDims;

    const INF: f64 = f64::INFINITY;

    #[test]
    fn step_flips_direction_at_an_upper_bound() {
        // x sits exactly on its upper bound; the step must go down.
        let s = forward_step(1.0, 1.0, 1e-6, 0.0, 1.0, true);
        assert!(s.h < 0.0, "step {} should be negative", s.h);
        assert!(s.x_plus <= 1.0);
    }

    #[test]
    fn step_flips_direction_at_a_lower_bound() {
        let s = forward_step(-1.0, 1.0, 1e-6, -1.0, 5.0, true);
        assert!(s.h > 0.0, "step {} should be positive", s.h);
        assert!(s.x_plus >= -1.0);
    }

    #[test]
    fn step_shrinks_when_both_directions_are_blocked() {
        // A box narrower than the nominal step.
        let s = forward_step(0.5, 1.0, 1e-1, 0.499, 0.501, true);
        assert!(s.x_plus >= 0.499 && s.x_plus <= 0.501, "x+ = {}", s.x_plus);
        assert!(s.h != 0.0);
    }

    #[test]
    fn a_pinned_variable_gets_a_zero_step() {
        let s = forward_step(2.0, 1.0, 1e-6, 2.0, 2.0, true);
        assert_eq!(s.h, 0.0);
    }

    #[test]
    fn ignoring_bounds_steps_outside_them() {
        let s = forward_step(1.0, 1.0, 1e-6, 0.0, 1.0, false);
        assert!(s.x_plus > 1.0);
    }

    #[test]
    fn realized_step_is_exact_after_rounding() {
        let s = forward_step(1e16, 1.0, 1e-8, -INF, INF, false);
        assert_eq!(s.h, s.x_plus - 1e16);
    }

    #[test]
    fn central_step_declines_when_the_box_is_tight() {
        assert!(central_step(0.5, 1.0, 1e-1, 0.49, 0.51, true).is_none());
        assert!(central_step(0.5, 1.0, 1e-6, 0.0, 1.0, true).is_some());
    }

    // ---- A model to differentiate ----

    struct Quadratic {
        n: usize,
        lb: Vec<f64>,
        ub: Vec<f64>,
        cl: Vec<f64>,
        cu: Vec<f64>,
        x0: Vec<f64>,
        pattern: Sparsity,
    }

    impl Quadratic {
        fn new(n: usize) -> Self {
            // c_i(x) = x_i^2 + x_{i+1},  i = 0..n-1  (tridiagonal-ish pattern)
            let mut t = Vec::new();
            for i in 0..n - 1 {
                t.push((i, i));
                t.push((i, i + 1));
            }
            Self {
                n,
                lb: vec![-INF; n],
                ub: vec![INF; n],
                cl: vec![0.0; n - 1],
                cu: vec![0.0; n - 1],
                x0: (0..n).map(|i| 0.5 + i as f64 * 0.1).collect(),
                pattern: Sparsity::from_triplets(n - 1, n, &t).unwrap(),
            }
        }
    }

    impl Nlp for Quadratic {
        fn dims(&self) -> NlpDims {
            NlpDims {
                n: self.n,
                m: self.n - 1,
            }
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
            Capabilities {
                parallel_safe: true,
                ..Capabilities::none()
            }
        }
        fn objective(&self, x: &[f64]) -> Result<f64, EvalError> {
            Ok(x.iter()
                .enumerate()
                .map(|(i, v)| (i as f64 + 1.0) * v * v)
                .sum())
        }
        fn constraints(&self, x: &[f64], out: &mut [f64]) -> Result<(), EvalError> {
            for i in 0..self.n - 1 {
                out[i] = x[i] * x[i] + x[i + 1];
            }
            Ok(())
        }
        fn jacobian_structure(&self) -> Option<&Sparsity> {
            Some(&self.pattern)
        }
    }

    #[test]
    fn gradient_matches_the_analytic_one() {
        let p = Quadratic::new(6);
        let fd = FiniteDifferences::new(FdConfig::default(), 6, 5, None, p.jacobian_structure());
        let x = p.x0().to_vec();
        let f0 = p.objective(&x).unwrap();
        let mut g = vec![0.0; 6];
        fd.gradient(&p, &x, f0, &mut g).unwrap();
        for i in 0..6 {
            let exact = 2.0 * (i as f64 + 1.0) * x[i];
            assert!((g[i] - exact).abs() < 1e-6, "g[{i}] = {} vs {exact}", g[i]);
        }
    }

    #[test]
    fn central_differences_are_more_accurate_than_forward() {
        let p = Quadratic::new(4);
        let x = p.x0().to_vec();
        let f0 = p.objective(&x).unwrap();
        let exact: Vec<f64> = (0..4).map(|i| 2.0 * (i as f64 + 1.0) * x[i]).collect();

        let mut fwd = vec![0.0; 4];
        let e_fwd = {
            let fd = FiniteDifferences::new(
                FdConfig {
                    fd_type: FdType::Forward,
                    ..FdConfig::default()
                },
                4,
                3,
                None,
                None,
            );
            fd.gradient(&p, &x, f0, &mut fwd).unwrap();
            (0..4).fold(0.0_f64, |a, i| a.max((fwd[i] - exact[i]).abs()))
        };
        let mut ctr = vec![0.0; 4];
        let e_ctr = {
            let fd = FiniteDifferences::new(
                FdConfig {
                    fd_type: FdType::Central,
                    ..FdConfig::default()
                },
                4,
                3,
                None,
                None,
            );
            fd.gradient(&p, &x, f0, &mut ctr).unwrap();
            (0..4).fold(0.0_f64, |a, i| a.max((ctr[i] - exact[i]).abs()))
        };
        assert!(
            e_ctr < e_fwd,
            "central error {e_ctr} should beat forward {e_fwd}"
        );
    }

    #[test]
    fn colored_jacobian_matches_the_analytic_one() {
        let n = 30;
        let p = Quadratic::new(n);
        let pattern = p.jacobian_structure().unwrap().clone();
        let fd = FiniteDifferences::new(FdConfig::default(), n, n - 1, None, Some(&pattern));
        // A tridiagonal-style pattern must compress hard.
        assert!(
            fd.jacobian_cost() <= 3,
            "expected <= 3 evaluations, got {}",
            fd.jacobian_cost()
        );

        let x = p.x0().to_vec();
        let mut c0 = vec![0.0; n - 1];
        p.constraints(&x, &mut c0).unwrap();
        let mut vals = vec![0.0; pattern.nnz()];
        fd.jacobian(&p, &x, &c0, &mut vals).unwrap();

        for j in 0..n {
            for pos in pattern.col_ptr()[j]..pattern.col_ptr()[j + 1] {
                let i = pattern.row_idx()[pos];
                let exact = if j == i { 2.0 * x[i] } else { 1.0 };
                assert!(
                    (vals[pos] - exact).abs() < 1e-5,
                    "J[{i},{j}] = {} vs {exact}",
                    vals[pos]
                );
            }
        }
    }

    #[test]
    fn uncolored_jacobian_agrees_with_the_colored_one() {
        let n = 12;
        let p = Quadratic::new(n);
        let pattern = p.jacobian_structure().unwrap().clone();
        let x = p.x0().to_vec();
        let mut c0 = vec![0.0; n - 1];
        p.constraints(&x, &mut c0).unwrap();

        let colored = FiniteDifferences::new(FdConfig::default(), n, n - 1, None, Some(&pattern));
        let plain = FiniteDifferences::new(
            FdConfig {
                use_coloring: false,
                ..FdConfig::default()
            },
            n,
            n - 1,
            None,
            Some(&pattern),
        );
        let mut a = vec![0.0; pattern.nnz()];
        let mut b = vec![0.0; pattern.nnz()];
        colored.jacobian(&p, &x, &c0, &mut a).unwrap();
        plain.jacobian(&p, &x, &c0, &mut b).unwrap();
        for k in 0..pattern.nnz() {
            assert!(
                (a[k] - b[k]).abs() < 1e-9,
                "entry {k}: {} vs {}",
                a[k],
                b[k]
            );
        }
        assert!(colored.jacobian_cost() < plain.jacobian_cost());
    }

    #[test]
    fn a_model_that_fails_outside_its_domain_still_yields_a_gradient() {
        struct Sqrt {
            lb: Vec<f64>,
            ub: Vec<f64>,
            x0: Vec<f64>,
        }
        impl Nlp for Sqrt {
            fn dims(&self) -> NlpDims {
                NlpDims { n: 1, m: 0 }
            }
            fn x_bounds(&self) -> (&[f64], &[f64]) {
                (&self.lb, &self.ub)
            }
            fn c_bounds(&self) -> (&[f64], &[f64]) {
                (&[], &[])
            }
            fn x0(&self) -> &[f64] {
                &self.x0
            }
            fn capabilities(&self) -> Capabilities {
                Capabilities::none()
            }
            fn objective(&self, x: &[f64]) -> Result<f64, EvalError> {
                if x[0] < 0.0 {
                    return Err(EvalError::OutOfDomain("negative argument".into()));
                }
                Ok(x[0].sqrt())
            }
            fn constraints(&self, _x: &[f64], _out: &mut [f64]) -> Result<(), EvalError> {
                Ok(())
            }
        }
        // x is exactly at the lower bound of the domain; bounds handling must
        // step up rather than down.
        let p = Sqrt {
            lb: vec![0.0],
            ub: vec![INF],
            x0: vec![0.0],
        };
        let fd = FiniteDifferences::new(FdConfig::default(), 1, 0, None, None);
        let mut g = vec![0.0; 1];
        fd.gradient(&p, &[0.0], 0.0, &mut g).unwrap();
        assert!(g[0].is_finite() && g[0] > 0.0, "g = {}", g[0]);
    }
}
