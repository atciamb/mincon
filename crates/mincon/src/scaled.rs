//! Variable scaling as a model wrapper.
//!
//! The solvers assume variables of order one: the finite-difference step is
//! `sqrt(eps) * max(1, |x_i|)`, the interior-point start push is 1e-2 of
//! `max(1, |bound|)`, the SQP step bound is relative to `max(1, |x_i|)`, and
//! the quasi-Newton model starts from the identity. A model with a pressure
//! of 1e6 next to an area of 1e-6 (`bench/results/s7-friction`, bad_scaling;
//! the corpus problem UNITS) breaks every one of those assumptions at once,
//! and every solver in that audit missed its optimum. `fmincon` leaves the
//! fix to the user (`TypicalX`); this wrapper does it from the start's
//! magnitudes: the solver sees `x~ = x / d`, the model sees `x = d * x~`.

use mincon_core::{is_free, Capabilities, EvalError, Nlp, NlpDims, Sparsity, VariableScaling};

/// `x = d * x~`, with the derivatives chain-ruled: gradient times `d`,
/// Jacobian columns times `d_j`, Hessian entries times `d_i d_j`.
pub struct ScaledNlp<'a, P: Nlp + ?Sized> {
    inner: &'a P,
    d: Vec<f64>,
    x0: Vec<f64>,
    lb: Vec<f64>,
    ub: Vec<f64>,
    ones: Vec<f64>,
    dense_jac: Option<Sparsity>,
}

/// The span `max / min` of the factors above which [`VariableScaling::Auto`]
/// scales.
pub const AUTO_SPAN: f64 = 1e4;

/// Scale factors from the start: `max(|x0_i|, typical_i)`, or 1 where both
/// vanish. Returns `None` when every factor is 1 (nothing to do), or, in
/// [`VariableScaling::Auto`], when the factors span less than [`AUTO_SPAN`].
#[must_use]
pub fn factors_from_start<P: Nlp + ?Sized>(nlp: &P, mode: VariableScaling) -> Option<Vec<f64>> {
    if mode == VariableScaling::Off {
        return None;
    }
    let x0 = nlp.x0();
    let typical = nlp.typical_x();
    let d: Vec<f64> = x0
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let t = typical.map_or(0.0, |t| t[i].abs());
            let s = v.abs().max(t);
            if s > 0.0 && s.is_finite() {
                s
            } else {
                1.0
            }
        })
        .collect();
    if d.iter().all(|v| (v - 1.0).abs() < 1e-12) {
        return None;
    }
    if mode == VariableScaling::Auto {
        let (lo, hi) = d.iter().fold((f64::INFINITY, 0.0_f64), |(lo, hi), v| {
            (lo.min(*v), hi.max(*v))
        });
        if hi < AUTO_SPAN * lo {
            return None;
        }
    }
    Some(d)
}

impl<'a, P: Nlp + ?Sized> ScaledNlp<'a, P> {
    /// Wrap `inner` with the factors `d` (all positive and finite).
    #[must_use]
    pub fn new(inner: &'a P, d: Vec<f64>) -> Self {
        let n = inner.dims().n;
        let m = inner.dims().m;
        let (lb, ub) = inner.x_bounds();
        let map = |v: f64, dj: f64| if is_free(v) { v } else { v / dj };
        let lb: Vec<f64> = lb.iter().zip(&d).map(|(v, dj)| map(*v, *dj)).collect();
        let ub: Vec<f64> = ub.iter().zip(&d).map(|(v, dj)| map(*v, *dj)).collect();
        let x0: Vec<f64> = inner.x0().iter().zip(&d).map(|(v, dj)| v / dj).collect();
        let dense_jac = if inner.jacobian_structure().is_none() && m > 0 {
            Some(Sparsity::dense(m, n))
        } else {
            None
        };
        Self {
            inner,
            d,
            x0,
            lb,
            ub,
            ones: vec![1.0; n],
            dense_jac,
        }
    }

    /// The factors.
    #[must_use]
    pub fn factors(&self) -> &[f64] {
        &self.d
    }

    /// The model's point for a solver point.
    #[must_use]
    pub fn unscale_x(&self, xt: &[f64]) -> Vec<f64> {
        xt.iter().zip(&self.d).map(|(v, dj)| v * dj).collect()
    }

    /// Bound multipliers in the model's units: the bound `x~ >= lb / d` carries
    /// `z~ (x~ - lb / d) = (z~ / d) (x - lb)`.
    #[must_use]
    pub fn unscale_bound_multipliers(&self, zt: &[f64]) -> Vec<f64> {
        zt.iter().zip(&self.d).map(|(v, dj)| v / dj).collect()
    }

    /// A dense row-major `n x n` Hessian in the user's variables -> the scaled
    /// ones (`H_ij d_i d_j`).
    pub fn scale_hessian(&self, h: &mut [f64]) {
        let n = self.d.len();
        if h.len() == n * n {
            for i in 0..n {
                for j in 0..n {
                    h[i * n + j] *= self.d[i] * self.d[j];
                }
            }
        }
    }

    /// The inverse of [`ScaledNlp::scale_hessian`].
    pub fn unscale_hessian(&self, h: &mut [f64]) {
        let n = self.d.len();
        if h.len() == n * n {
            for i in 0..n {
                for j in 0..n {
                    h[i * n + j] /= self.d[i] * self.d[j];
                }
            }
        }
    }

    fn jac_pattern(&self) -> &Sparsity {
        self.inner
            .jacobian_structure()
            .or(self.dense_jac.as_ref())
            .expect("a Jacobian pattern exists whenever m > 0")
    }
}

impl<P: Nlp + ?Sized> Nlp for ScaledNlp<'_, P> {
    fn dims(&self) -> NlpDims {
        self.inner.dims()
    }
    fn x_bounds(&self) -> (&[f64], &[f64]) {
        (&self.lb, &self.ub)
    }
    fn c_bounds(&self) -> (&[f64], &[f64]) {
        self.inner.c_bounds()
    }
    fn x0(&self) -> &[f64] {
        &self.x0
    }
    fn capabilities(&self) -> Capabilities {
        self.inner.capabilities()
    }
    fn objective(&self, x: &[f64]) -> Result<f64, EvalError> {
        self.inner.objective(&self.unscale_x(x))
    }
    fn constraints(&self, x: &[f64], out: &mut [f64]) -> Result<(), EvalError> {
        self.inner.constraints(&self.unscale_x(x), out)
    }
    fn gradient(&self, x: &[f64], out: &mut [f64]) -> Result<(), EvalError> {
        self.inner.gradient(&self.unscale_x(x), out)?;
        for (g, dj) in out.iter_mut().zip(&self.d) {
            *g *= dj;
        }
        Ok(())
    }
    fn jacobian_structure(&self) -> Option<&Sparsity> {
        self.inner.jacobian_structure()
    }
    fn jacobian(&self, x: &[f64], out: &mut [f64]) -> Result<(), EvalError> {
        self.inner.jacobian(&self.unscale_x(x), out)?;
        let p = self.jac_pattern();
        for j in 0..self.d.len() {
            for pos in p.col_ptr()[j]..p.col_ptr()[j + 1] {
                out[pos] *= self.d[j];
            }
        }
        Ok(())
    }
    fn hessian_structure(&self) -> Option<&Sparsity> {
        self.inner.hessian_structure()
    }
    fn hessian_lagrangian(
        &self,
        x: &[f64],
        sigma: f64,
        lambda: &[f64],
        out: &mut [f64],
    ) -> Result<(), EvalError> {
        self.inner
            .hessian_lagrangian(&self.unscale_x(x), sigma, lambda, out)?;
        if let Some(p) = self.inner.hessian_structure() {
            for j in 0..self.d.len() {
                for pos in p.col_ptr()[j]..p.col_ptr()[j + 1] {
                    let i = p.row_idx()[pos];
                    out[pos] *= self.d[i] * self.d[j];
                }
            }
        }
        Ok(())
    }
    fn hessian_vector(
        &self,
        x: &[f64],
        sigma: f64,
        lambda: &[f64],
        v: &[f64],
        out: &mut [f64],
    ) -> Result<(), EvalError> {
        let dv: Vec<f64> = v.iter().zip(&self.d).map(|(a, b)| a * b).collect();
        self.inner
            .hessian_vector(&self.unscale_x(x), sigma, lambda, &dv, out)?;
        for (o, dj) in out.iter_mut().zip(&self.d) {
            *o *= dj;
        }
        Ok(())
    }
    fn typical_x(&self) -> Option<&[f64]> {
        Some(&self.ones)
    }
}
