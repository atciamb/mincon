//! A closure-based problem builder, and the `fmincon`-compatible façade.
//!
//! # Two front doors
//!
//! [`Problem`] is the ergonomic Rust way in: closures, `lb`/`ub`, nonlinear
//! constraints written the way you would say them out loud.
//!
//! The second door is not built yet: an `fmincon`-shaped façade taking the
//! arguments in MATLAB's own order — `A x <= b`, `Aeq x = beq`, `lb`, `ub`, and
//! a `nonlcon` returning `(c, ceq)` with `c(x) <= 0` and `ceq(x) = 0` — so that
//! porting a MATLAB model is transcription rather than translation. See
//! `docs/07_API_DESIGN.md` §4. [`to_fmincon_multipliers`] is the piece of it
//! that exists, because the multiplier sign conventions are the part people get
//! wrong silently.

use mincon_core::{Capabilities, EvalError, Nlp, NlpDims, Sparsity, INF_BOUND};

type ObjFn<'a> = Box<dyn Fn(&[f64]) -> f64 + Sync + 'a>;
type ConFn<'a> = Box<dyn Fn(&[f64], &mut [f64]) + Sync + 'a>;
type GradFn<'a> = Box<dyn Fn(&[f64], &mut [f64]) + Sync + 'a>;

/// A problem assembled from closures.
///
/// ```
/// use mincon::Problem;
/// // minimize (x0 - 1)^2 + (x1 - 2)^2  subject to  x0 + x1 = 2,  x >= 0
/// let p = Problem::new(2, |x| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2))
///     .start_at(&[0.5, 0.5])
///     .lower_bounds(&[0.0, 0.0])
///     .equality(1, |x, c| c[0] = x[0] + x[1] - 2.0);
/// let report = mincon::minimize(&p, &mincon::Options::default()).unwrap();
/// assert!((report.solution.f - 0.5).abs() < 1e-6);
/// ```
pub struct Problem<'a> {
    n: usize,
    f: ObjFn<'a>,
    grad: Option<GradFn<'a>>,
    cons: Option<ConFn<'a>>,
    m: usize,
    x0: Vec<f64>,
    lb: Vec<f64>,
    ub: Vec<f64>,
    cl: Vec<f64>,
    cu: Vec<f64>,
    jac_pattern: Option<Sparsity>,
    typical: Option<Vec<f64>>,
    parallel_safe: bool,
}

impl<'a> Problem<'a> {
    /// A new unconstrained problem in `n` variables.
    pub fn new<F>(n: usize, f: F) -> Self
    where
        F: Fn(&[f64]) -> f64 + Sync + 'a,
    {
        Self {
            n,
            f: Box::new(f),
            grad: None,
            cons: None,
            m: 0,
            x0: vec![0.0; n],
            lb: vec![-INF_BOUND; n],
            ub: vec![INF_BOUND; n],
            cl: Vec::new(),
            cu: Vec::new(),
            jac_pattern: None,
            typical: None,
            parallel_safe: true,
        }
    }

    /// Set the starting point.
    ///
    /// Named `start_at` rather than `x0` because an inherent method shadows the
    /// trait method of the same name, and `problem.x0()` silently resolving to
    /// the builder setter is a footgun. Same reason for `with_gradient`,
    /// `with_constraints` and `with_typical_x` below.
    #[must_use]
    pub fn start_at(mut self, x: &[f64]) -> Self {
        assert_eq!(x.len(), self.n, "x0 has the wrong length");
        self.x0 = x.to_vec();
        self
    }

    /// Set lower bounds.
    #[must_use]
    pub fn lower_bounds(mut self, lb: &[f64]) -> Self {
        assert_eq!(lb.len(), self.n, "lower bounds have the wrong length");
        self.lb = lb.to_vec();
        self
    }

    /// Set upper bounds.
    #[must_use]
    pub fn upper_bounds(mut self, ub: &[f64]) -> Self {
        assert_eq!(ub.len(), self.n, "upper bounds have the wrong length");
        self.ub = ub.to_vec();
        self
    }

    /// Supply an analytic objective gradient.
    #[must_use]
    pub fn with_gradient<G>(mut self, g: G) -> Self
    where
        G: Fn(&[f64], &mut [f64]) + Sync + 'a,
    {
        self.grad = Some(Box::new(g));
        self
    }

    /// Add `m` general constraints with the given bounds.
    #[must_use]
    pub fn with_constraints<C>(mut self, cl: &[f64], cu: &[f64], c: C) -> Self
    where
        C: Fn(&[f64], &mut [f64]) + Sync + 'a,
    {
        assert_eq!(
            cl.len(),
            cu.len(),
            "constraint bounds must have equal length"
        );
        self.m = cl.len();
        self.cl = cl.to_vec();
        self.cu = cu.to_vec();
        self.cons = Some(Box::new(c));
        self
    }

    /// Add `m` equality constraints `c(x) = 0`.
    #[must_use]
    pub fn equality<C>(self, m: usize, c: C) -> Self
    where
        C: Fn(&[f64], &mut [f64]) + Sync + 'a,
    {
        let z = vec![0.0; m];
        self.with_constraints(&z.clone(), &z, c)
    }

    /// Add `m` inequality constraints `c(x) <= 0`, the `fmincon` convention.
    #[must_use]
    pub fn inequality<C>(self, m: usize, c: C) -> Self
    where
        C: Fn(&[f64], &mut [f64]) + Sync + 'a,
    {
        let lo = vec![-INF_BOUND; m];
        let hi = vec![0.0; m];
        self.with_constraints(&lo, &hi, c)
    }

    /// Declare the Jacobian sparsity. The highest-value thing to supply for a
    /// large model.
    #[must_use]
    pub fn with_jacobian_pattern(mut self, p: Sparsity) -> Self {
        self.jac_pattern = Some(p);
        self
    }

    /// Declare typical variable magnitudes (`fmincon`'s `TypicalX`).
    #[must_use]
    pub fn with_typical_x(mut self, t: &[f64]) -> Self {
        assert_eq!(t.len(), self.n);
        self.typical = Some(t.to_vec());
        self
    }

    /// Declare whether the closures may be called concurrently. Default `true`;
    /// set `false` for anything holding a lock, the Python GIL, or a
    /// non-reentrant simulator.
    #[must_use]
    pub fn parallel_safe(mut self, yes: bool) -> Self {
        self.parallel_safe = yes;
        self
    }
}

impl Nlp for Problem<'_> {
    fn dims(&self) -> NlpDims {
        NlpDims {
            n: self.n,
            m: self.m,
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
            gradient: self.grad.is_some(),
            parallel_safe: self.parallel_safe,
            ..Capabilities::none()
        }
    }
    fn objective(&self, x: &[f64]) -> Result<f64, EvalError> {
        let v = (self.f)(x);
        if v.is_finite() {
            Ok(v)
        } else {
            Err(EvalError::NonFinite(None))
        }
    }
    fn gradient(&self, x: &[f64], out: &mut [f64]) -> Result<(), EvalError> {
        match &self.grad {
            Some(g) => {
                g(x, out);
                if out.iter().all(|v| v.is_finite()) {
                    Ok(())
                } else {
                    Err(EvalError::NonFinite(None))
                }
            }
            None => Err(EvalError::Failed("no gradient supplied".into())),
        }
    }
    fn constraints(&self, x: &[f64], out: &mut [f64]) -> Result<(), EvalError> {
        match &self.cons {
            Some(c) => {
                c(x, out);
                if out.iter().all(|v| v.is_finite()) {
                    Ok(())
                } else {
                    Err(EvalError::NonFinite(None))
                }
            }
            None => Ok(()),
        }
    }
    fn jacobian_structure(&self) -> Option<&Sparsity> {
        self.jac_pattern.as_ref()
    }
    fn typical_x(&self) -> Option<&[f64]> {
        self.typical.as_deref()
    }
}

/// Multipliers in `fmincon`'s sign convention.
///
/// `fmincon` returns `lambda.ineqnonlin >= 0` for constraints written
/// `c(x) <= 0`, whereas the canonical form here uses a signed multiplier on
/// `c_L <= c(x) <= c_U`. Converting is a sign flip, and getting it wrong
/// produces multipliers that look plausible and are backwards — which is
/// exactly the kind of bug that survives review.
#[derive(Debug, Clone, Default)]
pub struct FminconMultipliers {
    /// Multipliers for `c(x) <= 0`, non-negative.
    pub ineqnonlin: Vec<f64>,
    /// Multipliers for `ceq(x) = 0`.
    pub eqnonlin: Vec<f64>,
    /// Multipliers for active lower bounds, non-negative.
    pub lower: Vec<f64>,
    /// Multipliers for active upper bounds, non-negative.
    pub upper: Vec<f64>,
}

/// Convert a canonical solution's multipliers into `fmincon`'s convention.
///
/// `ineq_rows` and `eq_rows` give the positions of the inequality and equality
/// constraints in the canonical constraint vector.
#[must_use]
pub fn to_fmincon_multipliers(
    solution: &mincon_core::Solution,
    ineq_rows: &[usize],
    eq_rows: &[usize],
) -> FminconMultipliers {
    FminconMultipliers {
        // Canonical lambda is positive at an active upper bound c <= c_U. For
        // fmincon's c(x) <= 0 that is the same active side, and fmincon reports
        // it non-negative, so the value carries over directly.
        ineqnonlin: ineq_rows
            .iter()
            .map(|&i| solution.lambda.get(i).copied().unwrap_or(0.0).max(0.0))
            .collect(),
        eqnonlin: eq_rows
            .iter()
            .map(|&i| solution.lambda.get(i).copied().unwrap_or(0.0))
            .collect(),
        lower: solution.z_l.clone(),
        upper: solution.z_u.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_produces_a_valid_nlp() {
        let p = Problem::new(2, |x| x[0] * x[0] + x[1] * x[1])
            .start_at(&[1.0, 1.0])
            .lower_bounds(&[-1.0, -1.0])
            .upper_bounds(&[1.0, 1.0])
            .equality(1, |x, c| c[0] = x[0] + x[1] - 1.0);
        assert_eq!(p.dims(), NlpDims { n: 2, m: 1 });
        assert_eq!(p.c_bounds().0, &[0.0]);
        assert_eq!(p.c_bounds().1, &[0.0]);
        mincon_core::validate(&p).unwrap();
    }

    #[test]
    fn inequality_uses_the_fmincon_convention() {
        let p = Problem::new(1, |x| x[0]).inequality(2, |x, c| {
            c[0] = x[0] - 1.0;
            c[1] = -x[0];
        });
        // c(x) <= 0 becomes (-inf, 0].
        assert_eq!(p.c_bounds().1, &[0.0, 0.0]);
        assert!(p.c_bounds().0.iter().all(|v| *v <= -INF_BOUND));
    }

    #[test]
    fn capabilities_track_what_was_supplied() {
        let p = Problem::new(1, |x| x[0] * x[0]);
        assert!(!p.capabilities().gradient);
        let p = p.with_gradient(|x, g| g[0] = 2.0 * x[0]);
        assert!(p.capabilities().gradient);
        let mut g = [0.0];
        Nlp::gradient(&p, &[3.0], &mut g).unwrap();
        assert_eq!(g[0], 6.0);
    }

    #[test]
    fn non_finite_objective_becomes_an_eval_error() {
        let p = Problem::new(1, |x| x[0].ln());
        assert!(p.objective(&[-1.0]).is_err());
        assert!(p.objective(&[1.0]).is_ok());
    }

    #[test]
    fn multiplier_conversion_keeps_signs_straight() {
        let s = mincon_core::Solution {
            x: vec![],
            f: 0.0,
            c: vec![],
            lambda: vec![2.0, -3.0, 0.5],
            z_l: vec![1.0],
            z_u: vec![0.0],
        };
        let m = to_fmincon_multipliers(&s, &[0, 2], &[1]);
        assert_eq!(m.ineqnonlin, vec![2.0, 0.5]);
        assert_eq!(m.eqnonlin, vec![-3.0]);
        assert!(m.ineqnonlin.iter().all(|v| *v >= 0.0));
    }
}
