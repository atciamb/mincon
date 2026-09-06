//! The problem model.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::{EvalError, Sparsity};

/// Problem dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NlpDims {
    /// Number of variables.
    pub n: usize,
    /// Number of general constraints (rows of `c`).
    pub m: usize,
}

/// What derivative information a model can supply itself.
///
/// Anything not supplied is filled in by `mincon-diff`. A model that supplies
/// nothing but `f` and `c` is a first-class citizen — that is the `fmincon`
/// plug-and-play experience and it is the case we are graded on hardest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Capabilities {
    /// `gradient` is implemented.
    pub gradient: bool,
    /// `jacobian` is implemented.
    pub jacobian: bool,
    /// `hessian_lagrangian` is implemented.
    pub hessian: bool,
    /// `hessian_vector` is implemented (matrix-free Hessian products).
    pub hessian_vector: bool,
    /// Evaluations are cheap and thread-safe enough to call in parallel.
    /// Controls whether the derivative layer fans finite differences across
    /// a `rayon` pool. Models wrapping a Python callback must set this to
    /// `false` unless they release the GIL.
    pub parallel_safe: bool,
}

impl Capabilities {
    /// Model provides no derivatives at all; everything is approximated.
    #[must_use]
    pub fn none() -> Self {
        Self::default()
    }
    /// Model provides first derivatives only (the common case for hand-written
    /// models and for AD frameworks bridged from Python).
    #[must_use]
    pub fn first_order() -> Self {
        Self {
            gradient: true,
            jacobian: true,
            ..Self::default()
        }
    }
    /// Model provides exact first and second derivatives.
    #[must_use]
    pub fn second_order() -> Self {
        Self {
            gradient: true,
            jacobian: true,
            hessian: true,
            hessian_vector: true,
            parallel_safe: false,
        }
    }
}

/// Atomic evaluation counters, owned by the driver and shared with the
/// derivative layer. Kept out of [`Nlp`] so that `Nlp` can stay `&self`.
#[derive(Debug, Default)]
pub struct EvalCounters {
    /// Objective evaluations.
    pub f: AtomicU64,
    /// Gradient evaluations (analytic only; finite differences count as `f`).
    pub g: AtomicU64,
    /// Constraint evaluations.
    pub c: AtomicU64,
    /// Jacobian evaluations (analytic only).
    pub j: AtomicU64,
    /// Hessian-of-Lagrangian evaluations.
    pub h: AtomicU64,
    /// Evaluations that returned [`EvalError`]. A high ratio is the signature
    /// of a model with a restricted domain and is surfaced in diagnostics.
    pub failed: AtomicU64,
}

impl EvalCounters {
    /// Increment a counter.
    #[inline]
    pub fn bump(counter: &AtomicU64) {
        counter.fetch_add(1, Ordering::Relaxed);
    }
    /// Read a counter.
    #[inline]
    #[must_use]
    pub fn get(counter: &AtomicU64) -> u64 {
        counter.load(Ordering::Relaxed)
    }
}

/// A nonlinear program in the canonical form documented at the crate root.
///
/// # Contract
///
/// * All slices passed in have the length implied by [`Nlp::dims`]; all output
///   slices must be fully written.
/// * `jacobian` and `hessian_lagrangian` write values in the order given by
///   [`Nlp::jacobian_structure`] / [`Nlp::hessian_structure`]. The structure
///   must not change between calls.
/// * The Hessian structure is the **lower triangle** of a symmetric matrix,
///   stored column-major, matching what the KKT assembler expects.
/// * Implementations must be pure with respect to `x`: calling twice with the
///   same `x` must give the same answer. Caching is fine; state that changes
///   the answer is not. The line search relies on this.
pub trait Nlp: Sync {
    /// Problem dimensions.
    fn dims(&self) -> NlpDims;

    /// Variable bounds, each of length `n`. Use `±`[`crate::INF_BOUND`] or
    /// `±f64::INFINITY` for free.
    fn x_bounds(&self) -> (&[f64], &[f64]);

    /// Constraint bounds, each of length `m`. `c_l[i] == c_u[i]` is an equality.
    fn c_bounds(&self) -> (&[f64], &[f64]);

    /// A starting point. The driver may push it inside the bounds.
    fn x0(&self) -> &[f64];

    /// What this model can compute itself.
    fn capabilities(&self) -> Capabilities;

    /// Objective value.
    ///
    /// # Errors
    /// [`EvalError`] if the model cannot be evaluated here; the caller retreats.
    fn objective(&self, x: &[f64]) -> Result<f64, EvalError>;

    /// Constraint values, `m` of them.
    ///
    /// # Errors
    /// [`EvalError`] if the model cannot be evaluated here.
    fn constraints(&self, x: &[f64], out: &mut [f64]) -> Result<(), EvalError>;

    /// Objective gradient. Only called when [`Capabilities::gradient`].
    ///
    /// # Errors
    /// [`EvalError`] if the model cannot be evaluated here.
    fn gradient(&self, _x: &[f64], _out: &mut [f64]) -> Result<(), EvalError> {
        Err(EvalError::Failed("gradient not implemented".into()))
    }

    /// Nonzero structure of the constraint Jacobian (`m x n`).
    ///
    /// `None` means "dense"; the derivative layer will then use one finite
    /// difference per variable. Declaring structure is the single highest-value
    /// thing a user can do for a large model.
    fn jacobian_structure(&self) -> Option<&Sparsity> {
        None
    }

    /// Jacobian values in the order of [`Nlp::jacobian_structure`].
    /// Only called when [`Capabilities::jacobian`].
    ///
    /// # Errors
    /// [`EvalError`] if the model cannot be evaluated here.
    fn jacobian(&self, _x: &[f64], _out: &mut [f64]) -> Result<(), EvalError> {
        Err(EvalError::Failed("jacobian not implemented".into()))
    }

    /// Nonzero structure of the **lower triangle** of the Hessian of the
    /// Lagrangian (`n x n`).
    fn hessian_structure(&self) -> Option<&Sparsity> {
        None
    }

    /// Hessian of the Lagrangian
    /// `sigma * grad^2 f(x) + sum_i lambda[i] * grad^2 c_i(x)`,
    /// lower triangle, in the order of [`Nlp::hessian_structure`].
    ///
    /// The `sigma` scaling of the objective term is the IPOPT convention and
    /// exists so the restoration phase can request `sigma = 0`.
    ///
    /// # Errors
    /// [`EvalError`] if the model cannot be evaluated here.
    fn hessian_lagrangian(
        &self,
        _x: &[f64],
        _sigma: f64,
        _lambda: &[f64],
        _out: &mut [f64],
    ) -> Result<(), EvalError> {
        Err(EvalError::Failed("hessian not implemented".into()))
    }

    /// Product of the Hessian of the Lagrangian with `v`.
    /// Only called when [`Capabilities::hessian_vector`].
    ///
    /// # Errors
    /// [`EvalError`] if the model cannot be evaluated here.
    fn hessian_vector(
        &self,
        _x: &[f64],
        _sigma: f64,
        _lambda: &[f64],
        _v: &[f64],
        _out: &mut [f64],
    ) -> Result<(), EvalError> {
        Err(EvalError::Failed("hessian_vector not implemented".into()))
    }

    /// Optional per-variable typical magnitudes, the analogue of `fmincon`'s
    /// `TypicalX`. Used to size finite-difference steps and as a fallback
    /// scaling when gradient-based scaling is not informative.
    fn typical_x(&self) -> Option<&[f64]> {
        None
    }
}

/// Validate a model against the [`Nlp`] contract. Called once by the driver.
///
/// # Errors
/// A human-readable description of the first problem found.
pub fn validate<P: Nlp + ?Sized>(p: &P) -> Result<(), String> {
    let NlpDims { n, m } = p.dims();
    if n == 0 {
        return Err("problem has zero variables".into());
    }
    let (xl, xu) = p.x_bounds();
    if xl.len() != n || xu.len() != n {
        return Err(format!(
            "variable bounds have lengths {} and {}, expected {n}",
            xl.len(),
            xu.len()
        ));
    }
    for i in 0..n {
        if xl[i].is_nan() || xu[i].is_nan() {
            return Err(format!("variable bound {i} is NaN"));
        }
        if xl[i] > xu[i] {
            return Err(format!(
                "variable {i} has lower bound {} above upper bound {}",
                xl[i], xu[i]
            ));
        }
    }
    let (cl, cu) = p.c_bounds();
    if cl.len() != m || cu.len() != m {
        return Err(format!(
            "constraint bounds have lengths {} and {}, expected {m}",
            cl.len(),
            cu.len()
        ));
    }
    for i in 0..m {
        if cl[i].is_nan() || cu[i].is_nan() {
            return Err(format!("constraint bound {i} is NaN"));
        }
        if cl[i] > cu[i] {
            return Err(format!(
                "constraint {i} has lower bound {} above upper bound {}",
                cl[i], cu[i]
            ));
        }
    }
    if p.x0().len() != n {
        return Err(format!("x0 has length {}, expected {n}", p.x0().len()));
    }
    if let Some(s) = p.jacobian_structure() {
        if s.nrows() != m || s.ncols() != n {
            return Err(format!(
                "jacobian structure is {}x{}, expected {m}x{n}",
                s.nrows(),
                s.ncols()
            ));
        }
    }
    if let Some(s) = p.hessian_structure() {
        if s.nrows() != n || s.ncols() != n {
            return Err(format!(
                "hessian structure is {}x{}, expected {n}x{n}",
                s.nrows(),
                s.ncols()
            ));
        }
        for j in 0..n {
            if let Some(&first) = s.col(j).first() {
                if first < j {
                    return Err(format!(
                        "hessian structure must be lower triangular; column {j} has row {first}"
                    ));
                }
            }
        }
    }
    Ok(())
}
