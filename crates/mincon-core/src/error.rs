//! Error types.
//!
//! The distinction that matters: [`EvalError`] is *recoverable* — it means "the
//! model could not be evaluated at this point", and every globalization loop in
//! the workspace must respond by shortening the step, never by aborting.
//! [`SolveError`] is a genuine failure of the solver or of the problem setup.
//!
//! This mirrors the single most under-appreciated robustness feature of
//! `fmincon`'s `sqp` algorithm: when the user function returns `Inf`, `NaN` or
//! a complex value, it retries with a smaller step rather than giving up.
//! Practical engineering models (implicit solves, log/sqrt of an intermediate,
//! FEA that diverges) hit this constantly, and it is a large part of why
//! `fmincon` feels more robust than solvers with better convergence theory.

use thiserror::Error;

/// A failure to evaluate the model at a point. Always recoverable by retreating.
#[derive(Debug, Clone, Error)]
pub enum EvalError {
    /// The model returned a non-finite value (`NaN` or `±Inf`).
    ///
    /// Carries the index of the first offending entry when known, for
    /// diagnostics; solvers must not branch on it.
    #[error("model returned a non-finite value{}", .0.map(|i| format!(" at index {i}")).unwrap_or_default())]
    NonFinite(Option<usize>),

    /// The point is outside the model's domain (e.g. a negative argument to a
    /// square root). Semantically identical to [`EvalError::NonFinite`] for the
    /// solver, but lets a user model say so explicitly.
    #[error("point is outside the model domain: {0}")]
    OutOfDomain(String),

    /// The model itself failed for a reason of its own (a simulation did not
    /// converge, a subprocess died, a Python callback raised).
    #[error("model evaluation failed: {0}")]
    Failed(String),

    /// The user requested that the solve stop (output/plot callback returned
    /// "stop"). This is *not* retried; it propagates to the driver, which
    /// finishes with [`crate::ExitFlag::StoppedByUser`].
    #[error("stopped by user callback")]
    UserAbort,
}

impl EvalError {
    /// Whether the globalization loop should retreat and retry, as opposed to
    /// unwinding to the driver.
    #[must_use]
    pub fn is_retryable(&self) -> bool {
        !matches!(self, EvalError::UserAbort)
    }
}

/// A failure of the solve as a whole.
#[derive(Debug, Clone, Error)]
pub enum SolveError {
    /// The problem definition is inconsistent (mismatched lengths, `lb > ub`,
    /// non-finite bounds where finite ones are required, and so on).
    #[error("invalid problem: {0}")]
    InvalidProblem(String),

    /// The options are inconsistent (e.g. an algorithm that requires a Hessian
    /// paired with a model that cannot supply one and derivative
    /// approximation disabled).
    #[error("invalid options: {0}")]
    InvalidOptions(String),

    /// The linear solver failed in a way the regularization loop could not
    /// repair. Carries the context so a bug report is actionable.
    #[error("linear algebra failure: {0}")]
    LinearAlgebra(String),

    /// An unrecoverable model evaluation failure at the starting point. Every
    /// other evaluation failure is handled by retreating.
    #[error("model could not be evaluated at the initial point: {0}")]
    InitialPoint(#[source] EvalError),

    /// Internal invariant violated. Always a bug in `mincon`; never triggered
    /// by user input.
    #[error("internal error: {0}")]
    Internal(String),
}
