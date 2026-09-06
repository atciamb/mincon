//! The quadratic programming subproblem solver used by SQP.
//!
//! # Status: specified, not implemented
//!
//! # The problem
//!
//! ```text
//!   min_d  1/2 d^T H d + g^T d
//!   s.t.   a_L <= A d <= a_U
//!          d_L <=  d  <= d_U
//! ```
//!
//! `H` is symmetric, positive semidefinite when it comes from a damped BFGS
//! update and possibly indefinite when it comes from an exact Hessian.
//!
//! # Design decision to make before writing any code
//!
//! Two families, and the choice is not obvious:
//!
//! **Active set (dual or primal).** What `fmincon`'s `active-set` and `sqp`
//! algorithms use, and what SNOPT uses. Warm starts beautifully — the whole
//! point of SQP is that consecutive subproblems differ slightly, and an
//! active-set method can start from the previous active set and finish in a
//! handful of pivots. Worst-case exponential, and on a degenerate QP it can
//! cycle without anti-cycling rules (Bland, or a lexicographic rule). This is
//! the right choice for the small dense subproblems SQP actually generates.
//!
//! **Interior point.** Polynomial, insensitive to the size of the active set,
//! and it can reuse `mincon-linalg` directly since the KKT system has the same
//! shape as the NLP one. Warm starts badly, which forfeits SQP's main
//! advantage.
//!
//! **Recommendation: a dual active-set method with warm starting, and an
//! interior-point fallback** when the active-set method exceeds an iteration
//! budget. That combination is what makes SQP worth having alongside the
//! interior-point NLP solver rather than a slower copy of it.
//!
//! Do not reach for an external QP solver. `osqp-rust` is Apache-2.0 and good,
//! but it is an ADMM method: first-order, excellent for moderate accuracy on
//! large problems, and not accurate enough for the tail of an SQP where the
//! subproblem must be solved to high precision for superlinear convergence. It
//! is a reasonable *fallback*, not the primary.
//!
//! # Acceptance gate
//!
//! * Solves a random dense QP to `1e-10` KKT residual, cross-checked against a
//!   dense KKT solve on the correct active set.
//! * Warm-started from a neighbouring QP, takes strictly fewer iterations.
//! * Terminates on the standard degenerate/cycling QPs rather than looping.
//! * Detects an infeasible QP and says so, since SQP relies on that to trigger
//!   its elastic reformulation.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use mincon_core::SolveError;

/// A quadratic program in the form documented at the crate root.
#[derive(Debug, Clone)]
pub struct QuadraticProgram {
    /// Number of variables.
    pub n: usize,
    /// Number of general constraints.
    pub m: usize,
}

/// The outcome of a QP solve.
#[derive(Debug, Clone)]
pub struct QpSolution {
    /// The step.
    pub d: Vec<f64>,
    /// Multipliers for the general constraints.
    pub lambda: Vec<f64>,
    /// Indices of the constraints active at the solution, for warm starting the
    /// next subproblem.
    pub active_set: Vec<usize>,
    /// Iterations taken.
    pub iterations: usize,
}

/// Solve a QP.
///
/// # Errors
/// Always, for now: not implemented.
pub fn solve(_qp: &QuadraticProgram) -> Result<QpSolution, SolveError> {
    Err(SolveError::InvalidOptions(
        "the QP subproblem solver is not implemented yet; see docs/03_SPEC_SQP.md".into(),
    ))
}

/// Whether this solver is available.
#[must_use]
pub fn is_available() -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_itself_unavailable() {
        assert!(!is_available());
        assert!(solve(&QuadraticProgram { n: 1, m: 0 }).is_err());
    }
}
