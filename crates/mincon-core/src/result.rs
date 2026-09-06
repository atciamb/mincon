//! Result types.

use std::time::Duration;

use crate::Algorithm;

/// Why the solve stopped.
///
/// Numeric values match `fmincon`'s exit flags where the meanings coincide, so
/// that a user porting code can compare `report.exit_flag as i32` against the
/// integer they were checking before. New states use values `fmincon` does not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ExitFlag {
    /// First-order optimality and feasibility both satisfied. `fmincon` 1.
    Optimal = 1,
    /// Step size fell below tolerance while feasible. `fmincon` 2.
    StepTolerance = 2,
    /// Objective change fell below tolerance while feasible. `fmincon` 3.
    FunctionTolerance = 3,
    /// Stopped at a point meeting the relaxed "acceptable" tolerances.
    /// No `fmincon` equivalent; reported separately rather than dressed up as
    /// success, because silently loosening tolerances is how benchmark tables
    /// get gamed.
    Acceptable = 6,
    /// Iteration or evaluation budget exhausted. `fmincon` 0.
    MaxReached = 0,
    /// Stopped by a user callback. `fmincon` -1.
    StoppedByUser = -1,
    /// No feasible point found. `fmincon` -2.
    Infeasible = -2,
    /// Objective appears unbounded below. `fmincon` -3.
    Unbounded = -3,
    /// Converged to a point that is a local minimum of infeasibility but not
    /// feasible — the restoration phase succeeded at its own problem and the
    /// original constraints are locally inconsistent.
    LocallyInfeasible = -4,
    /// The solver failed for a numerical reason and said so honestly.
    NumericalFailure = -5,
}

impl ExitFlag {
    /// Whether this outcome counts as a success for benchmarking. Deliberately
    /// strict: [`ExitFlag::Acceptable`] does **not** count. The benchmark
    /// harness reports both a strict and a relaxed success rate.
    #[must_use]
    pub fn is_success(self) -> bool {
        matches!(self, ExitFlag::Optimal)
    }

    /// Whether a usable point was returned, even if not certified optimal.
    #[must_use]
    pub fn returned_usable_point(self) -> bool {
        matches!(
            self,
            ExitFlag::Optimal
                | ExitFlag::StepTolerance
                | ExitFlag::FunctionTolerance
                | ExitFlag::Acceptable
        )
    }

    /// A short human-readable message.
    #[must_use]
    pub fn message(self) -> &'static str {
        match self {
            ExitFlag::Optimal => {
                "Local minimum found. First-order optimality and constraints satisfied."
            }
            ExitFlag::StepTolerance => {
                "Local minimum possible. Step size below tolerance; constraints satisfied."
            }
            ExitFlag::FunctionTolerance => {
                "Local minimum possible. Objective change below tolerance; constraints satisfied."
            }
            ExitFlag::Acceptable => {
                "Solved to acceptable tolerances, but not to the requested optimality tolerance."
            }
            ExitFlag::MaxReached => "Iteration or evaluation limit reached.",
            ExitFlag::StoppedByUser => "Stopped by user callback.",
            ExitFlag::Infeasible => "No feasible point found.",
            ExitFlag::Unbounded => "Objective appears unbounded below.",
            ExitFlag::LocallyInfeasible => {
                "Converged to a local minimum of constraint violation; problem is locally infeasible."
            }
            ExitFlag::NumericalFailure => "Numerical difficulties; solve abandoned.",
        }
    }
}

/// One row of the iteration trace.
///
/// This is deliberately the same set of columns `fmincon`'s `Display='iter'`
/// prints, plus the regularization and barrier internals that `fmincon` hides
/// and that are the first thing anyone debugging a hard model wants.
#[derive(Debug, Clone, Copy)]
pub struct IterationRecord {
    /// Iteration index.
    pub iter: usize,
    /// Cumulative objective evaluations.
    pub f_count: u64,
    /// Objective value (unscaled).
    pub f: f64,
    /// Max constraint violation (unscaled).
    pub constraint_violation: f64,
    /// Scaled KKT error `E_0`.
    pub optimality: f64,
    /// Step norm actually taken.
    pub step_norm: f64,
    /// Accepted step length in `(0, 1]`.
    pub alpha: f64,
    /// Barrier parameter, or `NaN` for algorithms without one.
    pub mu: f64,
    /// Primal regularization applied to the KKT matrix this iteration.
    pub delta_w: f64,
    /// Dual regularization applied this iteration.
    pub delta_c: f64,
    /// Whether the iteration was spent in the feasibility restoration phase.
    pub in_restoration: bool,
    /// Number of second-order corrections used.
    pub soc_count: usize,
}

/// Where the time went. Reported always, because "is it the solver or my
/// model?" is unanswerable without it, and it is the number that decides
/// whether a user's speed complaint is ours to fix.
#[derive(Debug, Clone, Copy, Default)]
pub struct Timings {
    /// Total wall clock.
    pub total: Duration,
    /// Inside user model evaluations.
    pub model: Duration,
    /// Inside symbolic + numeric factorization.
    pub factorization: Duration,
    /// Inside triangular solves and iterative refinement.
    pub solves: Duration,
    /// Inside derivative approximation (finite differences, coloring).
    pub derivatives: Duration,
}

/// The point returned by a solve.
#[derive(Debug, Clone)]
pub struct Solution {
    /// The variables.
    pub x: Vec<f64>,
    /// Objective value at `x`.
    pub f: f64,
    /// Constraint values at `x`.
    pub c: Vec<f64>,
    /// Multipliers for the general constraints, sign convention:
    /// stationarity is `grad f + J^T lambda - z_L + z_U = 0`, so `lambda[i]`
    /// is positive at an active upper bound `c_i <= c_U` and negative at an
    /// active lower bound. This is the IPOPT/AMPL convention.
    ///
    /// Note this is the **opposite sign** to `fmincon`'s `lambda.ineqnonlin`,
    /// which is non-negative for `c(x) <= 0`. The compatibility façade flips it.
    pub lambda: Vec<f64>,
    /// Multipliers for the lower variable bounds, non-negative.
    pub z_l: Vec<f64>,
    /// Multipliers for the upper variable bounds, non-negative.
    pub z_u: Vec<f64>,
}

/// Everything a solve produces.
#[derive(Debug, Clone)]
pub struct SolveReport {
    /// The point, multipliers and objective.
    pub solution: Solution,
    /// Why it stopped.
    pub exit_flag: ExitFlag,
    /// Which algorithm produced this answer. Meaningful when the portfolio ran.
    pub algorithm: Algorithm,
    /// Iterations taken.
    pub iterations: usize,
    /// Objective evaluations.
    pub f_evals: u64,
    /// Gradient evaluations.
    pub g_evals: u64,
    /// Constraint evaluations.
    pub c_evals: u64,
    /// Jacobian evaluations.
    pub j_evals: u64,
    /// Hessian evaluations.
    pub h_evals: u64,
    /// Model evaluations that failed and forced a retreat.
    pub failed_evals: u64,
    /// Final scaled KKT error.
    pub optimality: f64,
    /// Final maximum constraint violation, unscaled.
    pub constraint_violation: f64,
    /// Final complementarity residual.
    pub complementarity: f64,
    /// Per-iteration trace, when `Options::record_trace`.
    pub trace: Vec<IterationRecord>,
    /// Where the time went.
    pub timings: Timings,
    /// Human-readable notes: scaling factors applied, restoration entries,
    /// derivative-check discrepancies, why the portfolio chose this answer.
    pub notes: Vec<String>,
}

impl SolveReport {
    /// A one-line summary in the shape `fmincon` users expect from
    /// `Display='final'`.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "{}\n  f = {:.10e}   max constraint violation = {:.3e}   first-order optimality = {:.3e}\n  {} iterations, {} objective evaluations, {:.3}s ({:?})",
            self.exit_flag.message(),
            self.solution.f,
            self.constraint_violation,
            self.optimality,
            self.iterations,
            self.f_evals,
            self.timings.total.as_secs_f64(),
            self.algorithm,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exit_flags_match_fmincon_integers() {
        assert_eq!(ExitFlag::Optimal as i32, 1);
        assert_eq!(ExitFlag::MaxReached as i32, 0);
        assert_eq!(ExitFlag::Infeasible as i32, -2);
        assert_eq!(ExitFlag::Unbounded as i32, -3);
    }

    #[test]
    fn acceptable_is_not_success() {
        assert!(!ExitFlag::Acceptable.is_success());
        assert!(ExitFlag::Acceptable.returned_usable_point());
        assert!(ExitFlag::Optimal.is_success());
        assert!(!ExitFlag::MaxReached.returned_usable_point());
    }
}
