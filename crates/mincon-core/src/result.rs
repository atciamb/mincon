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
    /// Restoration reached stationary positive constraint violation. This is
    /// a first-order local diagnostic, not a proof of infeasibility or a
    /// second-order minimum certificate.
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
                "First-order optimality and constraint tolerances satisfied."
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
                "Restoration reached stationary positive constraint violation; feasibility is unresolved beyond this local diagnostic."
            }
            ExitFlag::NumericalFailure => "Numerical difficulties; solve abandoned.",
        }
    }
}

/// Which limit ended a solve that stopped with [`ExitFlag::MaxReached`].
///
/// Three limits share that exit flag, and a user who set one of them needs to
/// know whether it was the one that bound: a solve that stopped on the clock
/// after 17 % of its evaluation budget is a different situation from one that
/// spent the budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Limit {
    /// `Options::max_iterations`, or the `n`-scaled default in its place.
    Iterations,
    /// `Options::max_evaluations`.
    Evaluations,
    /// `Options::max_seconds`.
    Time,
}

impl Limit {
    /// The lowercase name used in reports and in the Python result.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Limit::Iterations => "iterations",
            Limit::Evaluations => "evaluations",
            Limit::Time => "time",
        }
    }

    /// Which of the three limits bound, given whether each fired. Several can
    /// fire on the same iteration; the evaluation budget is named first
    /// because it is the reproducible one, the clock second, and the
    /// iteration cap last.
    #[must_use]
    pub fn which(evaluations: bool, time: bool, iterations: bool) -> Option<Self> {
        if evaluations {
            Some(Limit::Evaluations)
        } else if time {
            Some(Limit::Time)
        } else if iterations {
            Some(Limit::Iterations)
        } else {
            None
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
    /// This matches `fmincon` for upper inequalities `c(x) <= 0`. Python's
    /// SciPy-style inequalities `c(x) >= 0` are lower bounds and have negative
    /// multipliers in this convention.
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
    /// Which limit bound when `exit_flag` is [`ExitFlag::MaxReached`];
    /// `None` for every other exit.
    pub limit: Option<Limit>,
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
    /// The quasi-Newton model of the Lagrangian Hessian at the exit, dense
    /// row-major `n x n` in the user's units and variables, when the member
    /// used one and `n <= 1000`; a warm start (`Options::warm_start`) resumes
    /// from it. `None` with an exact Hessian or a larger `n`.
    pub quasi_newton: Option<Vec<f64>>,
}

impl SolveReport {
    /// The termination message. The same as [`ExitFlag::message`] except at a
    /// budget exit, where it names the limit that bound and how much of it was
    /// used, since three limits share [`ExitFlag::MaxReached`].
    #[must_use]
    pub fn message(&self) -> String {
        match (self.exit_flag, self.limit) {
            (ExitFlag::MaxReached, Some(Limit::Iterations)) => {
                format!("Iteration limit reached ({} iterations).", self.iterations)
            }
            (ExitFlag::MaxReached, Some(Limit::Evaluations)) => format!(
                "Evaluation limit reached ({} objective evaluations).",
                self.f_evals
            ),
            (ExitFlag::MaxReached, Some(Limit::Time)) => format!(
                "Time limit reached ({:.1} s, {} iterations, {} objective evaluations).",
                self.timings.total.as_secs_f64(),
                self.iterations,
                self.f_evals
            ),
            _ => self.exit_flag.message().to_string(),
        }
    }

    /// A one-line summary in the shape `fmincon` users expect from
    /// `Display='final'`.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "{}\n  f = {:.10e}   max constraint violation = {:.3e}   first-order optimality = {:.3e}\n  {} iterations, {} objective evaluations, {:.3}s ({:?})",
            self.message(),
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
