//! # mincon
//!
//! A nonlinear constrained optimizer, aimed squarely at what MATLAB's
//! `fmincon` does well: solving the problem a working engineer or scientist
//! actually has, without being told how.
//!
//! ```text
//!   minimize    f(x)
//!   subject to  c_L <= c(x) <= c_U
//!               x_L <=   x  <= x_U
//! ```
//!
//! ```
//! use mincon::{minimize, Options, Problem};
//!
//! // Rosenbrock on the unit disc.
//! let p = Problem::new(2, |x| {
//!         100.0 * (x[1] - x[0] * x[0]).powi(2) + (1.0 - x[0]).powi(2)
//!     })
//!     .start_at(&[-1.2, 1.0])
//!     .inequality(1, |x, c| c[0] = x[0] * x[0] + x[1] * x[1] - 1.0);
//!
//! let r = minimize(&p, &Options::default()).unwrap();
//! assert!(r.exit_flag.returned_usable_point());
//! assert!(r.constraint_violation < 1e-6);
//! ```
//!
//! # Current status
//!
//! Experimental constrained optimizer with interior-point steps, feasibility
//! restoration, gradient-based scaling, finite differences and a portfolio
//! of interior-point configurations. See the repository README for current
//! validation results and implementation limitations.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod builder;
pub mod portfolio;

pub use builder::{to_fmincon_multipliers, FminconMultipliers, Problem};
pub use mincon_core::{
    Algorithm, Capabilities, Display, EvalError, ExitFlag, FdType, HessianMode, IterationRecord,
    LinearSolverKind, Nlp, NlpDims, Options, RegularizationMode, ScalingMode, Solution, SolveError,
    SolveReport, Sparsity, Timings, Tolerances,
};
pub use mincon_diff::{check_derivatives, CheckReport};
pub use portfolio::PortfolioReport;

/// Solve a problem.
///
/// With [`Algorithm::Auto`] (the default) this runs the portfolio described in
/// [`portfolio`]; with a specific algorithm it runs exactly that one.
///
/// # Errors
/// [`SolveError`] when the problem or options are malformed, or when every
/// algorithm failed outright. A solve that ran but did not converge is *not* an
/// error — check [`SolveReport::exit_flag`].
pub fn minimize<P: Nlp + Sync + ?Sized>(
    nlp: &P,
    opts: &Options,
) -> Result<SolveReport, SolveError> {
    match opts.algorithm {
        Algorithm::Auto => portfolio::solve(nlp, opts).map(|p| p.best),
        Algorithm::InteriorPoint => mincon_ip::solve(nlp, opts),
        Algorithm::Sqp | Algorithm::Slqp => mincon_sqp::solve(nlp, opts),
    }
}

/// Solve and get the full portfolio outcome, including what every member did.
///
/// # Errors
/// As [`minimize`].
pub fn minimize_with_portfolio<P: Nlp + Sync + ?Sized>(
    nlp: &P,
    opts: &Options,
) -> Result<PortfolioReport, SolveError> {
    portfolio::solve(nlp, opts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solves_an_equality_constrained_quadratic_exactly() {
        // min x0^2 + x1^2 s.t. x0 + x1 = 2  ->  x = (1, 1), f = 2.
        let p = Problem::new(2, |x| x[0] * x[0] + x[1] * x[1])
            .start_at(&[3.0, -1.0])
            .equality(1, |x, c| c[0] = x[0] + x[1] - 2.0);
        let r = minimize(&p, &Options::default()).unwrap();
        assert_eq!(r.exit_flag, ExitFlag::Optimal);
        assert!((r.solution.f - 2.0).abs() < 1e-8, "f = {}", r.solution.f);
        assert!((r.solution.x[0] - 1.0).abs() < 1e-6);
        assert!((r.solution.x[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn respects_bounds_at_every_iterate() {
        // The objective is NaN below zero; only a bound-honouring solver survives.
        let p = Problem::new(1, |x| {
            assert!(
                x[0] >= 0.0,
                "solver evaluated outside the bounds at {}",
                x[0]
            );
            (x[0] - 3.0).powi(2)
        })
        .start_at(&[0.5])
        .lower_bounds(&[0.0])
        .upper_bounds(&[2.0]);
        let r = minimize(&p, &Options::default()).unwrap();
        assert!(
            (r.solution.x[0] - 2.0).abs() < 1e-6,
            "x = {:?}",
            r.solution.x
        );
    }

    #[test]
    fn analytic_gradients_are_used_and_cost_fewer_evaluations() {
        let obj = |x: &[f64]| (x[0] - 1.0).powi(2) + 10.0 * (x[1] - 2.0).powi(2);
        let without = minimize(
            &Problem::new(2, obj).start_at(&[0.0, 0.0]),
            &Options {
                algorithm: Algorithm::InteriorPoint,
                ..Options::default()
            },
        )
        .unwrap();
        let with = minimize(
            &Problem::new(2, obj)
                .start_at(&[0.0, 0.0])
                .with_gradient(|x, g| {
                    g[0] = 2.0 * (x[0] - 1.0);
                    g[1] = 20.0 * (x[1] - 2.0);
                }),
            &Options {
                algorithm: Algorithm::InteriorPoint,
                ..Options::default()
            },
        )
        .unwrap();
        assert!(
            with.f_evals < without.f_evals,
            "{} vs {}",
            with.f_evals,
            without.f_evals
        );
        assert!(with.g_evals > 0, "analytic gradient should have been used");
        assert!((with.solution.f - without.solution.f).abs() < 1e-6);
    }

    #[test]
    fn portfolio_reports_every_member_and_the_true_cost() {
        let p = Problem::new(2, |x| (x[0] - 1.0).powi(2) + (x[1] + 1.0).powi(2))
            .start_at(&[5.0, 5.0])
            .inequality(1, |x, c| c[0] = x[0] + x[1] - 1.0);
        let r = minimize_with_portfolio(&p, &Options::default()).unwrap();
        assert!(r.best.exit_flag.returned_usable_point());
        assert!(!r.outcomes.is_empty());
        assert!(
            r.total_f_evals >= r.best.f_evals,
            "total cost must include every member"
        );
        assert!(r.best.notes.iter().any(|n| n.contains("portfolio")));
    }

    #[test]
    fn the_whole_test_set_still_passes_through_the_public_api() {
        // A coarse regression on the public entry point rather than the
        // internal one; catches façade-level mistakes such as options not
        // being forwarded.
        let mut solved = 0;
        let mut total = 0;
        let opts = Options {
            algorithm: Algorithm::InteriorPoint,
            max_seconds: Some(10.0),
            ..Options::default()
        };
        for prob in mincon_testset::hs::all() {
            total += 1;
            let nlp = prob.as_nlp();
            if let Ok(r) = minimize(&nlp, &opts) {
                let ok = match prob.expect {
                    mincon_testset::Expect::Optimum => prob.f_opt.is_some_and(|fo| {
                        r.constraint_violation < 1e-5
                            && (r.solution.f - fo).abs() / fo.abs().max(1.0) < 1e-4
                    }),
                    _ => r.constraint_violation < 1e-5 && r.exit_flag.returned_usable_point(),
                };
                if ok {
                    solved += 1;
                }
            }
        }
        assert!(
            solved * 100 >= total * 95,
            "Hock-Schittkowski pass rate regressed: {solved}/{total}"
        );
    }

    #[test]
    fn an_infeasible_problem_is_never_reported_as_solved() {
        let p = Problem::new(2, |x| x[0] * x[0] + x[1] * x[1])
            .start_at(&[0.0, 0.0])
            .equality(2, |x, c| {
                c[0] = x[0] + x[1] - 1.0;
                c[1] = x[0] + x[1] - 3.0;
            });
        let r = minimize(&p, &Options::default()).unwrap();
        assert!(
            !r.exit_flag.is_success(),
            "reported {:?} on an infeasible problem",
            r.exit_flag
        );
        assert!(r.constraint_violation > 0.1);
    }
}
