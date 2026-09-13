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
pub mod quadratic;
pub mod scaled;

pub use builder::{to_fmincon_multipliers, FminconMultipliers, Problem};
pub use mincon_core::{
    Algorithm, Capabilities, DerivativeCheck, Display, EvalError, ExitFlag, FdType, HessianMode,
    IterationRecord, LinearSolverKind, Nlp, NlpDims, Options, RegularizationMode, ScalingMode,
    Solution, SolveError, SolveReport, Sparsity, Timings, Tolerances, VariableScaling,
};
pub use mincon_diff::{check_derivatives, check_derivatives_directional, CheckReport};
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
    if opts.scale_variables != VariableScaling::Off {
        if let Some(d) = scaled::factors_from_start(nlp, opts.scale_variables) {
            let s = scaled::ScaledNlp::new(nlp, d);
            let scaled_opts;
            let opts = if let Some(w) = opts
                .warm_start
                .as_ref()
                .filter(|w| w.quasi_newton.is_some())
            {
                let mut w = w.clone();
                if let Some(q) = w.quasi_newton.as_mut() {
                    s.scale_hessian(q);
                }
                scaled_opts = Options {
                    warm_start: Some(w),
                    ..opts.clone()
                };
                &scaled_opts
            } else {
                opts
            };
            let mut r = minimize_unscaled(&s, opts)?;
            r.solution.x = s.unscale_x(&r.solution.x);
            r.solution.z_l = s.unscale_bound_multipliers(&r.solution.z_l);
            r.solution.z_u = s.unscale_bound_multipliers(&r.solution.z_u);
            if let Some(q) = r.quasi_newton.as_mut() {
                s.unscale_hessian(q);
            }
            let (lo, hi) = s
                .factors()
                .iter()
                .fold((f64::INFINITY, 0.0_f64), |(lo, hi), v| {
                    (lo.min(*v), hi.max(*v))
                });
            r.notes.insert(
                0,
                format!(
                    "Variables scaled by their starting magnitudes (factors from {lo:.2e} to {hi:.2e}); the trace's step norms are in scaled units."
                ),
            );
            return Ok(r);
        }
    }
    minimize_unscaled(nlp, opts)
}

fn minimize_unscaled<P: Nlp + Sync + ?Sized>(
    nlp: &P,
    opts: &Options,
) -> Result<SolveReport, SolveError> {
    let mut check_note: Option<String> = None;
    match opts.check_derivatives {
        DerivativeCheck::Off => {}
        DerivativeCheck::Full => {
            // `fmincon`'s CheckGradients semantics: a failed check stops the solve. An
            // inconclusive check (no analytic derivatives, or callbacks that failed) is
            // also a stop, because "could not verify" must never read as "verified".
            let report = check_derivatives(nlp, 1e-5, 3).map_err(SolveError::InitialPoint)?;
            if !report.passed() {
                return Err(SolveError::InvalidProblem(report.message()));
            }
        }
        DerivativeCheck::Directional => {
            let caps = nlp.capabilities();
            if caps.gradient || (caps.jacobian && nlp.dims().m > 0) {
                let report =
                    check_derivatives_directional(nlp, 1e-5).map_err(SolveError::InitialPoint)?;
                if report.conclusive() && !report.passed() {
                    if report.max_relative > GROSS_DERIVATIVE_ERROR || report.nonfinite > 0 {
                        // Name the components: the full check at x0 only.
                        let full = check_derivatives(nlp, 1e-5, 1)
                            .map(|r| r.message())
                            .unwrap_or_else(|e| format!("(the full check could not run: {e})"));
                        return Err(SolveError::InvalidProblem(format!(
                            "The supplied derivatives disagree with finite differences along a test direction at x0 (relative error {:.2e}). {full}",
                            report.max_relative
                        )));
                    }
                    check_note = Some(format!(
                        "Derivative check: the supplied derivatives disagree with a central difference along a test direction at x0 by {:.2e} relative (tolerance 1e-5). The solve continued; if it stalls, check the derivatives with options check_derivatives = Full, or expect this much noise from the model.",
                        report.max_relative
                    ));
                } else if let Some(reason) = report.inconclusive_reason {
                    check_note = Some(format!("Derivative check skipped: {reason}."));
                }
            }
        }
    }
    let mut report = match opts.algorithm {
        Algorithm::Auto => portfolio::solve(nlp, opts).map(|p| p.best),
        Algorithm::InteriorPoint => mincon_ip::solve(nlp, opts),
        Algorithm::Sqp | Algorithm::Slqp => mincon_sqp::solve(nlp, opts),
    }?;
    if let Some(note) = check_note {
        report.notes.insert(0, note);
    }
    Ok(report)
}

/// Relative disagreement along the test direction above which a supplied
/// derivative is treated as wrong rather than noisy.
const GROSS_DERIVATIVE_ERROR: f64 = 1e-2;

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
    fn check_derivatives_option_stops_a_solve_with_a_wrong_gradient() {
        let p = Problem::new(2, |x| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2))
            .start_at(&[0.0, 0.0])
            .with_gradient(|x, g| {
                g[0] = 2.0 * (x[0] - 1.0);
                g[1] = 20.0 * (x[1] - 2.0); // wrong by a factor of 10
            });
        let opts = Options {
            check_derivatives: DerivativeCheck::Full,
            ..Options::default()
        };
        let err = minimize(&p, &opts).expect_err("a wrong gradient must be rejected");
        assert!(err.to_string().contains("FAILED"), "{err}");
        // The default directional check catches the same gradient with two
        // evaluations and names the component through the full check at x0.
        let err = minimize(&p, &Options::default()).expect_err("the default check must reject it");
        assert!(err.to_string().contains("grad f [    1]"), "{err}");
    }

    #[test]
    fn variable_scaling_solves_the_units_problem() {
        // bench/results/s7-friction bad_scaling: x = (1e6, 1e-6), optimum f = 0.16935553839
        let p = Problem::new(2, |x| {
            (x[0] / 3.0e6 - 1.0).powi(2) + (x[1] / 1.0e-6 - 1.0).powi(2)
        })
        .start_at(&[1.0e6, 1.0e-6])
        .lower_bounds(&[1.0, 1.0e-9])
        .inequality(1, |x, c| c[0] = 5.0 - x[0] * x[1]);
        let off = minimize(
            &p,
            &Options {
                scale_variables: VariableScaling::Off,
                ..Options::default()
            },
        )
        .unwrap();
        assert!(
            (off.solution.f - 0.169_355_538).abs() > 1e-3,
            "without variable scaling this is not expected to be solved: f = {}",
            off.solution.f
        );
        assert!(
            !off.exit_flag.is_success() || off.solution.f < 20.0,
            "an uncertified or a first-order-stationary exit is the honest outcome: {:?} at f = {}",
            off.exit_flag,
            off.solution.f
        );
        let on = minimize(
            &p,
            &Options {
                scale_variables: VariableScaling::Auto, // the factors span 1e12
                ..Options::default()
            },
        )
        .unwrap();
        assert!(
            (on.solution.f - 0.169_355_538).abs() < 1e-6,
            "f = {} at {:?} ({:?})",
            on.solution.f,
            on.solution.x,
            on.exit_flag
        );
        assert!(on.notes[0].contains("Variables scaled"), "{:?}", on.notes);
        // bound multipliers come back in the model's units
        assert_eq!(on.solution.z_l.len(), 2);
    }

    #[test]
    fn a_mildly_noisy_gradient_is_noted_not_rejected() {
        let p = Problem::new(2, |x| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2))
            .start_at(&[0.0, 0.0])
            .with_gradient(|x, g| {
                g[0] = 2.0 * (x[0] - 1.0) * (1.0 + 1e-4);
                g[1] = 2.0 * (x[1] - 2.0);
            });
        let r = minimize(&p, &Options::default()).unwrap();
        assert!(r.notes[0].contains("Derivative check"), "{:?}", r.notes);
        assert!(r.exit_flag.returned_usable_point());
    }

    #[test]
    fn check_derivatives_option_is_inconclusive_without_analytic_derivatives() {
        let p =
            Problem::new(2, |x| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2)).start_at(&[0.0, 0.0]);
        let opts = Options {
            check_derivatives: DerivativeCheck::Full,
            ..Options::default()
        };
        let err = minimize(&p, &opts).expect_err("nothing to check must not pass silently");
        assert!(err.to_string().contains("INCONCLUSIVE"), "{err}");
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
