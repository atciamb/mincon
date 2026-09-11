//! Sequential quadratic programming.
//!
//! # Status: implemented (`solver`), qualified against the interior-point
//! member on the benchmark corpus; see `docs/20_SQP_MATHEMATICS.md` for the
//! derivations and `docs/19_SQP_RD_PLAN.md` for the measurements.
//!
//! # Why SQP is worth building even though interior point works
//!
//! Interior point is the better default and it is what `fmincon` defaults to.
//! But the two methods fail on *different* problems, and that is precisely what
//! makes an algorithm portfolio pay:
//!
//! | | interior point | SQP |
//! |---|---|---|
//! | Warm starting | poor — the barrier has to be re-raised | excellent |
//! | Small, dense, few active constraints | fine | usually faster |
//! | Degenerate / LICQ failure | multipliers blow up | often still fine |
//! | Very large sparse | excellent | active-set combinatorics dominate |
//! | Infeasible starting point | needs restoration | handles naturally |
//! | Highly nonlinear constraints | good | can cycle on the active set |
//!
//! `fmincon` makes the user choose between them with an `Algorithm` option and
//! a page of prose about when to pick which. A user who guesses wrong gets a
//! failed solve and concludes their problem is hard. Racing both and returning
//! the first success removes the guess, and no single `fmincon` call can do it.
//! That is the cheapest robustness win available to us and it is why this crate
//! is milestone M5 rather than M9.
//!
//! # Specification
//!
//! The full algorithm specification is in `docs/03_SPEC_SQP.md`. In outline:
//!
//! 1. **QP subproblem.** At `x_k` with multipliers `lambda_k`, solve
//!    ```text
//!      min_d  1/2 d^T B_k d + grad f(x_k)^T d
//!      s.t.   c_L - c(x_k) <= J_k d <= c_U - c(x_k)
//!             x_L - x_k    <=   d   <= x_U - x_k
//!    ```
//!    `B_k` is the damped-BFGS or exact Hessian of the Lagrangian.
//!
//! 2. **Always-feasible relaxation.** The linearized constraints can easily be
//!    inconsistent even when the NLP is fine. `fmincon`'s `sqp` algorithm
//!    reformulates the subproblem so it is always feasible, and that single
//!    choice is a large part of why it is more robust than `active-set`. Use
//!    the elastic form: add `rho * sum(p + n)` with `J_k d + p - n` in the
//!    constraints, `p, n >= 0`, `rho` driven by the multiplier norm.
//!
//! 3. **Globalization.** An `l1` merit function
//!    `f(x) + nu * ||violation(x)||_1` with the Han–Powell penalty update
//!    `nu_i = max(|lambda_i|, (nu_i + |lambda_i|)/2)`, a backtracking line
//!    search, and a **second-order correction** to defeat the Maratos effect.
//!    A filter is the alternative; use the merit function here specifically so
//!    the portfolio's two members fail differently.
//!
//! 4. **Bounds are honoured exactly at every iterate**, as in `fmincon`'s
//!    `sqp`. The `d` bounds above enforce it structurally.
//!
//! 5. **Non-finite retreat.** If the model returns `NaN`/`Inf`, halve the step
//!    and retry rather than failing.
//!
//! # Acceptance gate
//!
//! Implemented means: the Hock–Schittkowski set passes at no worse than the
//! interior-point rate, `HS13` and `TORTURE_INFEASIBLE` are *better* than
//! interior point (they are the cases SQP should own), and the portfolio's
//! combined success rate strictly exceeds each member's.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod solver;

pub use solver::solve;

/// Whether this algorithm is available to the portfolio.
#[must_use]
pub fn is_available() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use mincon_core::{ExitFlag, Options};

    #[test]
    fn reports_itself_available() {
        assert!(is_available());
    }

    #[test]
    fn solves_a_bound_constrained_quadratic_in_one_step() {
        struct Q;
        impl mincon_core::Nlp for Q {
            fn dims(&self) -> mincon_core::NlpDims {
                mincon_core::NlpDims { n: 2, m: 0 }
            }
            fn x_bounds(&self) -> (&[f64], &[f64]) {
                (&[0.0, 0.0], &[1.0, 1.0])
            }
            fn c_bounds(&self) -> (&[f64], &[f64]) {
                (&[], &[])
            }
            fn x0(&self) -> &[f64] {
                &[0.5, 0.5]
            }
            fn capabilities(&self) -> mincon_core::Capabilities {
                mincon_core::Capabilities::none()
            }
            fn objective(&self, x: &[f64]) -> Result<f64, mincon_core::EvalError> {
                Ok((x[0] - 2.0).powi(2) + (x[1] - 0.25).powi(2))
            }
            fn constraints(
                &self,
                _x: &[f64],
                _o: &mut [f64],
            ) -> Result<(), mincon_core::EvalError> {
                Ok(())
            }
        }
        let r = solve(&Q, &Options::default()).unwrap();
        assert_eq!(
            r.exit_flag,
            ExitFlag::Optimal,
            "{:?} {:?}",
            r.exit_flag,
            r.notes
        );
        assert!(
            (r.solution.x[0] - 1.0).abs() < 1e-8 && (r.solution.x[1] - 0.25).abs() < 1e-6,
            "{:?}",
            r.solution.x
        );
        assert!(r.solution.z_u[0] > 1.9, "{:?}", r.solution.z_u);
    }
}

#[cfg(test)]
mod diag {
    #[test]
    #[ignore]
    fn fixture_details() {
        let name = std::env::var("MINCON_FIXTURE").unwrap_or_else(|_| "HS13".into());
        let p = mincon_testset::by_name(&name).unwrap();
        let r = super::solve(&p.as_nlp(), &mincon_core::Options::default()).unwrap();
        eprintln!(
            "{:?} x={:?} f={} lam={:?} zl={:?} zu={:?} e0={}",
            r.exit_flag,
            r.solution.x,
            r.solution.f,
            r.solution.lambda,
            r.solution.z_l,
            r.solution.z_u,
            r.optimality
        );
        for n in &r.notes {
            eprintln!("  {n}");
        }
        for t in r.trace.iter() {
            eprintln!("  {t:?}");
        }
    }
}
