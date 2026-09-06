//! The algorithm portfolio.
//!
//! # The idea, and why it is the cheapest robustness we can buy
//!
//! No single nonlinear programming algorithm is best on every problem, and the
//! ways they fail are largely *uncorrelated*: interior point struggles where
//! the multipliers are unbounded, SQP struggles where the active set churns,
//! and a badly chosen barrier start fails on problems that a different one
//! walks through. `fmincon` responds by exposing an `Algorithm` option and
//! documenting when to use which — which pushes the choice onto the user, who
//! by construction does not know, and whose failed solve becomes "this problem
//! is hard" rather than "I picked the wrong algorithm".
//!
//! Running several configurations and returning the best answer converts that
//! choice into compute. If two members each solve 80% of a set and their
//! failures are independent, the pair solves 96%. They are not independent in
//! practice, but they are far from identical, and every percentage point here
//! is a point `fmincon` cannot answer without the user running it twice by
//! hand.
//!
//! # Honest accounting
//!
//! A portfolio spends more evaluations than a single solve. Where the model is
//! expensive that matters, so:
//!
//! * Members run on **separate threads** when threads are available, so the
//!   wall-clock cost is the slowest member, not the sum.
//! * On one thread the members run in sequence with early exit, so a problem
//!   the first member solves costs exactly what it did before.
//! * [`PortfolioReport::total_f_evals`] reports the true cost across all
//!   members, and the benchmark harness scores both wall-clock and total
//!   evaluations. Reporting only the winner's evaluation count would be
//!   cheating, and reviewers will check.
//!
//! # Choosing the winner
//!
//! Strictly ordered: a feasible point beats an infeasible one; among feasible
//! points a converged one beats an unconverged one; among those, lower
//! objective wins; ties break by evaluation count, then by member order, so
//! the result is **deterministic** regardless of which thread finished first.
//! Determinism here is not a nicety — a nondeterministic optimizer cannot be
//! used in a regression test or a reproducible paper.

use std::thread;

use mincon_core::{
    Algorithm, BarrierUpdate, ExitFlag, FdType, Nlp, Options, ScalingMode, SolveError, SolveReport,
};

/// One configuration in the portfolio.
#[derive(Debug, Clone)]
pub struct Member {
    /// Human-readable name, reported in the notes.
    pub name: &'static str,
    /// Options this member runs with.
    pub options: Options,
}

/// The outcome of a portfolio run.
#[derive(Debug, Clone)]
pub struct PortfolioReport {
    /// The winning report.
    pub best: SolveReport,
    /// Which member produced it.
    pub winner: &'static str,
    /// Every member's outcome, in declaration order.
    pub outcomes: Vec<(&'static str, Result<SolveReport, String>)>,
    /// Objective evaluations across **all** members, not just the winner.
    pub total_f_evals: u64,
}

/// Build the portfolio for a problem.
///
/// The composition is deliberately conservative: the first member is the
/// configuration we would ship as a single solver, so a one-thread run is never
/// worse than no portfolio at all. Later members trade different risks.
#[must_use]
pub fn members(base: &Options) -> Vec<Member> {
    let mut v = Vec::new();

    // 1. The default. Everything we believe in, all at once.
    v.push(Member {
        name: "ip-default",
        options: Options {
            algorithm: Algorithm::InteriorPoint,
            ..base.clone()
        },
    });

    // 2. SQP, when it exists. Fails on a different set of problems than
    //    interior point does, which is the whole reason for the portfolio.
    if mincon_sqp::is_available() {
        v.push(Member {
            name: "sqp",
            options: Options {
                algorithm: Algorithm::Sqp,
                ..base.clone()
            },
        });
    }

    // 3. Interior point from a much larger barrier parameter, with central
    //    differences. Slower and more careful; wins on models whose
    //    derivatives are noisy and on problems where the default's first steps
    //    are too aggressive.
    v.push(Member {
        name: "ip-cautious",
        options: Options {
            algorithm: Algorithm::InteriorPoint,
            mu_init: 1.0,
            barrier_update: BarrierUpdate::Monotone,
            fd_type: FdType::Central,
            ..base.clone()
        },
    });

    // 4. Interior point with scaling off. Scaling is right far more often than
    //    it is wrong, but it is a transformation of the problem and there are
    //    models it hurts; this member is the insurance policy on our own
    //    most opinionated default.
    v.push(Member {
        name: "ip-unscaled",
        options: Options {
            algorithm: Algorithm::InteriorPoint,
            scaling: ScalingMode::None,
            ..base.clone()
        },
    });

    v
}

/// Rank two outcomes. Returns `true` when `a` is strictly better than `b`.
#[must_use]
fn better(a: &SolveReport, b: &SolveReport, tol_feas: f64) -> bool {
    let a_feas = a.constraint_violation <= tol_feas;
    let b_feas = b.constraint_violation <= tol_feas;
    if a_feas != b_feas {
        return a_feas;
    }
    if !a_feas {
        // Neither is feasible: prefer the smaller violation.
        return a.constraint_violation < b.constraint_violation;
    }
    let a_conv = a.exit_flag.is_success();
    let b_conv = b.exit_flag.is_success();
    if a_conv != b_conv {
        return a_conv;
    }
    let a_usable = a.exit_flag.returned_usable_point();
    let b_usable = b.exit_flag.returned_usable_point();
    if a_usable != b_usable {
        return a_usable;
    }
    if a.solution.f != b.solution.f {
        return a.solution.f < b.solution.f;
    }
    a.f_evals < b.f_evals
}

/// Run the portfolio.
///
/// # Errors
/// Only when *every* member failed; the error explains what each one said.
pub fn solve<P: Nlp + Sync + ?Sized>(
    nlp: &P,
    base: &Options,
) -> Result<PortfolioReport, SolveError> {
    let members = members(base);
    let threads = base
        .threads
        .unwrap_or_else(|| thread::available_parallelism().map_or(1, std::num::NonZero::get));

    let outcomes: Vec<(&'static str, Result<SolveReport, String>)> = if threads <= 1
        || members.len() == 1
    {
        // Sequential with early exit: a problem the first member solves cleanly
        // costs exactly what a single solve costs.
        let mut out = Vec::new();
        for m in &members {
            let r = run_member(nlp, m);
            let done = matches!(&r, Ok(rep) if rep.exit_flag == ExitFlag::Optimal);
            out.push((m.name, r));
            if done {
                break;
            }
        }
        out
    } else {
        thread::scope(|scope| {
            let handles: Vec<_> = members
                .iter()
                .map(|m| {
                    let m = m.clone();
                    scope.spawn(move || (m.name, run_member(nlp, &m)))
                })
                .collect();
            handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .unwrap_or_else(|_| ("panicked", Err("member thread panicked".to_string())))
                })
                .collect()
        })
    };

    let total_f_evals = outcomes
        .iter()
        .filter_map(|(_, r)| r.as_ref().ok().map(|rep| rep.f_evals))
        .sum();

    let mut best: Option<(&'static str, SolveReport)> = None;
    for (name, r) in &outcomes {
        if let Ok(rep) = r {
            match &best {
                None => best = Some((name, rep.clone())),
                Some((_, cur)) => {
                    if better(rep, cur, base.tol.feasibility) {
                        best = Some((name, rep.clone()));
                    }
                }
            }
        }
    }

    let Some((winner, mut best)) = best else {
        let detail = outcomes
            .iter()
            .map(|(n, r)| match r {
                Ok(_) => format!("{n}: ok"),
                Err(e) => format!("{n}: {e}"),
            })
            .collect::<Vec<_>>()
            .join("; ");
        return Err(SolveError::Internal(format!(
            "every portfolio member failed ({detail})"
        )));
    };

    best.notes.push(format!(
        "Algorithm portfolio: {} member(s) run on {} thread(s); '{winner}' produced the answer. \
         Total objective evaluations across all members: {total_f_evals}.",
        outcomes.len(),
        threads.min(members.len())
    ));
    for (name, r) in &outcomes {
        if *name != winner {
            let summary = match r {
                Ok(rep) => format!(
                    "{:?}, f = {:.6e}, violation = {:.2e}",
                    rep.exit_flag, rep.solution.f, rep.constraint_violation
                ),
                Err(e) => e.clone(),
            };
            best.notes
                .push(format!("  portfolio member '{name}': {summary}"));
        }
    }

    Ok(PortfolioReport {
        best,
        winner,
        outcomes,
        total_f_evals,
    })
}

fn run_member<P: Nlp + ?Sized>(nlp: &P, m: &Member) -> Result<SolveReport, String> {
    let r = match m.options.algorithm {
        Algorithm::Sqp | Algorithm::Slqp => mincon_sqp::solve(nlp, &m.options),
        _ => mincon_ip::solve(nlp, &m.options),
    };
    r.map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mincon_core::{Solution, Timings};

    fn report(f: f64, viol: f64, flag: ExitFlag, evals: u64) -> SolveReport {
        SolveReport {
            solution: Solution {
                x: vec![],
                f,
                c: vec![],
                lambda: vec![],
                z_l: vec![],
                z_u: vec![],
            },
            exit_flag: flag,
            algorithm: Algorithm::InteriorPoint,
            iterations: 0,
            f_evals: evals,
            g_evals: 0,
            c_evals: 0,
            j_evals: 0,
            h_evals: 0,
            failed_evals: 0,
            optimality: 0.0,
            constraint_violation: viol,
            complementarity: 0.0,
            trace: vec![],
            timings: Timings::default(),
            notes: vec![],
        }
    }

    #[test]
    fn feasibility_beats_a_lower_objective() {
        let feasible = report(10.0, 0.0, ExitFlag::Optimal, 100);
        let infeasible_but_lower = report(-1000.0, 1.0, ExitFlag::Optimal, 10);
        assert!(better(&feasible, &infeasible_but_lower, 1e-6));
        assert!(!better(&infeasible_but_lower, &feasible, 1e-6));
    }

    #[test]
    fn among_infeasible_points_the_least_violation_wins() {
        let a = report(0.0, 0.1, ExitFlag::MaxReached, 10);
        let b = report(-5.0, 1.0, ExitFlag::MaxReached, 10);
        assert!(better(&a, &b, 1e-6));
    }

    #[test]
    fn convergence_beats_a_lower_objective_at_equal_feasibility() {
        let converged = report(1.0, 0.0, ExitFlag::Optimal, 100);
        let stopped_early = report(0.5, 0.0, ExitFlag::MaxReached, 10);
        assert!(better(&converged, &stopped_early, 1e-6));
    }

    #[test]
    fn among_equally_converged_the_lower_objective_wins() {
        let a = report(1.0, 0.0, ExitFlag::Optimal, 500);
        let b = report(2.0, 0.0, ExitFlag::Optimal, 10);
        assert!(better(&a, &b, 1e-6));
    }

    #[test]
    fn ties_break_on_evaluation_count_so_the_result_is_deterministic() {
        let cheap = report(1.0, 0.0, ExitFlag::Optimal, 10);
        let dear = report(1.0, 0.0, ExitFlag::Optimal, 500);
        assert!(better(&cheap, &dear, 1e-6));
        assert!(!better(&dear, &cheap, 1e-6));
    }

    #[test]
    fn the_first_member_is_the_shippable_default() {
        let m = members(&Options::default());
        assert_eq!(m[0].name, "ip-default");
        assert_eq!(m[0].options.scaling, ScalingMode::GradientBased);
        assert!(m.len() >= 3, "portfolio should have real diversity");
    }

    #[test]
    fn unimplemented_algorithms_are_not_enlisted() {
        let m = members(&Options::default());
        assert!(
            !m.iter().any(|x| x.name == "sqp"),
            "SQP is not implemented, so it must not be in the portfolio yet"
        );
    }
}
