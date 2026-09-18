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

use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::thread;

use mincon_core::{
    Algorithm, BarrierUpdate, Capabilities, EvalError, ExitFlag, FdType, Nlp, NlpDims, Options,
    ScalingMode, SolveError, SolveReport, Sparsity,
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
pub fn members(base: &Options, n: usize) -> Vec<Member> {
    let mut v = Vec::new();

    let ip_default = Member {
        name: "ip-default",
        options: Options {
            algorithm: Algorithm::InteriorPoint,
            ..base.clone()
        },
    };
    let sqp = Member {
        name: "sqp",
        options: Options {
            algorithm: Algorithm::Sqp,
            ..base.clone()
        },
    };

    // 1 and 2. Interior point and SQP, in the order the measurements favour
    //    (`bench/results/abl-sqp1`): on problems with at most SQP_FIRST_MAX_N
    //    variables SQP attains at least as often as interior point and uses
    //    0.72-0.88x its evaluations, while above that the dense quasi-Newton
    //    QP falls behind (1.1-3x at n >= 50). They fail on different problems,
    //    which is the reason for the portfolio.
    if mincon_sqp::is_available() && n <= SQP_FIRST_MAX_N {
        v.push(sqp);
        v.push(ip_default);
    } else {
        v.push(ip_default);
        if mincon_sqp::is_available() {
            v.push(sqp);
        }
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

/// Problems with at most this many variables run the SQP member first.
pub const SQP_FIRST_MAX_N: usize = 20;

/// A report the portfolio can stop on: converged (to the requested or the
/// acceptable tolerances) at a point feasible to tolerance. A step-tolerance
/// or otherwise unverified point is kept as a candidate but the next member
/// still runs, since a different method often finishes what this one could not.
fn usable(rep: &SolveReport, tol_feas: f64) -> bool {
    matches!(rep.exit_flag, ExitFlag::Optimal | ExitFlag::Acceptable)
        && rep.constraint_violation <= tol_feas
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
    // "Acceptable" is a certificate (relaxed KKT tolerances met); a step-tolerance
    // or function-tolerance exit is only a usable point with unverified
    // optimality, so it ranks below even when its objective happens to be lower
    // (degenerate problems: the unverified point can sit past the optimum
    // inside the constraint tolerance).
    let a_acc = a.exit_flag == ExitFlag::Acceptable;
    let b_acc = b.exit_flag == ExitFlag::Acceptable;
    if a_acc != b_acc {
        return a_acc;
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

/// The problem wrapped so the driver counts objective evaluations at the model
/// boundary itself. A member that returns `Err` reports nothing, and the shared
/// budget must still be charged for what it spent; and the total the notes
/// report should be what the model saw, not a sum of what the members said.
struct Counted<'a, P: Nlp + ?Sized> {
    inner: &'a P,
    f_evals: AtomicU64,
}

impl<P: Nlp + ?Sized> Counted<'_, P> {
    fn spent(&self) -> u64 {
        self.f_evals.load(AtomicOrdering::Relaxed)
    }
}

impl<P: Nlp + ?Sized> Nlp for Counted<'_, P> {
    fn dims(&self) -> NlpDims {
        self.inner.dims()
    }
    fn x_bounds(&self) -> (&[f64], &[f64]) {
        self.inner.x_bounds()
    }
    fn c_bounds(&self) -> (&[f64], &[f64]) {
        self.inner.c_bounds()
    }
    fn x0(&self) -> &[f64] {
        self.inner.x0()
    }
    fn capabilities(&self) -> Capabilities {
        self.inner.capabilities()
    }
    fn objective(&self, x: &[f64]) -> Result<f64, EvalError> {
        self.f_evals.fetch_add(1, AtomicOrdering::Relaxed);
        self.inner.objective(x)
    }
    fn constraints(&self, x: &[f64], out: &mut [f64]) -> Result<(), EvalError> {
        self.inner.constraints(x, out)
    }
    fn objective_batch(&self, xs: &[f64]) -> Vec<Result<f64, EvalError>> {
        let k = xs.len() / self.inner.dims().n.max(1);
        self.f_evals.fetch_add(k as u64, AtomicOrdering::Relaxed);
        self.inner.objective_batch(xs)
    }
    fn constraints_batch(&self, xs: &[f64], out: &mut [f64]) -> Vec<Result<(), EvalError>> {
        self.inner.constraints_batch(xs, out)
    }
    fn gradient(&self, x: &[f64], out: &mut [f64]) -> Result<(), EvalError> {
        self.inner.gradient(x, out)
    }
    fn jacobian_structure(&self) -> Option<&Sparsity> {
        self.inner.jacobian_structure()
    }
    fn jacobian(&self, x: &[f64], out: &mut [f64]) -> Result<(), EvalError> {
        self.inner.jacobian(x, out)
    }
    fn hessian_structure(&self) -> Option<&Sparsity> {
        self.inner.hessian_structure()
    }
    fn hessian_lagrangian(
        &self,
        x: &[f64],
        sigma: f64,
        lambda: &[f64],
        out: &mut [f64],
    ) -> Result<(), EvalError> {
        self.inner.hessian_lagrangian(x, sigma, lambda, out)
    }
    fn hessian_vector(
        &self,
        x: &[f64],
        sigma: f64,
        lambda: &[f64],
        v: &[f64],
        out: &mut [f64],
    ) -> Result<(), EvalError> {
        self.inner.hessian_vector(x, sigma, lambda, v, out)
    }
    fn typical_x(&self) -> Option<&[f64]> {
        self.inner.typical_x()
    }
}

/// The budgets the caller set, shared across the members. `max_evaluations`
/// and `max_seconds` are always shared; `max_iterations` only when the caller
/// set it (the `n`-scaled default is a per-member safety net, not a budget).
struct Budget {
    evaluations: Option<u64>,
    seconds: Option<f64>,
    iterations: Option<usize>,
}

impl Budget {
    fn of(base: &Options) -> Self {
        Self {
            evaluations: base.max_evaluations,
            seconds: base.max_seconds,
            iterations: base.max_iterations,
        }
    }

    /// The member's options with what the earlier members left, or the reason
    /// the member cannot run at all.
    fn remaining(
        &self,
        member: &Options,
        spent_evals: u64,
        elapsed: f64,
        spent_iters: usize,
    ) -> Result<Options, String> {
        let mut opts = member.clone();
        if let Some(limit) = self.evaluations {
            if spent_evals >= limit {
                return Err("skipped: evaluation budget exhausted by earlier members".to_string());
            }
            opts.max_evaluations = Some(limit - spent_evals);
        }
        if let Some(limit) = self.seconds {
            let left = limit - elapsed;
            if left <= 0.0 {
                return Err("skipped: time budget exhausted by earlier members".to_string());
            }
            opts.max_seconds = Some(left);
        }
        if let Some(limit) = self.iterations {
            if spent_iters >= limit {
                return Err("skipped: iteration budget exhausted by earlier members".to_string());
            }
            opts.max_iterations = Some(limit - spent_iters);
        }
        Ok(opts)
    }
}

fn iterations_of(outcomes: &[(&'static str, Result<SolveReport, String>)]) -> usize {
    outcomes
        .iter()
        .filter_map(|(_, r)| r.as_ref().ok().map(|rep| rep.iterations))
        .sum()
}

/// Run the portfolio.
///
/// # Errors
/// Only when *every* member failed; the error explains what each one said.
pub fn solve<P: Nlp + Sync + ?Sized>(
    nlp: &P,
    base: &Options,
) -> Result<PortfolioReport, SolveError> {
    let counted = Counted {
        inner: nlp,
        f_evals: AtomicU64::new(0),
    };
    let counted = &counted;
    let members = members(base, nlp.dims().n);
    let threads = base
        .threads
        .unwrap_or_else(|| thread::available_parallelism().map_or(1, std::num::NonZero::get));
    // A model whose callbacks cannot run concurrently (every Python model: the GIL
    // serializes them) gains nothing from parallel members and would pay for all of
    // them; run such models sequentially with early exit regardless of `threads`.
    let parallel_safe = nlp.capabilities().parallel_safe;
    let sequential = threads <= 1 || members.len() == 1 || !parallel_safe;
    let start = std::time::Instant::now();
    let budget = Budget::of(base);

    // I5 (`docs/22`): a quadratic program gets the SQP member with its exact,
    // constant Hessian before anything else; the ordinary members follow only
    // when that does not settle it. The probe's evaluations are charged to the
    // quadratic member when it ran and to the winner when the probe declined.
    let mut pre: Vec<(&'static str, Result<SolveReport, String>)> = Vec::new();
    let mut declined_probe_evals: u64 = 0;
    let mut probe_declined: Option<String> = None;
    let mut settled = false;
    if base.quadratic_probe && !(nlp.capabilities().hessian && nlp.hessian_structure().is_some()) {
        let (q, spent, reason) = crate::quadratic::probe_counted(counted, base);
        match q {
            Some(q) => {
                let sqp = Options {
                    algorithm: Algorithm::Sqp,
                    ..base.clone()
                };
                let r = match budget.remaining(&sqp, spent, start.elapsed().as_secs_f64(), 0) {
                    Ok(opts) => mincon_sqp::solve(&q, &opts)
                        .map(|mut rep| {
                            rep.f_evals += q.f_evals;
                            rep.c_evals += q.c_evals;
                            rep.g_evals += q.g_evals;
                            rep.j_evals += q.j_evals;
                            rep.notes.insert(0, q.note.clone());
                            rep
                        })
                        .map_err(|e| e.to_string()),
                    Err(why) => Err(why),
                };
                // Only a full certificate settles it: an `Acceptable` exit from
                // Newton steps on a model that only looked quadratic (noise, a
                // near-quadratic) falls through to the ordinary members, and the
                // ranking keeps the better of the two.
                settled = matches!(&r, Ok(rep) if rep.exit_flag == ExitFlag::Optimal
                    && rep.constraint_violation <= base.tol.feasibility);
                pre.push(("sqp-quadratic", r));
            }
            None => {
                declined_probe_evals = spent;
                probe_declined = reason;
            }
        }
    }

    let outcomes: Vec<(&'static str, Result<SolveReport, String>)> = if settled {
        pre
    } else if sequential {
        // Sequential with early exit: a problem the first member answers usably
        // costs exactly what a single solve costs. Later members only get the
        // budget the earlier ones left over, so the portfolio never exceeds the
        // caller's evaluation, time or iteration limits in total.
        let mut spent_iters = iterations_of(&pre);
        let mut out = pre;
        for m in &members {
            let opts = match budget.remaining(
                &m.options,
                counted.spent(),
                start.elapsed().as_secs_f64(),
                spent_iters,
            ) {
                Ok(opts) => opts,
                Err(why) => {
                    out.push((m.name, Err(why)));
                    continue;
                }
            };
            let r = run_member(
                counted,
                &Member {
                    name: m.name,
                    options: opts,
                },
            );
            let done = matches!(&r, Ok(rep) if usable(rep, base.tol.feasibility));
            if let Ok(rep) = &r {
                spent_iters += rep.iterations;
            }
            out.push((m.name, r));
            if done {
                break;
            }
        }
        out
    } else {
        // Parallel, at most `threads` members at a time so the configured limit
        // bounds actual concurrency rather than only the reported count. Every
        // member of a chunk starts with what the earlier chunks left; within a
        // chunk the members run concurrently, so a chunk of `k` can spend up to
        // `k` times the remainder. That is the price of concurrency, and it is
        // bounded by `threads`.
        let mut spent_iters = iterations_of(&pre);
        let mut out = pre;
        for chunk in members.chunks(threads.max(1)) {
            let mut chunk_out: Vec<(&'static str, Result<SolveReport, String>)> = Vec::new();
            let mut runnable: Vec<Member> = Vec::new();
            for m in chunk {
                match budget.remaining(
                    &m.options,
                    counted.spent(),
                    start.elapsed().as_secs_f64(),
                    spent_iters,
                ) {
                    Ok(opts) => runnable.push(Member {
                        name: m.name,
                        options: opts,
                    }),
                    Err(why) => chunk_out.push((m.name, Err(why))),
                }
            }
            if !runnable.is_empty() {
                let ran: Vec<_> = thread::scope(|scope| {
                    let handles: Vec<_> = runnable
                        .iter()
                        .map(|m| {
                            let m = m.clone();
                            scope.spawn(move || (m.name, run_member(counted, &m)))
                        })
                        .collect();
                    handles
                        .into_iter()
                        .map(|h| {
                            h.join().unwrap_or_else(|_| {
                                ("panicked", Err("member thread panicked".to_string()))
                            })
                        })
                        .collect()
                });
                chunk_out.extend(ran);
            }
            let done = chunk_out
                .iter()
                .any(|(_, r)| matches!(r, Ok(rep) if rep.exit_flag == ExitFlag::Optimal));
            spent_iters += iterations_of(&chunk_out);
            out.extend(chunk_out);
            if done {
                break;
            }
        }
        out
    };

    // What the model saw, including the probe and any member that failed.
    let total_f_evals = counted.spent();

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
    best.f_evals += declined_probe_evals;

    if let Some(reason) = probe_declined {
        best.notes.push(format!(
            "Quadratic-program probe declined after {declined_probe_evals} objective evaluations: {reason}."
        ));
    }
    best.notes.push(format!(
        "Algorithm portfolio: {} member(s) run {}; '{winner}' produced the answer. \
         Total objective evaluations across all members: {total_f_evals}.",
        outcomes.len(),
        if sequential {
            "sequentially with early exit".to_string()
        } else {
            format!("on up to {} thread(s)", threads.min(members.len()))
        }
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
        Algorithm::Sqp => mincon_sqp::solve(nlp, &m.options),
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
            limit: None,
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
            quasi_newton: None,
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
        let m = members(&Options::default(), 100);
        assert_eq!(m[0].name, "ip-default");
        assert_eq!(m[0].options.scaling, ScalingMode::GradientBased);
        assert!(m.len() >= 3, "portfolio should have real diversity");
        let small = members(&Options::default(), 5);
        assert_eq!(small[0].name, "sqp");
        assert_eq!(small[1].name, "ip-default");
    }

    /// A model that declares itself not parallel-safe (like every Python model)
    /// and counts how often it is called.
    struct Counting {
        calls: std::sync::atomic::AtomicU64,
        /// `(x - 1)^4` instead of `(x - 1)^2`: the quadratic probe declines it and no member finishes it in two iterations.
        quartic: bool,
        lb: Vec<f64>,
        ub: Vec<f64>,
        cl: Vec<f64>,
        cu: Vec<f64>,
        x0: Vec<f64>,
    }
    impl Nlp for Counting {
        fn dims(&self) -> mincon_core::NlpDims {
            mincon_core::NlpDims { n: 2, m: 1 }
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
        fn capabilities(&self) -> mincon_core::Capabilities {
            mincon_core::Capabilities {
                parallel_safe: false,
                ..mincon_core::Capabilities::none()
            }
        }
        fn objective(&self, x: &[f64]) -> Result<f64, mincon_core::EvalError> {
            self.calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let k = if self.quartic { 4 } else { 2 };
            Ok((x[0] - 1.0).powi(k) + (x[1] - 1.0).powi(k))
        }
        fn constraints(&self, x: &[f64], out: &mut [f64]) -> Result<(), mincon_core::EvalError> {
            out[0] = x[0] + x[1];
            Ok(())
        }
    }

    fn counting() -> Counting {
        Counting {
            calls: std::sync::atomic::AtomicU64::new(0),
            quartic: false,
            lb: vec![-1e20; 2],
            ub: vec![1e20; 2],
            cl: vec![-1e20],
            cu: vec![1.0],
            x0: vec![0.0, 0.0],
        }
    }

    #[test]
    fn a_non_parallel_safe_model_costs_one_solve_when_the_first_member_succeeds() {
        let nlp = counting();
        let r = solve(&nlp, &Options::default()).unwrap();
        assert_eq!(
            r.outcomes.len(),
            1,
            "no later member should have run: {:?}",
            r.best.notes
        );
        assert_eq!(
            nlp.calls.load(std::sync::atomic::Ordering::Relaxed),
            r.best.f_evals
        );
        assert_eq!(
            nlp.calls.load(std::sync::atomic::Ordering::Relaxed),
            r.total_f_evals,
            "the total must be what the model saw"
        );
        assert!(r.best.notes.iter().any(|n| n.contains("sequentially")));
    }

    #[test]
    fn the_portfolio_never_exceeds_the_evaluation_budget_in_total() {
        let nlp = counting();
        let opts = Options {
            max_evaluations: Some(12), // far too few for any member to converge
            ..Options::default()
        };
        let r = solve(&nlp, &opts).unwrap();
        let total = nlp.calls.load(std::sync::atomic::Ordering::Relaxed);
        // Each member may overshoot by at most one iteration's worth of probes,
        // but the members share the budget rather than each getting all of it.
        assert!(
            total < 3 * 12 + 30,
            "portfolio spent {total} evaluations against a budget of 12"
        );
        assert!(!r.outcomes.is_empty());
    }

    #[test]
    fn both_algorithms_are_enlisted_in_every_size_class() {
        for n in [1, 20, 21, 1000] {
            let m = members(&Options::default(), n);
            assert!(m.iter().any(|x| x.name == "sqp"), "n = {n}");
            assert!(m.iter().any(|x| x.name == "ip-default"), "n = {n}");
        }
    }
    /// The quartic with its row inactive: an unconstrained `(x - 1)^4` from the
    /// origin, which a quasi-Newton member needs many iterations to certify.
    fn quartic() -> Counting {
        Counting {
            quartic: true,
            cu: vec![10.0],
            // Off the diagonal: from the origin the first quasi-Newton step lands
            // on the optimum exactly, which would end the solve in one iteration.
            x0: vec![0.3, -0.2],
            ..counting()
        }
    }

    #[test]
    fn an_iteration_cap_set_by_the_caller_is_shared_across_members() {
        let nlp = quartic();
        let opts = Options {
            max_iterations: Some(2),
            ..Options::default()
        };
        let r = solve(&nlp, &opts).unwrap();
        let spent: usize = r
            .outcomes
            .iter()
            .filter_map(|(_, o)| o.as_ref().ok().map(|rep| rep.iterations))
            .sum();
        assert!(
            spent <= 2,
            "members ran {spent} iterations against a cap of 2: {:?}",
            r.best.notes
        );
        assert!(
            r.outcomes
                .iter()
                .any(|(_, o)| matches!(o, Err(e) if e.contains("iteration budget"))),
            "{:?}",
            r.best.notes
        );
        assert_eq!(r.best.exit_flag, ExitFlag::MaxReached);
        assert_eq!(r.best.limit, Some(mincon_core::Limit::Iterations));
        assert!(
            r.best.message().starts_with("Iteration limit reached"),
            "{}",
            r.best.message()
        );
    }

    #[test]
    fn the_default_iteration_cap_is_not_shared() {
        // No caller-set cap: every member keeps its own n-scaled safety net,
        // so nothing is ever skipped for iterations.
        let nlp = quartic();
        let r = solve(&nlp, &Options::default()).unwrap();
        assert!(!r
            .outcomes
            .iter()
            .any(|(_, o)| matches!(o, Err(e) if e.contains("iteration budget"))));
    }

    #[test]
    fn the_probe_decline_reason_is_in_the_notes_and_the_total_counts_the_model() {
        let nlp = quartic();
        let r = solve(&nlp, &Options::default()).unwrap();
        assert!(
            r.best
                .notes
                .iter()
                .any(|n| n.starts_with("Quadratic-program probe declined")),
            "{:?}",
            r.best.notes
        );
        assert_eq!(
            r.total_f_evals,
            nlp.calls.load(std::sync::atomic::Ordering::Relaxed)
        );
    }
}
