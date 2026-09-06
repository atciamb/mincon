//! Test problems with known solutions.
//!
//! # What is here
//!
//! * [`hs`] — problems from Hock and Schittkowski, *Test Examples for Nonlinear
//!   Programming Codes* (Springer, 1981), in the canonical form of
//!   [`mincon_core`]. Small, hand-checkable, and the set every NLP paper since
//!   1981 reports on, which makes them the right thing to regress against
//!   every commit.
//! * [`torture`] — problems designed to break solvers rather than to be solved:
//!   bad scaling, restricted domains, degenerate constraints, infeasible
//!   models, unbounded objectives. Some have no solution at all; the pass
//!   criterion there is *reporting the right failure*, not converging.
//!
//! # What is deliberately not here
//!
//! CUTEst. The 1075-problem collection is the real benchmark and it does not
//! belong compiled into a Rust crate — it is driven from Python through S2MPJ
//! by `bench/`. See `docs/08_BENCHMARK_PROTOCOL.md`. The set in this crate is
//! the *fast* gate that runs on every commit; CUTEst is the *slow* gate that
//! runs nightly and decides whether we are actually beating `fmincon`.
//!
//! # Honesty about the reference values
//!
//! `f_opt` values are as published. Several HS problems have known errata and a
//! few have better local minima than the published one; where we know of a
//! discrepancy it is recorded in the problem's `notes`. A solver that finds a
//! *lower* feasible objective than `f_opt` has not failed — check the note
//! before treating it as a bug.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod hs;
pub mod torture;

use mincon_core::{Capabilities, EvalError, Nlp, NlpDims};

/// A test problem: dimensions, bounds, closures and the published solution.
#[derive(Clone)]
pub struct TestProblem {
    /// Problem name, e.g. `"HS71"`.
    pub name: &'static str,
    /// Number of variables.
    pub n: usize,
    /// Number of general constraints.
    pub m: usize,
    /// Starting point.
    pub x0: Vec<f64>,
    /// Variable lower bounds.
    pub xl: Vec<f64>,
    /// Variable upper bounds.
    pub xu: Vec<f64>,
    /// Constraint lower bounds.
    pub cl: Vec<f64>,
    /// Constraint upper bounds.
    pub cu: Vec<f64>,
    /// Objective.
    pub f: fn(&[f64]) -> f64,
    /// Constraints; writes `m` values.
    pub c: fn(&[f64], &mut [f64]),
    /// Published optimal objective value, or `None` when the problem has no
    /// solution (infeasible or unbounded).
    pub f_opt: Option<f64>,
    /// What the correct outcome is, when it is not "find `f_opt`".
    pub expect: Expect,
    /// Anything a reader needs to know: errata, degeneracy, known better minima.
    pub notes: &'static str,
}

/// What a correct solver should do with a problem.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Expect {
    /// Converge to `f_opt`.
    Optimum,
    /// Detect and report local infeasibility.
    Infeasible,
    /// Detect and report an unbounded objective.
    Unbounded,
    /// Converge, but the published value is a *different* local minimum than
    /// the one a local method reaches from `x0`; only feasibility and
    /// stationarity are checked.
    LocalMinimum,
}

impl TestProblem {
    /// Wrap as an [`Nlp`].
    #[must_use]
    pub fn as_nlp(&self) -> TestNlp<'_> {
        TestNlp { p: self }
    }

    /// Maximum constraint violation at `x`, including variable bounds.
    #[must_use]
    pub fn violation(&self, x: &[f64]) -> f64 {
        let mut c = vec![0.0; self.m];
        (self.c)(x, &mut c);
        let mut worst = 0.0_f64;
        for i in 0..self.m {
            worst = worst.max(self.cl[i] - c[i]).max(c[i] - self.cu[i]);
        }
        for j in 0..self.n {
            worst = worst.max(self.xl[j] - x[j]).max(x[j] - self.xu[j]);
        }
        worst.max(0.0)
    }
}

/// An [`Nlp`] view of a [`TestProblem`]. Supplies no derivatives, which is the
/// point: this is the plug-and-play path we are graded on.
pub struct TestNlp<'a> {
    p: &'a TestProblem,
}

impl Nlp for TestNlp<'_> {
    fn dims(&self) -> NlpDims {
        NlpDims {
            n: self.p.n,
            m: self.p.m,
        }
    }
    fn x_bounds(&self) -> (&[f64], &[f64]) {
        (&self.p.xl, &self.p.xu)
    }
    fn c_bounds(&self) -> (&[f64], &[f64]) {
        (&self.p.cl, &self.p.cu)
    }
    fn x0(&self) -> &[f64] {
        &self.p.x0
    }
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            parallel_safe: true,
            ..Capabilities::none()
        }
    }
    fn objective(&self, x: &[f64]) -> Result<f64, EvalError> {
        let v = (self.p.f)(x);
        if v.is_finite() {
            Ok(v)
        } else {
            Err(EvalError::NonFinite(None))
        }
    }
    fn constraints(&self, x: &[f64], out: &mut [f64]) -> Result<(), EvalError> {
        (self.p.c)(x, out);
        if out.iter().all(|v| v.is_finite()) {
            Ok(())
        } else {
            Err(EvalError::NonFinite(None))
        }
    }
}

/// Every problem in the crate.
#[must_use]
pub fn all() -> Vec<TestProblem> {
    let mut v = hs::all();
    v.extend(torture::all());
    v
}

/// Look one up by name.
#[must_use]
pub fn by_name(name: &str) -> Option<TestProblem> {
    all()
        .into_iter()
        .find(|p| p.name.eq_ignore_ascii_case(name))
}

/// Positive infinity, for readable bound tables.
pub const INF: f64 = f64::INFINITY;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_problem_is_internally_consistent() {
        for p in all() {
            assert_eq!(p.x0.len(), p.n, "{}: x0 length", p.name);
            assert_eq!(p.xl.len(), p.n, "{}: xl length", p.name);
            assert_eq!(p.xu.len(), p.n, "{}: xu length", p.name);
            assert_eq!(p.cl.len(), p.m, "{}: cl length", p.name);
            assert_eq!(p.cu.len(), p.m, "{}: cu length", p.name);
            for j in 0..p.n {
                assert!(p.xl[j] <= p.xu[j], "{}: xl > xu at {j}", p.name);
            }
            for i in 0..p.m {
                assert!(p.cl[i] <= p.cu[i], "{}: cl > cu at {i}", p.name);
            }
            mincon_core::validate(&p.as_nlp()).unwrap_or_else(|e| panic!("{}: {e}", p.name));
        }
    }

    #[test]
    fn names_are_unique() {
        let mut names: Vec<&str> = all().iter().map(|p| p.name).collect();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(before, names.len(), "duplicate problem names");
    }

    #[test]
    fn objective_and_constraints_evaluate_at_the_starting_point() {
        for p in all() {
            let f = (p.f)(&p.x0);
            assert!(
                f.is_finite() || p.name.starts_with("TORTURE"),
                "{}: objective is {f} at x0",
                p.name
            );
            let mut c = vec![0.0; p.m];
            (p.c)(&p.x0, &mut c);
        }
    }
}
