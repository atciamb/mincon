//! Core types for `mincon`: the problem model, solver options, and result types.
//!
//! # The canonical form
//!
//! Every algorithm in this workspace operates on a single canonical NLP:
//!
//! ```text
//!   minimize    f(x)                         x in R^n
//!   subject to  c_L <= c(x) <= c_U           c: R^n -> R^m
//!               x_L <=   x  <= x_U
//! ```
//!
//! Equality constraints are expressed as `c_L[i] == c_U[i]`. One-sided
//! inequalities use `-inf` / `+inf` on the free side. Free variables use
//! infinite bounds. This is deliberately the IPOPT / CUTEst form rather than
//! the MATLAB `fmincon` form (`A x <= b`, `Aeq x = beq`, `c(x) <= 0`,
//! `ceq(x) = 0`), because:
//!
//! * range constraints `l <= c(x) <= u` need one row here and two there,
//! * it removes the eq/ineq branch from every kernel,
//! * CUTEst, AMPL, `.nl` and S2MPJ all speak it natively, so the benchmark
//!   bridge is a memcpy rather than a translation.
//!
//! The `fmincon`-compatible façade lives in the `mincon` crate and lowers into
//! this form. See `docs/07_API_DESIGN.md`.
//!
//! # Evaluation is `&self`
//!
//! [`Nlp`] methods take `&self`, not `&mut self`. Evaluation counting lives in
//! [`EvalCounters`] (atomics) held by the driver. This keeps `Nlp: Sync` cheap,
//! which is what makes the algorithm portfolio in `mincon::portfolio` possible:
//! interior-point and SQP can race on the same problem in parallel threads.
//! Do not "fix" this by adding `&mut self` for caching — cache behind a
//! `Mutex`/`OnceLock` inside the implementor instead.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod error;
mod options;
mod problem;
mod result;
mod sparsity;

pub use error::{EvalError, SolveError};
pub use options::{
    Algorithm, BarrierUpdate, Display, FdType, HessianMode, LinearSolverKind, Options,
    RegularizationMode, ScalingMode, Tolerances,
};
pub use problem::{validate, Capabilities, EvalCounters, Nlp, NlpDims};
pub use result::{ExitFlag, IterationRecord, Solution, SolveReport, Timings};
pub use sparsity::{ColoringKind, Sparsity};

/// Machine epsilon for `f64`, hoisted so algorithm code reads like the papers.
pub const EPS: f64 = f64::EPSILON;

/// The value treated as "infinite" for a bound. Bounds with magnitude at or
/// above this are treated as absent.
///
/// Matches the IPOPT convention (`nlp_lower_bound_inf` / `nlp_upper_bound_inf`)
/// so that models transcribed from AMPL or CUTEst behave identically.
pub const INF_BOUND: f64 = 1.0e20;

/// True when `v` should be treated as no bound at all.
#[inline]
pub fn is_free(v: f64) -> bool {
    !v.is_finite() || v.abs() >= INF_BOUND
}
