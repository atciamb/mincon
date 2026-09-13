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
    Algorithm, BarrierUpdate, DerivativeCheck, Display, FdType, HessianMode, IterationCallback,
    LinearSolverKind, Options, PivotSigns, RegularizationMode, ScalingMode, Tolerances,
    VariableScaling,
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

/// The gradient span above which a start that carries no scale gets a hint.
pub const NO_SCALE_HINT_GRADIENT_SPAN: f64 = 1e6;
/// The span of the start's magnitudes from which the automatic variable
/// scaling takes its factors, so no hint is needed.
pub const NO_SCALE_HINT_START_SPAN: f64 = 1e4;

/// A note for a start that carries no scale (round 5, `docs/22` §4 item 3):
/// the objective gradient at `x0` spans at least
/// [`NO_SCALE_HINT_GRADIENT_SPAN`] across the variables, the signature of
/// mismatched units, while the start's magnitudes cannot supply the scale
/// factors (a zero coordinate has no magnitude; nonzero magnitudes within
/// [`NO_SCALE_HINT_START_SPAN`] are what `VariableScaling::Auto` leaves
/// alone). A note only: it changes nothing about the solve.
#[must_use]
pub fn no_scale_hint(x0: &[f64], grad: &[f64]) -> Option<String> {
    let n = x0.len();
    if n < 2 || grad.len() != n {
        return None;
    }
    let (mut gmin, mut gmax, mut imin, mut imax) = (f64::INFINITY, 0.0_f64, 0usize, 0usize);
    for (i, g) in grad.iter().enumerate() {
        let a = g.abs();
        if !a.is_finite() || a == 0.0 {
            continue;
        }
        if a < gmin {
            gmin = a;
            imin = i;
        }
        if a > gmax {
            gmax = a;
            imax = i;
        }
    }
    if gmin == f64::INFINITY || gmax < NO_SCALE_HINT_GRADIENT_SPAN * gmin {
        return None;
    }
    let mut zeros = 0usize;
    let (mut xmin, mut xmax) = (f64::INFINITY, 0.0_f64);
    for x in x0 {
        let a = x.abs();
        if a == 0.0 || !a.is_finite() {
            zeros += 1;
        } else {
            xmin = xmin.min(a);
            xmax = xmax.max(a);
        }
    }
    if zeros == 0 && xmax >= NO_SCALE_HINT_START_SPAN * xmin {
        return None;
    }
    Some(format!(
        "The objective gradient at the start spans a factor {:.0e} across the variables (largest in x[{imax}], smallest in x[{imin}]), the signature of variables with different units, and the start carries no scale to correct it by ({zeros} of {n} coordinates are zero{}). If the units differ, start from values of typical magnitude (scale_variables = 'auto' then solves in scaled variables) or supply the typical magnitudes (typical_x, fmincon's TypicalX).",
        gmax / gmin,
        if zeros == n { "" } else { ", the others are within a factor 1e4 of each other" }
    ))
}
