//! Primal-dual interior-point method with a filter line search.
//!
//! This is the default algorithm, matching `fmincon`'s choice of
//! `interior-point` as its default, and for the same reason: it is the most
//! reliable single method across a broad test set, it handles bounds
//! gracefully, and it scales to sparse problems where an active-set method's
//! combinatorics do not.
//!
//! See [`solver::solve`] for the entry point and
//! `docs/02_SPEC_INTERIOR_POINT.md` for the specification, including the list
//! of what is not yet implemented.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod bfgs;
pub mod filter;
pub mod kkt;
pub mod solver;

pub use bfgs::DenseBfgs;
pub use filter::{Acceptance, Filter, FilterParams};
pub use kkt::{CorrectionParams, KktFailure, KktSystem};
pub use solver::{solve, BarrierParams};
