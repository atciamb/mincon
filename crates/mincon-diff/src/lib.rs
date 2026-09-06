//! Derivative approximation for `mincon`.
//!
//! Three capabilities, in increasing order of how much they differentiate us
//! from `fmincon`:
//!
//! * [`fd`] — bounds-aware finite differences. `fmincon` has this; most
//!   open-source solvers do not, and it matters on models with a restricted
//!   domain.
//! * [`coloring`] — Curtis–Powell–Reid compression of the finite-difference
//!   Jacobian. `fmincon` does not do this for nonlinear constraint Jacobians,
//!   so a banded model costs it `n` evaluations per iteration where it should
//!   cost three.
//! * [`detect`] — probing for sparsity when the user has not declared it.
//!   `fmincon` requires `JacobPattern`.
//!
//! # What is deliberately missing: automatic differentiation
//!
//! There is no AD here, and that is the largest single gap in the workspace.
//! Finite differences cost accuracy (`sqrt(eps)` relative error forward) and
//! evaluations, and the accuracy loss is what caps how tightly any
//! derivative-free-gradient solve can converge. The plan, in priority order,
//! is in `docs/05_SPEC_DERIVATIVES.md`:
//!
//! 1. **Bridge to the user's AD**, which is nearly always the right answer in
//!    practice: JAX, PyTorch, CasADi and SymPy can all hand us exact gradients
//!    and Jacobians through the Python layer, and sparsity with them.
//! 2. **Forward-mode dual numbers** in Rust for models written natively
//!    against a generic-over-scalar trait. Cheap to build, exact, and it makes
//!    Hessian-vector products free.
//! 3. **Reverse-mode over a tape** for `n >> m`.
//!
//! Do not treat finite differences as the destination.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod check;
pub mod coloring;
pub mod detect;
pub mod evaluator;
pub mod fd;

pub use check::{check_derivatives, CheckReport, Discrepancy};
pub use coloring::{distance1_coloring, star_coloring, verify_coloring, Coloring};
pub use detect::{detect_jacobian_sparsity, DetectConfig, Detected};
pub use evaluator::Evaluator;
pub use fd::{central_step, forward_step, FdConfig, FiniteDifferences, Step};
