//! Sparse linear algebra for `mincon`.
//!
//! # Why this crate exists at all
//!
//! Every serious interior-point NLP code needs to factor a symmetric
//! **indefinite** KKT matrix and know its inertia. The established answer is
//! `MA57` or `MA27` from HSL, and that answer is why IPOPT is awkward to
//! distribute: HSL is not redistributable under a permissive licence, so the
//! `pip install` story degrades into "obtain a licence, build from source".
//! MUMPS is LGPL-ish with a Fortran toolchain; PARDISO is proprietary.
//!
//! `faer` 0.24 — otherwise an excellent MIT-licensed Rust linear algebra
//! library — has sparse `LLT`, `LU` and `QR`, but **no sparse `LDL^T` or
//! Bunch–Kaufman**. So there is nothing to depend on. We write it.
//!
//! # The design that makes that tractable
//!
//! We do *not* implement Bunch–Kaufman with 2x2 pivots on a sparse matrix,
//! which is genuinely hard. Instead:
//!
//! 1. **Regularize into quasi-definiteness.** The KKT matrix
//!    `[[W + Sigma + delta_w I, A^T], [A, -delta_c I]]` is quasi-definite when
//!    `W + Sigma + delta_w I` is positive definite and `delta_c > 0`, and a
//!    quasi-definite matrix admits an `LDL^T` factorization under *any*
//!    symmetric permutation (Vanderbei 1995). No pivoting search needed.
//! 2. **Read the inertia off `D` for free.** `A = L D L^T` with unit lower
//!    triangular `L` is a congruence, so by Sylvester's law of inertia the
//!    signs of `D` *are* the inertia of `A` — provided no pivot broke down.
//!    We therefore get the Wächter–Biegler inertia test at zero extra cost
//!    whenever the factorization is numerically clean.
//! 3. **Fall back to a curvature test when it is not.** When pivots get small
//!    we perturb them (dynamic regularization) and the reported inertia is
//!    that of a perturbed matrix. Rather than trusting it, we hand the caller
//!    [`Factorization::inertia_is_certified`] and let the interior-point code
//!    switch to the Chiang–Zavala curvature test. That is
//!    [`mincon_core::RegularizationMode::Hybrid`].
//! 4. **Iterative refinement on the unperturbed matrix.** Dynamic
//!    regularization means we factored something we did not want to solve, so
//!    every solve is refined against the true matrix.
//!
//! The result is a pure-Rust, `unsafe`-free, permissively licensed, fully
//! redistributable KKT solver. That is not a compromise relative to MA57 — it
//! is the thing that makes `pip install mincon` possible at all.
//!
//! # Status
//!
//! Correct and tested, not yet tuned. The ordering is the known gap: see
//! [`Ordering`] and `docs/04_SPEC_LINEAR_ALGEBRA.md`.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod csc;
mod ldlt;
mod ordering;

pub use csc::{Csc, CscBuilder};
pub use ldlt::{Factorization, Inertia, LdltError, RegularizationParams, Symbolic};
pub use ordering::{compute_ordering, Ordering};
