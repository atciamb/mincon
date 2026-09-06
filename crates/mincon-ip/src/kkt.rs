//! Assembly, factorization and regularization of the primal-dual KKT system.
//!
//! # The system
//!
//! With `v = (x, s)` the primal variables (original variables plus slacks for
//! the inequality constraints), `A = grad_v c_hat(v)` the `nv x m` transposed
//! Jacobian and `Sigma` the barrier diagonal, each iteration solves
//!
//! ```text
//!   [ W + Sigma + delta_w I      A     ] [ d_v     ]     [ grad phi_mu + A lambda ]
//!   [        A^T            -delta_c I ] [ d_lambda ] = - [ c_hat                  ]
//! ```
//!
//! stored as the **upper triangle** in CSC, which is what
//! [`mincon_linalg::Symbolic`] wants. The structure is fixed for the whole
//! solve, so the symbolic analysis and the index maps below are computed once
//! and every iteration is a value refill — this is the single largest
//! constant-factor win available in an interior-point code and the reason the
//! assembler looks like bookkeeping rather than mathematics.
//!
//! # Regularization
//!
//! [`KktSystem::factor_with_correction`] implements Wächter–Biegler's
//! Algorithm IC over `delta_w` and `delta_c`, with the Chiang–Zavala
//! inertia-free curvature test standing in whenever the factorization declines
//! to certify its inertia (see [`mincon_linalg::Factorization::inertia_is_certified`]).
//! That combination is what lets the whole workspace avoid an HSL dependency.

use std::sync::Arc;

use mincon_core::{RegularizationMode, Sparsity};
use mincon_linalg::{
    Csc, CscBuilder, Factorization, Inertia, LdltError, Ordering, RegularizationParams, Symbolic,
};

/// Constants of the Wächter–Biegler inertia correction, Algorithm IC.
/// Names and defaults follow the paper (Section 3.1) and IPOPT's options.
#[derive(Debug, Clone, Copy)]
pub struct CorrectionParams {
    /// `delta_w^min`, the floor once a nonzero perturbation has been used.
    pub delta_w_min: f64,
    /// `delta_w^0`, the first perturbation tried in a fresh iteration.
    pub delta_w_0: f64,
    /// `delta_w^max`; exceeding it means the step cannot be computed and the
    /// solver must enter feasibility restoration.
    pub delta_w_max: f64,
    /// `kappa_w^+`, the growth factor when a previous iteration also perturbed.
    pub kappa_w_plus: f64,
    /// `kappa_w^+bar`, the (larger) growth factor on the first perturbation.
    pub kappa_w_plus_first: f64,
    /// `kappa_w^-`, the shrink factor applied to the remembered perturbation.
    pub kappa_w_minus: f64,
    /// `delta_c^bar`, the coefficient of the dual regularization.
    pub delta_c_bar: f64,
    /// `kappa_c`, the exponent on `mu` in the dual regularization.
    pub kappa_c: f64,
    /// Curvature-test threshold, used in inertia-free mode. Chiang–Zavala's
    /// `alpha_d`, scaled by `mu`.
    pub curvature_tol: f64,
    /// Maximum factorization attempts in one iteration before giving up.
    pub max_attempts: usize,
}

impl Default for CorrectionParams {
    fn default() -> Self {
        Self {
            delta_w_min: 1e-20,
            delta_w_0: 1e-4,
            delta_w_max: 1e40,
            kappa_w_plus: 8.0,
            kappa_w_plus_first: 100.0,
            kappa_w_minus: 1.0 / 3.0,
            delta_c_bar: 1e-8,
            kappa_c: 0.25,
            curvature_tol: 1e-12,
            max_attempts: 60,
        }
    }
}

/// Outcome of one regularized factorization.
#[derive(Debug, Clone, Copy)]
pub struct FactorOutcome {
    /// Primal regularization actually used.
    pub delta_w: f64,
    /// Dual regularization actually used.
    pub delta_c: f64,
    /// Inertia reported by the factorization.
    pub inertia: Inertia,
    /// Whether that inertia was certified.
    pub certified: bool,
    /// How many factorizations this cost.
    pub attempts: usize,
}

/// Why a regularized factorization gave up.
#[derive(Debug, Clone)]
pub enum KktFailure {
    /// `delta_w` grew past its maximum; the caller must enter restoration.
    RegularizationExhausted {
        /// The last value tried.
        delta_w: f64,
    },
    /// The linear solver failed outright.
    Linear(LdltError),
}

impl std::fmt::Display for KktFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RegularizationExhausted { delta_w } => write!(
                f,
                "inertia correction exhausted at delta_w = {delta_w:.3e}; the KKT matrix cannot be made suitable"
            ),
            Self::Linear(e) => write!(f, "{e}"),
        }
    }
}

/// Maps values from one sparsity pattern into another, for transposes and for
/// scattering blocks into the assembled KKT matrix.
#[derive(Debug, Clone)]
pub struct ValueMap {
    positions: Vec<usize>,
}

impl ValueMap {
    /// Scatter `src` into `dst` at the mapped positions.
    ///
    /// # Panics
    /// If `src` is shorter than the map.
    pub fn scatter(&self, src: &[f64], dst: &mut [f64]) {
        for (k, &p) in self.positions.iter().enumerate() {
            dst[p] = src[k];
        }
    }
    /// Add `alpha * src` into `dst` at the mapped positions.
    pub fn accumulate(&self, alpha: f64, src: &[f64], dst: &mut [f64]) {
        for (k, &p) in self.positions.iter().enumerate() {
            dst[p] += alpha * src[k];
        }
    }
    /// The raw positions.
    #[must_use]
    pub fn positions(&self) -> &[usize] {
        &self.positions
    }
}

/// Transpose a pattern and record where each value moves.
#[must_use]
pub fn transpose_with_map(p: &Sparsity) -> (Sparsity, ValueMap) {
    let t = p.transpose();
    let mut positions = vec![0usize; p.nnz()];
    for j in 0..p.ncols() {
        for pos in p.col_ptr()[j]..p.col_ptr()[j + 1] {
            let i = p.row_idx()[pos];
            // In the transpose this lives in column `i`, row `j`.
            let seg = t.col(i);
            let off = seg
                .binary_search(&j)
                .expect("transpose must contain every entry");
            positions[pos] = t.col_ptr()[i] + off;
        }
    }
    (t, ValueMap { positions })
}

/// The assembled KKT system for one problem, with its symbolic factorization
/// and the index maps needed to refill it cheaply.
pub struct KktSystem {
    /// Number of primal variables including slacks.
    pub nv: usize,
    /// Number of constraints.
    pub m: usize,
    matrix: Csc,
    factor: Factorization,
    signs: Vec<i8>,

    /// Position of the `(j, j)` diagonal for each primal variable.
    diag_v: Vec<usize>,
    /// Position of the `(nv + i, nv + i)` diagonal for each constraint.
    diag_c: Vec<usize>,
    /// Where the upper-triangular Hessian values go.
    hess_map: ValueMap,
    /// Where the transposed Jacobian values go.
    jac_map: ValueMap,
    /// Where the `-1` slack coefficients go.
    slack_positions: Vec<usize>,

    /// Scratch for the right-hand side and solution.
    rhs: Vec<f64>,
    sol: Vec<f64>,

    last_delta_w: f64,
}

impl KktSystem {
    /// Build the system.
    ///
    /// * `hess_upper` is the upper-triangular pattern of the `n x n` Hessian
    ///   block (dense for quasi-Newton, the model's for exact Hessians).
    /// * `jac_t` is the `nv x m` transposed Jacobian pattern **excluding** the
    ///   slack `-1` entries, which are added here.
    /// * `slack_row[i]` is `Some(row)` when constraint `i` has a slack at
    ///   primal index `row`.
    ///
    /// # Errors
    /// If the patterns are inconsistent or the symbolic analysis fails.
    pub fn new(
        nv: usize,
        n: usize,
        m: usize,
        hess_upper: &Sparsity,
        jac_t: &Sparsity,
        slack_row: &[Option<usize>],
        ordering: Ordering,
    ) -> Result<Self, String> {
        if hess_upper.nrows() != n || hess_upper.ncols() != n {
            return Err(format!(
                "hessian pattern is {}x{}, expected {n}x{n}",
                hess_upper.nrows(),
                hess_upper.ncols()
            ));
        }
        if jac_t.nrows() != nv || jac_t.ncols() != m {
            return Err(format!(
                "transposed jacobian is {}x{}, expected {nv}x{m}",
                jac_t.nrows(),
                jac_t.ncols()
            ));
        }
        let dim = nv + m;

        // Assemble the structure once with a builder, then locate every block.
        let mut b = CscBuilder::new(dim, dim);
        b.reserve(hess_upper.nnz() + nv + jac_t.nnz() + m + m);
        for j in 0..n {
            for &i in hess_upper.col(j) {
                b.push(i, j, 0.0);
            }
        }
        for j in 0..nv {
            b.push(j, j, 0.0); // Sigma + delta_w lives here
        }
        for i in 0..m {
            for &r in jac_t.col(i) {
                b.push(r, nv + i, 0.0);
            }
            if let Some(r) = slack_row[i] {
                b.push(r, nv + i, 0.0);
            }
            b.push(nv + i, nv + i, 0.0); // -delta_c
        }
        let matrix = b.build()?;
        let pattern = matrix.pattern().clone();

        let locate = |row: usize, col: usize| -> Result<usize, String> {
            let seg = pattern.col(col);
            seg.binary_search(&row)
                .map(|off| pattern.col_ptr()[col] + off)
                .map_err(|_| format!("KKT entry ({row}, {col}) missing from the assembled pattern"))
        };

        let mut diag_v = Vec::with_capacity(nv);
        for j in 0..nv {
            diag_v.push(locate(j, j)?);
        }
        let mut diag_c = Vec::with_capacity(m);
        for i in 0..m {
            diag_c.push(locate(nv + i, nv + i)?);
        }
        let mut hess_positions = Vec::with_capacity(hess_upper.nnz());
        for j in 0..n {
            for &i in hess_upper.col(j) {
                hess_positions.push(locate(i, j)?);
            }
        }
        let mut jac_positions = Vec::with_capacity(jac_t.nnz());
        for i in 0..m {
            for &r in jac_t.col(i) {
                jac_positions.push(locate(r, nv + i)?);
            }
        }
        let mut slack_positions = Vec::new();
        for i in 0..m {
            if let Some(r) = slack_row[i] {
                slack_positions.push(locate(r, nv + i)?);
            }
        }

        let symbolic = Arc::new(
            Symbolic::analyse(&pattern, ordering).map_err(|e| format!("symbolic analysis: {e}"))?,
        );
        let factor = Factorization::new(symbolic);

        let mut signs = vec![1i8; dim];
        for s in signs.iter_mut().skip(nv) {
            *s = -1;
        }

        Ok(Self {
            nv,
            m,
            matrix,
            factor,
            signs,
            diag_v,
            diag_c,
            hess_map: ValueMap {
                positions: hess_positions,
            },
            jac_map: ValueMap {
                positions: jac_positions,
            },
            slack_positions,
            rhs: vec![0.0; dim],
            sol: vec![0.0; dim],
            last_delta_w: 0.0,
        })
    }

    /// Dimension of the KKT matrix.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.nv + self.m
    }

    /// Nonzeros in the factor. Diagnostics and ordering comparisons.
    #[must_use]
    pub fn factor_nnz(&self) -> usize {
        self.factor.symbolic().l_nnz()
    }

    /// Refill the matrix for a new iteration.
    ///
    /// `hess_upper_values` correspond to the Hessian pattern given to
    /// [`KktSystem::new`]; `jac_t_values` to the transposed Jacobian pattern;
    /// `sigma` is the length-`nv` barrier diagonal.
    pub fn assemble(
        &mut self,
        hess_upper_values: &[f64],
        jac_t_values: &[f64],
        sigma: &[f64],
        delta_w: f64,
        delta_c: f64,
    ) {
        let vals = self.matrix.values_mut();
        vals.fill(0.0);
        self.hess_map.accumulate(1.0, hess_upper_values, vals);
        self.jac_map.accumulate(1.0, jac_t_values, vals);
        for &p in &self.slack_positions {
            vals[p] = -1.0;
        }
        for (j, &p) in self.diag_v.iter().enumerate() {
            vals[p] += sigma[j] + delta_w;
        }
        for &p in &self.diag_c {
            vals[p] = -delta_c;
        }
    }

    /// Change only the regularization, reusing the rest of the assembly.
    /// Saves rebuilding the Hessian and Jacobian blocks on a retry, which is
    /// where an inertia-correction loop spends most of its time.
    pub fn set_regularization(&mut self, previous_delta_w: f64, delta_w: f64, delta_c: f64) {
        let vals = self.matrix.values_mut();
        let shift = delta_w - previous_delta_w;
        if shift != 0.0 {
            for &p in &self.diag_v {
                vals[p] += shift;
            }
        }
        for &p in &self.diag_c {
            vals[p] = -delta_c;
        }
    }

    /// The assembled matrix, for residual computation.
    #[must_use]
    pub fn matrix(&self) -> &Csc {
        &self.matrix
    }

    /// Factor the currently assembled matrix, raising `delta_w` until the step
    /// it produces is usable.
    ///
    /// `rhs` is consumed to produce `sol`; both are length `nv + m`. `mu`
    /// scales the dual regularization and the curvature threshold.
    ///
    /// # Errors
    /// [`KktFailure::RegularizationExhausted`] when no `delta_w` works, which
    /// is the signal to enter feasibility restoration.
    pub fn factor_with_correction(
        &mut self,
        hess_upper_values: &[f64],
        jac_t_values: &[f64],
        sigma: &[f64],
        mu: f64,
        mode: RegularizationMode,
        params: &CorrectionParams,
    ) -> Result<FactorOutcome, KktFailure> {
        let (nv, m) = (self.nv, self.m);
        let mut delta_w = 0.0;
        let mut delta_c = 0.0;
        let mut applied = 0.0;
        self.assemble(hess_upper_values, jac_t_values, sigma, 0.0, 0.0);

        let reg = RegularizationParams::default();
        let mut first_perturbation = true;

        for attempt in 1..=params.max_attempts {
            let result = self.factor.factor(self.matrix.values(), &self.signs, &reg);
            let mut acceptable = false;
            let mut inertia = Inertia::default();
            let mut certified = false;

            match result {
                Ok(inert) => {
                    inertia = inert;
                    certified = self.factor.inertia_is_certified();
                    let singular = inert.zero > 0;
                    acceptable = match mode {
                        RegularizationMode::Inertia => certified && inert.is_kkt_correct(nv, m),
                        RegularizationMode::InertiaFree => !singular,
                        RegularizationMode::Hybrid => {
                            if certified {
                                inert.is_kkt_correct(nv, m)
                            } else {
                                !singular
                            }
                        }
                    };
                }
                Err(LdltError::ZeroPivot(_) | LdltError::NonFinite(_)) => {}
                Err(e) => return Err(KktFailure::Linear(e)),
            }

            if acceptable {
                // In inertia-free / uncertified mode the inertia is not a
                // certificate, so the caller must run the curvature test on the
                // computed direction. That happens in `solve_and_verify`.
                self.last_delta_w = delta_w;
                return Ok(FactorOutcome {
                    delta_w,
                    delta_c,
                    inertia,
                    certified,
                    attempts: attempt,
                });
            }

            // Algorithm IC steps 2-5.
            let previous = applied;
            if inertia.zero > 0 || !matches!(mode, RegularizationMode::Inertia) && delta_w > 0.0 {
                delta_c = params.delta_c_bar * mu.max(f64::MIN_POSITIVE).powf(params.kappa_c);
            }
            delta_w = if delta_w == 0.0 {
                if self.last_delta_w == 0.0 {
                    params.delta_w_0
                } else {
                    (params.delta_w_min).max(params.kappa_w_minus * self.last_delta_w)
                }
            } else if first_perturbation && self.last_delta_w == 0.0 {
                first_perturbation = false;
                params.kappa_w_plus_first * delta_w
            } else {
                params.kappa_w_plus * delta_w
            };

            if delta_w > params.delta_w_max {
                return Err(KktFailure::RegularizationExhausted { delta_w });
            }
            self.set_regularization(previous, delta_w, delta_c);
            applied = delta_w;
        }
        Err(KktFailure::RegularizationExhausted { delta_w })
    }

    /// Solve with the current factors, refining against the assembled matrix.
    ///
    /// # Errors
    /// A linear-algebra failure.
    pub fn solve(&mut self, rhs: &[f64], sol: &mut [f64], steps: usize) -> Result<f64, LdltError> {
        self.factor.solve_refined(&self.matrix, rhs, sol, steps)
    }

    /// Chiang–Zavala curvature test on a computed direction.
    ///
    /// Returns `true` when the direction has enough positive curvature to be a
    /// descent direction for the barrier problem, which is the inertia-free
    /// substitute for checking that the KKT matrix has inertia `(nv, m, 0)`.
    ///
    /// The test is on the full step `d_v`:
    /// `d^T (W + Sigma + delta_w I) d + max(-lambda^+ . c, 0) >= alpha ||d||^2`.
    #[must_use]
    #[allow(clippy::too_many_arguments)] // a curvature test needs the curvature
    pub fn curvature_is_sufficient(
        &self,
        d_v: &[f64],
        d_lambda: &[f64],
        c_hat: &[f64],
        hess_upper_values: &[f64],
        sigma: &[f64],
        delta_w: f64,
        n: usize,
        threshold: f64,
    ) -> bool {
        // d^T W d over the Hessian block (upper triangle, symmetric).
        let mut quad = 0.0;
        let hp = self.hess_map.positions();
        let vals = self.matrix.values();
        let _ = vals;
        // Recompute from the caller's Hessian values so this is independent of
        // whatever regularization was folded into the matrix.
        // The Hessian pattern is walked in the same order it was registered.
        // Reconstruct (row, col) from the stored positions via the matrix.
        for (k, &p) in hp.iter().enumerate() {
            let (row, col) = self.entry_coords(p);
            if row < n && col < n {
                let v = hess_upper_values[k];
                quad += if row == col {
                    v * d_v[row] * d_v[col]
                } else {
                    2.0 * v * d_v[row] * d_v[col]
                };
            }
        }
        for j in 0..self.nv {
            quad += (sigma[j] + delta_w) * d_v[j] * d_v[j];
        }
        let dual_term: f64 = d_lambda
            .iter()
            .zip(c_hat)
            .map(|(l, c)| -l * c)
            .sum::<f64>()
            .max(0.0);
        let norm2: f64 = d_v.iter().map(|v| v * v).sum();
        quad + dual_term >= threshold * norm2
    }

    /// Recover `(row, col)` for a position in the value array.
    fn entry_coords(&self, pos: usize) -> (usize, usize) {
        let cp = self.matrix.pattern().col_ptr();
        // Column is the last `j` with `col_ptr[j] <= pos`.
        let col = match cp.binary_search(&pos) {
            Ok(mut j) => {
                while j + 1 < cp.len() && cp[j + 1] == cp[j] {
                    j += 1;
                }
                j
            }
            Err(j) => j - 1,
        };
        (self.matrix.pattern().row_idx()[pos], col)
    }

    /// Scratch right-hand side buffer.
    pub fn rhs_mut(&mut self) -> &mut [f64] {
        &mut self.rhs
    }
    /// Scratch solution buffer.
    #[must_use]
    pub fn sol(&self) -> &[f64] {
        &self.sol
    }
    /// Solve using the internal scratch buffers.
    ///
    /// # Errors
    /// A linear-algebra failure.
    pub fn solve_scratch(&mut self, steps: usize) -> Result<f64, LdltError> {
        let rhs = std::mem::take(&mut self.rhs);
        let mut sol = std::mem::take(&mut self.sol);
        let out = self
            .factor
            .solve_refined(&self.matrix, &rhs, &mut sol, steps);
        self.rhs = rhs;
        self.sol = sol;
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mincon_core::Sparsity;

    #[test]
    fn transpose_map_moves_values_correctly() {
        let p = Sparsity::from_triplets(2, 3, &[(0, 0), (1, 1), (0, 2), (1, 2)]).unwrap();
        let (t, map) = transpose_with_map(&p);
        assert_eq!(t.nrows(), 3);
        assert_eq!(t.ncols(), 2);
        // Values in pattern order: col0 -> (0,0); col1 -> (1,1); col2 -> (0,2),(1,2)
        let vals = [10.0, 20.0, 30.0, 40.0];
        let mut tvals = vec![0.0; t.nnz()];
        map.scatter(&vals, &mut tvals);
        // Rebuild densely and compare.
        let m = Csc::new(t.clone(), tvals).unwrap().to_dense();
        assert_eq!(m[0][0], 10.0);
        assert_eq!(m[1][1], 20.0);
        assert_eq!(m[2][0], 30.0);
        assert_eq!(m[2][1], 40.0);
    }

    /// A tiny equality-constrained QP whose KKT solution is known by hand:
    ///   min 0.5(x0^2 + x1^2)  s.t.  x0 + x1 = 2
    /// KKT: x = (1,1), lambda = -1 with the convention grad f + A lambda = 0.
    #[test]
    fn assembles_and_solves_a_hand_checkable_kkt_system() {
        let (n, nv, m) = (2, 2, 1);
        let hess = Sparsity::from_triplets(2, 2, &[(0, 0), (1, 1)]).unwrap();
        let jac_t = Sparsity::from_triplets(2, 1, &[(0, 0), (1, 0)]).unwrap();
        let mut k = KktSystem::new(nv, n, m, &hess, &jac_t, &[None], Ordering::Natural).unwrap();
        assert_eq!(k.dim(), 3);

        let outcome = k
            .factor_with_correction(
                &[1.0, 1.0],
                &[1.0, 1.0],
                &[0.0, 0.0],
                0.1,
                RegularizationMode::Hybrid,
                &CorrectionParams::default(),
            )
            .unwrap();
        assert!(
            outcome.inertia.is_kkt_correct(2, 1),
            "{:?}",
            outcome.inertia
        );
        assert_eq!(
            outcome.delta_w, 0.0,
            "a well-posed system needs no perturbation"
        );

        // Solve [H A; A^T 0] [x; l] = [0; 2]
        let rhs = [0.0, 0.0, 2.0];
        let mut sol = vec![0.0; 3];
        let r = k.solve(&rhs, &mut sol, 2).unwrap();
        assert!(r < 1e-12, "residual {r}");
        assert!((sol[0] - 1.0).abs() < 1e-10, "x0 = {}", sol[0]);
        assert!((sol[1] - 1.0).abs() < 1e-10, "x1 = {}", sol[1]);
        // Stationarity is grad f + A lambda = 0 with grad f = H x = (1, 1)
        // and A = (1, 1)^T, so lambda = -1. The opposite sign here would mean
        // every multiplier the solver reports is backwards.
        assert!((sol[2] + 1.0).abs() < 1e-10, "lambda = {}", sol[2]);
    }

    #[test]
    fn indefinite_hessian_triggers_regularization_until_inertia_is_right() {
        let (n, nv, m) = (2, 2, 1);
        let hess = Sparsity::from_triplets(2, 2, &[(0, 0), (1, 1)]).unwrap();
        let jac_t = Sparsity::from_triplets(2, 1, &[(0, 0), (1, 0)]).unwrap();
        let mut k = KktSystem::new(nv, n, m, &hess, &jac_t, &[None], Ordering::Natural).unwrap();

        // Strongly negative curvature in both directions: no (2,1,0) inertia
        // is possible until delta_w exceeds 5.
        let outcome = k
            .factor_with_correction(
                &[-5.0, -5.0],
                &[1.0, 1.0],
                &[0.0, 0.0],
                0.1,
                RegularizationMode::Inertia,
                &CorrectionParams::default(),
            )
            .unwrap();
        assert!(outcome.delta_w > 5.0, "delta_w = {}", outcome.delta_w);
        assert!(outcome.inertia.is_kkt_correct(2, 1));
        assert!(outcome.attempts > 1);
    }

    #[test]
    fn slack_columns_get_their_minus_one() {
        let (n, nv, m) = (1, 2, 1);
        let hess = Sparsity::from_triplets(1, 1, &[(0, 0)]).unwrap();
        let jac_t = Sparsity::from_triplets(2, 1, &[(0, 0)]).unwrap();
        let mut k = KktSystem::new(nv, n, m, &hess, &jac_t, &[Some(1)], Ordering::Natural).unwrap();
        k.assemble(&[2.0], &[3.0], &[0.5, 0.5], 0.0, 1e-8);
        let dense = k.matrix().to_dense();
        assert_eq!(dense[0][0], 2.5);
        assert_eq!(dense[1][1], 0.5);
        assert_eq!(dense[0][2], 3.0, "jacobian entry");
        assert_eq!(dense[1][2], -1.0, "slack coefficient");
        assert_eq!(dense[2][2], -1e-8, "dual regularization");
    }

    #[test]
    fn entry_coords_round_trips_every_position() {
        let (n, nv, m) = (3, 4, 2);
        let hess = Sparsity::from_triplets(3, 3, &[(0, 0), (0, 1), (1, 1), (2, 2)]).unwrap();
        let jac_t = Sparsity::from_triplets(4, 2, &[(0, 0), (2, 0), (1, 1), (3, 1)]).unwrap();
        let k = KktSystem::new(
            nv,
            n,
            m,
            &hess,
            &jac_t,
            &[Some(2), Some(3)],
            Ordering::Natural,
        )
        .unwrap();
        let p = k.matrix().pattern();
        for col in 0..p.ncols() {
            for pos in p.col_ptr()[col]..p.col_ptr()[col + 1] {
                let (r, c) = k.entry_coords(pos);
                assert_eq!(c, col, "column mismatch at position {pos}");
                assert_eq!(r, p.row_idx()[pos]);
            }
        }
    }
}
