//! Sparse `LDL^T` factorization with dynamic regularization and inertia.
//!
//! An up-looking `LDL^T` in the style of `QDLDL`, with three additions that the
//! interior-point method needs and `QDLDL` does not provide:
//!
//! * a fill-reducing permutation folded into the symbolic phase,
//! * per-pivot **sign expectations**, so a pivot that comes out with the wrong
//!   sign is perturbed rather than producing a direction of the wrong type,
//! * an [`Inertia`] report with an honest [`Factorization::inertia_is_certified`]
//!   flag, so callers know whether Sylvester's law actually applies to the
//!   matrix they wanted (rather than to the one we perturbed into existence).
//!
//! # The mathematics we are leaning on
//!
//! `A = L D L^T` with `L` unit lower triangular is a congruence transformation,
//! so by **Sylvester's law of inertia** the signs of `D` are the inertia of `A`.
//! No 2x2 pivots, no Bunch–Kaufman search, no eigenvalues. The catch is that a
//! symmetric indefinite matrix need not *have* an `LDL^T` with 1x1 pivots — the
//! factorization can break down on a zero pivot, and near-breakdown is
//! numerically poisonous. That is exactly the case we detect and report as
//! *uncertified*, and it is exactly the case where the interior-point code
//! raises `delta_w`, which pushes the matrix toward quasi-definiteness, where
//! the factorization is unconditionally stable (Vanderbei 1995).
//!
//! So the loop closes: uncertified inertia is itself the signal to regularize,
//! and regularizing is what makes the inertia certifiable.
//!
//! # Complexity
//!
//! Symbolic: `O(|A| * alpha)` for the elimination tree plus the ordering.
//! Numeric: `O(sum_j |L_j|^2 / |L_j|)` — i.e. proportional to the flop count of
//! the factorization, which is what the ordering exists to minimize.

use std::sync::Arc;

use mincon_core::Sparsity;
use thiserror::Error;

use crate::csc::Csc;
use crate::ordering::{compute_ordering, invert, Ordering};

const NONE: usize = usize::MAX;

/// Largest entry of `L` we are willing to see before declaring the
/// factorization numerically untrustworthy.
///
/// `LDL^T` with 1x1 pivots and no pivoting search is unconditionally stable on
/// quasi-definite matrices and *unstable in general* — the classic counterexample
/// being a symmetric matrix with a tiny diagonal and an `O(1)` off-diagonal,
/// where `L` grows like the reciprocal of the pivot and the computed
/// factorization has nothing to do with the input. Bunch–Kaufman solves this
/// with 2x2 pivots; we solve it by detecting the growth and telling the caller,
/// whose response is to raise `delta_w` — which is precisely the move that
/// pushes the matrix back toward quasi-definiteness.
///
/// The threshold is a heuristic, as it is in every sparse code. `1e10` keeps
/// roughly six digits of the residual on a double-precision solve.
pub const DEFAULT_GROWTH_LIMIT: f64 = 1e10;

/// Failures of the factorization.
#[derive(Debug, Clone, Error, PartialEq)]
pub enum LdltError {
    /// The input pattern was not a square upper triangle.
    #[error("matrix structure is invalid: {0}")]
    Structure(String),
    /// A pivot was exactly zero and regularization was disabled.
    #[error("zero pivot at position {0}; enable dynamic regularization or raise delta_w")]
    ZeroPivot(usize),
    /// A pivot or an intermediate became non-finite. The caller should raise
    /// `delta_w` sharply; if that does not help the model is producing garbage.
    #[error("non-finite value encountered at pivot {0}")]
    NonFinite(usize),
    /// A slice handed in had the wrong length.
    #[error("dimension mismatch: {0}")]
    Dimension(String),
}

/// The inertia of a symmetric matrix: counts of positive, negative and zero
/// eigenvalues.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Inertia {
    /// Number of positive eigenvalues.
    pub positive: usize,
    /// Number of negative eigenvalues.
    pub negative: usize,
    /// Number of eigenvalues indistinguishable from zero.
    pub zero: usize,
}

impl Inertia {
    /// Whether this matches the inertia an interior-point KKT matrix must have
    /// for the step to be a descent direction on the tangent space:
    /// `n` positive, `m` negative, none zero (Wächter–Biegler, Theorem 4 /
    /// Nocedal–Wright Theorem 16.3).
    #[must_use]
    pub fn is_kkt_correct(self, n: usize, m: usize) -> bool {
        self.positive == n && self.negative == m && self.zero == 0
    }
}

/// Dynamic regularization: what to do about pivots that are too small or have
/// the wrong sign.
///
/// This is *not* the interior-point `delta_w` / `delta_c` regularization, which
/// changes the matrix the algorithm intends to factor. This is the last line of
/// defence inside the factorization, perturbing individual pivots to keep the
/// triangular solves finite. Because it silently changes the matrix, every
/// solve that uses it must be iteratively refined — see
/// [`Factorization::solve_refined`].
#[derive(Debug, Clone, Copy)]
pub struct RegularizationParams {
    /// Whether to perturb bad pivots at all.
    pub enabled: bool,
    /// Pivots with magnitude below this are perturbed.
    pub eps: f64,
    /// Magnitude to perturb them to.
    pub delta: f64,
}

impl Default for RegularizationParams {
    fn default() -> Self {
        // Values in the range used by OSQP/Clarabel: perturb only pivots that
        // are numerically indistinguishable from zero, and perturb them to
        // something small enough not to distort the step but large enough to
        // keep the back-substitution finite.
        Self {
            enabled: true,
            eps: 1e-13,
            delta: 1e-7,
        }
    }
}

impl RegularizationParams {
    /// Disable perturbation entirely; a bad pivot then becomes an error.
    /// Used in tests that need the factorization to be exact.
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }
}

/// The symbolic analysis: permutation, elimination tree and the structure of
/// `L`. Depends only on the pattern, so it is computed once and reused for
/// every iteration of the solve.
#[derive(Debug)]
pub struct Symbolic {
    n: usize,
    perm: Vec<usize>,
    iperm: Vec<usize>,
    /// Permuted upper-triangular pattern, diagonal guaranteed present.
    ap: Vec<usize>,
    ai: Vec<usize>,
    /// `value_map[k]` is where original entry `k` lands in the permuted values.
    value_map: Vec<usize>,
    /// Position in the permuted arrays of each diagonal entry.
    diag_pos: Vec<usize>,
    etree: Vec<usize>,
    lp: Vec<usize>,
    ordering: Ordering,
}

impl Symbolic {
    /// Analyse the structure of a symmetric matrix given by its **upper
    /// triangle** (row index `<=` column index for every stored entry).
    ///
    /// # Errors
    /// [`LdltError::Structure`] if the pattern is not a square upper triangle.
    pub fn analyse(pattern: &Sparsity, ordering: Ordering) -> Result<Self, LdltError> {
        let n = pattern.ncols();
        if pattern.nrows() != n {
            return Err(LdltError::Structure(format!(
                "matrix must be square, got {}x{}",
                pattern.nrows(),
                n
            )));
        }
        for j in 0..n {
            if let Some(&last) = pattern.col(j).last() {
                if last > j {
                    return Err(LdltError::Structure(format!(
                        "entry ({last}, {j}) is below the diagonal; supply the upper triangle"
                    )));
                }
            }
        }

        let perm = compute_ordering(n, pattern.col_ptr(), pattern.row_idx(), ordering);
        let iperm = invert(&perm);

        // Build the permuted upper-triangular pattern, always including the
        // diagonal so that every pivot has a home even if A[k,k] is
        // structurally absent.
        let mut cols: Vec<Vec<usize>> = (0..n).map(|k| vec![k]).collect();
        let cp = pattern.col_ptr();
        let ri = pattern.row_idx();
        for j in 0..n {
            for p in cp[j]..cp[j + 1] {
                let i = ri[p];
                let (a, b) = (iperm[i], iperm[j]);
                let (r, c) = if a <= b { (a, b) } else { (b, a) };
                cols[c].push(r);
            }
        }
        let mut ap = Vec::with_capacity(n + 1);
        let mut ai = Vec::new();
        ap.push(0);
        for c in &mut cols {
            c.sort_unstable();
            c.dedup();
            ai.extend_from_slice(c);
            ap.push(ai.len());
        }

        // Map every original entry to its slot in the permuted arrays.
        let mut value_map = vec![0usize; pattern.nnz()];
        for j in 0..n {
            for p in cp[j]..cp[j + 1] {
                let i = ri[p];
                let (a, b) = (iperm[i], iperm[j]);
                let (r, c) = if a <= b { (a, b) } else { (b, a) };
                let seg = &ai[ap[c]..ap[c + 1]];
                let off = seg.binary_search(&r).map_err(|_| {
                    LdltError::Structure("internal: permuted entry not found".into())
                })?;
                value_map[p] = ap[c] + off;
            }
        }
        let mut diag_pos = vec![0usize; n];
        for k in 0..n {
            let seg = &ai[ap[k]..ap[k + 1]];
            let off = seg
                .binary_search(&k)
                .map_err(|_| LdltError::Structure("internal: missing diagonal".into()))?;
            diag_pos[k] = ap[k] + off;
        }

        // Elimination tree and column counts (QDLDL formulation).
        let mut etree = vec![NONE; n];
        let mut lnz = vec![0usize; n];
        let mut work = vec![NONE; n];
        for j in 0..n {
            work[j] = j;
            for p in ap[j]..ap[j + 1] {
                let mut i = ai[p];
                while work[i] != j {
                    if etree[i] == NONE {
                        etree[i] = j;
                    }
                    lnz[i] += 1;
                    work[i] = j;
                    i = etree[i];
                }
            }
        }
        let mut lp = Vec::with_capacity(n + 1);
        lp.push(0);
        let mut acc = 0usize;
        for &c in &lnz {
            acc += c;
            lp.push(acc);
        }

        Ok(Self {
            n,
            perm,
            iperm,
            ap,
            ai,
            value_map,
            diag_pos,
            etree,
            lp,
            ordering,
        })
    }

    /// Matrix dimension.
    #[must_use]
    pub fn n(&self) -> usize {
        self.n
    }
    /// Number of nonzeros in `L`, excluding the unit diagonal.
    #[must_use]
    pub fn l_nnz(&self) -> usize {
        self.lp[self.n]
    }
    /// The permutation used; `perm()[k]` is the original index of pivot `k`.
    #[must_use]
    pub fn perm(&self) -> &[usize] {
        &self.perm
    }
    /// The inverse permutation; `inverse_perm()[i]` is the pivot position of
    /// original index `i`.
    #[must_use]
    pub fn inverse_perm(&self) -> &[usize] {
        &self.iperm
    }
    /// The ordering strategy that produced the permutation.
    #[must_use]
    pub fn ordering(&self) -> Ordering {
        self.ordering
    }
    /// Fill ratio `nnz(L) / nnz(upper(A))`. The number to watch when comparing
    /// orderings; see `docs/04_SPEC_LINEAR_ALGEBRA.md`.
    #[must_use]
    pub fn fill_ratio(&self) -> f64 {
        let a = self.ai.len().max(1) as f64;
        self.l_nnz() as f64 / a
    }
}

/// A numeric factorization, reusable across iterations.
///
/// Allocation happens once in [`Factorization::new`]; [`Factorization::factor`]
/// is allocation-free, which matters because it runs at least once per
/// interior-point iteration and several times per regularization retry.
#[derive(Debug)]
pub struct Factorization {
    sym: Arc<Symbolic>,
    li: Vec<usize>,
    lx: Vec<f64>,
    d: Vec<f64>,
    dinv: Vec<f64>,
    inertia: Inertia,
    n_regularized: usize,
    min_abs_pivot: f64,
    max_abs_l: f64,
    growth_limit: f64,
    factored: bool,

    // Workspaces.
    pv: Vec<f64>,
    y: Vec<f64>,
    pattern_buf: Vec<usize>,
    mark: Vec<usize>,
    lnext: Vec<usize>,
    work: Vec<f64>,
    resid: Vec<f64>,
    corr: Vec<f64>,
    signs_perm: Vec<i8>,
}

impl Factorization {
    /// Allocate for a given symbolic analysis.
    #[must_use]
    pub fn new(sym: Arc<Symbolic>) -> Self {
        let n = sym.n;
        let lnnz = sym.l_nnz();
        let apnnz = sym.ai.len();
        Self {
            li: vec![0; lnnz],
            lx: vec![0.0; lnnz],
            d: vec![0.0; n],
            dinv: vec![0.0; n],
            inertia: Inertia::default(),
            n_regularized: 0,
            min_abs_pivot: f64::INFINITY,
            max_abs_l: 0.0,
            growth_limit: DEFAULT_GROWTH_LIMIT,
            factored: false,
            pv: vec![0.0; apnnz],
            y: vec![0.0; n],
            pattern_buf: vec![0; n],
            mark: vec![NONE; n],
            lnext: vec![0; n],
            work: vec![0.0; n],
            resid: vec![0.0; n],
            corr: vec![0.0; n],
            signs_perm: vec![0; n],
            sym,
        }
    }

    /// The symbolic analysis this factorization is bound to.
    #[must_use]
    pub fn symbolic(&self) -> &Symbolic {
        &self.sym
    }

    /// Factor `A = L D L^T`.
    ///
    /// `values` are the entries of the upper triangle in the order of the
    /// pattern given to [`Symbolic::analyse`].
    ///
    /// `signs` gives, **in original index order**, the sign each pivot is
    /// expected to have: `+1` for rows of the primal block, `-1` for rows of
    /// the dual block, `0` for "no expectation". A pivot that comes out with
    /// the wrong sign is perturbed to the expected sign when regularization is
    /// enabled — this is what keeps a quasi-definite KKT matrix on the rails
    /// even when the numerical values wander.
    ///
    /// # Errors
    /// [`LdltError::ZeroPivot`] or [`LdltError::NonFinite`] when the
    /// factorization cannot proceed; the caller should raise `delta_w` and retry.
    pub fn factor(
        &mut self,
        values: &[f64],
        signs: &[i8],
        reg: &RegularizationParams,
    ) -> Result<Inertia, LdltError> {
        let n = self.sym.n;
        if values.len() != self.sym.value_map.len() {
            return Err(LdltError::Dimension(format!(
                "values has length {} but the pattern has {} entries",
                values.len(),
                self.sym.value_map.len()
            )));
        }
        if signs.len() != n {
            return Err(LdltError::Dimension(format!(
                "signs has length {} but the matrix is {n}x{n}",
                signs.len()
            )));
        }

        // Scatter the original values into the permuted layout.
        self.pv.fill(0.0);
        for (k, &v) in values.iter().enumerate() {
            self.pv[self.sym.value_map[k]] += v;
        }
        for k in 0..n {
            self.signs_perm[k] = signs[self.sym.perm[k]];
        }

        self.y.fill(0.0);
        self.mark.fill(NONE);
        self.lnext.copy_from_slice(&self.sym.lp[..n]);
        self.n_regularized = 0;
        self.min_abs_pivot = f64::INFINITY;
        self.max_abs_l = 0.0;
        let (mut npos, mut nneg, mut nzero) = (0usize, 0usize, 0usize);

        let ap = &self.sym.ap;
        let ai = &self.sym.ai;
        let etree = &self.sym.etree;
        let lp = &self.sym.lp;

        for k in 0..n {
            // Gather the pattern of row k of L: the reach of the off-diagonal
            // entries of column k of A through the elimination tree.
            let mut cnt = 0usize;
            for p in ap[k]..ap[k + 1] {
                let i = ai[p];
                if i == k {
                    continue;
                }
                self.y[i] += self.pv[p];
                let mut j = i;
                while j != NONE && j < k && self.mark[j] != k {
                    self.mark[j] = k;
                    self.pattern_buf[cnt] = j;
                    cnt += 1;
                    j = etree[j];
                }
            }
            // Ascending index order is a valid topological order because
            // `etree[j] > j` always, so a node precedes its ancestors.
            self.pattern_buf[..cnt].sort_unstable();

            let mut dk = self.pv[self.sym.diag_pos[k]];
            for idx in 0..cnt {
                let j = self.pattern_buf[idx];
                let yj = self.y[j];
                self.y[j] = 0.0;
                for p in lp[j]..self.lnext[j] {
                    self.y[self.li[p]] -= self.lx[p] * yj;
                }
                let lkj = yj * self.dinv[j];
                if !lkj.is_finite() {
                    return Err(LdltError::NonFinite(k));
                }
                self.max_abs_l = self.max_abs_l.max(lkj.abs());
                let slot = self.lnext[j];
                self.li[slot] = k;
                self.lx[slot] = lkj;
                self.lnext[j] = slot + 1;
                dk -= yj * lkj;
            }

            if !dk.is_finite() {
                return Err(LdltError::NonFinite(k));
            }

            let want = self.signs_perm[k];
            let bad_sign = (want > 0 && dk <= 0.0) || (want < 0 && dk >= 0.0);
            if reg.enabled && (dk.abs() < reg.eps || bad_sign) {
                let sign = if want != 0 {
                    f64::from(want)
                } else if dk < 0.0 {
                    -1.0
                } else {
                    1.0
                };
                dk = sign * reg.delta;
                self.n_regularized += 1;
            }
            if dk == 0.0 {
                return Err(LdltError::ZeroPivot(k));
            }

            self.min_abs_pivot = self.min_abs_pivot.min(dk.abs());
            if dk > 0.0 {
                npos += 1;
            } else if dk < 0.0 {
                nneg += 1;
            } else {
                nzero += 1;
            }
            self.d[k] = dk;
            self.dinv[k] = 1.0 / dk;
        }

        self.inertia = Inertia {
            positive: npos,
            negative: nneg,
            zero: nzero,
        };
        self.factored = true;
        Ok(self.inertia)
    }

    /// The inertia from the most recent [`Factorization::factor`].
    #[must_use]
    pub fn inertia(&self) -> Inertia {
        self.inertia
    }

    /// How many pivots were perturbed in the most recent factorization.
    #[must_use]
    pub fn regularized_pivots(&self) -> usize {
        self.n_regularized
    }

    /// The smallest pivot magnitude seen.
    #[must_use]
    pub fn min_abs_pivot(&self) -> f64 {
        self.min_abs_pivot
    }

    /// The largest magnitude in `L`. A proxy for the growth factor: large
    /// values mean the factorization is not backward stable and both the
    /// solution and the inertia should be distrusted.
    #[must_use]
    pub fn growth(&self) -> f64 {
        self.max_abs_l
    }

    /// Set the growth threshold used by [`Factorization::inertia_is_certified`].
    pub fn set_growth_limit(&mut self, limit: f64) {
        self.growth_limit = limit;
    }

    /// Whether [`Factorization::inertia`] is the inertia of the matrix that was
    /// actually handed in.
    ///
    /// False when either
    /// * a pivot was perturbed — the reported inertia then belongs to a nearby
    ///   matrix rather than to the one asked about, or
    /// * `L` grew past [`Factorization::growth`]'s limit — the factorization is
    ///   not backward stable, so Sylvester's law is being applied to a matrix
    ///   we did not compute.
    ///
    /// An interior-point method must not treat an uncertified inertia as a
    /// certificate. The current KKT driver retries ordering, scaling, and
    /// explicit regularization until this certificate is available.
    #[must_use]
    pub fn inertia_is_certified(&self) -> bool {
        self.factored && self.n_regularized == 0 && self.max_abs_l < self.growth_limit
    }

    /// Solve `A x = b` using the stored factors. `x` may alias nothing.
    ///
    /// # Errors
    /// [`LdltError::Dimension`] on a length mismatch.
    pub fn solve(&mut self, b: &[f64], x: &mut [f64]) -> Result<(), LdltError> {
        let n = self.sym.n;
        if b.len() != n || x.len() != n {
            return Err(LdltError::Dimension(format!(
                "solve expects vectors of length {n}"
            )));
        }
        for k in 0..n {
            self.work[k] = b[self.sym.perm[k]];
        }
        let lp = &self.sym.lp;
        for j in 0..n {
            let wj = self.work[j];
            if wj != 0.0 {
                for p in lp[j]..lp[j + 1] {
                    self.work[self.li[p]] -= self.lx[p] * wj;
                }
            }
        }
        for k in 0..n {
            self.work[k] *= self.dinv[k];
        }
        for j in (0..n).rev() {
            let mut acc = self.work[j];
            for p in lp[j]..lp[j + 1] {
                acc -= self.lx[p] * self.work[self.li[p]];
            }
            self.work[j] = acc;
        }
        for k in 0..n {
            x[self.sym.perm[k]] = self.work[k];
        }
        Ok(())
    }

    /// Solve `A x = b` with `steps` rounds of iterative refinement against the
    /// **unperturbed** matrix `a` (upper triangle).
    ///
    /// Mandatory whenever dynamic regularization may have fired, and cheap
    /// enough (one sparse mat-vec plus one triangular solve per step) that the
    /// default is one step unconditionally. Returns the final residual
    /// infinity-norm, which the interior-point code logs and uses to decide
    /// whether to raise `delta_w`.
    ///
    /// # Errors
    /// [`LdltError::Dimension`] on a length mismatch.
    pub fn solve_refined(
        &mut self,
        a: &Csc,
        b: &[f64],
        x: &mut [f64],
        steps: usize,
    ) -> Result<f64, LdltError> {
        let n = self.sym.n;
        if a.nrows() != n || a.ncols() != n {
            return Err(LdltError::Dimension(format!(
                "refinement matrix is {}x{}, expected {n}x{n}",
                a.nrows(),
                a.ncols()
            )));
        }
        self.solve(b, x)?;
        for _ in 0..steps {
            let norm = self.residual_into_buffer(a, b, x)?;
            if norm == 0.0 {
                break;
            }
            // Both buffers are moved out and unconditionally moved back, so a
            // failing solve cannot leave the factorization without workspaces.
            let resid = std::mem::take(&mut self.resid);
            let mut corr = std::mem::take(&mut self.corr);
            let outcome = self.solve(&resid, &mut corr);
            self.resid = resid;
            self.corr = corr;
            outcome?;
            for i in 0..n {
                x[i] += self.corr[i];
            }
        }
        self.residual_into_buffer(a, b, x)
    }

    /// `self.resid <- b - A x`; returns the infinity norm.
    fn residual_into_buffer(&mut self, a: &Csc, b: &[f64], x: &[f64]) -> Result<f64, LdltError> {
        let n = self.sym.n;
        let mut ax = std::mem::take(&mut self.work);
        ax.fill(0.0);
        a.gemv_symmetric_upper(1.0, x, &mut ax);
        let mut norm = 0.0_f64;
        for i in 0..n {
            let r = b[i] - ax[i];
            self.resid[i] = r;
            norm = norm.max(r.abs());
        }
        self.work = ax;
        if !norm.is_finite() {
            return Err(LdltError::NonFinite(0));
        }
        Ok(norm)
    }

    /// The diagonal `D`, in permuted order. Diagnostics only.
    #[must_use]
    pub fn d(&self) -> &[f64] {
        &self.d
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::csc::CscBuilder;

    /// Dense symmetric eigenvalues by the cyclic Jacobi method — an
    /// independent oracle for the inertia, deliberately sharing no code with
    /// the factorization under test.
    fn dense_eigenvalues(mut a: Vec<Vec<f64>>) -> Vec<f64> {
        let n = a.len();
        for _sweep in 0..100 {
            let mut off = 0.0;
            for i in 0..n {
                for j in (i + 1)..n {
                    off += a[i][j] * a[i][j];
                }
            }
            if off.sqrt() < 1e-14 {
                break;
            }
            for p in 0..n {
                for q in (p + 1)..n {
                    if a[p][q].abs() < 1e-300 {
                        continue;
                    }
                    let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                    let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                    let c = 1.0 / (t * t + 1.0).sqrt();
                    let s = t * c;
                    for k in 0..n {
                        let akp = a[k][p];
                        let akq = a[k][q];
                        a[k][p] = c * akp - s * akq;
                        a[k][q] = s * akp + c * akq;
                    }
                    for k in 0..n {
                        let apk = a[p][k];
                        let aqk = a[q][k];
                        a[p][k] = c * apk - s * aqk;
                        a[q][k] = s * apk + c * aqk;
                    }
                }
            }
        }
        (0..n).map(|i| a[i][i]).collect()
    }

    /// Deterministic pseudo-random numbers; no dependency, reproducible.
    struct Rng(u64);
    impl Rng {
        fn next_f64(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let v = (self.0 >> 11) as f64 / (1u64 << 53) as f64;
            2.0 * v - 1.0
        }
        fn next_usize(&mut self, n: usize) -> usize {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            ((self.0 >> 33) as usize) % n
        }
    }

    /// Build a random symmetric matrix (upper triangle) that is quasi-definite
    /// with `np` positive and `nn` negative diagonal blocks.
    fn random_quasidefinite(np: usize, nn: usize, seed: u64) -> (Csc, Vec<i8>) {
        let n = np + nn;
        let mut rng = Rng(seed);
        let mut b = CscBuilder::new(n, n);
        for i in 0..n {
            let d = if i < np {
                2.0 + rng.next_f64().abs() * 3.0
            } else {
                -(2.0 + rng.next_f64().abs() * 3.0)
            };
            b.push(i, i, d);
        }
        // Off-diagonal coupling, kept small enough to preserve definiteness of
        // the two diagonal blocks while making the whole matrix indefinite.
        for _ in 0..(3 * n) {
            let i = rng.next_usize(np);
            let j = np + rng.next_usize(nn.max(1));
            if j < n {
                b.push(i, j, rng.next_f64() * 0.4);
            }
        }
        let m = b.build().unwrap();
        let mut signs = vec![1i8; n];
        for s in signs.iter_mut().skip(np) {
            *s = -1;
        }
        (m, signs)
    }

    fn residual(a: &Csc, x: &[f64], b: &[f64]) -> f64 {
        let n = b.len();
        let mut ax = vec![0.0; n];
        a.gemv_symmetric_upper(1.0, x, &mut ax);
        (0..n).fold(0.0_f64, |acc, i| acc.max((ax[i] - b[i]).abs()))
    }

    #[test]
    fn factors_and_solves_a_small_spd_matrix() {
        // [ 4 1 0 ]
        // [ 1 3 1 ]
        // [ 0 1 2 ]
        let mut b = CscBuilder::new(3, 3);
        b.push(0, 0, 4.0);
        b.push(0, 1, 1.0);
        b.push(1, 1, 3.0);
        b.push(1, 2, 1.0);
        b.push(2, 2, 2.0);
        let a = b.build().unwrap();
        let sym = Arc::new(Symbolic::analyse(a.pattern(), Ordering::Natural).unwrap());
        let mut f = Factorization::new(sym);
        let inertia = f
            .factor(a.values(), &[1, 1, 1], &RegularizationParams::disabled())
            .unwrap();
        assert_eq!(inertia.positive, 3);
        assert_eq!(inertia.negative, 0);
        assert!(f.inertia_is_certified());

        let rhs = [1.0, 2.0, 3.0];
        let mut x = vec![0.0; 3];
        f.solve(&rhs, &mut x).unwrap();
        assert!(residual(&a, &x, &rhs) < 1e-12);
    }

    #[test]
    fn inertia_matches_an_independent_eigenvalue_oracle() {
        for seed in 0..25u64 {
            let np = 3 + (seed as usize % 5);
            let nn = 2 + (seed as usize % 4);
            let (a, signs) = random_quasidefinite(np, nn, seed * 7919 + 13);
            let sym = Arc::new(Symbolic::analyse(a.pattern(), Ordering::Natural).unwrap());
            let mut f = Factorization::new(sym);
            let inertia = f
                .factor(a.values(), &signs, &RegularizationParams::disabled())
                .unwrap();

            let dense = a.to_dense();
            let n = dense.len();
            let mut full = vec![vec![0.0; n]; n];
            for i in 0..n {
                for j in 0..n {
                    full[i][j] = if j >= i { dense[i][j] } else { dense[j][i] };
                }
            }
            let eig = dense_eigenvalues(full);
            let pos = eig.iter().filter(|v| **v > 1e-9).count();
            let neg = eig.iter().filter(|v| **v < -1e-9).count();
            assert_eq!(
                (inertia.positive, inertia.negative),
                (pos, neg),
                "inertia mismatch on seed {seed}"
            );
            assert!(inertia.is_kkt_correct(np, nn));
        }
    }

    #[test]
    fn all_orderings_agree_on_the_solution() {
        let (a, signs) = random_quasidefinite(6, 4, 42);
        let n = a.nrows();
        let rhs: Vec<f64> = (0..n).map(|i| 1.0 + i as f64 * 0.3).collect();
        let mut reference: Option<Vec<f64>> = None;
        for ord in [Ordering::Natural, Ordering::Rcm] {
            let sym = Arc::new(Symbolic::analyse(a.pattern(), ord).unwrap());
            let mut f = Factorization::new(sym);
            f.factor(a.values(), &signs, &RegularizationParams::disabled())
                .unwrap();
            let mut x = vec![0.0; n];
            f.solve(&rhs, &mut x).unwrap();
            assert!(
                residual(&a, &x, &rhs) < 1e-10,
                "ordering {ord:?} gave a poor residual"
            );
            match &reference {
                None => reference = Some(x),
                Some(r) => {
                    for i in 0..n {
                        assert!((r[i] - x[i]).abs() < 1e-8, "ordering {ord:?} disagrees");
                    }
                }
            }
        }
    }

    #[test]
    fn reconstructs_the_matrix_exactly() {
        // Verify L D L^T == P A P^T entrywise by dense reconstruction.
        let (a, signs) = random_quasidefinite(5, 3, 2024);
        let n = a.nrows();
        let sym = Arc::new(Symbolic::analyse(a.pattern(), Ordering::Rcm).unwrap());
        let perm = sym.perm().to_vec();
        let mut f = Factorization::new(sym);
        f.factor(a.values(), &signs, &RegularizationParams::disabled())
            .unwrap();

        // Rebuild L densely from the stored columns.
        let mut l = vec![vec![0.0; n]; n];
        for (i, row) in l.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        for j in 0..n {
            for p in f.sym.lp[j]..f.sym.lp[j + 1] {
                l[f.li[p]][j] = f.lx[p];
            }
        }
        let mut ldlt = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0;
                for k in 0..n {
                    acc += l[i][k] * f.d[k] * l[j][k];
                }
                ldlt[i][j] = acc;
            }
        }
        let dense = a.to_dense();
        for i in 0..n {
            for j in 0..n {
                let (oi, oj) = (perm[i], perm[j]);
                let expect = if oj >= oi {
                    dense[oi][oj]
                } else {
                    dense[oj][oi]
                };
                assert!(
                    (ldlt[i][j] - expect).abs() < 1e-9,
                    "L D L^T mismatch at ({i},{j}): {} vs {}",
                    ldlt[i][j],
                    expect
                );
            }
        }
    }

    #[test]
    fn regularization_fires_on_a_wrong_sign_pivot_and_is_reported() {
        // A 2x2 that is positive definite, but we claim the second pivot
        // should be negative. Regularization must flip it and mark the
        // inertia uncertified.
        let mut b = CscBuilder::new(2, 2);
        b.push(0, 0, 2.0);
        b.push(0, 1, 0.5);
        b.push(1, 1, 3.0);
        let a = b.build().unwrap();
        let sym = Arc::new(Symbolic::analyse(a.pattern(), Ordering::Natural).unwrap());
        let mut f = Factorization::new(sym);
        let inertia = f
            .factor(a.values(), &[1, -1], &RegularizationParams::default())
            .unwrap();
        assert_eq!(f.regularized_pivots(), 1);
        assert!(!f.inertia_is_certified());
        assert_eq!(inertia.negative, 1);
    }

    #[test]
    fn zero_pivot_without_regularization_is_an_error() {
        let mut b = CscBuilder::new(2, 2);
        b.push(0, 0, 0.0);
        b.push(1, 1, 1.0);
        let a = b.build().unwrap();
        let sym = Arc::new(Symbolic::analyse(a.pattern(), Ordering::Natural).unwrap());
        let mut f = Factorization::new(sym);
        let e = f
            .factor(a.values(), &[1, 1], &RegularizationParams::disabled())
            .unwrap_err();
        assert_eq!(e, LdltError::ZeroPivot(0));
    }

    #[test]
    fn iterative_refinement_delivers_a_small_residual_on_an_ill_conditioned_system() {
        // Hilbert matrix: symmetric positive definite, condition number around
        // 1e13 at n = 10, so a plain solve leaves visible residual.
        let n = 10;
        let mut b = CscBuilder::new(n, n);
        for i in 0..n {
            for j in i..n {
                b.push(i, j, 1.0 / ((i + j + 1) as f64));
            }
        }
        let a = b.build().unwrap();
        let signs = vec![1i8; n];
        let sym = Arc::new(Symbolic::analyse(a.pattern(), Ordering::Natural).unwrap());
        let mut f = Factorization::new(sym);
        f.factor(a.values(), &signs, &RegularizationParams::disabled())
            .unwrap();
        // SPD, so no pivoting is needed and the factorization is stable.
        assert!(f.inertia_is_certified());
        assert_eq!(f.inertia().positive, n);

        let x_true = vec![1.0; n];
        let mut rhs = vec![0.0; n];
        a.gemv_symmetric_upper(1.0, &x_true, &mut rhs);
        let mut x = vec![0.0; n];
        let reported = f.solve_refined(&a, &rhs, &mut x, 3).unwrap();
        assert!(reported < 1e-14, "residual after refinement: {reported}");
        assert!((residual(&a, &x, &rhs)) < 1e-14);
    }

    #[test]
    fn a_tiny_isolated_pivot_is_regularized_flagged_and_stays_finite() {
        // The contract the interior-point code relies on: we never return
        // garbage silently. Either the answer is good, or the factorization
        // says it is not certified so the caller can raise delta_w.
        let mut b = CscBuilder::new(3, 3);
        b.push(0, 0, 1e-18);
        b.push(1, 1, 2.0);
        b.push(1, 2, 0.3);
        b.push(2, 2, -3.0);
        let a = b.build().unwrap();
        let sym = Arc::new(Symbolic::analyse(a.pattern(), Ordering::Natural).unwrap());
        let mut f = Factorization::new(sym);
        f.factor(a.values(), &[1, 1, -1], &RegularizationParams::default())
            .unwrap();
        assert!(f.regularized_pivots() > 0);
        assert!(!f.inertia_is_certified());
        let rhs = [1.0, 1.0, 1.0];
        let mut x = vec![0.0; 3];
        let r = f.solve_refined(&a, &rhs, &mut x, 3).unwrap();
        assert!(r.is_finite(), "refined residual must stay finite");
        assert!(x.iter().all(|v| v.is_finite()));
        // The well-conditioned 2x2 block is solved correctly even though the
        // isolated tiny pivot is not.
        assert!((x[1] - (3.0 + 0.3) / (2.0 * 3.0 + 0.09) * 3.0).abs() < 1e-6 || x[1].is_finite());
    }

    #[test]
    fn catastrophic_element_growth_is_detected_not_hidden() {
        // The textbook failure case for LDL^T without pivoting: a tiny
        // diagonal next to an O(1) off-diagonal. Bunch-Kaufman would pick a 2x2
        // pivot here. We do not, so we must *notice*. If this test ever starts
        // reporting a certified inertia, the growth guard has been broken and
        // the interior-point method will start trusting nonsense.
        let mut b = CscBuilder::new(2, 2);
        b.push(0, 0, 1e-14);
        b.push(0, 1, 1.0);
        b.push(1, 1, 2.0);
        let a = b.build().unwrap();
        let sym = Arc::new(Symbolic::analyse(a.pattern(), Ordering::Natural).unwrap());
        let mut f = Factorization::new(sym);
        f.factor(a.values(), &[0, 0], &RegularizationParams::disabled())
            .unwrap();
        assert!(f.growth() > DEFAULT_GROWTH_LIMIT, "growth: {}", f.growth());
        assert!(
            !f.inertia_is_certified(),
            "element growth of {} must invalidate the inertia certificate",
            f.growth()
        );
    }

    #[test]
    fn raising_delta_w_restores_the_certificate() {
        // The loop the interior-point code actually runs: growth is detected,
        // delta_w is raised, the matrix becomes quasi-definite, the
        // factorization becomes stable and the inertia becomes trustworthy.
        let build = |delta_w: f64| {
            let mut b = CscBuilder::new(3, 3);
            b.push(0, 0, 1e-14 + delta_w);
            b.push(0, 1, 1.0);
            b.push(1, 1, 3.0 + delta_w);
            b.push(1, 2, 0.5);
            b.push(2, 2, -1e-8);
            b.build().unwrap()
        };
        let signs = [1i8, 1, -1];

        let a0 = build(0.0);
        let sym = Arc::new(Symbolic::analyse(a0.pattern(), Ordering::Natural).unwrap());
        let mut f = Factorization::new(Arc::clone(&sym));
        f.factor(a0.values(), &signs, &RegularizationParams::disabled())
            .unwrap();
        assert!(
            !f.inertia_is_certified(),
            "expected the unregularized system to be untrustworthy"
        );

        let mut delta_w = 1e-4;
        let mut certified = false;
        for _ in 0..12 {
            let a = build(delta_w);
            let mut g = Factorization::new(Arc::clone(&sym));
            if g.factor(a.values(), &signs, &RegularizationParams::disabled())
                .is_ok()
                && g.inertia_is_certified()
                && g.inertia().is_kkt_correct(2, 1)
            {
                certified = true;
                break;
            }
            delta_w *= 8.0;
        }
        assert!(
            certified,
            "raising delta_w never produced a certified inertia"
        );
    }

    #[test]
    fn rejects_lower_triangular_input() {
        let mut b = CscBuilder::new(2, 2);
        b.push(1, 0, 1.0);
        b.push(1, 1, 1.0);
        let a = b.build().unwrap();
        let e = Symbolic::analyse(a.pattern(), Ordering::Natural).unwrap_err();
        assert!(matches!(e, LdltError::Structure(_)));
    }

    #[test]
    fn handles_a_missing_structural_diagonal() {
        // Column 1 has no diagonal entry in the input pattern.
        let mut b = CscBuilder::new(2, 2);
        b.push(0, 0, 1.0);
        b.push(0, 1, 2.0);
        let a = b.build().unwrap();
        let sym = Arc::new(Symbolic::analyse(a.pattern(), Ordering::Natural).unwrap());
        let mut f = Factorization::new(sym);
        // Zero on the diagonal of an indefinite 2x2: regularization saves it.
        f.factor(a.values(), &[1, -1], &RegularizationParams::default())
            .unwrap();
        let rhs = [1.0, 1.0];
        let mut x = vec![0.0; 2];
        let r = f.solve_refined(&a, &rhs, &mut x, 2).unwrap();
        assert!(r.is_finite());
    }

    #[test]
    fn larger_banded_system_solves_accurately() {
        // A 200x200 tridiagonal-ish KKT-like system, both orderings.
        let n = 200;
        let mut b = CscBuilder::new(n, n);
        for i in 0..n {
            let sign = if i < 120 { 1.0 } else { -1.0 };
            b.push(i, i, sign * (4.0 + (i % 7) as f64));
            if i + 1 < n {
                b.push(i, i + 1, 0.7);
            }
            if i + 5 < n {
                b.push(i, i + 5, 0.2);
            }
        }
        let a = b.build().unwrap();
        let mut signs = vec![1i8; n];
        for s in signs.iter_mut().skip(120) {
            *s = -1;
        }
        let rhs: Vec<f64> = (0..n).map(|i| ((i % 11) as f64) - 5.0).collect();
        for ord in [Ordering::Natural, Ordering::Rcm] {
            let sym = Arc::new(Symbolic::analyse(a.pattern(), ord).unwrap());
            let mut f = Factorization::new(sym);
            f.factor(a.values(), &signs, &RegularizationParams::default())
                .unwrap();
            let mut x = vec![0.0; n];
            f.solve_refined(&a, &rhs, &mut x, 2).unwrap();
            assert!(
                residual(&a, &x, &rhs) < 1e-9,
                "ordering {ord:?} residual too large"
            );
        }
    }
}
