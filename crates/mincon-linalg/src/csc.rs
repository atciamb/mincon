//! A compressed-sparse-column matrix with values, sharing
//! [`mincon_core::Sparsity`] for structure.

use mincon_core::Sparsity;

/// A sparse matrix in CSC form.
#[derive(Clone, Debug)]
pub struct Csc {
    pattern: Sparsity,
    values: Vec<f64>,
}

impl Csc {
    /// Wrap a pattern and its values.
    ///
    /// # Errors
    /// If `values.len() != pattern.nnz()`.
    pub fn new(pattern: Sparsity, values: Vec<f64>) -> Result<Self, String> {
        if values.len() != pattern.nnz() {
            return Err(format!(
                "values has length {} but pattern has {} nonzeros",
                values.len(),
                pattern.nnz()
            ));
        }
        Ok(Self { pattern, values })
    }

    /// A matrix of zeros with the given structure.
    #[must_use]
    pub fn zeros(pattern: Sparsity) -> Self {
        let values = vec![0.0; pattern.nnz()];
        Self { pattern, values }
    }

    /// The structure.
    #[must_use]
    pub fn pattern(&self) -> &Sparsity {
        &self.pattern
    }
    /// The values, in pattern order.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }
    /// The values, mutably.
    pub fn values_mut(&mut self) -> &mut [f64] {
        &mut self.values
    }
    /// Number of rows.
    #[must_use]
    pub fn nrows(&self) -> usize {
        self.pattern.nrows()
    }
    /// Number of columns.
    #[must_use]
    pub fn ncols(&self) -> usize {
        self.pattern.ncols()
    }
    /// Number of stored entries.
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// `y <- y + alpha * self * x`.
    ///
    /// # Panics
    /// If `x.len() != ncols` or `y.len() != nrows`.
    pub fn gemv(&self, alpha: f64, x: &[f64], y: &mut [f64]) {
        assert_eq!(x.len(), self.ncols(), "gemv: x has wrong length");
        assert_eq!(y.len(), self.nrows(), "gemv: y has wrong length");
        let cp = self.pattern.col_ptr();
        let ri = self.pattern.row_idx();
        for j in 0..self.ncols() {
            let xj = alpha * x[j];
            if xj == 0.0 {
                continue;
            }
            for p in cp[j]..cp[j + 1] {
                y[ri[p]] += self.values[p] * xj;
            }
        }
    }

    /// `y <- y + alpha * self^T * x`.
    ///
    /// # Panics
    /// If `x.len() != nrows` or `y.len() != ncols`.
    pub fn gemv_transpose(&self, alpha: f64, x: &[f64], y: &mut [f64]) {
        assert_eq!(x.len(), self.nrows(), "gemv_transpose: x has wrong length");
        assert_eq!(y.len(), self.ncols(), "gemv_transpose: y has wrong length");
        let cp = self.pattern.col_ptr();
        let ri = self.pattern.row_idx();
        for j in 0..self.ncols() {
            let mut acc = 0.0;
            for p in cp[j]..cp[j + 1] {
                acc += self.values[p] * x[ri[p]];
            }
            y[j] += alpha * acc;
        }
    }

    /// `y <- y + alpha * S * x` where `S` is the symmetric matrix whose upper
    /// triangle (including the diagonal) is stored in `self`.
    ///
    /// # Panics
    /// If the matrix is not square or the vectors have the wrong length.
    pub fn gemv_symmetric_upper(&self, alpha: f64, x: &[f64], y: &mut [f64]) {
        assert_eq!(self.nrows(), self.ncols(), "matrix must be square");
        assert_eq!(x.len(), self.ncols());
        assert_eq!(y.len(), self.nrows());
        let cp = self.pattern.col_ptr();
        let ri = self.pattern.row_idx();
        for j in 0..self.ncols() {
            for p in cp[j]..cp[j + 1] {
                let i = ri[p];
                let v = alpha * self.values[p];
                y[i] += v * x[j];
                if i != j {
                    y[j] += v * x[i];
                }
            }
        }
    }

    /// Dense conversion. Test and debugging aid; never on a hot path.
    #[must_use]
    pub fn to_dense(&self) -> Vec<Vec<f64>> {
        let mut d = vec![vec![0.0; self.ncols()]; self.nrows()];
        let cp = self.pattern.col_ptr();
        let ri = self.pattern.row_idx();
        for j in 0..self.ncols() {
            for p in cp[j]..cp[j + 1] {
                d[ri[p]][j] = self.values[p];
            }
        }
        d
    }
}

/// Accumulates triplets into a [`Csc`], summing duplicates.
///
/// Used by the KKT assembler, where the same `(i, j)` can be touched by the
/// Hessian block and by a regularization term.
#[derive(Debug, Clone)]
pub struct CscBuilder {
    nrows: usize,
    ncols: usize,
    triplets: Vec<(usize, usize, f64)>,
}

impl CscBuilder {
    /// A builder for an `nrows x ncols` matrix.
    #[must_use]
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            nrows,
            ncols,
            triplets: Vec::new(),
        }
    }

    /// Reserve room for `n` more entries.
    pub fn reserve(&mut self, n: usize) {
        self.triplets.reserve(n);
    }

    /// Add `v` at `(i, j)`. Duplicates are summed.
    ///
    /// # Panics
    /// If the index is out of range.
    pub fn push(&mut self, i: usize, j: usize, v: f64) {
        assert!(i < self.nrows && j < self.ncols, "index out of range");
        self.triplets.push((i, j, v));
    }

    /// Number of triplets pushed so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.triplets.len()
    }
    /// Whether nothing has been pushed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.triplets.is_empty()
    }

    /// Build the matrix, summing duplicates.
    ///
    /// # Errors
    /// If the resulting pattern is invalid, which would be an internal bug.
    pub fn build(self) -> Result<Csc, String> {
        let mut cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); self.ncols];
        for (i, j, v) in self.triplets {
            cols[j].push((i, v));
        }
        let mut col_ptr = Vec::with_capacity(self.ncols + 1);
        let mut row_idx = Vec::new();
        let mut values = Vec::new();
        col_ptr.push(0);
        for c in &mut cols {
            c.sort_unstable_by_key(|&(i, _)| i);
            let mut k = 0;
            while k < c.len() {
                let i = c[k].0;
                let mut acc = 0.0;
                while k < c.len() && c[k].0 == i {
                    acc += c[k].1;
                    k += 1;
                }
                row_idx.push(i);
                values.push(acc);
            }
            col_ptr.push(row_idx.len());
        }
        let pattern = Sparsity::new(self.nrows, self.ncols, col_ptr, row_idx)?;
        Csc::new(pattern, values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small() -> Csc {
        // [ 2  1  0 ]
        // [ 1  3  1 ]
        // [ 0  1  4 ]   stored in full
        let mut b = CscBuilder::new(3, 3);
        b.push(0, 0, 2.0);
        b.push(1, 0, 1.0);
        b.push(0, 1, 1.0);
        b.push(1, 1, 3.0);
        b.push(2, 1, 1.0);
        b.push(1, 2, 1.0);
        b.push(2, 2, 4.0);
        b.build().unwrap()
    }

    #[test]
    fn builder_sums_duplicates() {
        let mut b = CscBuilder::new(2, 2);
        b.push(0, 0, 1.0);
        b.push(0, 0, 2.5);
        b.push(1, 1, -1.0);
        let m = b.build().unwrap();
        assert_eq!(m.nnz(), 2);
        assert_eq!(m.to_dense()[0][0], 3.5);
        assert_eq!(m.to_dense()[1][1], -1.0);
    }

    #[test]
    fn gemv_matches_dense() {
        let m = small();
        let x = [1.0, 2.0, 3.0];
        let mut y = vec![0.0; 3];
        m.gemv(1.0, &x, &mut y);
        assert_eq!(y, vec![4.0, 10.0, 14.0]);
    }

    #[test]
    fn gemv_transpose_matches_dense_for_symmetric() {
        let m = small();
        let x = [1.0, 2.0, 3.0];
        let mut y = vec![0.0; 3];
        m.gemv_transpose(1.0, &x, &mut y);
        assert_eq!(y, vec![4.0, 10.0, 14.0]);
    }

    #[test]
    fn symmetric_upper_gemv_matches_full() {
        // Same matrix, upper triangle only.
        let mut b = CscBuilder::new(3, 3);
        b.push(0, 0, 2.0);
        b.push(0, 1, 1.0);
        b.push(1, 1, 3.0);
        b.push(1, 2, 1.0);
        b.push(2, 2, 4.0);
        let upper = b.build().unwrap();
        let x = [1.0, 2.0, 3.0];
        let mut y = vec![0.0; 3];
        upper.gemv_symmetric_upper(1.0, &x, &mut y);
        assert_eq!(y, vec![4.0, 10.0, 14.0]);
    }

    #[test]
    fn nonsquare_gemv() {
        let mut b = CscBuilder::new(2, 3);
        b.push(0, 0, 1.0);
        b.push(1, 2, 2.0);
        let m = b.build().unwrap();
        let mut y = vec![0.0; 2];
        m.gemv(1.0, &[3.0, 5.0, 7.0], &mut y);
        assert_eq!(y, vec![3.0, 14.0]);
        let mut z = vec![0.0; 3];
        m.gemv_transpose(1.0, &[1.0, 1.0], &mut z);
        assert_eq!(z, vec![1.0, 0.0, 2.0]);
    }
}
