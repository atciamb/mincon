//! Sparsity patterns in compressed sparse column (CSC) form.
//!
//! One pattern type is shared by the derivative layer, the linear algebra layer
//! and the algorithms, so a Jacobian pattern discovered once is reused for
//! coloring, for the KKT assembly and for the symbolic factorization.

use std::fmt;

/// A CSC sparsity pattern for an `nrows x ncols` matrix.
///
/// Invariants (checked by [`Sparsity::new`]):
/// * `col_ptr.len() == ncols + 1`, non-decreasing, `col_ptr[0] == 0`
/// * `col_ptr[ncols] == row_idx.len()`
/// * within each column, `row_idx` is strictly increasing and `< nrows`
///
/// Sorted, duplicate-free columns are relied upon throughout; do not relax it.
#[derive(Clone, PartialEq, Eq)]
pub struct Sparsity {
    nrows: usize,
    ncols: usize,
    col_ptr: Vec<usize>,
    row_idx: Vec<usize>,
}

impl Sparsity {
    /// Build a pattern, validating the invariants.
    ///
    /// # Errors
    /// Returns a description of the first invariant violated.
    pub fn new(
        nrows: usize,
        ncols: usize,
        col_ptr: Vec<usize>,
        row_idx: Vec<usize>,
    ) -> Result<Self, String> {
        if col_ptr.len() != ncols + 1 {
            return Err(format!(
                "col_ptr has length {} but ncols + 1 = {}",
                col_ptr.len(),
                ncols + 1
            ));
        }
        if col_ptr[0] != 0 {
            return Err("col_ptr[0] must be 0".into());
        }
        if *col_ptr.last().unwrap() != row_idx.len() {
            return Err(format!(
                "col_ptr[ncols] = {} but row_idx has length {}",
                col_ptr[ncols],
                row_idx.len()
            ));
        }
        for j in 0..ncols {
            if col_ptr[j] > col_ptr[j + 1] {
                return Err(format!("col_ptr not non-decreasing at column {j}"));
            }
            let seg = &row_idx[col_ptr[j]..col_ptr[j + 1]];
            for (k, &i) in seg.iter().enumerate() {
                if i >= nrows {
                    return Err(format!("row index {i} out of range in column {j}"));
                }
                if k > 0 && seg[k - 1] >= i {
                    return Err(format!(
                        "row indices in column {j} must be strictly increasing"
                    ));
                }
            }
        }
        Ok(Self {
            nrows,
            ncols,
            col_ptr,
            row_idx,
        })
    }

    /// Build a pattern from unsorted, possibly duplicated `(row, col)` pairs.
    pub fn from_triplets(
        nrows: usize,
        ncols: usize,
        triplets: &[(usize, usize)],
    ) -> Result<Self, String> {
        let mut cols: Vec<Vec<usize>> = vec![Vec::new(); ncols];
        for &(i, j) in triplets {
            if i >= nrows || j >= ncols {
                return Err(format!("triplet ({i}, {j}) out of range {nrows}x{ncols}"));
            }
            cols[j].push(i);
        }
        let mut col_ptr = Vec::with_capacity(ncols + 1);
        let mut row_idx = Vec::with_capacity(triplets.len());
        col_ptr.push(0);
        for c in &mut cols {
            c.sort_unstable();
            c.dedup();
            row_idx.extend_from_slice(c);
            col_ptr.push(row_idx.len());
        }
        Ok(Self {
            nrows,
            ncols,
            col_ptr,
            row_idx,
        })
    }

    /// A fully dense pattern. Used as the fallback when a model declines to
    /// declare structure; correct but quadratic, hence the warning in
    /// `docs/05_SPEC_DERIVATIVES.md`.
    #[must_use]
    pub fn dense(nrows: usize, ncols: usize) -> Self {
        let mut col_ptr = Vec::with_capacity(ncols + 1);
        let mut row_idx = Vec::with_capacity(nrows * ncols);
        col_ptr.push(0);
        for _ in 0..ncols {
            row_idx.extend(0..nrows);
            col_ptr.push(row_idx.len());
        }
        Self {
            nrows,
            ncols,
            col_ptr,
            row_idx,
        }
    }

    /// Number of rows.
    #[must_use]
    pub fn nrows(&self) -> usize {
        self.nrows
    }
    /// Number of columns.
    #[must_use]
    pub fn ncols(&self) -> usize {
        self.ncols
    }
    /// Number of stored entries.
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.row_idx.len()
    }
    /// Column pointers, length `ncols + 1`.
    #[must_use]
    pub fn col_ptr(&self) -> &[usize] {
        &self.col_ptr
    }
    /// Row indices, length `nnz`.
    #[must_use]
    pub fn row_idx(&self) -> &[usize] {
        &self.row_idx
    }
    /// Row indices of column `j`, sorted ascending.
    #[must_use]
    pub fn col(&self, j: usize) -> &[usize] {
        &self.row_idx[self.col_ptr[j]..self.col_ptr[j + 1]]
    }

    /// Fraction of entries that are stored. Used to decide dense-vs-sparse
    /// dispatch; see `docs/04_SPEC_LINEAR_ALGEBRA.md`.
    #[must_use]
    pub fn density(&self) -> f64 {
        let total = (self.nrows as f64) * (self.ncols as f64);
        if total == 0.0 {
            0.0
        } else {
            self.nnz() as f64 / total
        }
    }

    /// Transpose the pattern. `O(nnz + n)`, output columns sorted.
    #[must_use]
    pub fn transpose(&self) -> Self {
        let mut counts = vec![0usize; self.nrows + 1];
        for &i in &self.row_idx {
            counts[i + 1] += 1;
        }
        for i in 0..self.nrows {
            counts[i + 1] += counts[i];
        }
        let col_ptr = counts.clone();
        let mut row_idx = vec![0usize; self.nnz()];
        let mut next = counts;
        for j in 0..self.ncols {
            for &i in self.col(j) {
                row_idx[next[i]] = j;
                next[i] += 1;
            }
        }
        Self {
            nrows: self.ncols,
            ncols: self.nrows,
            col_ptr,
            row_idx,
        }
    }
}

impl fmt::Debug for Sparsity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Sparsity({}x{}, nnz={}, density={:.3}%)",
            self.nrows,
            self.ncols,
            self.nnz(),
            100.0 * self.density()
        )
    }
}

/// Which structural-orthogonality relation a coloring must satisfy.
///
/// See `docs/05_SPEC_DERIVATIVES.md`. Getting this wrong silently produces
/// wrong derivatives, which then look like a convergence bug three layers up —
/// so every coloring is verified against a dense reference in the test suite.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColoringKind {
    /// Distance-1 coloring of the column intersection graph. Correct for
    /// **Jacobians** (Curtis–Powell–Reid): two columns share a color iff they
    /// have no row in common.
    Distance1,
    /// Star coloring, for **symmetric** matrices recovered by direct
    /// substitution (Hessians, Coleman–Moré). Strictly stronger than
    /// distance-1; a distance-1 coloring of a Hessian is *not* sufficient.
    Star,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_pattern_is_valid() {
        let s = Sparsity::dense(3, 4);
        assert_eq!(s.nnz(), 12);
        assert_eq!(s.col(2), &[0, 1, 2]);
        assert!((s.density() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn triplets_sort_and_dedup() {
        let s = Sparsity::from_triplets(3, 2, &[(2, 0), (0, 0), (2, 0), (1, 1)]).unwrap();
        assert_eq!(s.col(0), &[0, 2]);
        assert_eq!(s.col(1), &[1]);
        assert_eq!(s.nnz(), 3);
    }

    #[test]
    fn transpose_roundtrips() {
        let s = Sparsity::from_triplets(4, 3, &[(0, 0), (3, 0), (1, 1), (2, 2), (0, 2)]).unwrap();
        let t = s.transpose();
        assert_eq!(t.nrows(), 3);
        assert_eq!(t.ncols(), 4);
        assert_eq!(t.transpose(), s);
    }

    #[test]
    fn rejects_unsorted_columns() {
        let e = Sparsity::new(3, 1, vec![0, 2], vec![2, 0]).unwrap_err();
        assert!(e.contains("strictly increasing"));
    }
}
