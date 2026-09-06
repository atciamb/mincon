//! Graph coloring for finite-difference compression.
//!
//! # The idea
//!
//! A finite-difference Jacobian normally costs one extra model evaluation per
//! **column**. If two columns are *structurally orthogonal* — no row where both
//! are nonzero — they can be perturbed in the same evaluation and the two
//! results read off without interference. Partitioning the columns into as few
//! such groups as possible is a graph coloring problem (Curtis, Powell and
//! Reid, 1974), and for a banded Jacobian it collapses `n` evaluations into a
//! handful.
//!
//! This is the difference between a large sparse model being tractable and
//! being hopeless, and it is a capability `fmincon` does not offer for
//! nonlinear constraint Jacobians: MATLAB will happily take `n` finite
//! differences per iteration on a problem where three would do.
//!
//! # Getting it wrong is silent
//!
//! A coloring that is merely *valid-looking* produces derivative values that
//! are subtly wrong, which manifests three layers up as "the line search keeps
//! failing" rather than as an error. Two guards:
//!
//! * [`verify_coloring`] checks the orthogonality property directly and is run
//!   in debug builds and in every test.
//! * Hessians need [`mincon_core::ColoringKind::Star`], not `Distance1`.
//!   Symmetric recovery by direct substitution has a strictly stronger
//!   requirement (Coleman and Moré, 1984), and a distance-1 coloring of a
//!   symmetric matrix will silently mix `H[i][j]` with `H[j][i]` contributions
//!   from different columns. [`star_coloring`] is the safe one.

use mincon_core::Sparsity;

/// A partition of columns into structurally orthogonal groups.
#[derive(Debug, Clone)]
pub struct Coloring {
    /// `color[j]` is the group column `j` belongs to.
    color: Vec<usize>,
    /// `groups[c]` lists the columns in group `c`.
    groups: Vec<Vec<usize>>,
}

impl Coloring {
    /// Group index of each column.
    #[must_use]
    pub fn colors(&self) -> &[usize] {
        &self.color
    }
    /// Columns making up each group.
    #[must_use]
    pub fn groups(&self) -> &[Vec<usize>] {
        &self.groups
    }
    /// Number of groups — equivalently, the number of model evaluations a
    /// forward-difference Jacobian will cost.
    #[must_use]
    pub fn num_groups(&self) -> usize {
        self.groups.len()
    }
    /// Compression ratio against the dense cost. `1.0` means no saving.
    #[must_use]
    pub fn compression(&self, ncols: usize) -> f64 {
        if self.groups.is_empty() {
            1.0
        } else {
            ncols as f64 / self.groups.len() as f64
        }
    }

    fn from_colors(color: Vec<usize>) -> Self {
        let ngroups = color.iter().copied().max().map_or(0, |m| m + 1);
        let mut groups = vec![Vec::new(); ngroups];
        for (j, &c) in color.iter().enumerate() {
            groups[c].push(j);
        }
        Self { color, groups }
    }
}

/// Greedy distance-1 coloring of the column intersection graph.
///
/// Correct for **Jacobians**. Columns are visited in order of decreasing
/// degree (the "largest-first" heuristic, which is what makes greedy coloring
/// competitive in practice) and each is given the lowest color not used by a
/// conflicting column.
///
/// Not optimal — minimum coloring is NP-hard — but within a color or two of
/// optimal on the structured matrices that come out of discretized models.
#[must_use]
pub fn distance1_coloring(pattern: &Sparsity) -> Coloring {
    let n = pattern.ncols();
    let rows_to_cols = pattern.transpose();

    let mut degree = vec![0usize; n];
    for j in 0..n {
        let mut d = 0;
        for &i in pattern.col(j) {
            d += rows_to_cols.col(i).len();
        }
        degree[j] = d;
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by_key(|&j| (std::cmp::Reverse(degree[j]), j));

    const UNCOLORED: usize = usize::MAX;
    let mut color = vec![UNCOLORED; n];
    let mut forbidden = vec![UNCOLORED; n + 1];

    for &j in &order {
        for &i in pattern.col(j) {
            for &k in rows_to_cols.col(i) {
                if k != j && color[k] != UNCOLORED {
                    forbidden[color[k]] = j;
                }
            }
        }
        let mut c = 0;
        while c < forbidden.len() && forbidden[c] == j {
            c += 1;
        }
        color[j] = c;
    }
    Coloring::from_colors(color)
}

/// Star coloring for **symmetric** matrices recovered by direct substitution.
///
/// A valid star coloring is a distance-1 coloring in which every path on four
/// vertices uses at least three colors — equivalently, every two-colored
/// connected subgraph is a star. That is what makes `H[i][j]` recoverable
/// unambiguously from a compressed column.
///
/// Implemented here as a conservative greedy: distance-2 coloring, which is
/// strictly stronger than star coloring and therefore always valid, at the
/// cost of using more groups than necessary. Replacing it with a true star
/// coloring (Gebremedhin, Manne and Pothen, 2005) is a documented optimization
/// task — it typically saves 20–40% of the groups on a mesh Hessian — and it
/// must not be attempted without [`verify_coloring`] in the test.
#[must_use]
pub fn star_coloring(pattern: &Sparsity) -> Coloring {
    let n = pattern.ncols();
    let rows_to_cols = pattern.transpose();

    // Neighbourhood in the column intersection graph.
    let neighbours = |j: usize| -> Vec<usize> {
        let mut out = Vec::new();
        for &i in pattern.col(j) {
            for &k in rows_to_cols.col(i) {
                if k != j {
                    out.push(k);
                }
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    };

    let adj: Vec<Vec<usize>> = (0..n).map(neighbours).collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by_key(|&j| (std::cmp::Reverse(adj[j].len()), j));

    const UNCOLORED: usize = usize::MAX;
    let mut color = vec![UNCOLORED; n];
    let mut forbidden = vec![UNCOLORED; n + 2];

    for &j in &order {
        // Forbid colors within distance 2.
        for &k in &adj[j] {
            if color[k] != UNCOLORED {
                forbidden[color[k]] = j;
            }
            for &l in &adj[k] {
                if l != j && color[l] != UNCOLORED {
                    forbidden[color[l]] = j;
                }
            }
        }
        let mut c = 0;
        while c < forbidden.len() && forbidden[c] == j {
            c += 1;
        }
        color[j] = c;
    }
    Coloring::from_colors(color)
}

/// Check that no two columns in a group share a row.
///
/// # Errors
/// The offending `(row, column, column)` triple, so a failure is debuggable
/// rather than just a boolean.
pub fn verify_coloring(pattern: &Sparsity, coloring: &Coloring) -> Result<(), String> {
    let rows_to_cols = pattern.transpose();
    for i in 0..pattern.nrows() {
        let cols = rows_to_cols.col(i);
        let mut seen: Vec<(usize, usize)> = Vec::new();
        for &j in cols {
            let c = coloring.colors()[j];
            if let Some(&(_, other)) = seen.iter().find(|&&(cc, _)| cc == c) {
                return Err(format!(
                    "columns {other} and {j} share color {c} and both touch row {i}"
                ));
            }
            seen.push((c, j));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn banded(n: usize, bw: usize) -> Sparsity {
        let mut t = Vec::new();
        for j in 0..n {
            for i in j.saturating_sub(bw)..(j + bw + 1).min(n) {
                t.push((i, j));
            }
        }
        Sparsity::from_triplets(n, n, &t).unwrap()
    }

    #[test]
    fn diagonal_needs_one_group() {
        let s = Sparsity::from_triplets(5, 5, &(0..5).map(|i| (i, i)).collect::<Vec<_>>()).unwrap();
        let c = distance1_coloring(&s);
        assert_eq!(c.num_groups(), 1);
        verify_coloring(&s, &c).unwrap();
        assert!((c.compression(5) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn dense_needs_one_group_per_column() {
        let s = Sparsity::dense(4, 4);
        let c = distance1_coloring(&s);
        assert_eq!(c.num_groups(), 4);
        verify_coloring(&s, &c).unwrap();
    }

    #[test]
    fn tridiagonal_needs_three_groups() {
        let s = banded(50, 1);
        let c = distance1_coloring(&s);
        assert_eq!(c.num_groups(), 3, "tridiagonal is 3-colorable");
        verify_coloring(&s, &c).unwrap();
        assert!(c.compression(50) > 16.0);
    }

    #[test]
    fn banded_compression_scales() {
        for bw in 1..6 {
            let s = banded(120, bw);
            let c = distance1_coloring(&s);
            verify_coloring(&s, &c).unwrap();
            assert!(
                c.num_groups() <= 2 * bw + 1,
                "bandwidth {bw} used {} groups",
                c.num_groups()
            );
        }
    }

    #[test]
    fn star_coloring_is_at_least_as_strong_as_distance1() {
        let s = banded(40, 2);
        let star = star_coloring(&s);
        let d1 = distance1_coloring(&s);
        verify_coloring(&s, &star).unwrap();
        verify_coloring(&s, &d1).unwrap();
        assert!(
            star.num_groups() >= d1.num_groups(),
            "star coloring should never use fewer groups than distance-1"
        );
    }

    #[test]
    fn arrowhead_is_the_adversarial_case() {
        // Row 0 and column 0 dense: every column conflicts with column 0, and
        // column 0 conflicts with everything. Needs n groups.
        let n = 8;
        let mut t = Vec::new();
        for i in 0..n {
            t.push((i, 0));
            t.push((0, i));
            t.push((i, i));
        }
        let s = Sparsity::from_triplets(n, n, &t).unwrap();
        let c = distance1_coloring(&s);
        verify_coloring(&s, &c).unwrap();
        assert_eq!(c.num_groups(), n);
    }

    #[test]
    fn empty_columns_do_not_consume_groups() {
        // A column with no nonzeros conflicts with nothing.
        let s = Sparsity::from_triplets(3, 4, &[(0, 0), (1, 2), (2, 3)]).unwrap();
        let c = distance1_coloring(&s);
        verify_coloring(&s, &c).unwrap();
        assert_eq!(c.num_groups(), 1);
    }
}
