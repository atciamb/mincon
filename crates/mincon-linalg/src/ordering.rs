//! Fill-reducing orderings for the symmetric factorization.
//!
//! # Known gap — read before benchmarking
//!
//! The production ordering must be **AMD** (approximate minimum degree,
//! Amestoy–Davis–Duff). It is what MA57, `CHOLMOD`, `QDLDL` and every serious
//! sparse code uses, and on large KKT systems it is worth an order of
//! magnitude in fill against anything here. It is deliberately *not*
//! implemented in this scaffolding, because a hand-rolled minimum-degree code
//! is a multi-week project whose failure mode is "quietly 10x slower", which
//! is exactly the kind of thing that should be measured rather than guessed.
//!
//! The task is to wire in the `amd` crate (v0.2.2, a port of SuiteSparse AMD,
//! BSD-3-Clause, so licence-compatible) behind a new `Ordering::Amd` variant
//! (there is deliberately none today: a variant that fell back to RCM was an
//! option that did not do what its name said), and to gate the switch on the
//! fill and factor-time comparison in `docs/08_BENCHMARK_PROTOCOL.md`. Until
//! then [`Ordering::Rcm`] is the
//! default: it is genuinely good on the banded systems that come out of
//! discretized optimal control, which is a large share of real NLPs, and it is
//! 60 lines that are obviously correct.
//!
//! Ordering affects only speed, never the answer. Every test in this crate
//! runs under all available orderings.

/// Which fill-reducing permutation to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Ordering {
    /// Identity. Use for reproducing hand-worked examples and for debugging.
    Natural,
    /// Reverse Cuthill–McKee. Bandwidth-reducing; strong on banded systems,
    /// mediocre on unstructured ones.
    #[default]
    Rcm,
}

/// Compute a permutation for the symmetric matrix whose **upper triangle**
/// (including the diagonal) has the given CSC structure.
///
/// Returns `perm` where `perm[k]` is the original index of the `k`-th pivot.
#[must_use]
pub fn compute_ordering(
    n: usize,
    col_ptr: &[usize],
    row_idx: &[usize],
    kind: Ordering,
) -> Vec<usize> {
    match kind {
        Ordering::Natural => (0..n).collect(),
        Ordering::Rcm => rcm(n, col_ptr, row_idx),
    }
}

/// Build the full (both triangles, no diagonal) adjacency of a symmetric
/// matrix given its upper triangle.
fn adjacency(n: usize, col_ptr: &[usize], row_idx: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let mut deg = vec![0usize; n + 1];
    for j in 0..n {
        for p in col_ptr[j]..col_ptr[j + 1] {
            let i = row_idx[p];
            if i != j {
                deg[i + 1] += 1;
                deg[j + 1] += 1;
            }
        }
    }
    for i in 0..n {
        deg[i + 1] += deg[i];
    }
    let adj_ptr = deg.clone();
    let mut adj = vec![0usize; adj_ptr[n]];
    let mut next = deg;
    for j in 0..n {
        for p in col_ptr[j]..col_ptr[j + 1] {
            let i = row_idx[p];
            if i != j {
                adj[next[i]] = j;
                next[i] += 1;
                adj[next[j]] = i;
                next[j] += 1;
            }
        }
    }
    (adj_ptr, adj)
}

/// Reverse Cuthill–McKee over all connected components, starting each
/// component from a low-degree vertex.
fn rcm(n: usize, col_ptr: &[usize], row_idx: &[usize]) -> Vec<usize> {
    let (adj_ptr, adj) = adjacency(n, col_ptr, row_idx);
    let degree: Vec<usize> = (0..n).map(|i| adj_ptr[i + 1] - adj_ptr[i]).collect();

    let mut visited = vec![false; n];
    let mut order = Vec::with_capacity(n);
    let mut neighbours: Vec<usize> = Vec::new();

    loop {
        // Seed: lowest-degree unvisited vertex. (A pseudo-peripheral seed via
        // the George-Liu algorithm is better and is part of the AMD task.)
        let seed = (0..n)
            .filter(|&i| !visited[i])
            .min_by_key(|&i| (degree[i], i));
        let Some(seed) = seed else { break };

        let mut head = order.len();
        visited[seed] = true;
        order.push(seed);
        while head < order.len() {
            let v = order[head];
            head += 1;
            neighbours.clear();
            for p in adj_ptr[v]..adj_ptr[v + 1] {
                let w = adj[p];
                if !visited[w] {
                    neighbours.push(w);
                }
            }
            neighbours.sort_unstable_by_key(|&w| (degree[w], w));
            for &w in &neighbours {
                if !visited[w] {
                    visited[w] = true;
                    order.push(w);
                }
            }
        }
    }

    order.reverse();
    order
}

/// Invert a permutation: `inverse[perm[k]] == k`.
#[must_use]
pub(crate) fn invert(perm: &[usize]) -> Vec<usize> {
    let mut inv = vec![0usize; perm.len()];
    for (k, &p) in perm.iter().enumerate() {
        inv[p] = k;
    }
    inv
}

#[cfg(test)]
mod tests {
    use super::*;
    use mincon_core::Sparsity;

    fn upper_of(n: usize, edges: &[(usize, usize)]) -> Sparsity {
        let mut t: Vec<(usize, usize)> = (0..n).map(|i| (i, i)).collect();
        for &(a, b) in edges {
            let (i, j) = if a <= b { (a, b) } else { (b, a) };
            t.push((i, j));
        }
        Sparsity::from_triplets(n, n, &t).unwrap()
    }

    #[test]
    fn natural_is_identity() {
        let s = upper_of(4, &[(0, 1), (1, 2)]);
        let p = compute_ordering(4, s.col_ptr(), s.row_idx(), Ordering::Natural);
        assert_eq!(p, vec![0, 1, 2, 3]);
    }

    #[test]
    fn rcm_is_a_permutation() {
        let s = upper_of(6, &[(0, 3), (3, 5), (1, 4), (4, 2), (2, 5)]);
        let p = compute_ordering(6, s.col_ptr(), s.row_idx(), Ordering::Rcm);
        let mut q = p.clone();
        q.sort_unstable();
        assert_eq!(q, (0..6).collect::<Vec<_>>());
    }

    #[test]
    fn rcm_handles_disconnected_components() {
        let s = upper_of(5, &[(0, 1), (3, 4)]);
        let p = compute_ordering(5, s.col_ptr(), s.row_idx(), Ordering::Rcm);
        let mut q = p.clone();
        q.sort_unstable();
        assert_eq!(q, (0..5).collect::<Vec<_>>());
    }

    #[test]
    fn rcm_reduces_bandwidth_on_a_reversed_path() {
        // A path graph presented in an adversarial order.
        let n: usize = 40;
        let mut edges = Vec::new();
        for k in 0..n - 1 {
            // interleave so the natural ordering has large bandwidth
            let a = if k % 2 == 0 { k / 2 } else { n - 1 - k / 2 };
            let b = if (k + 1) % 2 == 0 {
                k.div_ceil(2)
            } else {
                n - 1 - k.div_ceil(2)
            };
            edges.push((a, b));
        }
        let s = upper_of(n, &edges);
        let p = compute_ordering(n, s.col_ptr(), s.row_idx(), Ordering::Rcm);
        let inv = invert(&p);
        let bw = |f: &dyn Fn(usize) -> usize| {
            edges
                .iter()
                .map(|&(a, b)| f(a).abs_diff(f(b)))
                .max()
                .unwrap_or(0)
        };
        let natural = bw(&|i| i);
        let permuted = bw(&|i| inv[i]);
        assert!(
            permuted < natural,
            "RCM bandwidth {permuted} should beat natural {natural}"
        );
    }

    #[test]
    fn invert_roundtrips() {
        let p = vec![3, 0, 2, 1];
        let inv = invert(&p);
        for (k, &pk) in p.iter().enumerate() {
            assert_eq!(inv[pk], k);
        }
    }
}
