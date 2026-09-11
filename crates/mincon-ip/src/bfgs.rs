//! Quasi-Newton approximations to the Hessian of the Lagrangian.
//!
//! # Powell damping is the whole trick
//!
//! BFGS needs `s^T y > 0` to stay positive definite. On a *constrained*
//! problem `y` is the change in the gradient of the **Lagrangian**, which is
//! indefinite at the solution, so `s^T y` goes negative regularly and a plain
//! BFGS update destroys the approximation.
//!
//! Powell's fix (1978) replaces `y` with a convex combination of `y` and `B s`
//! chosen so that `s^T y_damped >= 0.2 * s^T B s`. It is what `fmincon`'s
//! `sqp` and `active-set` algorithms use, and skipping it is one of the
//! easiest ways to build a constrained quasi-Newton method that works on
//! textbook problems and falls apart on real ones.
//!
//! # Status
//!
//! Dense BFGS only, which is `O(n^2)` memory and `O(n^2)` per update — the same
//! limit `fmincon`'s interior-point default has. Limited-memory BFGS is
//! specified in `docs/02_SPEC_INTERIOR_POINT.md` and is required before any
//! problem with `n > ~2000` is in scope. The compact representation
//! (Byrd–Nocedal–Schnabel) is the one to implement, because it slots into the
//! KKT matrix as a low-rank update rather than forcing a matrix-free method.

use mincon_core::Sparsity;

/// A dense BFGS approximation to the Hessian of the Lagrangian.
#[derive(Debug, Clone)]
pub struct DenseBfgs {
    n: usize,
    /// Row-major `n x n`, symmetric.
    b: Vec<f64>,
    pattern: Sparsity,
    /// Upper-triangular values in `pattern` order, refreshed by
    /// [`DenseBfgs::upper_values`].
    upper: Vec<f64>,
    /// Updates applied.
    updates: usize,
    /// Updates skipped because the curvature was hopeless.
    skipped: usize,
}

impl DenseBfgs {
    /// A fresh approximation, initialized to the identity.
    #[must_use]
    pub fn new(n: usize) -> Self {
        let mut b = vec![0.0; n * n];
        for i in 0..n {
            b[i * n + i] = 1.0;
        }
        let mut triplets = Vec::with_capacity(n * (n + 1) / 2);
        for j in 0..n {
            for i in 0..=j {
                triplets.push((i, j));
            }
        }
        let pattern = Sparsity::from_triplets(n, n, &triplets)
            .expect("dense upper triangle is a valid pattern");
        let nnz = pattern.nnz();
        Self {
            n,
            b,
            pattern,
            upper: vec![0.0; nnz],
            updates: 0,
            skipped: 0,
        }
    }

    /// The upper-triangular sparsity of the approximation (dense).
    #[must_use]
    pub fn pattern(&self) -> &Sparsity {
        &self.pattern
    }

    /// Current values in `pattern` order.
    pub fn upper_values(&mut self) -> &[f64] {
        let mut k = 0;
        for j in 0..self.n {
            for i in 0..=j {
                self.upper[k] = self.b[i * self.n + j];
                k += 1;
            }
        }
        &self.upper
    }

    /// Number of accepted updates.
    #[must_use]
    pub fn updates(&self) -> usize {
        self.updates
    }
    /// Number of skipped updates. A high ratio means the Lagrangian curvature
    /// is badly behaved and an exact Hessian would pay for itself.
    #[must_use]
    pub fn skipped(&self) -> usize {
        self.skipped
    }

    /// Reset to a scaled identity. Called when restoration has moved the
    /// iterate far enough that the accumulated curvature is meaningless.
    pub fn reset(&mut self, scale: f64) {
        self.b.fill(0.0);
        let s = if scale.is_finite() && scale > 0.0 {
            scale
        } else {
            1.0
        };
        for i in 0..self.n {
            self.b[i * self.n + i] = s;
        }
    }

    /// Multiply the whole matrix by `k` (used when the objective scale changes
    /// mid-solve: the Lagrangian, and therefore its curvature model, scales
    /// with it).
    pub fn scale(&mut self, k: f64) {
        if k.is_finite() && k > 0.0 {
            for b in &mut self.b {
                *b *= k;
            }
        }
    }

    /// The full matrix, row-major `n * n`.
    #[must_use]
    pub fn dense(&self) -> Vec<f64> {
        self.b.clone()
    }

    /// `out <- B * v`.
    pub fn multiply(&self, v: &[f64], out: &mut [f64]) {
        for i in 0..self.n {
            let mut acc = 0.0;
            for j in 0..self.n {
                acc += self.b[i * self.n + j] * v[j];
            }
            out[i] = acc;
        }
    }

    /// Damped BFGS update from a step `s` and a Lagrangian-gradient change `y`.
    ///
    /// Returns `true` if the update was applied.
    pub fn update(&mut self, s: &[f64], y: &[f64]) -> bool {
        self.update_guarded(s, y, false)
    }

    /// As [`DenseBfgs::update`], optionally replacing the still-unit initial
    /// matrix by a diagonal built from this first curvature pair before the
    /// update (a per-coordinate refinement of Nocedal & Wright eq. 6.20). The
    /// caller enables it only when the unit matrix has demonstrably misjudged
    /// the scale — the first line search had to cut the step hard.
    pub fn update_guarded(&mut self, s: &[f64], y: &[f64], scale_initial: bool) -> bool {
        let n = self.n;
        debug_assert_eq!(s.len(), n);
        debug_assert_eq!(y.len(), n);

        let s_norm2: f64 = s.iter().map(|v| v * v).sum();
        if !s_norm2.is_finite() || s_norm2 <= 1e-300 {
            self.skipped += 1;
            return false;
        }

        let mut bs = vec![0.0; n];
        self.multiply(s, &mut bs);
        let s_bs: f64 = s.iter().zip(&bs).map(|(a, b)| a * b).sum();
        let s_y: f64 = s.iter().zip(y).map(|(a, b)| a * b).sum();

        if !s_bs.is_finite() || !s_y.is_finite() || s_bs <= 0.0 {
            self.skipped += 1;
            return false;
        }

        // Powell damping: theta = 1 keeps the plain BFGS update.
        const POWELL: f64 = 0.2;
        let theta = if s_y >= POWELL * s_bs {
            1.0
        } else {
            let denom = s_bs - s_y;
            if denom.abs() <= 1e-300 {
                self.skipped += 1;
                return false;
            }
            (1.0 - POWELL) * s_bs / denom
        };
        if !theta.is_finite() || theta <= 0.0 {
            self.skipped += 1;
            return false;
        }

        let r: Vec<f64> = (0..n)
            .map(|i| theta * y[i] + (1.0 - theta) * bs[i])
            .collect();
        let s_r: f64 = s.iter().zip(&r).map(|(a, b)| a * b).sum();
        // NaN must fail this test, so it is written to reject anything that is
        // not demonstrably above the floor rather than to accept anything that
        // is not below it.
        if s_r.is_nan() || s_r <= 1e-300 {
            self.skipped += 1;
            return false;
        }

        let (bs, s_bs) = if scale_initial && self.updates == 0 {
            let r_r: f64 = r.iter().map(|v| v * v).sum();
            let gamma = r_r / s_r;
            if gamma.is_finite() && gamma > 0.0 {
                // Diagonal, not scalar: the per-coordinate curvature quotient r_i / s_i,
                // clamped to four orders of magnitude around the scalar estimate. A
                // scalar rescale was measured to help chained Rosenbrock and hurt
                // diagonally ill-conditioned quadratics in equal measure; the diagonal
                // form keeps both (0.94x evaluations on 143 problems, one more attained).
                self.reset(gamma);
                for i in 0..n {
                    let q = if s[i].abs() > 1e-12 * (1.0 + s[i].abs()) {
                        r[i] / s[i]
                    } else {
                        gamma
                    };
                    let q = if q.is_finite() && q > 0.0 {
                        q.clamp(gamma / 1e4, 1e4 * gamma)
                    } else {
                        gamma
                    };
                    self.b[i * n + i] = q;
                }
                let mut bs2 = vec![0.0; n];
                self.multiply(s, &mut bs2);
                let s_bs2: f64 = s.iter().zip(&bs2).map(|(a, b)| a * b).sum();
                if !(s_bs2.is_finite() && s_bs2 > 0.0) {
                    self.skipped += 1;
                    return false;
                }
                (bs2, s_bs2)
            } else {
                (bs, s_bs)
            }
        } else {
            (bs, s_bs)
        };
        // B <- B - (B s)(B s)^T / (s^T B s) + r r^T / (s^T r)
        for i in 0..n {
            for j in 0..n {
                let v = self.b[i * n + j] - bs[i] * bs[j] / s_bs + r[i] * r[j] / s_r;
                self.b[i * n + j] = v;
            }
        }
        // Re-symmetrize to keep rounding from drifting.
        for i in 0..n {
            for j in (i + 1)..n {
                let avg = 0.5 * (self.b[i * n + j] + self.b[j * n + i]);
                self.b[i * n + j] = avg;
                self.b[j * n + i] = avg;
            }
        }
        if !self.b.iter().all(|v| v.is_finite()) {
            self.reset(1.0);
            self.skipped += 1;
            return false;
        }
        self.updates += 1;
        true
    }

    /// Entry `(i, j)`. Test helper.
    #[must_use]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.b[i * self.n + j]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_as_the_identity() {
        let b = DenseBfgs::new(3);
        assert_eq!(b.get(0, 0), 1.0);
        assert_eq!(b.get(0, 1), 0.0);
    }

    #[test]
    fn upper_values_follow_the_pattern() {
        let mut b = DenseBfgs::new(3);
        let p = b.pattern().clone();
        let vals = b.upper_values().to_vec();
        assert_eq!(vals.len(), 6);
        // Reconstruct and check it is the identity's upper triangle.
        for j in 0..3 {
            for pos in p.col_ptr()[j]..p.col_ptr()[j + 1] {
                let i = p.row_idx()[pos];
                let expect = if i == j { 1.0 } else { 0.0 };
                assert_eq!(vals[pos], expect, "entry ({i},{j})");
            }
        }
    }

    #[test]
    fn secant_condition_holds_after_one_update() {
        // With B0 = I and a positive-curvature pair, BFGS satisfies B s = y.
        let mut b = DenseBfgs::new(2);
        let s = [1.0, 0.0];
        let y = [4.0, 1.0];
        assert!(b.update(&s, &y));
        let mut bs = vec![0.0; 2];
        b.multiply(&s, &mut bs);
        assert!((bs[0] - y[0]).abs() < 1e-10, "B s = {bs:?}, y = {y:?}");
        assert!((bs[1] - y[1]).abs() < 1e-10);
    }

    #[test]
    fn stays_symmetric() {
        let mut b = DenseBfgs::new(4);
        for k in 0..10 {
            let s: Vec<f64> = (0..4).map(|i| ((i + k) % 3) as f64 - 1.0 + 0.1).collect();
            let y: Vec<f64> = (0..4).map(|i| ((i * k) % 5) as f64 * 0.3 + 0.2).collect();
            b.update(&s, &y);
        }
        for i in 0..4 {
            for j in 0..4 {
                assert!((b.get(i, j) - b.get(j, i)).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn powell_damping_keeps_positive_definiteness_on_negative_curvature() {
        // s^T y < 0: an undamped BFGS update would destroy definiteness.
        let mut b = DenseBfgs::new(2);
        let s = [1.0, 0.0];
        let y = [-3.0, 0.0];
        assert!(b.update(&s, &y), "damping should let the update proceed");
        // Check positive definiteness via the 2x2 leading minors.
        let (a, c, d) = (b.get(0, 0), b.get(0, 1), b.get(1, 1));
        assert!(a > 0.0, "B00 = {a}");
        assert!(a * d - c * c > 0.0, "determinant = {}", a * d - c * c);
    }

    #[test]
    fn a_zero_step_is_skipped_not_applied() {
        let mut b = DenseBfgs::new(2);
        assert!(!b.update(&[0.0, 0.0], &[1.0, 1.0]));
        assert_eq!(b.skipped(), 1);
        assert_eq!(b.updates(), 0);
        assert_eq!(b.get(0, 0), 1.0);
    }

    #[test]
    fn non_finite_input_never_corrupts_the_approximation() {
        let mut b = DenseBfgs::new(2);
        assert!(!b.update(&[1.0, f64::NAN], &[1.0, 1.0]));
        assert!(!b.update(&[1.0, 0.0], &[f64::INFINITY, 1.0]));
        for i in 0..2 {
            for j in 0..2 {
                assert!(b.get(i, j).is_finite());
            }
        }
    }

    #[test]
    fn reset_restores_a_scaled_identity() {
        let mut b = DenseBfgs::new(3);
        b.update(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        b.reset(2.5);
        assert_eq!(b.get(1, 1), 2.5);
        assert_eq!(b.get(0, 2), 0.0);
    }

    #[test]
    fn converges_to_the_hessian_of_a_quadratic() {
        // BFGS's hereditary property only preserves earlier secant conditions
        // when the steps are conjugate, so a handful of arbitrary directions
        // does NOT reproduce H exactly - it converges to it. Asserting
        // exactness here would be a mathematically false test that happens to
        // pass on a lucky example. What IS exact is the most recent secant
        // condition, so both are checked.
        let h = [[3.0, 1.0], [1.0, 2.0]];
        let mut b = DenseBfgs::new(2);
        let dirs = [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, -1.0],
            [0.7, 0.3],
            [-0.4, 0.9],
            [1.0, 0.25],
            [0.1, 1.0],
        ];
        let hv = |s: [f64; 2]| {
            [
                h[0][0] * s[0] + h[0][1] * s[1],
                h[1][0] * s[0] + h[1][1] * s[1],
            ]
        };
        for s in dirs {
            b.update(&s, &hv(s));
        }
        let mut worst = 0.0_f64;
        for i in 0..2 {
            for j in 0..2 {
                worst = worst.max((b.get(i, j) - h[i][j]).abs());
            }
        }
        assert!(worst < 1e-2, "approximation is {worst} away from H");

        // The last secant condition holds exactly.
        let last = dirs[dirs.len() - 1];
        let y = hv(last);
        let mut bs = vec![0.0; 2];
        b.multiply(&last, &mut bs);
        for i in 0..2 {
            assert!(
                (bs[i] - y[i]).abs() < 1e-10,
                "secant condition violated: B s = {bs:?}, y = {y:?}"
            );
        }
    }
}
