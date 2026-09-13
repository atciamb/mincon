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
    /// Curvature-tracking rebuild threshold (`f64::INFINITY` disables it):
    /// when the model's curvature along an accepted step is off by more than
    /// this factor and the per-coordinate quotients `y_i / s_i` say so
    /// consistently, the matrix is rebuilt from those quotients.
    rescale_factor: f64,
    /// Accepted updates required before another rebuild; doubles after each.
    rescale_cooldown: usize,
    since_rebuild: usize,
    rebuilds: usize,
    /// Rebuilds taken through the scale route (quotients consistent with
    /// the error on 80 % of the pair's weight).
    rebuilds_scale: usize,
    /// Rebuilds taken through the shape route only (the model's diagonal
    /// explains the pair, its off-diagonal does not).
    rebuilds_shape: usize,
    /// `(accepted updates before the rebuild, tau, via the scale route)` for
    /// the first [`REBUILD_LOG_CAP`] rebuilds, reported in the notes.
    rebuild_log: Vec<(usize, f64, bool)>,
    /// `tau` observed at the accepted updates that followed each rebuild
    /// (up to two per rebuild), to see whether the rebuild fixed the model.
    post_rebuild_tau: Vec<f64>,
    /// Updates still to be logged into `post_rebuild_tau`.
    post_rebuild_pending: usize,
}

/// Rebuilds remembered in [`DenseBfgs::rebuild_log`].
const REBUILD_LOG_CAP: usize = 16;

/// Per-coordinate curvature quotients `y_i / s_i` (a diagonal model), clamped
/// to four orders of magnitude around the scalar estimate `gamma` and falling
/// back to `gamma` where the step component is negligible or the quotient is
/// not a positive finite number. A scalar rescale was measured to help chained
/// Rosenbrock and hurt diagonally ill-conditioned quadratics in equal measure;
/// the diagonal form keeps both (`bench/results/s5-bfgs-guarded-diagonal`).
fn diagonal_from_pair(s: &[f64], y: &[f64], gamma: f64) -> Vec<f64> {
    s.iter()
        .zip(y)
        .map(|(&si, &yi)| {
            let q = if si.abs() > 1e-12 * (1.0 + si.abs()) {
                yi / si
            } else {
                gamma
            };
            if q.is_finite() && q > 0.0 {
                q.clamp(gamma / 1e4, 1e4 * gamma)
            } else {
                gamma
            }
        })
        .collect()
}

/// Curvature-tracking rebuilds are enabled only for problems with at least
/// this many variables (see [`DenseBfgs::with_curvature_rescale`]).
const CURVATURE_RESCALE_MIN_N: usize = 10;

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
            rescale_factor: f64::INFINITY,
            rescale_cooldown: 5,
            since_rebuild: 5,
            rebuilds: 0,
            rebuilds_scale: 0,
            rebuilds_shape: 0,
            rebuild_log: Vec::new(),
            post_rebuild_tau: Vec::new(),
            post_rebuild_pending: 0,
        }
    }

    /// As [`DenseBfgs::new`] with the curvature-tracking rebuild of
    /// [`DenseBfgs::set_curvature_rescale`] enabled at `factor`.
    #[must_use]
    pub fn with_curvature_rescale(n: usize, factor: f64) -> Self {
        let mut b = Self::new(n);
        // The repair pathology the rule targets costs of order `n` iterations
        // (one direction fixed per rank-two update), so it only dominates once
        // `n` exceeds the handful of updates a dense BFGS needs to build useful
        // curvature. Below that a mid-course diagonal rebuild only perturbs a
        // path that is about to converge, and on a *coupled* small problem the
        // diagonal it rebuilds to is simply wrong (HS97, HS98 at n = 6 lost an
        // attainment to it in `bench/results/abl-c7`). Gate it on `n`; the
        // smallest problem the rule helps on this corpus is QUADSPHERE_10.
        if n < CURVATURE_RESCALE_MIN_N {
            b.set_curvature_rescale(f64::INFINITY);
        } else {
            b.set_curvature_rescale(factor);
        }
        b
    }

    /// Enable (factor > 1) or disable (`f64::INFINITY`) the curvature-tracking
    /// rebuild: whenever the curvature the model predicts along an accepted
    /// step, `s^T B s`, is off from the measured `s^T y` by more than `factor`
    /// in either direction, and either the per-coordinate quotients
    /// `y_i / s_i` disagree with the model's diagonal in the same direction on
    /// at least 80 % of the pair's weight `|s_i y_i|` (the scale is wrong) or
    /// the model's own diagonal explains the pair at least twice as well as
    /// the full matrix (its off-diagonal part is wrong), the matrix is
    /// replaced by a diagonal built from those quotients before the ordinary
    /// update (see [`DenseBfgs::rebuild_diagonal`]). A rank-two
    /// update can repair one direction per iteration; on problems whose
    /// curvature grows by two orders of magnitude along the path (entropy
    /// terms `x log x`, a constraint multiplier climbing from 0 to 100) both
    /// members were spending hundreds of iterations on that repair
    /// (`bench/results/r5-large-n`). Rebuilds are rate-limited: a cooldown of
    /// five accepted updates that doubles after each rebuild.
    pub fn set_curvature_rescale(&mut self, factor: f64) {
        self.rescale_factor = if factor.is_nan() || factor <= 1.0 {
            f64::INFINITY
        } else {
            factor
        };
    }

    /// Number of curvature-tracking rebuilds performed.
    #[must_use]
    pub fn rebuilds(&self) -> usize {
        self.rebuilds
    }

    /// Rebuilds split by route: `(scale, shape-only)`. Reported in the
    /// solver notes so every record says which test admitted each rebuild.
    /// The shape route rebuilds to the quotients although its test validated
    /// the model's diagonal; guarding it (keep the model's entry wherever a
    /// quotient disagrees by more than the factor) was tried and rejected —
    /// it left PORTFOLIO_100 unchanged, whose rebuild comes through the scale
    /// route, and nearly doubled ELLIPSOID2_200 under the SQP member
    /// (`bench/results/abl-c8-rejected`).
    #[must_use]
    pub fn rebuild_routes(&self) -> (usize, usize) {
        (self.rebuilds_scale, self.rebuilds_shape)
    }

    /// Where the rebuilds happened: `(accepted updates before it, tau =
    /// s^T y / s^T B s at that pair, via the scale route)`, first
    /// [`REBUILD_LOG_CAP`] only.
    #[must_use]
    pub fn rebuild_log(&self) -> &[(usize, f64, bool)] {
        &self.rebuild_log
    }

    /// `s^T y / s^T B s` at the (up to two) accepted updates after each
    /// rebuild, in order: near 1 means the rebuilt model explains the next
    /// pairs, far from 1 means it did not.
    #[must_use]
    pub fn post_rebuild_tau(&self) -> &[f64] {
        &self.post_rebuild_tau
    }

    /// Replace the matrix by a diagonal.
    fn set_diagonal(&mut self, d: &[f64]) {
        self.b.fill(0.0);
        for (i, &v) in d.iter().enumerate() {
            self.b[i * self.n + i] = v;
        }
    }

    /// The diagonal the curvature-tracking rule rebuilds from, when the pair
    /// says the model is off by more than the factor along `s` (`under`: the
    /// model underestimates the curvature) and one of two things holds:
    ///
    /// * **scale**: the per-coordinate quotients `y_i / s_i` disagree with the
    ///   model's diagonal in the same direction on at least 80 % of the pair's
    ///   weight `|s_i y_i|` — the diagonal itself is wrong (a unit matrix on a
    ///   problem with curvature 1000; curvature that grew 100× along the path);
    /// * **shape**: the model's own diagonal explains the pair at least twice
    ///   as well (in log terms) as the full matrix does — the off-diagonal part
    ///   accumulated from earlier, inconsistent pairs cancels the curvature
    ///   along the directions the step now takes (MAXENT_200: the full model
    ///   15× off along every step for 300 iterations, its diagonal within 2×).
    ///
    /// The rebuilt diagonal is the quotient where the step component is
    /// significant and the quotient is a positive finite number, clamped to
    /// four orders of magnitude around the scalar estimate; elsewhere the
    /// model's own diagonal entry, clamped the same way.
    fn rebuild_diagonal(
        &self,
        s: &[f64],
        y: &[f64],
        tau: f64,
        under: bool,
    ) -> Option<(Vec<f64>, bool)> {
        let n = self.n;
        let f = self.rescale_factor;
        let s_y: f64 = s.iter().zip(y).map(|(a, b)| a * b).sum();
        let y_y: f64 = y.iter().map(|v| v * v).sum();
        if s_y.is_nan() || s_y <= 0.0 {
            return None;
        }
        let gamma = y_y / s_y;
        if !gamma.is_finite() || gamma <= 0.0 {
            return None;
        }
        let mut w_total = 0.0;
        let mut w_consistent = 0.0;
        let mut s_diag_s = 0.0;
        for i in 0..n {
            let bii = self.b[i * n + i];
            s_diag_s += bii * s[i] * s[i];
            let w = (s[i] * y[i]).abs();
            w_total += w;
            if s[i] * y[i] > 0.0 && s[i].abs() > 1e-12 * (1.0 + s[i].abs()) {
                let q = y[i] / s[i];
                let ok = if under { q > f * bii } else { q < bii / f };
                if ok {
                    w_consistent += w;
                }
            }
        }
        if w_total.is_nan() || w_total <= 0.0 {
            return None;
        }
        let scale_route = w_consistent >= 0.8 * w_total;
        let shape_route = s_diag_s.is_finite()
            && s_diag_s > 0.0
            && (s_y / s_diag_s).ln().abs() <= 0.5 * tau.ln().abs();
        if !(scale_route || shape_route) {
            return None;
        }
        let lo = gamma / 1e4;
        let hi = 1e4 * gamma;
        let d = (0..n)
            .map(|i| {
                let bii = self.b[i * n + i];
                let q = if s[i].abs() > 1e-12 * (1.0 + s[i].abs()) {
                    y[i] / s[i]
                } else {
                    bii
                };
                let q = if q.is_finite() && q > 0.0 { q } else { bii };
                if q.is_finite() && q > 0.0 {
                    q.clamp(lo, hi)
                } else {
                    gamma
                }
            })
            .collect();
        Some((d, scale_route))
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

    /// Replace the matrix by a full row-major `n * n` one (symmetrised); a
    /// warm start hands a previous solve's curvature model over this way.
    /// Ignored when the length is wrong or a value is not finite.
    pub fn set_dense(&mut self, full: &[f64]) -> bool {
        let n = self.n;
        if full.len() != n * n || full.iter().any(|v| !v.is_finite()) {
            return false;
        }
        for i in 0..n {
            for j in 0..n {
                self.b[i * n + j] = 0.5 * (full[i * n + j] + full[j * n + i]);
            }
        }
        true
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
        let mut s_bs: f64 = s.iter().zip(&bs).map(|(a, b)| a * b).sum();
        let s_y: f64 = s.iter().zip(y).map(|(a, b)| a * b).sum();

        if !s_bs.is_finite() || !s_y.is_finite() || s_bs <= 0.0 {
            self.skipped += 1;
            return false;
        }

        if self.post_rebuild_pending > 0 && s_y > 0.0 {
            self.post_rebuild_pending -= 1;
            if self.post_rebuild_tau.len() < 2 * REBUILD_LOG_CAP {
                self.post_rebuild_tau.push(s_y / s_bs);
            }
        }
        // Curvature-tracking rebuild (see `set_curvature_rescale`). The raw
        // `y` is used: the damped `r` below is bounded away from zero relative
        // to `s^T B s` and would hide the size of an underestimate.
        let first_update_scaling = scale_initial && self.updates == 0;
        if !first_update_scaling
            && self.rescale_factor.is_finite()
            && s_y > 0.0
            && self.since_rebuild >= self.rescale_cooldown
        {
            let tau = s_y / s_bs;
            let f = self.rescale_factor;
            if tau > f || tau < 1.0 / f {
                if let Some((d, via_scale)) = self.rebuild_diagonal(s, y, tau, tau > f) {
                    self.set_diagonal(&d);
                    self.rebuilds += 1;
                    if via_scale {
                        self.rebuilds_scale += 1;
                    } else {
                        self.rebuilds_shape += 1;
                    }
                    if self.rebuild_log.len() < REBUILD_LOG_CAP {
                        self.rebuild_log.push((self.updates, tau, via_scale));
                    }
                    self.post_rebuild_pending = 2;
                    self.since_rebuild = 0;
                    self.rescale_cooldown = self.rescale_cooldown.saturating_mul(2);
                    self.multiply(s, &mut bs);
                    s_bs = s.iter().zip(&bs).map(|(a, b)| a * b).sum();
                    if !s_bs.is_finite() || s_bs <= 0.0 {
                        self.skipped += 1;
                        return false;
                    }
                }
            }
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
                self.set_diagonal(&diagonal_from_pair(s, &r, gamma));
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
        self.since_rebuild += 1;
        true
    }

    /// Entry `(i, j)`. Test helper.
    #[must_use]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.b[i * self.n + j]
    }
}

/// The rebuild log as `"3 [scale, tau 1.2e3], 17 [shape, tau 4.5e-2]"` for the notes.
pub fn rebuild_log_text(log: &[(usize, f64, bool)]) -> String {
    log.iter()
        .map(|(k, tau, scale)| {
            format!(
                "{k} [{}, tau {tau:.1e}]",
                if *scale { "scale" } else { "shape" }
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
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
    fn curvature_rescale_rebuilds_the_diagonal_of_a_separable_quadratic() {
        // A full first step on a separable problem whose curvature is 1..1000
        // times the unit matrix: the guarded first-update scaling does not fire
        // (no cut), the curvature-tracking rule does, and because the quotients
        // are the exact diagonal the ordinary update that follows leaves it
        // unchanged: the model IS the Hessian after one pair.
        let h = [1.0, 10.0, 100.0, 1000.0];
        let s = [0.3, -0.2, 0.5, 0.1];
        let y: Vec<f64> = (0..4).map(|i| h[i] * s[i]).collect();
        let mut off = DenseBfgs::new(4);
        assert!(off.update(&s, &y));
        assert_eq!(off.rebuilds(), 0);
        assert!(
            (off.get(3, 3) - 1000.0).abs() > 100.0,
            "plain BFGS cannot learn the diagonal from one pair"
        );

        let mut b = DenseBfgs::new(4);
        b.set_curvature_rescale(10.0); // bypass the n gate: this tests the rebuild mechanism
        assert!(b.update(&s, &y));
        assert_eq!(b.rebuilds(), 1);
        for i in 0..4 {
            for j in 0..4 {
                let expect = if i == j { h[i] } else { 0.0 };
                assert!(
                    (b.get(i, j) - expect).abs() <= 1e-9 * 1000.0,
                    "entry ({i},{j}) = {} vs {expect}",
                    b.get(i, j)
                );
            }
        }
        // The cooldown doubled: the same pair straight away is not rebuilt again
        // (it does not need to be: the model now explains it).
        assert!(b.update(&s, &y));
        assert_eq!(b.rebuilds(), 1);
    }

    #[test]
    fn curvature_rescale_is_gated_on_problem_size() {
        // The same separable pair that rebuilds at n = 10 does nothing below the
        // gate, because `with_curvature_rescale` disables the rule for small n
        // (small problems a dense BFGS handles in a handful of updates; a
        // mid-course rebuild there only perturbs them - HS97/HS98).
        let h = [1.0, 10.0, 100.0, 1000.0];
        let s4 = [0.3, -0.2, 0.5, 0.1];
        let y4: Vec<f64> = (0..4).map(|i| h[i] * s4[i]).collect();
        let mut small = DenseBfgs::with_curvature_rescale(4, 10.0);
        assert!(small.update(&s4, &y4));
        assert_eq!(
            small.rebuilds(),
            0,
            "the rule must be off below the size gate"
        );

        // At n = 10 it is on: a separable pair with curvature far from the unit
        // matrix rebuilds.
        let mut s10 = vec![0.0; 10];
        let mut y10 = vec![0.0; 10];
        for i in 0..10 {
            let hi = 10f64.powi(i as i32 % 4);
            s10[i] = 0.1 * ((i % 3) as f64 - 1.0).abs().max(0.1);
            y10[i] = hi * s10[i];
        }
        let mut big = DenseBfgs::with_curvature_rescale(10, 10.0);
        assert!(big.update(&s10, &y10));
        assert_eq!(big.rebuilds(), 1, "the rule must be on at the size gate");
    }

    #[test]
    fn curvature_rescale_keeps_the_matrix_when_the_quotients_disagree() {
        // Dense, strongly coupled Hessian 1000 * [[1, 0.95], [0.95, 1]] and a
        // step for which the model is off by 55x along s, but the per-coordinate
        // quotients carry opposite signs (24 % of the weight is inconsistent):
        // the diagonal would be a lie, so the matrix is kept and updated.
        let s = [1.0, -0.9];
        let y = [1000.0 * (1.0 - 0.855), 1000.0 * (0.95 - 0.9)];
        let mut b = DenseBfgs::new(2);
        b.set_curvature_rescale(10.0);
        assert!(b.update(&s, &y));
        assert_eq!(b.rebuilds(), 0);
        assert_eq!(b.updates(), 1);
        assert!(
            b.get(0, 1).abs() > 1e-6,
            "the ordinary update produced an off-diagonal entry"
        );
        // The secant condition still holds.
        let mut bs = vec![0.0; 2];
        b.multiply(&s, &mut bs);
        for i in 0..2 {
            assert!((bs[i] - y[i]).abs() < 1e-9 * 1000.0);
        }
    }

    #[test]
    fn curvature_rescale_drops_off_diagonals_that_cancel_the_curvature() {
        // One update from a pair along (1, -1) with curvature 199 leaves
        // B = [[100, -99], [-99, 100]]: the right diagonal, but only curvature 1
        // along (1, 1). The true Hessian is diag(100, 100). The scale route does
        // not fire (quotients equal the diagonal); the shape route does, because
        // the diagonal alone explains the pair exactly while the full model is
        // 100x off; the rebuilt model is the true Hessian.
        let mut b = DenseBfgs::new(2);
        assert!(b.update(&[1.0, -1.0], &[199.0, -199.0]));
        assert!((b.get(0, 0) - 100.0).abs() < 1e-9 && (b.get(0, 1) + 99.0).abs() < 1e-9);
        b.set_curvature_rescale(10.0);
        assert!(b.update(&[1.0, 1.0], &[100.0, 100.0]));
        assert_eq!(b.rebuilds(), 1);
        for (i, j, expect) in [(0, 0, 100.0), (1, 1, 100.0), (0, 1, 0.0), (1, 0, 0.0)] {
            assert!(
                (b.get(i, j) - expect).abs() < 1e-9,
                "entry ({i},{j}) = {}",
                b.get(i, j)
            );
        }
    }

    #[test]
    fn rebuild_routes_are_counted_separately() {
        // Same accumulated matrix as above, B = [[100, -99], [-99, 100]], and a
        // pair along (1, 1) whose second quotient (0.005) is nothing like the
        // diagonal (100): the scale route cannot fire (no coordinate is
        // consistent with the 50x error), the shape route does, and the
        // rebuild installs that stray quotient, clamped to 0.01 - the matrix
        // the shape test never examined. Recorded here so the behaviour is
        // explicit; guarding it was tried and rejected
        // (`bench/results/abl-c8-rejected`).
        let mut b = DenseBfgs::new(2);
        assert!(b.update(&[1.0, -1.0], &[199.0, -199.0]));
        b.set_curvature_rescale(10.0);
        let s = [1.0, 1.0];
        let y = [100.0, 0.005];
        let tau = 100.005 / 2.0;
        let (d, via_scale) = b
            .rebuild_diagonal(&s, &y, tau, true)
            .expect("shape route fires");
        assert!(!via_scale);
        assert!((d[0] - 100.0).abs() < 1e-9);
        assert!(d[1] < 0.02, "the stray quotient is installed ({})", d[1]);
        assert!(b.update(&s, &y));
        assert_eq!(b.rebuilds(), 1);
        assert_eq!(b.rebuild_routes(), (0, 1));

        // A separable pair on a unit matrix goes through the scale route.
        let mut c = DenseBfgs::new(2);
        c.set_curvature_rescale(10.0);
        assert!(c.update(&[0.5, 0.5], &[50.0, 500.0]));
        assert_eq!(c.rebuild_routes(), (1, 0));
    }

    #[test]
    fn curvature_rescale_requires_an_order_of_magnitude_error() {
        // Curvature 3x the model along every coordinate: within the factor, no
        // rebuild; the ordinary update handles it.
        let s = [0.5, 0.5, 0.5];
        let y = [1.5, 1.5, 1.5];
        let mut b = DenseBfgs::new(3);
        b.set_curvature_rescale(10.0);
        assert!(b.update(&s, &y));
        assert_eq!(b.rebuilds(), 0);
        let mut nan_safe = DenseBfgs::new(3);
        nan_safe.set_curvature_rescale(f64::NAN);
        assert!(nan_safe.update(&s, &[150.0, 150.0, 150.0]));
        assert_eq!(nan_safe.rebuilds(), 0, "NaN disables the rule");
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
