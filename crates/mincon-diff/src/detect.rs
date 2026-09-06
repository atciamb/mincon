//! Sparsity detection for black-box models.
//!
//! # What this can and cannot do
//!
//! Detecting the sparsity of a function you can only *evaluate* is, in the
//! worst case, `n` evaluations — there is no way to rule out a dependence you
//! never probed. This module does exactly that: one probe per variable, at two
//! different base points, taking the union of what moved. Two points rather
//! than one because a single probe can land on a stationary point of a genuine
//! dependence (`d/dx x^2` at `x = 0`) and report a structural zero that is not
//! one, and a wrongly-narrow pattern is a *silent wrong answer*, not a slow one.
//!
//! Even two points is a heuristic. The honest hierarchy, best first:
//!
//! 1. The user declares the pattern. Free and exact.
//! 2. Automatic differentiation over an expression graph derives it. Exact.
//! 3. Probing. `2n` evaluations, and only *probably* right.
//!
//! So detection is worth it when `2n` evaluations are cheap relative to the
//! whole solve — which is the common case for a model with more than a few
//! iterations ahead of it — and it is gated on
//! [`DetectConfig::max_probe_variables`] so that a huge model does not spend
//! its entire budget here.
//!
//! `fmincon` does not attempt this at all: `JacobPattern` must be supplied or
//! the Jacobian is treated as dense. Detecting it is a real capability
//! difference, provided we are honest about the failure mode.

use mincon_core::{Nlp, Sparsity};

/// Configuration for probing.
#[derive(Debug, Clone, Copy)]
pub struct DetectConfig {
    /// Do not probe at all above this many variables; assume dense instead.
    pub max_probe_variables: usize,
    /// Relative perturbation used for probing. Deliberately much larger than a
    /// finite-difference step: we are asking "does this move at all", not
    /// "by how much", and a large probe is far less likely to be lost in
    /// rounding.
    pub probe_scale: f64,
    /// A response smaller than this (relative to the constraint's own scale) is
    /// treated as a structural zero.
    pub zero_threshold: f64,
    /// Number of base points to probe from.
    pub num_base_points: usize,
}

impl Default for DetectConfig {
    fn default() -> Self {
        Self {
            max_probe_variables: 5_000,
            probe_scale: 1e-3,
            zero_threshold: 1e-12,
            num_base_points: 2,
        }
    }
}

/// Outcome of a detection attempt.
#[derive(Debug, Clone)]
pub enum Detected {
    /// A pattern was found, with the number of evaluations it cost.
    Pattern {
        /// The detected structure.
        pattern: Sparsity,
        /// Model evaluations consumed.
        evaluations: u64,
    },
    /// Detection was skipped; treat the Jacobian as dense.
    Dense {
        /// Why it was skipped.
        reason: String,
    },
}

/// Probe a model for its Jacobian sparsity.
///
/// Returns [`Detected::Dense`] rather than an error whenever probing is
/// inadvisable or inconclusive, so a caller never has to decide what a failure
/// means: dense is always correct, merely slow.
pub fn detect_jacobian_sparsity<P: Nlp + ?Sized>(nlp: &P, cfg: &DetectConfig) -> Detected {
    let dims = nlp.dims();
    let (n, m) = (dims.n, dims.m);
    if m == 0 {
        return Detected::Pattern {
            pattern: Sparsity::from_triplets(0, n, &[]).unwrap_or_else(|_| Sparsity::dense(0, n)),
            evaluations: 0,
        };
    }
    if n > cfg.max_probe_variables {
        return Detected::Dense {
            reason: format!(
                "{n} variables exceeds the probing limit of {}; declare jacobian_structure() to avoid a dense Jacobian",
                cfg.max_probe_variables
            ),
        };
    }

    let (lb, ub) = nlp.x_bounds();
    let x0 = nlp.x0();
    let typical = nlp.typical_x();

    let mut evaluations = 0u64;
    let mut triplets: Vec<(usize, usize)> = Vec::new();
    let mut base = x0.to_vec();
    let mut probe = vec![0.0; m];
    let mut reference = vec![0.0; m];

    for point in 0..cfg.num_base_points.max(1) {
        if point > 0 {
            // A second base point, displaced deterministically so the run is
            // reproducible, and clipped into the box.
            for i in 0..n {
                let scale = typical.map_or(1.0, |t| t[i].abs().max(1.0));
                let shift = 0.31_f64.mul_add(scale, 0.017 * ((i % 7) as f64 - 3.0) * scale);
                base[i] = clamp(x0[i] + shift, lb[i], ub[i]);
            }
        }
        if nlp.constraints(&base, &mut reference).is_err() {
            continue;
        }
        evaluations += 1;
        if !reference.iter().all(|v| v.is_finite()) {
            continue;
        }
        let row_scale: Vec<f64> = reference.iter().map(|v| v.abs().max(1.0)).collect();

        for j in 0..n {
            let scale = typical
                .map_or(1.0, |t| t[j].abs().max(1.0))
                .max(base[j].abs());
            let mut h = cfg.probe_scale * scale;
            if base[j] + h > ub[j] {
                h = -h;
            }
            if base[j] + h < lb[j] {
                // Pinned: cannot probe, so assume every row may depend on it.
                for i in 0..m {
                    triplets.push((i, j));
                }
                continue;
            }
            let saved = base[j];
            base[j] = saved + h;
            let ok = nlp.constraints(&base, &mut probe).is_ok();
            evaluations += 1;
            base[j] = saved;
            if !ok {
                for i in 0..m {
                    triplets.push((i, j));
                }
                continue;
            }
            for i in 0..m {
                if !probe[i].is_finite()
                    || (probe[i] - reference[i]).abs() > cfg.zero_threshold * row_scale[i]
                {
                    triplets.push((i, j));
                }
            }
        }
    }

    match Sparsity::from_triplets(m, n, &triplets) {
        Ok(pattern) => {
            // If probing found nothing at all the model is probably constant or
            // the threshold is wrong; dense is the safe answer.
            if pattern.nnz() == 0 {
                Detected::Dense {
                    reason: "probing detected no dependence at all; falling back to dense".into(),
                }
            } else {
                Detected::Pattern {
                    pattern,
                    evaluations,
                }
            }
        }
        Err(e) => Detected::Dense { reason: e },
    }
}

fn clamp(v: f64, lo: f64, hi: f64) -> f64 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mincon_core::{Capabilities, EvalError, NlpDims};

    const INF: f64 = f64::INFINITY;

    struct Banded {
        n: usize,
        lb: Vec<f64>,
        ub: Vec<f64>,
        cl: Vec<f64>,
        cu: Vec<f64>,
        x0: Vec<f64>,
    }

    impl Banded {
        fn new(n: usize) -> Self {
            Self {
                n,
                lb: vec![-INF; n],
                ub: vec![INF; n],
                cl: vec![0.0; n - 1],
                cu: vec![0.0; n - 1],
                x0: (0..n).map(|i| 0.3 + 0.2 * i as f64).collect(),
            }
        }
    }

    impl Nlp for Banded {
        fn dims(&self) -> NlpDims {
            NlpDims {
                n: self.n,
                m: self.n - 1,
            }
        }
        fn x_bounds(&self) -> (&[f64], &[f64]) {
            (&self.lb, &self.ub)
        }
        fn c_bounds(&self) -> (&[f64], &[f64]) {
            (&self.cl, &self.cu)
        }
        fn x0(&self) -> &[f64] {
            &self.x0
        }
        fn capabilities(&self) -> Capabilities {
            Capabilities::none()
        }
        fn objective(&self, x: &[f64]) -> Result<f64, EvalError> {
            Ok(x.iter().map(|v| v * v).sum())
        }
        fn constraints(&self, x: &[f64], out: &mut [f64]) -> Result<(), EvalError> {
            for i in 0..self.n - 1 {
                out[i] = x[i] * x[i] - x[i + 1];
            }
            Ok(())
        }
    }

    #[test]
    fn finds_a_banded_pattern_exactly() {
        let p = Banded::new(10);
        match detect_jacobian_sparsity(&p, &DetectConfig::default()) {
            Detected::Pattern {
                pattern,
                evaluations,
            } => {
                assert_eq!(pattern.nnz(), 2 * 9, "expected two entries per row");
                for j in 0..10 {
                    for &i in pattern.col(j) {
                        assert!(j == i || j == i + 1, "unexpected entry ({i},{j})");
                    }
                }
                assert!(evaluations <= 2 * (10 + 1));
            }
            Detected::Dense { reason } => panic!("fell back to dense: {reason}"),
        }
    }

    #[test]
    fn two_base_points_survive_a_stationary_first_probe() {
        // c(x) = x0^2, probed from x0 = 0 where the derivative vanishes.
        // A single-point detector would report no dependence at all.
        struct Stationary {
            lb: Vec<f64>,
            ub: Vec<f64>,
            cl: Vec<f64>,
            cu: Vec<f64>,
            x0: Vec<f64>,
        }
        impl Nlp for Stationary {
            fn dims(&self) -> NlpDims {
                NlpDims { n: 2, m: 1 }
            }
            fn x_bounds(&self) -> (&[f64], &[f64]) {
                (&self.lb, &self.ub)
            }
            fn c_bounds(&self) -> (&[f64], &[f64]) {
                (&self.cl, &self.cu)
            }
            fn x0(&self) -> &[f64] {
                &self.x0
            }
            fn capabilities(&self) -> Capabilities {
                Capabilities::none()
            }
            fn objective(&self, x: &[f64]) -> Result<f64, EvalError> {
                Ok(x[1])
            }
            fn constraints(&self, x: &[f64], out: &mut [f64]) -> Result<(), EvalError> {
                out[0] = x[0] * x[0];
                Ok(())
            }
        }
        let p = Stationary {
            lb: vec![-INF; 2],
            ub: vec![INF; 2],
            cl: vec![0.0],
            cu: vec![0.0],
            x0: vec![0.0, 0.0],
        };
        let d = detect_jacobian_sparsity(&p, &DetectConfig::default());
        match d {
            Detected::Pattern { pattern, .. } => {
                assert_eq!(pattern.col(0), &[0], "must detect the x0 dependence");
                assert!(
                    pattern.col(1).is_empty(),
                    "x1 does not enter the constraint"
                );
            }
            Detected::Dense { reason } => panic!("fell back to dense: {reason}"),
        }
    }

    #[test]
    fn declines_to_probe_a_huge_model() {
        let p = Banded::new(10);
        let cfg = DetectConfig {
            max_probe_variables: 2,
            ..DetectConfig::default()
        };
        assert!(matches!(
            detect_jacobian_sparsity(&p, &cfg),
            Detected::Dense { .. }
        ));
    }

    #[test]
    fn unconstrained_model_gives_an_empty_pattern() {
        struct Unconstrained {
            lb: Vec<f64>,
            ub: Vec<f64>,
            x0: Vec<f64>,
        }
        impl Nlp for Unconstrained {
            fn dims(&self) -> NlpDims {
                NlpDims { n: 3, m: 0 }
            }
            fn x_bounds(&self) -> (&[f64], &[f64]) {
                (&self.lb, &self.ub)
            }
            fn c_bounds(&self) -> (&[f64], &[f64]) {
                (&[], &[])
            }
            fn x0(&self) -> &[f64] {
                &self.x0
            }
            fn capabilities(&self) -> Capabilities {
                Capabilities::none()
            }
            fn objective(&self, x: &[f64]) -> Result<f64, EvalError> {
                Ok(x[0])
            }
            fn constraints(&self, _x: &[f64], _out: &mut [f64]) -> Result<(), EvalError> {
                Ok(())
            }
        }
        let p = Unconstrained {
            lb: vec![-INF; 3],
            ub: vec![INF; 3],
            x0: vec![0.0; 3],
        };
        match detect_jacobian_sparsity(&p, &DetectConfig::default()) {
            Detected::Pattern { pattern, .. } => assert_eq!(pattern.nnz(), 0),
            Detected::Dense { reason } => panic!("unexpected dense fallback: {reason}"),
        }
    }
}
