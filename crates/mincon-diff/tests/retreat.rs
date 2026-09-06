//! Analytical derivative oracles for domain-limited finite-difference probes.
use std::sync::atomic::{AtomicU64, Ordering};

use mincon_core::{Capabilities, EvalError, FdType, Nlp, NlpDims, Sparsity};
use mincon_diff::fd::{FdConfig, FiniteDifferences};

struct Window {
    x: [f64; 2],
    lower: [f64; 2],
    upper: [f64; 2],
    domain_lower: [f64; 2],
    domain_upper: [f64; 2],
    quadratic: bool,
    abort: bool,
    calls: AtomicU64,
}

impl Window {
    fn new() -> Self {
        Self {
            x: [0.0, 0.0],
            lower: [-1.0; 2],
            upper: [1.0; 2],
            domain_lower: [-0.0625; 2],
            domain_upper: [0.03125; 2],
            quadratic: false,
            abort: false,
            calls: AtomicU64::new(0),
        }
    }
    fn values(&self, x: &[f64]) -> Result<[f64; 2], EvalError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        for (j, value) in x.iter().enumerate() {
            assert!(
                *value >= self.lower[j] && *value <= self.upper[j],
                "probe outside box: {x:?}"
            );
        }
        if self.abort {
            return Err(EvalError::UserAbort);
        }
        let mut out = [0.0; 2];
        for j in 0..2 {
            let d = x[j] - self.x[j];
            if d < self.domain_lower[j] || d > self.domain_upper[j] {
                return Err(EvalError::OutOfDomain(
                    "probe outside evaluation window".into(),
                ));
            }
            out[j] = [3.0, -2.0][j] * d + if self.quadratic { 5.0 * d * d } else { 0.0 };
        }
        Ok(out)
    }
}

impl Nlp for Window {
    fn dims(&self) -> NlpDims {
        NlpDims { n: 2, m: 2 }
    }
    fn x0(&self) -> &[f64] {
        &self.x
    }
    fn x_bounds(&self) -> (&[f64], &[f64]) {
        (&self.lower, &self.upper)
    }
    fn c_bounds(&self) -> (&[f64], &[f64]) {
        (&[0.0, 0.0], &[0.0, 0.0])
    }
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            parallel_safe: true,
            ..Capabilities::none()
        }
    }
    fn objective(&self, x: &[f64]) -> Result<f64, EvalError> {
        Ok(self.values(x)?.iter().sum())
    }
    fn constraints(&self, x: &[f64], out: &mut [f64]) -> Result<(), EvalError> {
        out.copy_from_slice(&self.values(x)?);
        Ok(())
    }
}

fn engine(kind: FdType, parallel: bool, coloring: bool, rel: f64) -> FiniteDifferences {
    let pattern = Sparsity::from_triplets(2, 2, &[(0, 0), (1, 1)]).unwrap();
    FiniteDifferences::new(
        FdConfig {
            fd_type: kind,
            rel_step: Some(rel),
            parallel,
            use_coloring: coloring,
            ..FdConfig::default()
        },
        2,
        2,
        None,
        Some(&pattern),
    )
}

fn check(p: &Window, fd: &FiniteDifferences, jacobian: bool) {
    let mut out = [0.0; 2];
    p.calls.store(0, Ordering::Relaxed);
    let count = if jacobian {
        fd.jacobian(p, &p.x, &[0.0, 0.0], &mut out)
    } else {
        fd.gradient(p, &p.x, 0.0, &mut out)
    }
    .unwrap();
    assert_eq!(count, p.calls.load(Ordering::Relaxed));
    for (got, want) in out.iter().zip([3.0, -2.0]) {
        assert!((got - want).abs() < 1e-10, "{out:?} vs [3,-2]");
    }
}

#[test]
fn forward_retreat_uses_actual_displacement_for_gradient_and_jacobian() {
    let p = Window::new();
    for parallel in [false, true] {
        for coloring in [false, true] {
            let fd = engine(FdType::Forward, parallel, coloring, 0.125);
            check(&p, &fd, false);
            check(&p, &fd, true);
        }
    }
}

#[test]
fn unequal_central_retreat_is_exact_for_a_quadratic() {
    let mut p = Window::new();
    p.quadratic = true;
    for parallel in [false, true] {
        for coloring in [false, true] {
            let fd = engine(FdType::Central, parallel, coloring, 0.125);
            check(&p, &fd, false);
            check(&p, &fd, true);
        }
    }
}

#[test]
fn central_coloring_never_reflects_a_column_outside_its_box() {
    let mut p = Window::new();
    p.lower = [0.0, -1.0];
    p.upper = [1.0, 0.0];
    for parallel in [false, true] {
        let fd = engine(FdType::Central, parallel, true, 0.125);
        check(&p, &fd, false);
        check(&p, &fd, true);
    }
}

#[test]
fn central_gradient_and_jacobian_can_use_the_only_valid_side() {
    let mut p = Window::new();
    p.domain_upper = [0.0; 2];
    let fd = engine(FdType::Central, false, true, 0.125);
    check(&p, &fd, false);
    check(&p, &fd, true);
}

#[test]
fn central_rounding_uses_the_actual_unequal_float_spacings() {
    let mut p = Window::new();
    p.x = [2.0_f64.powi(53); 2];
    p.lower = [f64::NEG_INFINITY; 2];
    p.upper = [f64::INFINITY; 2];
    p.domain_lower = [-100.0; 2];
    p.domain_upper = [100.0; 2];
    p.quadratic = true;
    let fd = engine(FdType::Central, false, true, 3.0 / p.x[0]);
    check(&p, &fd, false);
    check(&p, &fd, true);
}

#[test]
fn unrepresentable_nonfixed_probe_is_not_a_zero_derivative() {
    let mut p = Window::new();
    p.x = [1.0; 2];
    p.upper = [2.0; 2];
    let fd = engine(FdType::Forward, false, true, f64::EPSILON / 8.0);
    assert!(fd.gradient(&p, &p.x, 0.0, &mut [0.0; 2]).is_err());
    assert!(fd.jacobian(&p, &p.x, &[0.0; 2], &mut [0.0; 2]).is_err());
    assert_eq!(p.calls.load(Ordering::Relaxed), 0);
}

#[test]
fn user_abort_is_not_retried_or_replaced_with_another_side() {
    let mut p = Window::new();
    p.abort = true;
    let fd = engine(FdType::Central, false, true, 0.125);
    assert!(matches!(
        fd.gradient(&p, &p.x, 0.0, &mut [0.0; 2]),
        Err(EvalError::UserAbort)
    ));
    assert_eq!(p.calls.swap(0, Ordering::Relaxed), 1);
    assert!(matches!(
        fd.jacobian(&p, &p.x, &[0.0; 2], &mut [0.0; 2]),
        Err(EvalError::UserAbort)
    ));
    assert_eq!(p.calls.load(Ordering::Relaxed), 1);
}

#[test]
fn narrow_box_uses_the_available_side_at_either_bound() {
    let mut p = Window::new();
    p.lower = [0.0, -0.01];
    p.upper = [0.01, 0.0];
    for kind in [FdType::Forward, FdType::Central] {
        let fd = engine(kind, false, true, 0.125);
        check(&p, &fd, false);
        check(&p, &fd, true);
    }
}

#[test]
fn one_color_can_mix_central_and_one_sided_columns() {
    let mut p = Window::new();
    p.lower[0] = 0.0;
    for parallel in [false, true] {
        let fd = engine(FdType::Central, parallel, true, 0.125);
        check(&p, &fd, false);
        check(&p, &fd, true);
    }
}

#[test]
fn central_boundary_stencils_retain_quadratic_accuracy_after_retreat() {
    let mut p = Window::new();
    p.quadratic = true;
    for (lower, upper) in [([0.0, -1.0], [1.0, 0.0]), ([0.0, -1.0], [1.0, 1.0])] {
        p.lower = lower;
        p.upper = upper;
        for parallel in [false, true] {
            for coloring in [false, true] {
                let fd = engine(FdType::Central, parallel, coloring, 0.125);
                check(&p, &fd, false);
                check(&p, &fd, true);
            }
        }
    }
}

#[test]
fn fixed_coordinates_need_no_probes() {
    let mut p = Window::new();
    p.lower = p.x;
    p.upper = p.x;
    p.abort = true; // Any callback would fail this test.
    for kind in [FdType::Forward, FdType::Central] {
        let fd = engine(kind, false, true, 0.125);
        let mut out = [f64::NAN; 2];
        assert_eq!(fd.gradient(&p, &p.x, 0.0, &mut out).unwrap(), 0);
        assert_eq!(out, [0.0; 2]);
        out.fill(f64::NAN);
        assert_eq!(fd.jacobian(&p, &p.x, &[0.0; 2], &mut out).unwrap(), 0);
        assert_eq!(out, [0.0; 2]);
    }
    assert_eq!(p.calls.load(Ordering::Relaxed), 0);
}

#[test]
fn dense_jacobian_uses_the_same_retreat_rules() {
    let mut p = Window::new();
    p.quadratic = true;
    for parallel in [false, true] {
        let fd = FiniteDifferences::new(
            FdConfig {
                fd_type: FdType::Central,
                rel_step: Some(0.125),
                parallel,
                ..FdConfig::default()
            },
            2,
            2,
            None,
            None,
        );
        let mut out = [f64::NAN; 4];
        p.calls.store(0, Ordering::Relaxed);
        let count = fd.jacobian(&p, &p.x, &[0.0; 2], &mut out).unwrap();
        assert_eq!(count, p.calls.load(Ordering::Relaxed));
        for (got, want) in out.iter().zip([3.0, 0.0, 0.0, -2.0]) {
            assert!((got - want).abs() < 1e-10, "{out:?}");
        }
    }
}
