//! The batched finite-difference path evaluates the serial path's points and
//! does the serial path's arithmetic: every derivative must agree to the last
//! bit, on the easy columns and on the awkward ones (a variable on a bound, a
//! pinned one, a box too tight for a central pair, a probe the model refuses).
use std::sync::atomic::{AtomicU64, Ordering};

use mincon_core::{Capabilities, EvalError, FdType, Nlp, NlpDims, Sparsity};
use mincon_diff::fd::{FdConfig, FiniteDifferences};

const N: usize = 6;
const M: usize = 4;

/// Six variables in awkward places and a model that refuses points past a
/// limit in the last one, so its probes retreat.
struct Rough {
    x: [f64; N],
    lower: [f64; N],
    upper: [f64; N],
    /// The model is undefined for `x[5] > limit`.
    limit: f64,
    calls: AtomicU64,
}

impl Rough {
    fn new() -> Self {
        let x = [0.3, 1.0, -2.0, 0.5, 0.25, 7.0];
        Self {
            x,
            //       free   on ub  on lb  pinned tight          free, domain-limited
            lower: [-10.0, -4.0, -2.0, 0.5, 0.25 - 1e-9, -10.0],
            upper: [10.0, 1.0, 3.0, 0.5, 0.25 + 1e-9, 10.0],
            limit: 7.0 + 3e-8,
            calls: AtomicU64::new(0),
        }
    }
    fn check(&self, x: &[f64]) -> Result<(), EvalError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        for j in 0..N {
            assert!(
                x[j] >= self.lower[j] && x[j] <= self.upper[j],
                "probe outside the box: {x:?}"
            );
        }
        if x[5] > self.limit {
            return Err(EvalError::OutOfDomain("past the limit".into()));
        }
        Ok(())
    }
}

impl Nlp for Rough {
    fn dims(&self) -> NlpDims {
        NlpDims { n: N, m: M }
    }
    fn x0(&self) -> &[f64] {
        &self.x
    }
    fn x_bounds(&self) -> (&[f64], &[f64]) {
        (&self.lower, &self.upper)
    }
    fn c_bounds(&self) -> (&[f64], &[f64]) {
        (&[0.0; M], &[0.0; M])
    }
    fn capabilities(&self) -> Capabilities {
        Capabilities::none()
    }
    fn objective(&self, x: &[f64]) -> Result<f64, EvalError> {
        self.check(x)?;
        Ok(x.iter()
            .enumerate()
            .map(|(i, v)| ((i + 1) as f64 * v).sin())
            .sum::<f64>()
            + x[0] * x[5]
            + (0.1 * x[1]).exp())
    }
    fn constraints(&self, x: &[f64], out: &mut [f64]) -> Result<(), EvalError> {
        self.check(x)?;
        out[0] = x[0] * x[0] + x[1].cos();
        out[1] = x[1] * x[2] - x[3].exp();
        out[2] = (x[3] + x[4]).sin() * x[2];
        out[3] = x[4] * x[5] + x[5].sqrt();
        Ok(())
    }
}

/// `inner`, advertising that it evaluates batches (it answers them with the
/// trait's default loop) and counting the batch calls.
struct Batched<'a> {
    inner: &'a Rough,
    batches: AtomicU64,
}

impl Nlp for Batched<'_> {
    fn dims(&self) -> NlpDims {
        self.inner.dims()
    }
    fn x0(&self) -> &[f64] {
        self.inner.x0()
    }
    fn x_bounds(&self) -> (&[f64], &[f64]) {
        self.inner.x_bounds()
    }
    fn c_bounds(&self) -> (&[f64], &[f64]) {
        self.inner.c_bounds()
    }
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            batch: true,
            ..Capabilities::none()
        }
    }
    fn objective(&self, _: &[f64]) -> Result<f64, EvalError> {
        panic!("a batching model's gradient must go through objective_batch");
    }
    fn constraints(&self, _: &[f64], _: &mut [f64]) -> Result<(), EvalError> {
        panic!("a batching model's Jacobian must go through constraints_batch");
    }
    fn objective_batch(&self, xs: &[f64]) -> Vec<Result<f64, EvalError>> {
        self.batches.fetch_add(1, Ordering::Relaxed);
        xs.chunks_exact(N)
            .map(|x| self.inner.objective(x))
            .collect()
    }
    fn constraints_batch(&self, xs: &[f64], out: &mut [f64]) -> Vec<Result<(), EvalError>> {
        self.batches.fetch_add(1, Ordering::Relaxed);
        xs.chunks_exact(N)
            .zip(out.chunks_exact_mut(M))
            .map(|(x, o)| self.inner.constraints(x, o))
            .collect()
    }
}

fn engine(kind: FdType, coloring: bool, pattern: &Sparsity) -> FiniteDifferences {
    FiniteDifferences::new(
        FdConfig {
            fd_type: kind,
            use_coloring: coloring,
            ..FdConfig::default()
        },
        N,
        M,
        None,
        Some(pattern),
    )
}

fn sparse_pattern() -> Sparsity {
    Sparsity::from_triplets(
        M,
        N,
        &[
            (0, 0),
            (0, 1),
            (1, 1),
            (1, 2),
            (1, 3),
            (2, 2),
            (2, 3),
            (2, 4),
            (3, 4),
            (3, 5),
        ],
    )
    .unwrap()
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Serial and batched results, the evaluation counts of each, and how many
/// batch calls the batched one made.
fn both(
    kind: FdType,
    coloring: bool,
    pattern: &Sparsity,
    jacobian: bool,
) -> (Vec<f64>, Vec<f64>, u64, u64, u64) {
    let p = Rough::new();
    let fd = engine(kind, coloring, pattern);
    let f0 = p.objective(&p.x).unwrap();
    let mut c0 = [0.0; M];
    p.constraints(&p.x, &mut c0).unwrap();
    let len = if jacobian { pattern.nnz() } else { N };

    let mut serial = vec![0.0; len];
    p.calls.store(0, Ordering::Relaxed);
    let reported = if jacobian {
        fd.jacobian(&p, &p.x, &c0, &mut serial)
    } else {
        fd.gradient(&p, &p.x, f0, &mut serial)
    }
    .unwrap();
    let serial_calls = p.calls.load(Ordering::Relaxed);
    assert_eq!(reported, serial_calls);

    let b = Batched {
        inner: &p,
        batches: AtomicU64::new(0),
    };
    let mut batched = vec![0.0; len];
    p.calls.store(0, Ordering::Relaxed);
    let reported = if jacobian {
        fd.jacobian(&b, &p.x, &c0, &mut batched)
    } else {
        fd.gradient(&b, &p.x, f0, &mut batched)
    }
    .unwrap();
    let batched_calls = p.calls.load(Ordering::Relaxed);
    assert_eq!(reported, batched_calls);
    (
        serial,
        batched,
        serial_calls,
        batched_calls,
        b.batches.load(Ordering::Relaxed),
    )
}

#[test]
fn the_batched_gradient_is_the_serial_one_to_the_last_bit() {
    let dense = Sparsity::dense(M, N);
    for kind in [FdType::Forward, FdType::Central] {
        let (serial, batched, serial_calls, batched_calls, _) = both(kind, true, &dense, false);
        assert_eq!(bits(&serial), bits(&batched), "{kind:?}");
        assert_eq!(serial_calls, batched_calls, "{kind:?}");
        assert_eq!(serial[3], 0.0, "the pinned variable has a zero derivative");
        assert!(serial.iter().all(|v| v.is_finite()));
    }
}

#[test]
fn the_batched_jacobian_is_the_serial_one_to_the_last_bit() {
    let dense = Sparsity::dense(M, N);
    let sparse = sparse_pattern();
    for pattern in [&dense, &sparse] {
        for coloring in [false, true] {
            for kind in [FdType::Forward, FdType::Central] {
                let (serial, batched, serial_calls, batched_calls, _) =
                    both(kind, coloring, pattern, true);
                let what = format!("{kind:?}, coloring {coloring}, nnz {}", pattern.nnz());
                assert_eq!(bits(&serial), bits(&batched), "{what}");
                assert_eq!(serial_calls, batched_calls, "{what}");
            }
        }
    }
}

#[test]
fn a_gradient_is_one_batch_call_per_round() {
    let dense = Sparsity::dense(M, N);
    // Forward: the first attempts, then two more rounds for the one probe the
    // model refuses twice (7e-8 and 3.5e-8 past x[5] against a 3e-8 window...
    // the count is whatever the serial retreat needs, plus one).
    let (_, _, serial_calls, _, batches) = both(FdType::Forward, true, &dense, false);
    let retreats = serial_calls - 5; // five movable variables, one probe each
    assert!(retreats >= 1, "the fixture is meant to force a retreat");
    assert_eq!(batches, 1 + retreats);
    // Central: the same, and the inward second probes are one more wave.
    let (_, _, _, _, batches) = both(FdType::Central, true, &dense, false);
    assert!(batches >= 2);
}

#[test]
fn a_batch_that_comes_back_short_is_a_failed_probe_not_a_hang() {
    struct Short(Rough);
    impl Nlp for Short {
        fn dims(&self) -> NlpDims {
            self.0.dims()
        }
        fn x0(&self) -> &[f64] {
            self.0.x0()
        }
        fn x_bounds(&self) -> (&[f64], &[f64]) {
            self.0.x_bounds()
        }
        fn c_bounds(&self) -> (&[f64], &[f64]) {
            self.0.c_bounds()
        }
        fn capabilities(&self) -> Capabilities {
            Capabilities {
                batch: true,
                ..Capabilities::none()
            }
        }
        fn objective(&self, x: &[f64]) -> Result<f64, EvalError> {
            self.0.objective(x)
        }
        fn constraints(&self, x: &[f64], out: &mut [f64]) -> Result<(), EvalError> {
            self.0.constraints(x, out)
        }
        fn objective_batch(&self, _: &[f64]) -> Vec<Result<f64, EvalError>> {
            Vec::new()
        }
    }
    let p = Short(Rough::new());
    let fd = engine(FdType::Forward, true, &Sparsity::dense(M, N));
    let mut g = vec![0.0; N];
    assert!(fd.gradient(&p, &p.0.x, 0.0, &mut g).is_err());
}
