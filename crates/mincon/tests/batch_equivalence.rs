//! A model that evaluates its finite-difference probes as a batch must be
//! solved along exactly the iterates of the same model evaluated one call at a
//! time: the probes are the same points and only their scheduling changes. Every
//! fixture, every algorithm, compared to the last bit.
use std::sync::atomic::{AtomicU64, Ordering};

use mincon_core::{
    Algorithm, Capabilities, EvalError, Nlp, NlpDims, Options, SolveError, SolveReport, Sparsity,
};

/// `inner`, advertising [`Capabilities::batch`] and answering a batch with the
/// trait's default loop, counting the batches and the points in them.
struct Batching<'a, P: Nlp>(&'a P, AtomicU64, AtomicU64);

impl<P: Nlp> Nlp for Batching<'_, P> {
    fn dims(&self) -> NlpDims {
        self.0.dims()
    }
    fn x_bounds(&self) -> (&[f64], &[f64]) {
        self.0.x_bounds()
    }
    fn c_bounds(&self) -> (&[f64], &[f64]) {
        self.0.c_bounds()
    }
    fn x0(&self) -> &[f64] {
        self.0.x0()
    }
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            batch: true,
            ..self.0.capabilities()
        }
    }
    fn objective(&self, x: &[f64]) -> Result<f64, EvalError> {
        self.0.objective(x)
    }
    fn constraints(&self, x: &[f64], out: &mut [f64]) -> Result<(), EvalError> {
        self.0.constraints(x, out)
    }
    fn objective_batch(&self, xs: &[f64]) -> Vec<Result<f64, EvalError>> {
        let n = self.0.dims().n;
        self.1.fetch_add(1, Ordering::Relaxed);
        self.2.fetch_add((xs.len() / n) as u64, Ordering::Relaxed);
        xs.chunks_exact(n).map(|x| self.0.objective(x)).collect()
    }
    fn jacobian_structure(&self) -> Option<&Sparsity> {
        self.0.jacobian_structure()
    }
    fn typical_x(&self) -> Option<&[f64]> {
        self.0.typical_x()
    }
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Everything about a solve that the path of iterates determines.
fn fingerprint(r: &Result<SolveReport, SolveError>) -> String {
    match r {
        Err(e) => format!("error: {e}"),
        Ok(r) => {
            let trace: Vec<String> = r
                .trace
                .iter()
                .map(|t| {
                    format!(
                        "{}:{}:{:x}:{:x}:{:x}:{:x}:{:x}",
                        t.iter,
                        t.f_count,
                        t.f.to_bits(),
                        t.constraint_violation.to_bits(),
                        t.optimality.to_bits(),
                        t.step_norm.to_bits(),
                        t.alpha.to_bits()
                    )
                })
                .collect();
            format!(
                "{:?} it={} f={} c={} failed={} x={:?} lambda={:?} trace={}",
                r.exit_flag,
                r.iterations,
                r.f_evals,
                r.c_evals,
                r.failed_evals,
                bits(&r.solution.x),
                bits(&r.solution.lambda),
                trace.join(" ")
            )
        }
    }
}

#[test]
fn every_fixture_takes_the_same_iterates_with_batched_probes() {
    let problems = mincon_testset::all();
    assert!(problems.len() >= 56);
    let mut compared = 0;
    let (mut batches, mut points, mut evaluations) = (0, 0, 0);
    for algorithm in [Algorithm::Auto, Algorithm::InteriorPoint, Algorithm::Sqp] {
        for central in [false, true] {
            let opts = Options {
                algorithm,
                threads: Some(1),
                fd_type: if central {
                    mincon_core::FdType::Central
                } else {
                    Options::default().fd_type
                },
                ..Options::default()
            };
            for p in &problems {
                // Central differences double the cost: a sample of the set is enough there.
                if central && p.n > 10 {
                    continue;
                }
                let nlp = p.as_nlp();
                let serial = mincon::minimize(&nlp, &opts);
                let wrapped = Batching(&nlp, AtomicU64::new(0), AtomicU64::new(0));
                let batched = mincon::minimize(&wrapped, &opts);
                batches += wrapped.1.load(Ordering::Relaxed);
                points += wrapped.2.load(Ordering::Relaxed);
                evaluations += batched.as_ref().map_or(0, |r| r.f_evals);
                assert_eq!(
                    fingerprint(&serial),
                    fingerprint(&batched),
                    "{} under {algorithm:?} (central: {central})",
                    p.name
                );
                compared += 1;
            }
        }
    }
    assert!(compared >= 3 * 56);
    // Not vacuous: the batched path carried most of the evaluations.
    println!("{compared} solves compared; {points} of {evaluations} objective evaluations in {batches} batches");
    assert!(batches > 1000 && 2 * points > evaluations);
}
