//! A uniform derivative interface over "what the model provides" and "what we
//! have to approximate".
//!
//! Every algorithm asks the [`Evaluator`] for `f`, `grad f`, `c` and `J` and
//! never learns whether the answer came from the user's analytic code or from
//! finite differences. That separation is what keeps the plug-and-play path
//! (no derivatives at all) from being a second-class citizen with its own code
//! path and its own bugs — the case `fmincon` is strongest on is the case we
//! must not treat as an afterthought.
//!
//! The evaluator also owns the counters, so evaluation accounting is correct by
//! construction rather than by every call site remembering to increment.

use std::time::{Duration, Instant};

use mincon_core::{EvalCounters, EvalError, Nlp, Options, Sparsity};

use crate::detect::{detect_jacobian_sparsity, DetectConfig, Detected};
use crate::fd::{FdConfig, FiniteDifferences};

/// Wraps a model with whatever derivative machinery it needs.
pub struct Evaluator<'a, P: Nlp + ?Sized> {
    nlp: &'a P,
    fd: FiniteDifferences,
    jac_pattern: Sparsity,
    /// Whether the Jacobian pattern came from the model, was detected, or is a
    /// dense fallback. Reported in the solve notes.
    pub jac_pattern_origin: &'static str,
    counters: EvalCounters,
    model_time: std::sync::Mutex<Duration>,
    setup_notes: Vec<String>,
}

impl<'a, P: Nlp + ?Sized> Evaluator<'a, P> {
    /// Build an evaluator, detecting sparsity if the options ask for it and the
    /// model has not declared any.
    pub fn new(nlp: &'a P, opts: &Options) -> Self {
        let dims = nlp.dims();
        let caps = nlp.capabilities();
        let mut notes = Vec::new();

        let (jac_pattern, origin) = match nlp.jacobian_structure() {
            Some(p) => (p.clone(), "declared by the model"),
            None if dims.m == 0 => (
                Sparsity::from_triplets(0, dims.n, &[])
                    .unwrap_or_else(|_| Sparsity::dense(0, dims.n)),
                "no constraints",
            ),
            None if opts.detect_sparsity && !caps.jacobian => {
                match detect_jacobian_sparsity(nlp, &DetectConfig::default()) {
                    Detected::Pattern {
                        pattern,
                        evaluations,
                    } => {
                        notes.push(format!(
                            "Detected a Jacobian with {} of {} possible nonzeros ({:.1}% dense) using {evaluations} probe evaluations.",
                            pattern.nnz(),
                            dims.m * dims.n,
                            100.0 * pattern.density()
                        ));
                        (pattern, "detected by probing")
                    }
                    Detected::Dense { reason } => {
                        notes.push(format!("Sparsity detection skipped: {reason}"));
                        (Sparsity::dense(dims.m, dims.n), "dense fallback")
                    }
                }
            }
            None => (Sparsity::dense(dims.m, dims.n), "dense fallback"),
        };

        let fd = FiniteDifferences::new(
            FdConfig {
                fd_type: opts.fd_type,
                rel_step: opts.fd_step,
                respect_bounds: opts.fd_respect_bounds,
                use_coloring: opts.fd_coloring,
                parallel: opts.threads.map_or(true, |t| t > 1) && caps.parallel_safe,
            },
            dims.n,
            dims.m,
            nlp.typical_x(),
            Some(&jac_pattern),
        );

        if !caps.jacobian && dims.m > 0 {
            notes.push(format!(
                "Constraint Jacobian by finite differences: {} model evaluations per Jacobian (dense would be {}).",
                fd.jacobian_cost(),
                dims.n
            ));
        }

        Self {
            nlp,
            fd,
            jac_pattern,
            jac_pattern_origin: origin,
            counters: EvalCounters::default(),
            model_time: std::sync::Mutex::new(Duration::ZERO),
            setup_notes: notes,
        }
    }

    /// The model.
    pub fn nlp(&self) -> &'a P {
        self.nlp
    }
    /// The Jacobian sparsity in use.
    #[must_use]
    pub fn jacobian_pattern(&self) -> &Sparsity {
        &self.jac_pattern
    }
    /// Notes accumulated during setup, for the solve report.
    #[must_use]
    pub fn setup_notes(&self) -> &[String] {
        &self.setup_notes
    }
    /// The counters.
    #[must_use]
    pub fn counters(&self) -> &EvalCounters {
        &self.counters
    }
    /// Total time spent inside the user's model.
    #[must_use]
    pub fn model_time(&self) -> Duration {
        *self.model_time.lock().unwrap_or_else(|e| e.into_inner())
    }
    /// Switch finite differences to central. Called when progress stalls.
    pub fn escalate_accuracy(&mut self) {
        self.fd.escalate();
    }
    /// Whether central differences are currently in use.
    #[must_use]
    pub fn uses_central_differences(&self) -> bool {
        self.fd.use_central()
    }

    fn timed<T>(&self, f: impl FnOnce() -> T) -> T {
        let start = Instant::now();
        let out = f();
        if let Ok(mut t) = self.model_time.lock() {
            *t += start.elapsed();
        }
        out
    }

    /// Objective value.
    ///
    /// # Errors
    /// Whatever the model returned; a non-finite value is converted into
    /// [`EvalError::NonFinite`] so callers have one thing to check.
    pub fn f(&self, x: &[f64]) -> Result<f64, EvalError> {
        EvalCounters::bump(&self.counters.f);
        let r = self.timed(|| self.nlp.objective(x));
        match r {
            Ok(v) if v.is_finite() => Ok(v),
            Ok(_) => {
                EvalCounters::bump(&self.counters.failed);
                Err(EvalError::NonFinite(None))
            }
            Err(e) => {
                EvalCounters::bump(&self.counters.failed);
                Err(e)
            }
        }
    }

    /// Constraint values.
    ///
    /// # Errors
    /// As [`Evaluator::f`].
    pub fn c(&self, x: &[f64], out: &mut [f64]) -> Result<(), EvalError> {
        if out.is_empty() {
            return Ok(());
        }
        EvalCounters::bump(&self.counters.c);
        let r = self.timed(|| self.nlp.constraints(x, out));
        match r {
            Ok(()) if out.iter().all(|v| v.is_finite()) => Ok(()),
            Ok(()) => {
                EvalCounters::bump(&self.counters.failed);
                Err(EvalError::NonFinite(None))
            }
            Err(e) => {
                EvalCounters::bump(&self.counters.failed);
                Err(e)
            }
        }
    }

    /// Objective gradient, analytic or by finite differences.
    ///
    /// # Errors
    /// As [`Evaluator::f`].
    pub fn grad(&self, x: &[f64], f0: f64, out: &mut [f64]) -> Result<(), EvalError> {
        if self.nlp.capabilities().gradient {
            EvalCounters::bump(&self.counters.g);
            let r = self.timed(|| self.nlp.gradient(x, out));
            return match r {
                Ok(()) if out.iter().all(|v| v.is_finite()) => Ok(()),
                Ok(()) => Err(EvalError::NonFinite(None)),
                Err(e) => Err(e),
            };
        }
        let used = self.timed(|| self.fd.gradient(self.nlp, x, f0, out))?;
        self.counters
            .f
            .fetch_add(used, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Constraint Jacobian values, in [`Evaluator::jacobian_pattern`] order.
    ///
    /// # Errors
    /// As [`Evaluator::f`].
    pub fn jac(&self, x: &[f64], c0: &[f64], out: &mut [f64]) -> Result<(), EvalError> {
        if self.jac_pattern.nrows() == 0 {
            return Ok(());
        }
        if self.nlp.capabilities().jacobian {
            EvalCounters::bump(&self.counters.j);
            let r = self.timed(|| self.nlp.jacobian(x, out));
            return match r {
                Ok(()) if out.iter().all(|v| v.is_finite()) => Ok(()),
                Ok(()) => Err(EvalError::NonFinite(None)),
                Err(e) => Err(e),
            };
        }
        let used = self.timed(|| self.fd.jacobian(self.nlp, x, c0, out))?;
        self.counters
            .c
            .fetch_add(used, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Whether the model supplies an exact Hessian of the Lagrangian.
    #[must_use]
    pub fn has_exact_hessian(&self) -> bool {
        self.nlp.capabilities().hessian && self.nlp.hessian_structure().is_some()
    }

    /// Exact Hessian of the Lagrangian, lower triangle in the model's order.
    ///
    /// # Errors
    /// As [`Evaluator::f`].
    pub fn hess(
        &self,
        x: &[f64],
        sigma: f64,
        lambda: &[f64],
        out: &mut [f64],
    ) -> Result<(), EvalError> {
        EvalCounters::bump(&self.counters.h);
        self.timed(|| self.nlp.hessian_lagrangian(x, sigma, lambda, out))
    }
}
