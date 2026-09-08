//! The derivative checker.
//!
//! `fmincon`'s `CheckGradients` compares user derivatives against finite
//! differences and prints the worst discrepancy. It is the single most useful
//! diagnostic in the toolbox, because a wrong analytic gradient is the most
//! common cause of "the solver doesn't work on my problem" and it is almost
//! impossible to diagnose from convergence behaviour alone.
//!
//! Two improvements on the original here:
//!
//! * We check at several points, not just the starting point. A gradient that
//!   is wrong only where a branch flips is invisible at `x0` and catastrophic
//!   in the line search.
//! * We report a **relative** discrepancy with an absolute floor, and rank the
//!   worst offenders, rather than a single number. "Component 47 of the
//!   gradient is 3.2x too large" is actionable; "max discrepancy 1e-3" is not.

use mincon_core::{EvalError, Nlp, Sparsity};

use crate::fd::{FdConfig, FiniteDifferences};

/// One disagreement between an analytic derivative and its finite-difference
/// estimate.
#[derive(Debug, Clone, Copy)]
pub struct Discrepancy {
    /// Row index (constraint), or `usize::MAX` for the objective gradient.
    pub row: usize,
    /// Column index (variable).
    pub col: usize,
    /// What the model said.
    pub analytic: f64,
    /// What finite differences said.
    pub numerical: f64,
    /// `|a - n| / max(|a|, |n|, 1)`.
    pub relative: f64,
    /// Which test point exposed it.
    pub point: usize,
}

/// The verdict.
#[derive(Debug, Clone)]
pub struct CheckReport {
    /// Worst offenders, sorted by relative discrepancy, at most 20.
    pub worst: Vec<Discrepancy>,
    /// Largest relative discrepancy seen anywhere.
    pub max_relative: f64,
    /// Threshold used.
    pub tolerance: f64,
    /// Number of points checked.
    pub points_checked: usize,
    /// Model evaluations spent.
    pub evaluations: u64,
    /// Number of (entry, point) comparisons actually made.
    pub comparisons: usize,
    /// Comparisons whose analytic or numerical value was not finite.
    pub nonfinite: usize,
    /// Derivative callbacks that failed, so their entries could not be compared.
    pub derivative_failures: usize,
    /// Why the check could not reach a verdict, if it could not.
    pub inconclusive_reason: Option<String>,
}

impl CheckReport {
    /// Whether the check reached a verdict at all: at least one finite comparison
    /// was made, no analytic entry was non-finite and no derivative callback failed.
    /// A model without analytic derivatives is inconclusive, not passing.
    #[must_use]
    pub fn conclusive(&self) -> bool {
        self.inconclusive_reason.is_none()
    }

    /// Whether every derivative agreed to within the tolerance **and** the check
    /// was conclusive. A check that compared nothing, saw a NaN derivative or
    /// could not evaluate a derivative never passes.
    #[must_use]
    pub fn passed(&self) -> bool {
        self.conclusive() && self.nonfinite == 0 && self.max_relative <= self.tolerance
    }

    /// A message suitable for printing or attaching to the solve notes.
    #[must_use]
    pub fn message(&self) -> String {
        if let Some(reason) = &self.inconclusive_reason {
            return format!(
                "Derivative check INCONCLUSIVE: {reason} ({} comparison(s) at {} point(s), {} non-finite, {} derivative failure(s)).",
                self.comparisons, self.points_checked, self.nonfinite, self.derivative_failures
            );
        }
        if self.passed() {
            return format!(
                "Derivative check passed at {} point(s); largest relative discrepancy {:.2e} (tolerance {:.1e}).",
                self.points_checked, self.max_relative, self.tolerance
            );
        }
        let mut s = format!(
            "Derivative check FAILED. Largest relative discrepancy {:.2e} exceeds tolerance {:.1e}.\n\
             The solver will behave erratically until this is fixed - a wrong derivative is not a\n\
             tolerance problem, it is a different optimization problem.\n\n\
             Worst disagreements (analytic vs finite difference):\n",
            self.max_relative, self.tolerance
        );
        for d in self.worst.iter().take(10) {
            if d.row == usize::MAX {
                s.push_str(&format!(
                    "  grad f [{:>5}]        {:>14.6e}  vs {:>14.6e}   (rel {:.2e}, point {})\n",
                    d.col, d.analytic, d.numerical, d.relative, d.point
                ));
            } else {
                s.push_str(&format!(
                    "  J [{:>5},{:>5}]       {:>14.6e}  vs {:>14.6e}   (rel {:.2e}, point {})\n",
                    d.row, d.col, d.analytic, d.numerical, d.relative, d.point
                ));
            }
        }
        s
    }
}

/// Compare a model's analytic derivatives against finite differences.
///
/// Points are the starting point plus deterministic displacements of it, so the
/// check is reproducible.
///
/// # Errors
/// Propagates a model failure that finite differences could not work around.
pub fn check_derivatives<P: Nlp + ?Sized>(
    nlp: &P,
    tolerance: f64,
    num_points: usize,
) -> Result<CheckReport, EvalError> {
    let dims = nlp.dims();
    let (n, m) = (dims.n, dims.m);
    let caps = nlp.capabilities();
    let (lb, ub) = nlp.x_bounds();
    let x0 = nlp.x0();

    // Central differences: we are measuring correctness, not speed.
    let fd = FiniteDifferences::new(
        FdConfig {
            fd_type: mincon_core::FdType::Central,
            ..FdConfig::default()
        },
        n,
        m,
        nlp.typical_x(),
        nlp.jacobian_structure(),
    );

    let jac_pattern: Sparsity = nlp
        .jacobian_structure()
        .cloned()
        .unwrap_or_else(|| Sparsity::dense(m, n));

    let mut all: Vec<Discrepancy> = Vec::new();
    let mut evaluations = 0u64;
    let mut points_checked = 0usize;
    let mut derivative_failures = 0usize;
    let mut objective_failures = 0usize;

    for point in 0..num_points.max(1) {
        let x: Vec<f64> = if point == 0 {
            x0.to_vec()
        } else {
            (0..n)
                .map(|i| {
                    let scale = nlp.typical_x().map_or(1.0, |t| t[i].abs().max(1.0));
                    let d = 0.137 * scale * ((point as f64) + 0.4 * ((i % 5) as f64 - 2.0));
                    clamp(x0[i] + d, lb[i], ub[i])
                })
                .collect()
        };

        let f0 = match nlp.objective(&x) {
            Ok(f) if f.is_finite() => f,
            _ => {
                objective_failures += 1;
                continue;
            }
        };
        points_checked += 1;

        if caps.gradient {
            let mut analytic = vec![0.0; n];
            if nlp.gradient(&x, &mut analytic).is_ok() {
                let mut numeric = vec![0.0; n];
                evaluations += fd.gradient(nlp, &x, f0, &mut numeric)?;
                for j in 0..n {
                    all.push(make(usize::MAX, j, analytic[j], numeric[j], point));
                }
            } else {
                derivative_failures += 1;
            }
        }

        if caps.jacobian && m > 0 {
            let mut c0 = vec![0.0; m];
            if nlp.constraints(&x, &mut c0).is_err() {
                objective_failures += 1;
                continue;
            }
            let mut analytic = vec![0.0; jac_pattern.nnz()];
            if nlp.jacobian(&x, &mut analytic).is_ok() {
                let mut numeric = vec![0.0; jac_pattern.nnz()];
                evaluations += fd.jacobian(nlp, &x, &c0, &mut numeric)?;
                for j in 0..n {
                    for pos in jac_pattern.col_ptr()[j]..jac_pattern.col_ptr()[j + 1] {
                        let i = jac_pattern.row_idx()[pos];
                        all.push(make(i, j, analytic[pos], numeric[pos], point));
                    }
                }
            } else {
                derivative_failures += 1;
            }
        }
    }

    let comparisons = all.len();
    let nonfinite = all.iter().filter(|d| !d.relative.is_finite()).count();
    // NaN must not be discarded by `f64::max`; a non-finite discrepancy is the worst possible.
    let max_relative = all.iter().fold(0.0_f64, |a, d| {
        if d.relative.is_finite() {
            a.max(d.relative)
        } else {
            f64::INFINITY
        }
    });
    // Non-finite entries sort first, then by relative discrepancy.
    all.sort_by(
        |a, b| match (a.relative.is_finite(), b.relative.is_finite()) {
            (false, true) => std::cmp::Ordering::Less,
            (true, false) => std::cmp::Ordering::Greater,
            _ => b.relative.total_cmp(&a.relative),
        },
    );
    all.truncate(20);

    let has_any = caps.gradient || (caps.jacobian && m > 0);
    let inconclusive_reason = if !has_any {
        Some("the model supplies no analytic derivatives to check".to_string())
    } else if points_checked == 0 {
        Some(format!(
            "the objective or constraints could not be evaluated at any of the {} test point(s)",
            num_points.max(1)
        ))
    } else if comparisons == 0 {
        Some(format!(
            "no derivative could be compared: {derivative_failures} derivative callback(s) failed and \
             {objective_failures} model evaluation(s) failed"
        ))
    } else if derivative_failures > 0 {
        Some(format!(
            "{derivative_failures} derivative callback(s) failed, so part of the derivatives went unchecked"
        ))
    } else {
        None
    };

    Ok(CheckReport {
        worst: all,
        max_relative,
        tolerance,
        points_checked,
        evaluations,
        comparisons,
        nonfinite,
        derivative_failures,
        inconclusive_reason,
    })
}

fn make(row: usize, col: usize, analytic: f64, numerical: f64, point: usize) -> Discrepancy {
    let denom = analytic.abs().max(numerical.abs()).max(1.0);
    Discrepancy {
        row,
        col,
        analytic,
        numerical,
        relative: (analytic - numerical).abs() / denom,
        point,
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
    use mincon_core::{Capabilities, NlpDims};

    const INF: f64 = f64::INFINITY;

    struct Model {
        sabotage: bool,
        lb: Vec<f64>,
        ub: Vec<f64>,
        cl: Vec<f64>,
        cu: Vec<f64>,
        x0: Vec<f64>,
        pattern: Sparsity,
    }

    impl Model {
        fn new(sabotage: bool) -> Self {
            Self {
                sabotage,
                lb: vec![-INF; 3],
                ub: vec![INF; 3],
                cl: vec![0.0],
                cu: vec![0.0],
                x0: vec![0.7, -1.3, 2.1],
                pattern: Sparsity::from_triplets(1, 3, &[(0, 0), (0, 1), (0, 2)]).unwrap(),
            }
        }
    }

    impl Nlp for Model {
        fn dims(&self) -> NlpDims {
            NlpDims { n: 3, m: 1 }
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
            Capabilities::first_order()
        }
        fn objective(&self, x: &[f64]) -> Result<f64, EvalError> {
            Ok(x[0] * x[0] + 3.0 * x[1] * x[2] + x[2].powi(3))
        }
        fn gradient(&self, x: &[f64], out: &mut [f64]) -> Result<(), EvalError> {
            out[0] = 2.0 * x[0];
            out[1] = 3.0 * x[2];
            // The classic mistake: a dropped chain-rule factor.
            out[2] = if self.sabotage {
                3.0 * x[1] + x[2] * x[2]
            } else {
                3.0 * x[1] + 3.0 * x[2] * x[2]
            };
            Ok(())
        }
        fn constraints(&self, x: &[f64], out: &mut [f64]) -> Result<(), EvalError> {
            out[0] = x[0] * x[1] + x[2];
            Ok(())
        }
        fn jacobian_structure(&self) -> Option<&Sparsity> {
            Some(&self.pattern)
        }
        fn jacobian(&self, x: &[f64], out: &mut [f64]) -> Result<(), EvalError> {
            out[0] = x[1];
            out[1] = x[0];
            out[2] = 1.0;
            Ok(())
        }
    }

    #[test]
    fn correct_derivatives_pass() {
        let r = check_derivatives(&Model::new(false), 1e-5, 3).unwrap();
        assert!(r.passed(), "{}", r.message());
        assert_eq!(r.points_checked, 3);
        assert!(r.message().contains("passed"));
    }

    #[test]
    fn a_dropped_chain_rule_factor_is_caught_and_localized() {
        let r = check_derivatives(&Model::new(true), 1e-5, 3).unwrap();
        assert!(!r.passed());
        let worst = r.worst[0];
        assert_eq!(
            worst.row,
            usize::MAX,
            "the objective gradient is the culprit"
        );
        assert_eq!(worst.col, 2, "component 2 is the wrong one");
        assert!(r.message().contains("grad f"));
        assert!(r.message().contains("FAILED"));
    }

    #[test]
    fn a_model_with_no_analytic_derivatives_is_inconclusive_not_passing() {
        struct Bare {
            lb: Vec<f64>,
            ub: Vec<f64>,
            x0: Vec<f64>,
        }
        impl Nlp for Bare {
            fn dims(&self) -> NlpDims {
                NlpDims { n: 1, m: 0 }
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
                Ok(x[0] * x[0])
            }
            fn constraints(&self, _x: &[f64], _o: &mut [f64]) -> Result<(), EvalError> {
                Ok(())
            }
        }
        let r = check_derivatives(
            &Bare {
                lb: vec![-INF],
                ub: vec![INF],
                x0: vec![1.0],
            },
            1e-6,
            2,
        )
        .unwrap();
        assert!(
            !r.passed(),
            "nothing was compared, so nothing can have passed"
        );
        assert!(!r.conclusive());
        assert_eq!(r.comparisons, 0);
        assert!(r.message().contains("INCONCLUSIVE"), "{}", r.message());
    }

    /// A gradient callback that returns NaN in one component.
    struct NanGradient(Model);
    impl Nlp for NanGradient {
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
                gradient: true,
                ..Capabilities::none()
            }
        }
        fn objective(&self, x: &[f64]) -> Result<f64, EvalError> {
            self.0.objective(x)
        }
        fn gradient(&self, x: &[f64], out: &mut [f64]) -> Result<(), EvalError> {
            self.0.gradient(x, out)?;
            out[1] = f64::NAN;
            Ok(())
        }
        fn constraints(&self, x: &[f64], out: &mut [f64]) -> Result<(), EvalError> {
            self.0.constraints(x, out)
        }
    }

    #[test]
    fn a_nan_analytic_derivative_fails_the_check() {
        let r = check_derivatives(&NanGradient(Model::new(false)), 1e-5, 2).unwrap();
        assert!(!r.passed(), "{}", r.message());
        assert!(r.nonfinite > 0);
        assert!(r.max_relative.is_infinite());
        assert_eq!(
            r.worst[0].col, 1,
            "the non-finite entry must be reported first"
        );
    }

    /// A gradient callback that always fails.
    struct FailingGradient(Model);
    impl Nlp for FailingGradient {
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
                gradient: true,
                ..Capabilities::none()
            }
        }
        fn objective(&self, x: &[f64]) -> Result<f64, EvalError> {
            self.0.objective(x)
        }
        fn gradient(&self, _x: &[f64], _out: &mut [f64]) -> Result<(), EvalError> {
            Err(EvalError::Failed("gradient unavailable".into()))
        }
        fn constraints(&self, x: &[f64], out: &mut [f64]) -> Result<(), EvalError> {
            self.0.constraints(x, out)
        }
    }

    #[test]
    fn a_failing_derivative_callback_is_inconclusive_not_passing() {
        let r = check_derivatives(&FailingGradient(Model::new(false)), 1e-5, 2).unwrap();
        assert!(!r.passed());
        assert!(!r.conclusive());
        assert_eq!(r.derivative_failures, 2);
        assert_eq!(r.comparisons, 0);
    }

    /// The objective is NaN everywhere, so no point can be checked.
    struct NanObjective(Model);
    impl Nlp for NanObjective {
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
            Capabilities::first_order()
        }
        fn objective(&self, _x: &[f64]) -> Result<f64, EvalError> {
            Ok(f64::NAN)
        }
        fn gradient(&self, x: &[f64], out: &mut [f64]) -> Result<(), EvalError> {
            self.0.gradient(x, out)
        }
        fn constraints(&self, x: &[f64], out: &mut [f64]) -> Result<(), EvalError> {
            self.0.constraints(x, out)
        }
        fn jacobian_structure(&self) -> Option<&Sparsity> {
            self.0.jacobian_structure()
        }
        fn jacobian(&self, x: &[f64], out: &mut [f64]) -> Result<(), EvalError> {
            self.0.jacobian(x, out)
        }
    }

    #[test]
    fn no_evaluable_point_is_inconclusive_not_passing() {
        let r = check_derivatives(&NanObjective(Model::new(false)), 1e-5, 3).unwrap();
        assert!(!r.passed());
        assert_eq!(r.points_checked, 0);
        assert!(
            r.message().contains("could not be evaluated"),
            "{}",
            r.message()
        );
    }
}
