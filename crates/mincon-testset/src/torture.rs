//! Problems built to break solvers.
//!
//! The Hock–Schittkowski set measures whether the algorithm is correct. This
//! set measures whether the *implementation* is honest, which is a different
//! and in practice more important question. Everything here reproduces a
//! failure mode seen on real engineering models:
//!
//! | Failure mode | Problem |
//! |---|---|
//! | Objective and variables span many orders of magnitude | `TORTURE_SCALING` |
//! | Two variables twelve orders of magnitude apart, each mattering (D11) | `TORTURE_UNITS` |
//! | Model returns `NaN` outside its domain | `TORTURE_DOMAIN` |
//! | Model returns `NaN` in a region *inside* the feasible set | `TORTURE_NAN_POCKET` |
//! | Constraints are inconsistent | `TORTURE_INFEASIBLE` |
//! | Objective is unbounded below on the feasible set | `TORTURE_UNBOUNDED` |
//! | Jacobian is rank deficient at the solution | `TORTURE_DEGENERATE` |
//! | Constraints are redundant many times over | `TORTURE_REDUNDANT` |
//! | The feasible set is a single point | `TORTURE_PINNED` |
//! | Objective is flat to machine precision near the solution | `TORTURE_FLAT` |
//! | Derivatives are noisy, as from an inner solve | `TORTURE_NOISY` |
//!
//! # The pass criterion is not "converged"
//!
//! Three of these have no solution. For those, success is *reporting the right
//! thing*: [`Expect::Infeasible`] or [`Expect::Unbounded`]. A solver that
//! returns `ExitFlag::Optimal` on `TORTURE_INFEASIBLE` has done something far
//! worse than fail — it has lied, and a user will build on that answer.
//!
//! This is where `fmincon` is genuinely strong and where naive open-source
//! solvers lose: MATLAB's exit flag `-2` ("no feasible point found") is
//! trustworthy. Ours must be too.

use crate::{Expect, TestProblem, INF};

#[allow(clippy::too_many_arguments)]
fn p(
    name: &'static str,
    x0: &[f64],
    xl: &[f64],
    xu: &[f64],
    cl: &[f64],
    cu: &[f64],
    f: fn(&[f64]) -> f64,
    c: fn(&[f64], &mut [f64]),
    f_opt: Option<f64>,
    expect: Expect,
    notes: &'static str,
) -> TestProblem {
    TestProblem {
        name,
        n: x0.len(),
        m: cl.len(),
        x0: x0.to_vec(),
        xl: xl.to_vec(),
        xu: xu.to_vec(),
        cl: cl.to_vec(),
        cu: cu.to_vec(),
        f,
        c,
        f_opt,
        expect,
        notes,
    }
}

fn none(_x: &[f64], _c: &mut [f64]) {}

/// Every torture problem.
#[must_use]
pub fn all() -> Vec<TestProblem> {
    vec![
        p(
            "TORTURE_SCALING",
            &[1.0e-4, 1.0e4],
            &[-INF, -INF],
            &[INF, INF],
            &[0.0],
            &[0.0],
            // Variables differ by 1e8 in natural magnitude; the objective by 1e12.
            |x| 1.0e6 * (x[0] - 1.0e-4).powi(2) + 1.0e-6 * (x[1] - 1.0e4).powi(2),
            |x, c| c[0] = 1.0e4 * x[0] + 1.0e-4 * x[1] - 2.0,
            Some(0.0),
            Expect::LocalMinimum,
            "Objective and constraint gradients differ by twelve orders of magnitude. With \
             scaling off this is close to unsolvable; with gradient-based scaling on it is easy. \
             This single problem is the clearest demonstration of why our default differs from \
             fmincon's ScaleProblem = false.",
        ),
        p(
            "TORTURE_UNITS",
            &[1.0e6, 1.0e-6],
            &[1.0, 1.0e-9],
            &[INF, INF],
            &[-INF],
            &[0.0],
            // A pressure (1e6) and an area (1e-6), each mattering equally, coupled by
            // x0 x1 >= 5 (written 5 - x0 x1 <= 0). Convex; the Hessian's condition
            // number is 1e25. The optimum is on the row at x = (2.8998e6, 1.7243e-6).
            |x| (x[0] / 3.0e6 - 1.0).powi(2) + (x[1] / 1.0e-6 - 1.0).powi(2),
            |x, c| c[0] = 5.0 - x[0] * x[1],
            Some(0.169_355_538_390_110_78),
            Expect::NoFalseCertificate,
            "The friction audit's bad_scaling problem (bench/results/s7-friction). Every solver              tested misses the optimum from this start; the fixture asserts that the answer is              not certified: the interior-point member once reported Optimal at f = 120.7 because              its start push moved x1 to 1e-2, the objective factor (5e-9) was chosen there, and              the scaled complementarity let a bound 1e-5 away carry a multiplier of 8e6 (D11).",
        ),
        p(
            "TORTURE_DOMAIN",
            &[0.5, 0.5],
            &[1e-8, 1e-8],
            &[INF, INF],
            &[0.0],
            &[INF],
            // log and sqrt: NaN the instant a step leaves the box.
            |x| -x[0].ln() - x[1].sqrt() + x[0] * x[0],
            |x, c| c[0] = 4.0 - x[0] - x[1],
            None,
            Expect::LocalMinimum,
            "Objective is NaN for any non-positive variable. Bounds-respecting finite differences \
             and a bound-honouring line search are both mandatory; without either, this fails on \
             the first iteration.",
        ),
        p(
            "TORTURE_NAN_POCKET",
            &[3.0, 3.0],
            &[-10.0, -10.0],
            &[10.0, 10.0],
            &[],
            &[],
            // A disc of undefined behaviour strictly inside the feasible box,
            // between the start and the minimum. Simulations that fail to
            // converge behave exactly like this.
            |x| {
                let r2 = (x[0] - 1.5) * (x[0] - 1.5) + (x[1] - 1.5) * (x[1] - 1.5);
                if r2 < 0.25 {
                    f64::NAN
                } else {
                    (x[0] - 1.0).powi(2) + (x[1] - 1.0).powi(2)
                }
            },
            none,
            Some(0.0),
            Expect::LocalMinimum,
            "A hole in the objective's domain that is NOT at a bound, lying directly between the \
             start and the minimum. The only survivable response is to shorten the step and try \
             again - which is precisely what fmincon's sqp algorithm does and what most solvers \
             do not. Reaching any finite stationary point counts as a pass.",
        ),
        p(
            "TORTURE_INFEASIBLE",
            &[0.0, 0.0],
            &[-INF, -INF],
            &[INF, INF],
            &[0.0, 0.0],
            &[0.0, 0.0],
            |x| x[0] * x[0] + x[1] * x[1],
            |x, c| {
                // x0 + x1 = 1 and x0 + x1 = 3 cannot both hold.
                c[0] = x[0] + x[1] - 1.0;
                c[1] = x[0] + x[1] - 3.0;
            },
            None,
            Expect::Infeasible,
            "Two inconsistent linear equalities. The Jacobian is rank deficient everywhere, so \
             delta_c is exercised on every iteration. Correct behaviour is to converge to the \
             least-squares point and report infeasibility, NOT to report success.",
        ),
        p(
            "TORTURE_UNBOUNDED",
            &[1.0, 1.0],
            &[-INF, -INF],
            &[INF, INF],
            &[0.0],
            &[INF],
            |x| -x[0] - x[1],
            |x, c| c[0] = x[1] - x[0],
            None,
            Expect::Unbounded,
            "The objective decreases without bound along a feasible ray. Correct behaviour is to \
             report unboundedness rather than to iterate until the budget runs out.",
        ),
        p(
            "TORTURE_DEGENERATE",
            &[1.0, 1.0],
            &[-INF, 0.0],
            &[INF, INF],
            &[0.0, 0.0],
            &[INF, INF],
            |x| x[0] + x[1],
            |x, c| {
                // Both constraints are active at the origin and their gradients
                // are parallel there, so LICQ fails at the solution.
                c[0] = x[0];
                c[1] = x[0] * 2.0;
            },
            Some(0.0),
            Expect::LocalMinimum,
            "LICQ fails at the solution (0, 0): the two active constraint gradients are (1, 0) \
             and (2, 0), linearly dependent, so the multipliers are not unique and can diverge. \
             A solver whose termination test is unscaled will never satisfy it.\n\
             NOTE: an earlier version of this problem left x2 unbounded below, which made it \
             genuinely unbounded rather than degenerate. The solver caught that by correctly \
             reporting ExitFlag::Unbounded, which is why the lower bound on x2 is here. Torture \
             problems need the same scrutiny as the solver.",
        ),
        p(
            "TORTURE_REDUNDANT",
            &[2.0, 2.0],
            &[-INF, -INF],
            &[INF, INF],
            &[0.0; 20],
            &[INF; 20],
            |x| (x[0] - 1.0).powi(2) + (x[1] - 1.0).powi(2),
            |x, c| {
                // Twenty copies of the same half-plane, scaled differently.
                for (k, ci) in c.iter_mut().enumerate() {
                    let s = 10f64.powi(k as i32 % 7 - 3);
                    *ci = s * (x[0] + x[1] - 1.0);
                }
            },
            Some(0.0),
            Expect::LocalMinimum,
            "Twenty structurally identical constraints at wildly different scales. The KKT matrix \
             is rank deficient by 19; without dual regularization the factorization fails, and \
             without row scaling the conditioning is hopeless.",
        ),
        p(
            "TORTURE_PINNED",
            &[5.0, 5.0],
            &[2.0, 3.0],
            &[2.0, 3.0],
            &[],
            &[],
            |x| (x[0] - 1.0).powi(2) + (x[1] - 1.0).powi(2),
            none,
            Some(5.0),
            Expect::Optimum,
            "Every variable is fixed by lb == ub, so the barrier interval is empty before bound \
             relaxation. The answer is forced: x = (2, 3), f = 5. A solver that divides by the \
             interval width produces inf here.",
        ),
        p(
            "TORTURE_FLAT",
            &[1.0, 1.0],
            &[-INF, -INF],
            &[INF, INF],
            &[],
            &[],
            // Eighth power: the gradient is below machine epsilon well before x
            // is anywhere near the minimum.
            |x| (x[0] - 0.3).powi(8) + (x[1] + 0.7).powi(8),
            none,
            Some(0.0),
            Expect::LocalMinimum,
            "The objective is flat to machine precision over a wide neighbourhood of the \
             minimum, so finite-difference gradients are pure noise there. Any exit that reports \
             a small gradient is defensible; reporting a tiny objective is not enough on its own.",
        ),
        p(
            "TORTURE_NOISY",
            &[2.0, 2.0],
            &[-INF, -INF],
            &[INF, INF],
            &[0.0],
            &[INF],
            // Deterministic but high-frequency perturbation, the signature of an
            // objective computed by an inner iterative solve with a loose tolerance.
            |x| {
                let base = (x[0] - 1.0).powi(2) + (x[1] - 1.0).powi(2);
                let noise = 1e-9 * ((1e6 * x[0]).sin() + (1e6 * x[1]).sin());
                base + noise
            },
            |x, c| c[0] = x[0] + x[1] - 1.0,
            Some(0.0),
            Expect::LocalMinimum,
            "Objective carries 1e-9 of deterministic noise, so forward differences with a \
             sqrt(eps) step return roughly 1e-1 of relative error in the gradient. The adaptive \
             switch to central differences is what makes this solvable; the achievable optimality \
             tolerance is bounded below by the noise, so demanding 1e-8 here is a specification \
             error, not a solver failure.",
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get(name: &str) -> TestProblem {
        all().into_iter().find(|p| p.name == name).unwrap()
    }

    #[test]
    fn infeasible_problem_really_is_infeasible() {
        let p = get("TORTURE_INFEASIBLE");
        // No point can satisfy both; the best possible violation is 1.
        for x in [[0.0, 0.0], [1.0, 0.0], [0.0, 3.0], [1.5, 1.5], [-4.0, 9.0]] {
            assert!(
                p.violation(&x) >= 0.999,
                "found a nearly feasible point {x:?} with violation {}",
                p.violation(&x)
            );
        }
    }

    #[test]
    fn unbounded_problem_really_is_unbounded() {
        let p = get("TORTURE_UNBOUNDED");
        let mut last = f64::INFINITY;
        for t in [1.0, 10.0, 100.0, 1000.0] {
            let x = [t, t];
            assert!(p.violation(&x) < 1e-12, "ray must stay feasible");
            let f = (p.f)(&x);
            assert!(f < last, "objective must keep decreasing along the ray");
            last = f;
        }
        assert!(last < -1000.0);
    }

    #[test]
    fn nan_pocket_is_where_we_claim_it_is() {
        let p = get("TORTURE_NAN_POCKET");
        assert!((p.f)(&[1.5, 1.5]).is_nan(), "centre of the pocket");
        assert!((p.f)(&[1.0, 1.0]).is_finite(), "the minimum itself");
        assert!((p.f)(&[3.0, 3.0]).is_finite(), "the starting point");
        // The straight line from x0 to the minimum passes through the hole.
        let mut hit = false;
        for k in 0..=100 {
            let t = f64::from(k) / 100.0;
            let x = [3.0 + t * (1.0 - 3.0), 3.0 + t * (1.0 - 3.0)];
            if (p.f)(&x).is_nan() {
                hit = true;
            }
        }
        assert!(hit, "the direct path must cross the undefined region");
    }

    #[test]
    fn pinned_problem_has_exactly_one_feasible_point() {
        let p = get("TORTURE_PINNED");
        assert!(p.violation(&[2.0, 3.0]) < 1e-15);
        assert!(p.violation(&[2.0, 3.1]) > 0.0);
        assert!(((p.f)(&[2.0, 3.0]) - 5.0).abs() < 1e-15);
    }

    #[test]
    fn scaling_problem_spans_the_advertised_range() {
        let p = get("TORTURE_SCALING");
        // Numerically differentiate to show the gradient range at x0.
        let x = p.x0.clone();
        let h = 1e-9;
        let mut g = [0.0; 2];
        for j in 0..2 {
            let mut xp = x.clone();
            xp[j] += h * x[j].abs().max(1.0);
            g[j] = ((p.f)(&xp) - (p.f)(&x)) / (h * x[j].abs().max(1.0));
        }
        let _ = g;
        // The constraint row alone spans 1e8 between its two coefficients.
        let mut c1 = [0.0];
        let mut c2 = [0.0];
        (p.c)(&[1.0, 0.0], &mut c1);
        (p.c)(&[0.0, 1.0], &mut c2);
        let ratio = (c1[0] + 2.0).abs() / (c2[0] + 2.0).abs();
        assert!(ratio > 1e7, "constraint coefficient ratio is only {ratio}");
    }

    #[test]
    fn redundant_problem_has_the_advertised_rank_deficiency() {
        let p = get("TORTURE_REDUNDANT");
        let mut c = vec![0.0; p.m];
        (p.c)(&[2.0, 3.0], &mut c);
        // Every row is a multiple of the same expression, value 4 here.
        for (k, ci) in c.iter().enumerate() {
            let s = 10f64.powi(k as i32 % 7 - 3);
            assert!((ci - s * 4.0).abs() < 1e-12);
        }
    }

    #[test]
    fn noisy_problem_noise_is_deterministic() {
        let p = get("TORTURE_NOISY");
        let x = [1.234, -0.567];
        assert_eq!((p.f)(&x), (p.f)(&x), "the model must be reproducible");
    }
}
