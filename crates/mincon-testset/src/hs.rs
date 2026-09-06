//! Problems from Hock and Schittkowski (1981).
//!
//! Constraints in the book are written `g(x) >= 0` for inequalities and
//! `h(x) = 0` for equalities. Here they become `c_L <= c(x) <= c_U` with
//! `(0, +inf)` and `(0, 0)` respectively, so the transcription is mechanical
//! and checkable against the book by eye.
//!
//! Every problem records the published optimal objective. Where a problem is
//! known to be degenerate, to have a better local minimum than the published
//! one, or to be reported inconsistently across sources, the `notes` field says
//! so — a benchmark whose reference values are wrong is worse than no benchmark.

use crate::{Expect, TestProblem, INF};

use std::f64::consts::PI;

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

fn rosen(x: &[f64]) -> f64 {
    100.0 * (x[1] - x[0] * x[0]).powi(2) + (1.0 - x[0]).powi(2)
}

fn no_constraints(_x: &[f64], _c: &mut [f64]) {}

/// Every Hock–Schittkowski problem in the crate.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn all() -> Vec<TestProblem> {
    let mut v = Vec::new();

    // ---- bound-constrained ----

    v.push(p(
        "HS1",
        &[-2.0, 1.0],
        &[-INF, -1.5],
        &[INF, INF],
        &[],
        &[],
        rosen,
        no_constraints,
        Some(0.0),
        Expect::Optimum,
        "Rosenbrock with an inactive lower bound. Tests the banana valley, not the constraint machinery.",
    ));

    v.push(p(
        "HS2",
        &[-2.0, 1.0],
        &[-INF, 1.5],
        &[INF, INF],
        &[],
        &[],
        rosen,
        no_constraints,
        Some(0.050_426_187_9),
        Expect::Optimum,
        "Rosenbrock with an ACTIVE lower bound at the solution. x* = (1.224370749, 1.5).",
    ));

    v.push(p(
        "HS3",
        &[10.0, 1.0],
        &[-INF, 0.0],
        &[INF, INF],
        &[],
        &[],
        |x| x[1] + 1e-5 * (x[1] - x[0]).powi(2),
        no_constraints,
        Some(0.0),
        Expect::Optimum,
        "Nearly flat in x0: the objective changes by 1e-5 over a unit step, so a solver with a \
         poorly scaled optimality test declares victory far from the solution.",
    ));

    v.push(p(
        "HS4",
        &[1.125, 0.125],
        &[1.0, 0.0],
        &[INF, INF],
        &[],
        &[],
        |x| (x[0] + 1.0).powi(3) / 3.0 + x[1],
        no_constraints,
        Some(8.0 / 3.0),
        Expect::Optimum,
        "Both bounds active at x* = (1, 0).",
    ));

    v.push(p(
        "HS5",
        &[0.0, 0.0],
        &[-1.5, -3.0],
        &[4.0, 3.0],
        &[],
        &[],
        |x| (x[0] + x[1]).sin() + (x[0] - x[1]).powi(2) - 1.5 * x[0] + 2.5 * x[1] + 1.0,
        no_constraints,
        Some(-1.913_222_954_981_0),
        Expect::Optimum,
        "Multiple local minima inside the box; x* = (-pi/3 + 1/2, -pi/3 - 1/2).",
    ));

    v.push(p(
        "HS38",
        &[-3.0, -1.0, -3.0, -1.0],
        &[-10.0; 4],
        &[10.0; 4],
        &[],
        &[],
        |x| {
            100.0 * (x[1] - x[0] * x[0]).powi(2)
                + (1.0 - x[0]).powi(2)
                + 90.0 * (x[3] - x[2] * x[2]).powi(2)
                + (1.0 - x[2]).powi(2)
                + 10.1 * ((x[1] - 1.0).powi(2) + (x[3] - 1.0).powi(2))
                + 19.8 * (x[1] - 1.0) * (x[3] - 1.0)
        },
        no_constraints,
        Some(0.0),
        Expect::Optimum,
        "Colville's function. Long curved valley; a good test of the quasi-Newton update.",
    ));

    v.push(p(
        "HS45",
        &[2.0; 5],
        &[0.0; 5],
        &[1.0, 2.0, 3.0, 4.0, 5.0],
        &[],
        &[],
        |x| 2.0 - x[0] * x[1] * x[2] * x[3] * x[4] / 120.0,
        no_constraints,
        Some(1.0),
        Expect::Optimum,
        "Every bound active at x* = (1,2,3,4,5). Starts infeasible with respect to the first bound.",
    ));

    v.push(p(
        "HS110",
        &[9.0; 10],
        &[2.001; 10],
        &[9.999; 10],
        &[],
        &[],
        |x| {
            let mut s = 0.0;
            let mut prod = 1.0;
            for &xi in x {
                // Undefined outside the box: the whole point of this problem.
                s += (xi - 2.0).ln().powi(2) + (10.0 - xi).ln().powi(2);
                prod *= xi;
            }
            s - prod.powf(0.2)
        },
        no_constraints,
        Some(-45.778_470_4),
        Expect::Optimum,
        "The objective is NaN outside the bounds. A solver that finite-differences across a \
         bound, or lets a line search leave the box, fails immediately. fmincon's HonorBounds \
         exists for problems like this.",
    ));

    // ---- equality constrained ----

    v.push(p(
        "HS6",
        &[-1.2, 1.0],
        &[-INF; 2],
        &[INF; 2],
        &[0.0],
        &[0.0],
        |x| (1.0 - x[0]).powi(2),
        |x, c| c[0] = 10.0 * (x[1] - x[0] * x[0]),
        Some(0.0),
        Expect::Optimum,
        "The Hessian of the objective is singular; convergence relies on the constraint curvature.",
    ));

    v.push(p(
        "HS7",
        &[2.0, 2.0],
        &[-INF; 2],
        &[INF; 2],
        &[0.0],
        &[0.0],
        |x| (1.0 + x[0] * x[0]).ln() - x[1],
        |x, c| c[0] = (1.0 + x[0] * x[0]).powi(2) + x[1] * x[1] - 4.0,
        Some(-1.732_050_807_568_877_2),
        Expect::Optimum,
        "f* = -sqrt(3). The starting point is infeasible.",
    ));

    v.push(p(
        "HS8",
        &[2.0, 1.0],
        &[-INF; 2],
        &[INF; 2],
        &[0.0, 0.0],
        &[0.0, 0.0],
        |_x| -1.0,
        |x, c| {
            c[0] = x[0] * x[0] + x[1] * x[1] - 25.0;
            c[1] = x[0] * x[1] - 9.0;
        },
        Some(-1.0),
        Expect::Optimum,
        "Constant objective: a pure feasibility problem wearing an optimization costume. Any \
         solver relying on objective decrease to make progress will stall.",
    ));

    v.push(p(
        "HS9",
        &[0.0, 0.0],
        &[-INF; 2],
        &[INF; 2],
        &[0.0],
        &[0.0],
        |x| (PI * x[0] / 12.0).sin() * (PI * x[1] / 16.0).cos(),
        |x, c| c[0] = 4.0 * x[0] - 3.0 * x[1],
        Some(-0.5),
        Expect::Optimum,
        "Infinitely many global minima along the constraint line; x0 is a stationary point of \
         the objective, so the first step must come from the constraint.",
    ));

    v.push(p(
        "HS26",
        &[-2.6, 2.0, 2.0],
        &[-INF; 3],
        &[INF; 3],
        &[0.0],
        &[0.0],
        |x| (x[0] - x[1]).powi(2) + (x[1] - x[2]).powi(4),
        |x, c| c[0] = (1.0 + x[1] * x[1]) * x[0] + x[2].powi(4) - 3.0,
        Some(0.0),
        Expect::Optimum,
        "The solution set is a curve, so the Jacobian is rank deficient there. Tests delta_c.",
    ));

    v.push(p(
        "HS27",
        &[2.0, 2.0, 2.0],
        &[-INF; 3],
        &[INF; 3],
        &[0.0],
        &[0.0],
        |x| 0.01 * (x[0] - 1.0).powi(2) + (x[1] - x[0] * x[0]).powi(2),
        |x, c| c[0] = x[0] + x[2] * x[2] + 1.0,
        Some(0.04),
        Expect::Optimum,
        "x* = (-1, 1, 0).",
    ));

    v.push(p(
        "HS28",
        &[-4.0, 1.0, 1.0],
        &[-INF; 3],
        &[INF; 3],
        &[0.0],
        &[0.0],
        |x| (x[0] + x[1]).powi(2) + (x[1] + x[2]).powi(2),
        |x, c| c[0] = x[0] + 2.0 * x[1] + 3.0 * x[2] - 1.0,
        Some(0.0),
        Expect::Optimum,
        "Convex quadratic with one linear equality: should take a single Newton step.",
    ));

    v.push(p(
        "HS39",
        &[2.0; 4],
        &[-INF; 4],
        &[INF; 4],
        &[0.0, 0.0],
        &[0.0, 0.0],
        |x| -x[0],
        |x, c| {
            c[0] = x[1] - x[0].powi(3) - x[2] * x[2];
            c[1] = x[0] * x[0] - x[1] - x[3] * x[3];
        },
        Some(-1.0),
        Expect::Optimum,
        "Linear objective, so all curvature comes from the constraints.",
    ));

    v.push(p(
        "HS40",
        &[0.8; 4],
        &[-INF; 4],
        &[INF; 4],
        &[0.0; 3],
        &[0.0; 3],
        |x| -x[0] * x[1] * x[2] * x[3],
        |x, c| {
            c[0] = x[0].powi(3) + x[1] * x[1] - 1.0;
            c[1] = x[0] * x[0] * x[3] - x[2];
            c[2] = x[3] * x[3] - x[1];
        },
        Some(-0.25),
        Expect::Optimum,
        "Three equalities in four variables.",
    ));

    v.push(p(
        "HS42",
        &[1.0; 4],
        &[-INF; 4],
        &[INF; 4],
        &[0.0, 0.0],
        &[0.0, 0.0],
        |x| {
            (x[0] - 1.0).powi(2)
                + (x[1] - 2.0).powi(2)
                + (x[2] - 3.0).powi(2)
                + (x[3] - 4.0).powi(2)
        },
        |x, c| {
            c[0] = x[0] - 2.0;
            c[1] = x[2] * x[2] + x[3] * x[3] - 2.0;
        },
        Some(28.0 - 10.0 * std::f64::consts::SQRT_2),
        Expect::Optimum,
        "f* = 28 - 10*sqrt(2). One linear and one nonlinear equality.",
    ));

    // ---- inequality constrained ----

    v.push(p(
        "HS10",
        &[-10.0, 10.0],
        &[-INF; 2],
        &[INF; 2],
        &[0.0],
        &[INF],
        |x| x[0] - x[1],
        |x, c| c[0] = -3.0 * x[0] * x[0] + 2.0 * x[0] * x[1] - x[1] * x[1] + 1.0,
        Some(-1.0),
        Expect::Optimum,
        "Starts badly infeasible (g(x0) = -501).",
    ));

    v.push(p(
        "HS11",
        &[4.9, 0.1],
        &[-INF; 2],
        &[INF; 2],
        &[0.0],
        &[INF],
        |x| (x[0] - 5.0).powi(2) + x[1] * x[1] - 25.0,
        |x, c| c[0] = -x[0] * x[0] + x[1],
        Some(-8.498_464_223_2),
        Expect::Optimum,
        "",
    ));

    v.push(p(
        "HS12",
        &[0.0, 0.0],
        &[-INF; 2],
        &[INF; 2],
        &[0.0],
        &[INF],
        |x| 0.5 * x[0] * x[0] + x[1] * x[1] - x[0] * x[1] - 7.0 * x[0] - 7.0 * x[1],
        |x, c| c[0] = 25.0 - 4.0 * x[0] * x[0] - x[1] * x[1],
        Some(-30.0),
        Expect::Optimum,
        "x* = (2, 3).",
    ));

    v.push(p(
        "HS13",
        &[-2.0, -2.0],
        &[0.0, 0.0],
        &[INF, INF],
        &[0.0],
        &[INF],
        |x| (x[0] - 2.0).powi(2) + x[1] * x[1],
        |x, c| c[0] = (1.0 - x[0]).powi(3) - x[1],
        Some(1.0),
        Expect::DegenerateOptimum,
        "DEGENERATE: the Mangasarian-Fromovitz constraint qualification fails at x* = (1, 0), so \
         the KKT conditions do not hold there and no first-order method can certify optimality. \
         The correct behaviour is to reach x* and report a step-size or acceptable-point exit, \
         NOT ExitFlag::Optimal. Counting this as a failure is a benchmarking error.",
    ));

    v.push(p(
        "HS14",
        &[2.0, 2.0],
        &[-INF; 2],
        &[INF; 2],
        &[0.0, 0.0],
        &[0.0, INF],
        |x| (x[0] - 2.0).powi(2) + (x[1] - 1.0).powi(2),
        |x, c| {
            c[0] = x[0] - 2.0 * x[1] + 1.0;
            c[1] = -0.25 * x[0] * x[0] - x[1] * x[1] + 1.0;
        },
        Some(1.393_464_620_9),
        Expect::Optimum,
        "One equality and one inequality: exercises the mixed path.",
    ));

    v.push(p(
        "HS15",
        &[-2.0, 1.0],
        &[-INF, -INF],
        &[0.5, INF],
        &[0.0, 0.0],
        &[INF, INF],
        rosen,
        |x, c| {
            c[0] = x[0] * x[1] - 1.0;
            c[1] = x[0] + x[1] * x[1];
        },
        Some(306.5),
        Expect::Optimum,
        "The feasible region has two disconnected components; the global optimum is in the one \
         NOT containing x0 under a naive step. f* = 306.5 at x* = (0.5, 2).",
    ));

    v.push(p(
        "HS16",
        &[-2.0, 1.0],
        &[-2.0, -INF],
        &[0.5, 1.0],
        &[0.0, 0.0],
        &[INF, INF],
        rosen,
        |x, c| {
            c[0] = x[0] + x[1] * x[1];
            c[1] = x[0] * x[0] + x[1];
        },
        Some(0.25),
        Expect::LocalMinimum,
        "MULTIPLE LOCAL MINIMA. The published global optimum is f* = 0.25 at x* = (0.5, 0.25), but \
         a local method started from the published x0 = (-2, 1) converges instead to \
         x = (-0.99097, 0.99547), f = 3.98206, where g1 = x1 + x2^2 is active. That point was \
         verified to be a genuine local minimum by exhaustive search of its feasible \
         neighbourhood, and it satisfies the KKT conditions to 1e-8. Scoring a local solver \
         against 0.25 here measures luck, not quality, so this is classified LocalMinimum.",
    ));

    v.push(p(
        "HS18",
        &[2.0, 2.0],
        &[2.0, 0.0],
        &[50.0, 50.0],
        &[0.0, 0.0],
        &[INF, INF],
        |x| 0.01 * x[0] * x[0] + x[1] * x[1],
        |x, c| {
            c[0] = x[0] * x[1] - 25.0;
            c[1] = x[0] * x[0] + x[1] * x[1] - 25.0;
        },
        Some(5.0),
        Expect::Optimum,
        "Starts infeasible; the objective is 100x worse conditioned in x0 than in x1.",
    ));

    v.push(p(
        "HS21",
        &[-1.0, -1.0],
        &[2.0, -50.0],
        &[50.0, 50.0],
        &[0.0],
        &[INF],
        |x| 0.01 * x[0] * x[0] + x[1] * x[1] - 100.0,
        |x, c| c[0] = 10.0 * x[0] - x[1] - 10.0,
        Some(-99.96),
        Expect::Optimum,
        "x0 violates its own lower bound, which every solver must repair before the first step.",
    ));

    v.push(p(
        "HS22",
        &[2.0, 2.0],
        &[-INF; 2],
        &[INF; 2],
        &[0.0, 0.0],
        &[INF, INF],
        |x| (x[0] - 2.0).powi(2) + (x[1] - 1.0).powi(2),
        |x, c| {
            c[0] = -x[0] - x[1] + 2.0;
            c[1] = -x[0] * x[0] + x[1];
        },
        Some(1.0),
        Expect::Optimum,
        "Both constraints active at x* = (1, 1).",
    ));

    v.push(p(
        "HS23",
        &[3.0, 1.0],
        &[-50.0, -50.0],
        &[50.0, 50.0],
        &[0.0; 5],
        &[INF; 5],
        |x| x[0] * x[0] + x[1] * x[1],
        |x, c| {
            c[0] = x[0] + x[1] - 1.0;
            c[1] = x[0] * x[0] + x[1] * x[1] - 1.0;
            c[2] = 9.0 * x[0] * x[0] + x[1] * x[1] - 9.0;
            c[3] = x[0] * x[0] - x[1];
            c[4] = x[1] * x[1] - x[0];
        },
        Some(2.0),
        Expect::Optimum,
        "Five inequalities in two variables: heavily redundant, so the active set is ambiguous.",
    ));

    v.push(p(
        "HS24",
        &[1.0, 0.5],
        &[0.0, 0.0],
        &[INF, INF],
        &[0.0; 3],
        &[INF; 3],
        |x| ((x[0] - 3.0).powi(2) - 9.0) * x[1].powi(3) / (27.0 * 3.0_f64.sqrt()),
        |x, c| {
            let r3 = 3.0_f64.sqrt();
            c[0] = x[0] / r3 - x[1];
            c[1] = x[0] + r3 * x[1];
            c[2] = -x[0] - r3 * x[1] + 6.0;
        },
        Some(-1.0),
        Expect::Optimum,
        "x* = (3, sqrt(3)).",
    ));

    v.push(p(
        "HS29",
        &[1.0, 1.0, 1.0],
        &[-INF; 3],
        &[INF; 3],
        &[0.0],
        &[INF],
        |x| -x[0] * x[1] * x[2],
        |x, c| c[0] = -x[0] * x[0] - 2.0 * x[1] * x[1] - 4.0 * x[2] * x[2] + 48.0,
        Some(-16.0 * std::f64::consts::SQRT_2),
        Expect::Optimum,
        "f* = -16*sqrt(2). Four symmetric global minima; sign conventions matter.",
    ));

    v.push(p(
        "HS30",
        &[1.0, 1.0, 1.0],
        &[1.0, -10.0, -10.0],
        &[10.0, 10.0, 10.0],
        &[0.0],
        &[INF],
        |x| x[0] * x[0] + x[1] * x[1] + x[2] * x[2],
        |x, c| c[0] = x[0] * x[0] + x[1] * x[1] - 1.0,
        Some(1.0),
        Expect::Optimum,
        "x* = (1, 0, 0); the nonlinear constraint is active and the bound on x0 is too.",
    ));

    v.push(p(
        "HS31",
        &[1.0, 1.0, 1.0],
        &[-10.0, 1.0, -10.0],
        &[10.0, 10.0, 1.0],
        &[0.0],
        &[INF],
        |x| 9.0 * x[0] * x[0] + x[1] * x[1] + 9.0 * x[2] * x[2],
        |x, c| c[0] = x[0] * x[1] - 1.0,
        Some(6.0),
        Expect::Optimum,
        "",
    ));

    v.push(p(
        "HS32",
        &[0.1, 0.7, 0.2],
        &[0.0; 3],
        &[INF; 3],
        &[0.0, 0.0],
        &[INF, 0.0],
        |x| (x[0] + 3.0 * x[1] + x[2]).powi(2) + 4.0 * (x[0] - x[1]).powi(2),
        |x, c| {
            c[0] = 6.0 * x[1] + 4.0 * x[2] - x[0].powi(3) - 3.0;
            c[1] = 1.0 - x[0] - x[1] - x[2];
        },
        Some(1.0),
        Expect::Optimum,
        "x* = (0, 0, 1).",
    ));

    v.push(p(
        "HS33",
        &[0.0, 0.0, 3.0],
        &[0.0, 0.0, 0.0],
        &[INF, INF, 5.0],
        &[0.0, 0.0],
        &[INF, INF],
        |x| (x[0] - 1.0) * (x[0] - 2.0) * (x[0] - 3.0) + x[2],
        |x, c| {
            c[0] = x[2] * x[2] - x[1] * x[1] - x[0] * x[0];
            c[1] = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] - 4.0;
        },
        Some(std::f64::consts::SQRT_2 - 6.0),
        Expect::Optimum,
        "f* = sqrt(2) - 6. Nonconvex objective in x0.",
    ));

    v.push(p(
        "HS34",
        &[0.0, 1.05, 2.9],
        &[0.0, 0.0, 0.0],
        &[100.0, 100.0, 10.0],
        &[0.0, 0.0],
        &[INF, INF],
        |x| -x[0],
        |x, c| {
            c[0] = x[1] - x[0].exp();
            c[1] = x[2] - x[1].exp();
        },
        Some(-0.834_032_445_9),
        Expect::Optimum,
        "Nested exponentials overflow readily: exp(exp(x)) is inf for x above about 4.6, and the \
         upper bound of 100 on x0 lets a careless step go there. Tests non-finite handling.",
    ));

    v.push(p(
        "HS35",
        &[0.5, 0.5, 0.5],
        &[0.0; 3],
        &[INF; 3],
        &[0.0],
        &[INF],
        |x| {
            9.0 - 8.0 * x[0] - 6.0 * x[1] - 4.0 * x[2]
                + 2.0 * x[0] * x[0]
                + 2.0 * x[1] * x[1]
                + x[2] * x[2]
                + 2.0 * x[0] * x[1]
                + 2.0 * x[0] * x[2]
        },
        |x, c| c[0] = 3.0 - x[0] - x[1] - 2.0 * x[2],
        Some(1.0 / 9.0),
        Expect::Optimum,
        "Beale's problem. Convex QP; f* = 1/9.",
    ));

    v.push(p(
        "HS36",
        &[10.0, 10.0, 10.0],
        &[0.0; 3],
        &[20.0, 11.0, 42.0],
        &[0.0],
        &[INF],
        |x| -x[0] * x[1] * x[2],
        |x, c| c[0] = 72.0 - x[0] - 2.0 * x[1] - 2.0 * x[2],
        Some(-3300.0),
        Expect::Optimum,
        "x* = (20, 11, 15).",
    ));

    v.push(p(
        "HS37",
        &[10.0, 10.0, 10.0],
        &[0.0; 3],
        &[42.0; 3],
        &[0.0, 0.0],
        &[INF, INF],
        |x| -x[0] * x[1] * x[2],
        |x, c| {
            c[0] = 72.0 - x[0] - 2.0 * x[1] - 2.0 * x[2];
            c[1] = x[0] + 2.0 * x[1] + 2.0 * x[2];
        },
        Some(-3456.0),
        Expect::Optimum,
        "x* = (24, 12, 12).",
    ));

    v.push(p(
        "HS41",
        &[2.0; 4],
        &[0.0; 4],
        &[1.0, 1.0, 1.0, 2.0],
        &[0.0],
        &[0.0],
        |x| 2.0 - x[0] * x[1] * x[2],
        |x, c| c[0] = x[0] + 2.0 * x[1] + 2.0 * x[2] - x[3],
        Some(52.0 / 27.0),
        Expect::Optimum,
        "x0 violates every upper bound. f* = 52/27.",
    ));

    v.push(p(
        "HS43",
        &[0.0; 4],
        &[-INF; 4],
        &[INF; 4],
        &[0.0; 3],
        &[INF; 3],
        |x| {
            x[0] * x[0] + x[1] * x[1] + 2.0 * x[2] * x[2] + x[3] * x[3]
                - 5.0 * x[0]
                - 5.0 * x[1]
                - 21.0 * x[2]
                + 7.0 * x[3]
        },
        |x, c| {
            c[0] = 8.0 - x[0] * x[0] - x[1] * x[1] - x[2] * x[2] - x[3] * x[3] - x[0] + x[1] - x[2]
                + x[3];
            c[1] = 10.0 - x[0] * x[0] - 2.0 * x[1] * x[1] - x[2] * x[2] - 2.0 * x[3] * x[3]
                + x[0]
                + x[3];
            c[2] = 5.0 - 2.0 * x[0] * x[0] - x[1] * x[1] - x[2] * x[2] - 2.0 * x[0] + x[1] + x[3];
        },
        Some(-44.0),
        Expect::Optimum,
        "Rosen-Suzuki, the standard SQP benchmark. x* = (0, 1, 2, -1).",
    ));

    v.push(p(
        "HS44",
        &[0.0; 4],
        &[0.0; 4],
        &[INF; 4],
        &[0.0; 6],
        &[INF; 6],
        |x| x[0] - x[1] - x[2] - x[0] * x[2] + x[0] * x[3] + x[1] * x[2] - x[1] * x[3],
        |x, c| {
            c[0] = 8.0 - x[0] - 2.0 * x[1];
            c[1] = 12.0 - 4.0 * x[0] - x[1];
            c[2] = 12.0 - 3.0 * x[0] - 4.0 * x[1];
            c[3] = 8.0 - 2.0 * x[2] - x[3];
            c[4] = 8.0 - x[2] - 2.0 * x[3];
            c[5] = 5.0 - x[2] - x[3];
        },
        Some(-15.0),
        Expect::Optimum,
        "Bilinear objective over a polytope; the published minimum is at a vertex. Local methods \
         starting from the origin sometimes stop at f = -13, which is a genuine local minimum.",
    ));

    v.push(p(
        "HS71",
        &[1.0, 5.0, 5.0, 1.0],
        &[1.0; 4],
        &[5.0; 4],
        &[0.0, 40.0],
        &[INF, 40.0],
        |x| x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2],
        |x, c| {
            c[0] = x[0] * x[1] * x[2] * x[3] - 25.0;
            c[1] = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3];
        },
        Some(17.014_017_293_5),
        Expect::Optimum,
        "The IPOPT tutorial problem. One inequality, one equality, all variables bounded. \
         x* = (1, 4.743, 3.821, 1.379).",
    ));

    v.push(p(
        "HS100",
        &[1.0, 2.0, 0.0, 4.0, 0.0, 1.0, 1.0],
        &[-INF; 7],
        &[INF; 7],
        &[0.0; 4],
        &[INF; 4],
        |x| {
            (x[0] - 10.0).powi(2)
                + 5.0 * (x[1] - 12.0).powi(2)
                + x[2].powi(4)
                + 3.0 * (x[3] - 11.0).powi(2)
                + 10.0 * x[4].powi(6)
                + 7.0 * x[5] * x[5]
                + x[6].powi(4)
                - 4.0 * x[5] * x[6]
                - 10.0 * x[5]
                - 8.0 * x[6]
        },
        |x, c| {
            c[0] = 127.0
                - 2.0 * x[0] * x[0]
                - 3.0 * x[1].powi(4)
                - x[2]
                - 4.0 * x[3] * x[3]
                - 5.0 * x[4];
            c[1] = 282.0 - 7.0 * x[0] - 3.0 * x[1] - 10.0 * x[2] * x[2] - x[3] + x[4];
            c[2] = 196.0 - 23.0 * x[0] - x[1] * x[1] - 6.0 * x[5] * x[5] + 8.0 * x[6];
            c[3] = -4.0 * x[0] * x[0] - x[1] * x[1] + 3.0 * x[0] * x[1]
                - 2.0 * x[2] * x[2]
                - 5.0 * x[5]
                + 11.0 * x[6];
        },
        Some(680.630_057_3),
        Expect::Optimum,
        "Seven variables with sixth powers in the objective: the gradient spans many orders of \
         magnitude, which is exactly what gradient-based scaling is for.",
    ));

    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn published_optima_are_attainable_where_we_know_the_solution() {
        // Spot-check that our transcription of the objective agrees with the
        // published optimum at the published solution. A transcription typo is
        // otherwise invisible until it silently corrupts a benchmark.
        let cases: &[(&str, &[f64], f64)] = &[
            ("HS1", &[1.0, 1.0], 0.0),
            ("HS4", &[1.0, 0.0], 8.0 / 3.0),
            ("HS6", &[1.0, 1.0], 0.0),
            ("HS12", &[2.0, 3.0], -30.0),
            ("HS16", &[0.5, 0.25], 0.25),
            ("HS22", &[1.0, 1.0], 1.0),
            ("HS28", &[0.5, -0.5, 0.5], 0.0),
            ("HS35", &[4.0 / 3.0, 7.0 / 9.0, 4.0 / 9.0], 1.0 / 9.0),
            ("HS36", &[20.0, 11.0, 15.0], -3300.0),
            ("HS37", &[24.0, 12.0, 12.0], -3456.0),
            ("HS43", &[0.0, 1.0, 2.0, -1.0], -44.0),
            ("HS45", &[1.0, 2.0, 3.0, 4.0, 5.0], 1.0),
        ];
        let problems = all();
        for (name, x, expect) in cases {
            let p = problems
                .iter()
                .find(|p| p.name == *name)
                .unwrap_or_else(|| panic!("{name} missing"));
            let f = (p.f)(x);
            assert!(
                (f - expect).abs() < 1e-6,
                "{name}: objective at the published solution is {f}, expected {expect}"
            );
            let viol = p.violation(x);
            assert!(viol < 1e-6, "{name}: published solution violates by {viol}");
            if let Some(fopt) = p.f_opt {
                assert!(
                    (fopt - expect).abs() < 1e-6,
                    "{name}: recorded f_opt {fopt} disagrees with the checked value {expect}"
                );
            }
        }
    }

    #[test]
    fn hs71_published_solution_checks_out() {
        let p = all().into_iter().find(|p| p.name == "HS71").unwrap();
        let x = [1.0, 4.742_999_635_7, 3.821_149_997_4, 1.379_408_293_9];
        let f = (p.f)(&x);
        assert!((f - 17.014_017_293_5).abs() < 1e-6, "f = {f}");
        assert!(p.violation(&x) < 1e-6, "violation {}", p.violation(&x));
    }

    #[test]
    fn hs110_is_undefined_outside_its_box() {
        let p = all().into_iter().find(|p| p.name == "HS110").unwrap();
        assert!((p.f)(&[9.0; 10]).is_finite());
        let mut bad = vec![9.0; 10];
        bad[0] = 1.5; // below the lower bound of 2.001
        assert!(
            !(p.f)(&bad).is_finite(),
            "HS110 must be non-finite below its bounds, or it does not test what it claims to"
        );
    }
}
