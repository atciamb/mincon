//! Independent analytical regression gates for feasibility restoration.
use mincon_core::{ExitFlag, Options};

#[test]
fn scaled_nonzero_constraint_bound_and_multiplier_match_analytic_solution() {
    let p = mincon_testset::TestProblem {
        name: "scaled upper bound",
        n: 1,
        m: 1,
        x0: vec![0.0],
        xl: vec![f64::NEG_INFINITY],
        xu: vec![f64::INFINITY],
        cl: vec![f64::NEG_INFINITY],
        cu: vec![1000.0],
        f: |x| 1e6 * (x[0] - 2.0).powi(2),
        c: |x, c| c[0] = 1000.0 * x[0],
        f_opt: Some(1e6),
        expect: mincon_testset::Expect::Optimum,
        notes: "x=1, lambda=2000",
    };
    let r = mincon_ip::solve(&p.as_nlp(), &Options::default()).unwrap();
    assert!(p.violation(&r.solution.x) <= 1e-6, "{}", r.summary());
    assert!((r.solution.x[0] - 1.0).abs() < 1e-6, "{}", r.summary());
    assert!(
        (r.solution.lambda[0] - 2000.0).abs() < 0.1,
        "{:?}",
        r.solution.lambda
    );
    assert!((r.solution.c[0] - 1000.0 * r.solution.x[0]).abs() < 1e-8);
}

#[test]
fn inconsistent_affine_constraints_report_stationary_infeasibility() {
    let p = mincon_testset::by_name("TORTURE_INFEASIBLE").unwrap();
    let r = mincon_ip::solve(&p.as_nlp(), &Options::default()).unwrap();
    // The testset separately proves these two affine constraints inconsistent.
    assert!(p.violation(&r.solution.x) > 0.1);
    assert_eq!(r.exit_flag, ExitFlag::LocallyInfeasible, "{}", r.summary());
    assert!(!r.exit_flag.is_success());
    assert!(r.trace.iter().any(|t| t.in_restoration));
}

#[test]
fn hs13_reaches_the_degenerate_solution_without_claiming_kkt_success() {
    let p = mincon_testset::by_name("HS13").unwrap();
    let r = mincon_ip::solve(&p.as_nlp(), &Options::default()).unwrap();
    // Analytical boundary solution (1,0), f=1; ordinary KKT fails there.
    assert!(p.violation(&r.solution.x) <= 1e-6);
    assert!(
        ((p.f)(&r.solution.x) - 1.0).abs() <= 1e-4,
        "{}",
        r.summary()
    );
    assert!(matches!(
        r.exit_flag,
        ExitFlag::StepTolerance | ExitFlag::Acceptable
    ));
    assert!(!r.exit_flag.is_success());
}

#[test]
fn strict_box_model_is_never_evaluated_outside_bounds_including_setup() {
    let p = mincon_testset::TestProblem {
        name: "strict box",
        n: 1,
        m: 1,
        x0: vec![-2.0],
        xl: vec![0.0],
        xu: vec![1.0],
        cl: vec![0.0],
        cu: vec![f64::INFINITY],
        f: |x| {
            assert!((0.0..=1.0).contains(&x[0]));
            x[0]
        },
        c: |x, c| {
            assert!((0.0..=1.0).contains(&x[0]));
            c[0] = x[0];
        },
        f_opt: Some(0.0),
        expect: mincon_testset::Expect::Optimum,
        notes: "exact domain",
    };
    let r = mincon_ip::solve(&p.as_nlp(), &Options::default()).unwrap();
    assert!((0.0..=1.0).contains(&r.solution.x[0]));
    assert!(r.solution.f < 1e-6);
}

#[test]
fn hs33_and_hs35_converge_with_independently_checked_kkt_residuals() {
    for name in ["HS33", "HS35"] {
        let p = mincon_testset::by_name(name).unwrap();
        let opts = Options::default();
        let r = mincon_ip::solve(&p.as_nlp(), &opts).unwrap();
        let s = &r.solution;
        let x = &s.x;
        assert_eq!(r.exit_flag, ExitFlag::Optimal, "{name}: {}", r.summary());
        assert!(p.violation(x) <= opts.tol.feasibility);
        assert!(((p.f)(x) - p.f_opt.unwrap()).abs() < 1e-7);
        // Algebraic derivatives, never the finite-difference engine or its
        // reported residual. Jacobian convention is grad f + J^T lambda-zl+zu.
        let mut residual = if name == "HS33" {
            vec![
                3.0 * x[0] * x[0] - 12.0 * x[0] + 11.0 + 2.0 * x[0] * (s.lambda[1] - s.lambda[0]),
                2.0 * x[1] * (s.lambda[1] - s.lambda[0]),
                1.0 + 2.0 * x[2] * (s.lambda[0] + s.lambda[1]),
            ]
        } else {
            vec![
                -8.0 + 4.0 * x[0] + 2.0 * x[1] + 2.0 * x[2] - s.lambda[0],
                -6.0 + 4.0 * x[1] + 2.0 * x[0] - s.lambda[0],
                -4.0 + 2.0 * x[2] + 2.0 * x[0] - 2.0 * s.lambda[0],
            ]
        };
        for j in 0..3 {
            residual[j] += s.z_u[j] - s.z_l[j];
            assert!(s.z_l[j] >= 0.0 && s.z_u[j] >= 0.0);
            assert!((s.z_l[j] * (x[j] - p.xl[j])).abs() <= opts.tol.complementarity);
            if p.xu[j].is_finite() {
                assert!((s.z_u[j] * (p.xu[j] - x[j])).abs() <= opts.tol.complementarity);
            }
        }
        let stationarity = residual.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        assert!(
            stationarity <= opts.tol.optimality,
            "{name}: analytical residual {residual:?}"
        );
        let mut c = vec![0.0; p.m];
        (p.c)(x, &mut c);
        for (ci, li) in c.iter().zip(&s.lambda) {
            assert!(*li <= 0.0); // All constraints have lower bound zero.
            assert!((ci * li).abs() <= opts.tol.complementarity);
        }
    }
}
