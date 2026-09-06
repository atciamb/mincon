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
