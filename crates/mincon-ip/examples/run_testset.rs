//! Run the whole test set and print a table.
//!
//! `cargo run --release -p mincon-ip --example run_testset`
//!
//! This is the inner development loop: every change to the algorithm should be
//! judged against this table before anything else. The nightly CUTEst run in
//! `bench/` is the outer loop.

use mincon_core::{Algorithm, ExitFlag, Options};
use mincon_testset::{Expect, TestProblem};

fn verdict(p: &TestProblem, r: &mincon_core::SolveReport) -> (&'static str, String) {
    let viol = p.violation(&r.solution.x);
    let f = if r.solution.x.len() == p.n && r.solution.x.iter().all(|v| v.is_finite()) {
        (p.f)(&r.solution.x)
    } else {
        f64::NAN
    };
    let feasible = viol <= 1e-5 && f.is_finite();
    if r.exit_flag.is_success() && !feasible {
        return (
            "LIE ",
            format!("claimed success at an independently invalid/infeasible point ({viol:.1e})"),
        );
    }
    match p.expect {
        Expect::Infeasible => {
            if matches!(
                r.exit_flag,
                ExitFlag::Infeasible | ExitFlag::LocallyInfeasible
            ) {
                ("PASS", "correctly reported infeasible".into())
            } else if r.exit_flag.is_success() {
                (
                    "LIE ",
                    format!("claimed success on an infeasible problem (violation {viol:.1e})"),
                )
            } else {
                (
                    "soft",
                    format!("did not converge ({:?}), but did not lie", r.exit_flag),
                )
            }
        }
        Expect::Unbounded => {
            if r.exit_flag == ExitFlag::Unbounded {
                ("PASS", "correctly reported unbounded".into())
            } else if r.exit_flag.is_success() {
                (
                    "LIE ",
                    "claimed a finite minimum on an unbounded problem".into(),
                )
            } else {
                (
                    "soft",
                    format!("did not detect unboundedness ({:?})", r.exit_flag),
                )
            }
        }
        Expect::Optimum => match p.f_opt {
            Some(fopt) => {
                let rel = (f - fopt).abs() / fopt.abs().max(1.0);
                if feasible && rel <= 1e-4 {
                    ("PASS", format!("f = {f:.8e}  (rel err {rel:.1e})"))
                } else if feasible && f < fopt - 1e-6 * fopt.abs().max(1.0) {
                    (
                        "BETR",
                        format!("found a LOWER objective {f:.8e} < {fopt:.8e}"),
                    )
                } else {
                    (
                        "FAIL",
                        format!(
                            "f = {:.8e} vs {fopt:.8e} (rel {rel:.1e}), violation {viol:.1e}, {:?}",
                            f, r.exit_flag
                        ),
                    )
                }
            }
            None => (
                "FAIL",
                "optimum fixture is missing its reference value".into(),
            ),
        },
        Expect::DegenerateOptimum => {
            if r.exit_flag.is_success() {
                (
                    "LIE ",
                    "claimed ordinary KKT success on a fixture requiring a degenerate exit".into(),
                )
            } else if feasible
                && p.f_opt
                    .is_some_and(|fo| (f - fo).abs() <= 1e-4 * fo.abs().max(1.0))
                && matches!(r.exit_flag, ExitFlag::Acceptable | ExitFlag::StepTolerance)
            {
                (
                    "PASS",
                    format!(
                        "f = {f:.8e}, feasible degenerate solution ({:?})",
                        r.exit_flag
                    ),
                )
            } else {
                (
                    "FAIL",
                    format!("f = {f:.8e}, violation {viol:.1e}, {:?}", r.exit_flag),
                )
            }
        }
        Expect::LocalMinimum => {
            if feasible && r.exit_flag.returned_usable_point() {
                (
                    "PASS",
                    format!("f = {f:.8e}, feasible with a usable-point status"),
                )
            } else {
                ("FAIL", format!("violation {viol:.1e}, {:?}", r.exit_flag))
            }
        }
    }
}

fn main() {
    let filter: Option<String> = std::env::args().nth(1);
    let opts = Options {
        algorithm: Algorithm::InteriorPoint,
        max_seconds: Some(20.0),
        ..Options::default()
    };

    let problems = mincon_testset::all();
    let mut pass = 0;
    let mut fail = 0;
    let mut lies = 0;
    let mut soft = 0;
    let mut strict = 0;
    let mut rows: Vec<String> = Vec::new();

    println!(
        "{:<18} {:>4} {:>4} {:>6} {:>7} {:>10} {:<18} {:<5} outcome",
        "problem", "n", "m", "iters", "f-evals", "opt", "status", "verdict"
    );
    println!("{}", "-".repeat(120));

    for p in &problems {
        if let Some(f) = &filter {
            if !p.name.to_lowercase().contains(&f.to_lowercase()) {
                continue;
            }
        }
        let nlp = p.as_nlp();
        match mincon_ip::solve(&nlp, &opts) {
            Ok(r) => {
                strict += usize::from(r.exit_flag.is_success());
                let (tag, msg) = verdict(p, &r);
                match tag {
                    "PASS" => pass += 1,
                    "BETR" => pass += 1,
                    "LIE " => lies += 1,
                    "soft" => soft += 1,
                    _ => fail += 1,
                }
                rows.push(format!(
                    "{:<18} {:>4} {:>4} {:>6} {:>7} {:>10.2e} {:<18?} {tag:<5} {msg}",
                    p.name, p.n, p.m, r.iterations, r.f_evals, r.optimality, r.exit_flag
                ));
            }
            Err(e) => {
                fail += 1;
                rows.push(format!(
                    "{:<18} {:>4} {:>4} {:>6} {:>7} {:>10}  {:<5} solver error: {e}",
                    p.name, p.n, p.m, "-", "-", "-", "ERR"
                ));
            }
        }
    }

    for r in &rows {
        println!("{r}");
    }
    println!("{}", "-".repeat(120));
    let total = pass + fail + lies + soft;
    println!("{strict}/{total} reported strict Optimal; fixture PASS also covers usable points and expected failure diagnostics.");
    println!(
        "{pass}/{total} pass, {fail} fail, {soft} soft-fail (did not converge but did not lie), {lies} LIES",
    );
    if lies > 0 {
        println!(
            "\nA LIE is a solver reporting success on a problem with no solution. Fix these first."
        );
    }
    if fail + lies + soft > 0 || total == 0 {
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn verdict_recomputes_feasibility_instead_of_trusting_the_report() {
        let p = mincon_testset::by_name("HS1").unwrap();
        let mut r = mincon_ip::solve(&p.as_nlp(), &Options::default()).unwrap();
        r.solution.x[1] = -10.0;
        r.constraint_violation = 0.0;
        r.exit_flag = ExitFlag::Optimal;
        assert_eq!(verdict(&p, &r).0, "LIE ");
        r.solution.x[1] = f64::NAN;
        assert_eq!(verdict(&p, &r).0, "LIE ");
    }
    #[test]
    fn degenerate_fixture_cannot_pass_on_a_success_flag_alone() {
        let p = mincon_testset::by_name("HS13").unwrap();
        let mut r = mincon_ip::solve(&p.as_nlp(), &Options::default()).unwrap();
        assert_eq!(verdict(&p, &r).0, "PASS");
        r.exit_flag = ExitFlag::Optimal;
        assert_eq!(verdict(&p, &r).0, "LIE ");
    }
}
