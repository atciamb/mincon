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
    let viol = r.constraint_violation;
    let feasible = viol <= 1e-5;
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
                let rel = (r.solution.f - fopt).abs() / fopt.abs().max(1.0);
                if feasible && rel <= 1e-4 {
                    (
                        "PASS",
                        format!("f = {:.8e}  (rel err {rel:.1e})", r.solution.f),
                    )
                } else if feasible && r.solution.f < fopt - 1e-6 * fopt.abs().max(1.0) {
                    (
                        "BETR",
                        format!("found a LOWER minimum {:.8e} < {fopt:.8e}", r.solution.f),
                    )
                } else {
                    (
                        "FAIL",
                        format!(
                            "f = {:.8e} vs {fopt:.8e} (rel {rel:.1e}), violation {viol:.1e}, {:?}",
                            r.solution.f, r.exit_flag
                        ),
                    )
                }
            }
            None => ("PASS", "no reference value".into()),
        },
        Expect::LocalMinimum => {
            if feasible && r.exit_flag.returned_usable_point() {
                (
                    "PASS",
                    format!("f = {:.8e}, stationary and feasible", r.solution.f),
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
    let mut rows: Vec<String> = Vec::new();

    println!(
        "{:<18} {:>4} {:>4} {:>6} {:>7} {:>10}  {:<5} outcome",
        "problem", "n", "m", "iters", "f-evals", "opt", "verdict"
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
                let (tag, msg) = verdict(p, &r);
                match tag {
                    "PASS" => pass += 1,
                    "BETR" => pass += 1,
                    "LIE " => lies += 1,
                    "soft" => soft += 1,
                    _ => fail += 1,
                }
                rows.push(format!(
                    "{:<18} {:>4} {:>4} {:>6} {:>7} {:>10.2e}  {tag:<5} {msg}",
                    p.name, p.n, p.m, r.iterations, r.f_evals, r.optimality
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
    println!(
        "{pass}/{total} pass, {fail} fail, {soft} soft-fail (did not converge but did not lie), {lies} LIES",
    );
    if lies > 0 {
        println!(
            "\nA LIE is a solver reporting success on a problem with no solution. Fix these first."
        );
    }
}
