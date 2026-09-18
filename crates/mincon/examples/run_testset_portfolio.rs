//! Run the whole test set through one algorithm and print a table.
//!
//! `cargo run --release -p mincon --example run_testset_portfolio -- [ip|sqp|auto] [name-filter]`
//!
//! Same verdict rules as `mincon-ip`'s `run_testset` (which stays the CI gate
//! for the interior-point member); this one exists so the SQP member and the
//! portfolio are judged by the same fixtures.

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
        Expect::NoFalseCertificate => {
            let attained = p
                .f_opt
                .is_some_and(|fo| feasible && (f - fo).abs() <= 1e-4 * fo.abs().max(1.0));
            if attained {
                (
                    "PASS",
                    format!("f = {f:.8e}, optimum attained ({:?})", r.exit_flag),
                )
            } else if r.exit_flag.is_success() {
                // Optimal away from the optimum is a lie unless the point is a
                // first-order stationary point by an independent check: a badly
                // scaled problem can have a whole curve of such points.
                let stat = p.relative_stationarity(&r.solution.x);
                if feasible && stat <= 1e-5 {
                    (
                        "PASS",
                        format!(
                            "f = {f:.8e}: not the optimum, but first-order stationary (relative residual {stat:.1e})"
                        ),
                    )
                } else {
                    (
                        "LIE ",
                        format!(
                            "claimed Optimal at f = {f:.8e} against {:?}; relative stationarity {stat:.1e}, violation {viol:.1e}",
                            p.f_opt
                        ),
                    )
                }
            } else if feasible && r.exit_flag.returned_usable_point() {
                (
                    "PASS",
                    format!(
                        "f = {f:.8e}, feasible, honest non-success ({:?})",
                        r.exit_flag
                    ),
                )
            } else {
                ("FAIL", format!("violation {viol:.1e}, {:?}", r.exit_flag))
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
    let alg = std::env::args().nth(1).unwrap_or_else(|| "ip".into());
    let filter: Option<String> = std::env::args().nth(2);
    let algorithm = match alg.as_str() {
        "ip" | "interior-point" => Algorithm::InteriorPoint,
        "sqp" => Algorithm::Sqp,
        "auto" => Algorithm::Auto,
        other => {
            eprintln!("unknown algorithm {other}; use ip, sqp or auto");
            std::process::exit(2);
        }
    };
    let opts = Options {
        algorithm,
        max_seconds: Some(20.0),
        threads: Some(1),
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
        match mincon::minimize(&nlp, &opts) {
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
