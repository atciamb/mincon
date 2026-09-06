//! Print everything about one problem's solve. `cargo run -p mincon-ip --example diagnose HS16`

use mincon_core::{Algorithm, Options};

fn main() {
    let name = std::env::args().nth(1).unwrap_or_else(|| "HS16".into());
    let p = mincon_testset::by_name(&name).expect("unknown problem");
    let opts = Options {
        algorithm: Algorithm::InteriorPoint,
        max_seconds: Some(30.0),
        ..Options::default()
    };
    let nlp = p.as_nlp();
    let r = mincon_ip::solve(&nlp, &opts).expect("solve failed");

    println!("=== {} ===\n{}", p.name, p.notes);
    println!("\n{}", r.summary());
    println!("\nx  = {:?}", r.solution.x);
    println!("c  = {:?}", r.solution.c);
    println!("lambda = {:?}", r.solution.lambda);
    println!("z_l    = {:?}", r.solution.z_l);
    println!("z_u    = {:?}", r.solution.z_u);
    println!("published f* = {:?}", p.f_opt);
    for n in &r.notes {
        println!("note: {n}");
    }

    // Independent check: is this a local minimum? Sample the feasible
    // neighbourhood and see whether anything nearby is better.
    let x = &r.solution.x;
    let mut best = r.solution.f;
    let mut best_x = x.clone();
    let mut improved = 0;
    for k in 0..200_000u64 {
        let mut y = x.clone();
        let mut h = k;
        for yi in y.iter_mut() {
            let d = ((h % 21) as f64 - 10.0) / 10.0;
            h /= 21;
            *yi += d * 0.05;
        }
        for j in 0..p.n {
            y[j] = y[j].clamp(p.xl[j], p.xu[j]);
        }
        if p.violation(&y) < 1e-9 {
            let f = (p.f)(&y);
            if f < best - 1e-12 {
                best = f;
                best_x = y;
                improved += 1;
            }
        }
    }
    println!("\nlocal search over a 0.05 neighbourhood: {improved} improvements found");
    if improved > 0 {
        println!("  best nearby feasible: f = {best:.10e} at {best_x:?}");
        println!("  => a lower sampled feasible point exists; investigate convergence and basins.");
    } else {
        println!("  => no improvement in these samples; this does not certify a local minimum.");
    }

    println!("\nlast 8 iterations:");
    println!(
        "{:>5} {:>13} {:>10} {:>10} {:>9} {:>9} {:>9}",
        "iter", "f", "viol", "optimality", "alpha", "mu", "delta_w"
    );
    for t in r.trace.iter().rev().take(8).rev() {
        println!(
            "{:>5} {:>13.6e} {:>10.2e} {:>10.2e} {:>9.2e} {:>9.2e} {:>9.2e}",
            t.iter, t.f, t.constraint_violation, t.optimality, t.alpha, t.mu, t.delta_w
        );
    }
}
