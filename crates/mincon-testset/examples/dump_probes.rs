//! Dump objective and constraint values of every test-set problem at the
//! starting point and at deterministic probe points, as JSON, so an
//! independently written corpus can be cross-checked against this crate.
fn main() {
    let mut out = String::from("{");
    let mut first = true;
    for p in mincon_testset::hs::all() {
        if !first {
            out.push(',');
        }
        first = false;
        out.push_str(&format!("\"{}\":[", p.name));
        for k in 0..4 {
            let x: Vec<f64> = (0..p.n)
                .map(|i| {
                    let d = 0.05 * (k as f64) * (1.0 + (i % 3) as f64);
                    (p.x0[i] + d).clamp(p.xl[i], p.xu[i])
                })
                .collect();
            let f = (p.f)(&x);
            let mut c = vec![0.0; p.m];
            (p.c)(&x, &mut c);
            if k > 0 {
                out.push(',');
            }
            out.push_str(&format!(
                "{{\"x\":{:?},\"f\":{:e},\"c\":[{}]}}",
                x,
                f,
                c.iter().map(|v| format!("{v:e}")).collect::<Vec<_>>().join(",")
            ));
        }
        out.push(']');
    }
    out.push('}');
    println!("{out}");
}
