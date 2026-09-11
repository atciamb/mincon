//! Fixtures for the dual active-set QP solver (plan `docs/19` §4 S-D).

use super::*;

/// Small deterministic generator (no dependency): xorshift64*.
struct Rng(u64);
impl Rng {
    fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let v = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (v >> 11) as f64 / (1u64 << 53) as f64
    }
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }
}

/// Random symmetric positive definite matrix `B^T B + shift I`.
fn spd(rng: &mut Rng, n: usize, shift: f64) -> Vec<f64> {
    let b: Vec<f64> = (0..n * n).map(|_| rng.uniform(-1.0, 1.0)).collect();
    let mut h = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            h[i * n + j] = (0..n).map(|k| b[k * n + i] * b[k * n + j]).sum::<f64>()
                + if i == j { shift } else { 0.0 };
        }
    }
    h
}

fn kkt_residual(qp: &DenseQp<'_>, s: &QpSolution) -> f64 {
    let n = qp.n;
    let mut r = 0.0_f64;
    for i in 0..n {
        let mut v = qp.g[i] + (0..n).map(|j| qp.h[i * n + j] * s.x[j]).sum::<f64>();
        for k in 0..qp.m {
            v += qp.a[k * n + i] * s.lambda[k];
        }
        v += -s.z_l[i] + s.z_u[i];
        r = r.max(v.abs());
    }
    r
}

fn max_violation(qp: &DenseQp<'_>, x: &[f64]) -> f64 {
    let n = qp.n;
    let mut v = 0.0_f64;
    for k in 0..qp.m {
        let ax: f64 = (0..n).map(|j| qp.a[k * n + j] * x[j]).sum();
        v = v.max(qp.a_l[k] - ax).max(ax - qp.a_u[k]);
    }
    for j in 0..n {
        v = v.max(qp.x_l[j] - x[j]).max(x[j] - qp.x_u[j]);
    }
    v
}

/// Complementarity and sign checks in mincon's convention.
fn signs_and_complementarity(qp: &DenseQp<'_>, s: &QpSolution, tol: f64) {
    let n = qp.n;
    for k in 0..qp.m {
        let ax: f64 = (0..n).map(|j| qp.a[k * n + j] * s.x[j]).sum();
        if qp.a_l[k] == qp.a_u[k] {
            continue;
        }
        if s.lambda[k] > tol {
            assert!(
                (ax - qp.a_u[k]).abs() <= tol,
                "row {k}: positive multiplier but upper side not active"
            );
        }
        if s.lambda[k] < -tol {
            assert!(
                (ax - qp.a_l[k]).abs() <= tol,
                "row {k}: negative multiplier but lower side not active"
            );
        }
    }
    for j in 0..n {
        assert!(s.z_l[j] >= -tol && s.z_u[j] >= -tol);
        if s.z_l[j] > tol {
            assert!((s.x[j] - qp.x_l[j]).abs() <= tol);
        }
        if s.z_u[j] > tol {
            assert!((s.x[j] - qp.x_u[j]).abs() <= tol);
        }
    }
}

/// Brute force: enumerate active sets (rows: lower/upper/inactive; bounds likewise),
/// solve the equality-constrained KKT system by Gaussian elimination, keep the
/// candidates that are primal feasible with correctly signed multipliers, and
/// return the smallest objective. Exponential, so n and m are tiny.
fn brute_force(qp: &DenseQp<'_>) -> Option<f64> {
    let n = qp.n;
    let items = qp.m + n; // each item: 0 inactive, 1 lower, 2 upper
    let mut best: Option<f64> = None;
    let total = 3usize.pow(items as u32);
    for code in 0..total {
        let mut c = code;
        let mut rows: Vec<(Vec<f64>, f64, i8)> = Vec::new(); // (normal, rhs, side)
        let mut ok = true;
        for it in 0..items {
            let choice = c % 3;
            c /= 3;
            let (normal, lo, hi) = if it < qp.m {
                (qp.a[it * n..(it + 1) * n].to_vec(), qp.a_l[it], qp.a_u[it])
            } else {
                let mut e = vec![0.0; n];
                e[it - qp.m] = 1.0;
                (e, qp.x_l[it - qp.m], qp.x_u[it - qp.m])
            };
            let eq = lo == hi;
            match choice {
                0 => {
                    if eq {
                        ok = false;
                    }
                }
                1 => {
                    if !lo.is_finite() {
                        ok = false;
                    } else {
                        rows.push((normal, lo, if eq { 0 } else { -1 }));
                    }
                }
                _ => {
                    if !hi.is_finite() || eq {
                        ok = false;
                    } else {
                        rows.push((normal, hi, 1));
                    }
                }
            }
            if !ok {
                break;
            }
        }
        if !ok || rows.len() > n {
            continue;
        }
        let q = rows.len();
        let dim = n + q;
        // KKT: [H N^T; N 0] [x; y] = [-g; b]  with stationarity H x + g + N^T y = 0
        let mut k = vec![0.0; dim * dim];
        let mut rhs = vec![0.0; dim];
        for i in 0..n {
            for j in 0..n {
                k[i * dim + j] = qp.h[i * n + j];
            }
            for (r, row) in rows.iter().enumerate() {
                k[i * dim + n + r] = row.0[i];
                k[(n + r) * dim + i] = row.0[i];
            }
            rhs[i] = -qp.g[i];
        }
        for (r, row) in rows.iter().enumerate() {
            rhs[n + r] = row.1;
        }
        let Some(sol) = gauss(&mut k, &mut rhs, dim) else {
            continue;
        };
        let x = &sol[..n];
        if max_violation(qp, x) > 1e-8 {
            continue;
        }
        let mut signs_ok = true;
        for (r, row) in rows.iter().enumerate() {
            let y = sol[n + r];
            // y multiplies +normal in stationarity; lower side needs y <= 0, upper y >= 0
            if row.2 < 0 && y > 1e-9 {
                signs_ok = false;
            }
            if row.2 > 0 && y < -1e-9 {
                signs_ok = false;
            }
        }
        if !signs_ok {
            continue;
        }
        let mut hx = vec![0.0; n];
        for i in 0..n {
            hx[i] = (0..n).map(|j| qp.h[i * n + j] * x[j]).sum();
        }
        let obj = 0.5 * dot(x, &hx) + dot(qp.g, x);
        best = Some(best.map_or(obj, |b: f64| b.min(obj)));
    }
    best
}

fn gauss(a: &mut [f64], b: &mut [f64], n: usize) -> Option<Vec<f64>> {
    for col in 0..n {
        let mut piv = col;
        for r in col + 1..n {
            if a[r * n + col].abs() > a[piv * n + col].abs() {
                piv = r;
            }
        }
        if a[piv * n + col].abs() < 1e-12 {
            return None;
        }
        if piv != col {
            for j in 0..n {
                a.swap(col * n + j, piv * n + j);
            }
            b.swap(col, piv);
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let f = a[r * n + col] / a[col * n + col];
            if f != 0.0 {
                for j in col..n {
                    a[r * n + j] -= f * a[col * n + j];
                }
                b[r] -= f * b[col];
            }
        }
    }
    Some((0..n).map(|i| b[i] / a[i * n + i]).collect())
}

#[test]
fn random_small_qps_match_brute_force_enumeration() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut solved = 0;
    for trial in 0..400 {
        let n = 1 + (rng.next_f64() * 3.0) as usize; // 1..3
        let m = (rng.next_f64() * 3.0) as usize; // 0..2
        let h = spd(&mut rng, n, 0.1);
        let g: Vec<f64> = (0..n).map(|_| rng.uniform(-2.0, 2.0)).collect();
        let a: Vec<f64> = (0..m * n).map(|_| rng.uniform(-1.0, 1.0)).collect();
        let mut a_l = vec![0.0; m];
        let mut a_u = vec![0.0; m];
        for i in 0..m {
            let kind = rng.next_f64();
            if kind < 0.25 {
                let v = rng.uniform(-1.0, 1.0);
                a_l[i] = v;
                a_u[i] = v; // equality
            } else if kind < 0.5 {
                a_l[i] = rng.uniform(-1.0, 0.5);
                a_u[i] = f64::INFINITY;
            } else if kind < 0.75 {
                a_l[i] = f64::NEG_INFINITY;
                a_u[i] = rng.uniform(-0.5, 1.0);
            } else {
                let lo = rng.uniform(-1.0, 0.0);
                a_l[i] = lo;
                a_u[i] = lo + rng.uniform(0.1, 1.5);
            }
        }
        let mut x_l = vec![f64::NEG_INFINITY; n];
        let mut x_u = vec![f64::INFINITY; n];
        for j in 0..n {
            if rng.next_f64() < 0.5 {
                x_l[j] = rng.uniform(-2.0, 0.0);
            }
            if rng.next_f64() < 0.5 {
                x_u[j] = rng.uniform(0.0, 2.0);
            }
        }
        let qp = DenseQp {
            n,
            m,
            h: &h,
            g: &g,
            a: &a,
            a_l: &a_l,
            a_u: &a_u,
            x_l: &x_l,
            x_u: &x_u,
        };
        let reference = brute_force(&qp);
        match solve_dense(&qp, &[], &QpOptions::default()) {
            Ok(s) => {
                let r = reference.unwrap_or_else(|| {
                    panic!("trial {trial}: solver found a point, brute force found none")
                });
                assert!(
                    (s.objective - r).abs() <= 1e-7 * (1.0 + r.abs()),
                    "trial {trial}: objective {} vs brute force {r}",
                    s.objective
                );
                assert!(
                    kkt_residual(&qp, &s) <= 1e-8,
                    "trial {trial}: KKT residual {}",
                    kkt_residual(&qp, &s)
                );
                assert!(
                    max_violation(&qp, &s.x) <= 1e-8,
                    "trial {trial}: violation {}",
                    max_violation(&qp, &s.x)
                );
                signs_and_complementarity(&qp, &s, 1e-7);
                solved += 1;
            }
            Err(QpError::Infeasible(_)) => {
                assert!(
                    reference.is_none(),
                    "trial {trial}: solver says infeasible, brute force found {reference:?}"
                );
            }
            Err(e) => panic!("trial {trial}: {e}"),
        }
    }
    assert!(solved > 250, "only {solved} feasible trials");
}

#[test]
fn larger_random_qps_satisfy_kkt_to_high_accuracy() {
    let mut rng = Rng(42);
    for trial in 0..40 {
        let n = 5 + (rng.next_f64() * 40.0) as usize;
        let m = (rng.next_f64() * n as f64 * 0.8) as usize;
        let h = spd(&mut rng, n, 1e-3);
        let g: Vec<f64> = (0..n).map(|_| rng.uniform(-5.0, 5.0)).collect();
        let a: Vec<f64> = (0..m * n).map(|_| rng.uniform(-1.0, 1.0)).collect();
        let a_l: Vec<f64> = (0..m).map(|_| rng.uniform(-3.0, 0.0)).collect();
        let a_u: Vec<f64> = a_l.iter().map(|lo| lo + rng.uniform(0.0, 2.0)).collect();
        let x_l = vec![-1.0; n];
        let x_u = vec![1.0; n];
        let qp = DenseQp {
            n,
            m,
            h: &h,
            g: &g,
            a: &a,
            a_l: &a_l,
            a_u: &a_u,
            x_l: &x_l,
            x_u: &x_u,
        };
        match solve_dense(&qp, &[], &QpOptions::default()) {
            Ok(s) => {
                let scale = 1.0 + g.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
                assert!(
                    kkt_residual(&qp, &s) <= 1e-9 * scale,
                    "trial {trial}: KKT {}",
                    kkt_residual(&qp, &s)
                );
                assert!(
                    max_violation(&qp, &s.x) <= 1e-9,
                    "trial {trial}: violation {}",
                    max_violation(&qp, &s.x)
                );
                signs_and_complementarity(&qp, &s, 1e-7);
                assert!(s.iterations <= 10 * (n + 2 * m + 2 * n) + 100);
            }
            Err(QpError::Infeasible(_)) => {}
            Err(e) => panic!("trial {trial}: {e}"),
        }
    }
}

#[test]
fn warm_hint_reduces_iterations_on_a_perturbed_neighbour() {
    let mut rng = Rng(7);
    let n = 12;
    let m = 8;
    let h = spd(&mut rng, n, 0.5);
    let g: Vec<f64> = (0..n).map(|_| rng.uniform(-5.0, 5.0)).collect();
    let a: Vec<f64> = (0..m * n).map(|_| rng.uniform(-1.0, 1.0)).collect();
    let a_l: Vec<f64> = (0..m).map(|_| rng.uniform(-1.0, 0.0)).collect();
    let a_u = vec![f64::INFINITY; m];
    let x_l = vec![-0.5; n];
    let x_u = vec![0.5; n];
    let qp = DenseQp {
        n,
        m,
        h: &h,
        g: &g,
        a: &a,
        a_l: &a_l,
        a_u: &a_u,
        x_l: &x_l,
        x_u: &x_u,
    };
    let first = solve_dense(&qp, &[], &QpOptions::default()).unwrap();
    let g2: Vec<f64> = g.iter().map(|v| v + rng.uniform(-0.05, 0.05)).collect();
    let qp2 = DenseQp {
        g: &g2,
        ..qp.clone()
    };
    let cold = solve_dense(&qp2, &[], &QpOptions::default()).unwrap();
    let warm = solve_dense(&qp2, &first.active, &QpOptions::default()).unwrap();
    assert!((cold.objective - warm.objective).abs() <= 1e-9 * (1.0 + cold.objective.abs()));
    assert!(
        warm.iterations <= cold.iterations,
        "warm {} > cold {} iterations",
        warm.iterations,
        cold.iterations
    );
}

#[test]
fn infeasible_constraints_are_certified() {
    // x >= 1 and x <= 0
    let h = [1.0];
    let g = [0.0];
    let a = [1.0, 1.0];
    let qp = DenseQp {
        n: 1,
        m: 2,
        h: &h,
        g: &g,
        a: &a,
        a_l: &[1.0, f64::NEG_INFINITY],
        a_u: &[f64::INFINITY, 0.0],
        x_l: &[f64::NEG_INFINITY],
        x_u: &[f64::INFINITY],
    };
    assert!(matches!(
        solve_dense(&qp, &[], &QpOptions::default()),
        Err(QpError::Infeasible(_))
    ));
    // inconsistent equalities: x1 + x2 = 1 and x1 + x2 = 2
    let h = [1.0, 0.0, 0.0, 1.0];
    let g = [0.0, 0.0];
    let a = [1.0, 1.0, 1.0, 1.0];
    let qp = DenseQp {
        n: 2,
        m: 2,
        h: &h,
        g: &g,
        a: &a,
        a_l: &[1.0, 2.0],
        a_u: &[1.0, 2.0],
        x_l: &[f64::NEG_INFINITY; 2],
        x_u: &[f64::INFINITY; 2],
    };
    assert!(matches!(
        solve_dense(&qp, &[], &QpOptions::default()),
        Err(QpError::Infeasible(_))
    ));
}

#[test]
fn redundant_equalities_are_tolerated() {
    // x1 + x2 = 1 twice, minimize ||x||^2 -> (0.5, 0.5)
    let h = [2.0, 0.0, 0.0, 2.0];
    let g = [0.0, 0.0];
    let a = [1.0, 1.0, 1.0, 1.0];
    let qp = DenseQp {
        n: 2,
        m: 2,
        h: &h,
        g: &g,
        a: &a,
        a_l: &[1.0, 1.0],
        a_u: &[1.0, 1.0],
        x_l: &[f64::NEG_INFINITY; 2],
        x_u: &[f64::INFINITY; 2],
    };
    let s = solve_dense(&qp, &[], &QpOptions::default()).unwrap();
    assert!(
        (s.x[0] - 0.5).abs() < 1e-12 && (s.x[1] - 0.5).abs() < 1e-12,
        "{:?}",
        s.x
    );
    assert!(kkt_residual(&qp, &s) < 1e-12);
}

#[test]
fn degenerate_vertex_with_more_active_constraints_than_variables() {
    // min (x1-1)^2 + (x2-1)^2 s.t. x1 <= 0, x2 <= 0, x1 + x2 <= 0 (all active at the origin)
    let h = [2.0, 0.0, 0.0, 2.0];
    let g = [-2.0, -2.0];
    let a = [1.0, 1.0];
    let qp = DenseQp {
        n: 2,
        m: 1,
        h: &h,
        g: &g,
        a: &a,
        a_l: &[f64::NEG_INFINITY],
        a_u: &[0.0],
        x_l: &[f64::NEG_INFINITY; 2],
        x_u: &[0.0, 0.0],
    };
    let s = solve_dense(&qp, &[], &QpOptions::default()).unwrap();
    assert!(s.x[0].abs() < 1e-12 && s.x[1].abs() < 1e-12, "{:?}", s.x);
    assert!(kkt_residual(&qp, &s) < 1e-12, "{}", kkt_residual(&qp, &s));
    signs_and_complementarity(&qp, &s, 1e-9);
}

#[test]
fn every_bound_active_and_fixed_variables() {
    // minimize with the unconstrained minimizer far outside the box
    let n = 6;
    let mut rng = Rng(3);
    let h = spd(&mut rng, n, 1.0);
    let g = vec![-100.0; n];
    let mut x_l = vec![0.0; n];
    let mut x_u = vec![1.0; n];
    x_l[2] = 0.3;
    x_u[2] = 0.3; // fixed variable
    let qp = DenseQp {
        n,
        m: 0,
        h: &h,
        g: &g,
        a: &[],
        a_l: &[],
        a_u: &[],
        x_l: &x_l,
        x_u: &x_u,
    };
    let s = solve_dense(&qp, &[], &QpOptions::default()).unwrap();
    for j in 0..n {
        if j == 2 {
            assert!((s.x[j] - 0.3).abs() < 1e-12);
        } else {
            assert!((s.x[j] - 1.0).abs() < 1e-12, "x[{j}] = {}", s.x[j]);
        }
    }
    assert!(kkt_residual(&qp, &s) < 1e-9, "{}", kkt_residual(&qp, &s));
}

#[test]
fn indefinite_hessian_is_reported_not_solved() {
    let h = [1.0, 0.0, 0.0, -1.0];
    let qp = DenseQp {
        n: 2,
        m: 0,
        h: &h,
        g: &[0.0, 0.0],
        a: &[],
        a_l: &[],
        a_u: &[],
        x_l: &[f64::NEG_INFINITY; 2],
        x_u: &[f64::INFINITY; 2],
    };
    assert!(matches!(
        solve_dense(&qp, &[], &QpOptions::default()),
        Err(QpError::NotPositiveDefinite(_))
    ));
}

#[test]
fn ill_conditioned_hessian_still_meets_kkt() {
    let n = 8;
    let mut h = vec![0.0; n * n];
    for i in 0..n {
        h[i * n + i] = 10f64.powi(-(i as i32)); // condition number 1e7
    }
    let g = vec![1.0; n];
    let x_l = vec![-2.0; n];
    let x_u = vec![2.0; n];
    let a = vec![1.0; n];
    let qp = DenseQp {
        n,
        m: 1,
        h: &h,
        g: &g,
        a: &a,
        a_l: &[-1.0],
        a_u: &[1.0],
        x_l: &x_l,
        x_u: &x_u,
    };
    let s = solve_dense(&qp, &[], &QpOptions::default()).unwrap();
    assert!(kkt_residual(&qp, &s) < 1e-9, "{}", kkt_residual(&qp, &s));
    assert!(max_violation(&qp, &s.x) < 1e-9);
    signs_and_complementarity(&qp, &s, 1e-8);
}

#[test]
fn reports_itself_available() {
    assert!(is_available());
}

#[test]
fn a_300_variable_bound_constrained_qp_is_fast_enough() {
    let n = 300;
    let mut rng = Rng(11);
    let mut h = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            h[i * n + j] = if i == j {
                2.0 + rng.uniform(0.0, 1.0)
            } else {
                0.01 * rng.uniform(-1.0, 1.0)
            };
        }
    }
    for i in 0..n {
        for j in 0..i {
            h[i * n + j] = h[j * n + i];
        }
    }
    let g: Vec<f64> = (0..n).map(|_| rng.uniform(-5.0, 5.0)).collect();
    let a = vec![1.0; n];
    let x_l = vec![0.0; n];
    let x_u = vec![1.0; n];
    let qp = DenseQp {
        n,
        m: 1,
        h: &h,
        g: &g,
        a: &a,
        a_l: &[1.0],
        a_u: &[1.0],
        x_l: &x_l,
        x_u: &x_u,
    };
    let t = std::time::Instant::now();
    let s = solve_dense(&qp, &[], &QpOptions::default()).unwrap();
    let secs = t.elapsed().as_secs_f64();
    eprintln!(
        "n = 300 simplex QP: {} active-set steps in {secs:.3} s",
        s.iterations
    );
    assert!(kkt_residual(&qp, &s) < 1e-8, "{}", kkt_residual(&qp, &s));
    assert!(max_violation(&qp, &s.x) < 1e-9);
    assert!(secs < 20.0, "{secs} s");
}
