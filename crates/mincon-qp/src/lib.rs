//! The quadratic programming subproblem solver used by SQP.
//!
//! # The problem
//!
//! ```text
//!   min_x  1/2 x^T H x + g^T x
//!   s.t.   a_L <= A x <= a_U        (m general rows, a two-sided row may be an equality a_L = a_U)
//!          x_L <=  x  <= x_U        (bounds; +-inf allowed)
//! ```
//!
//! `H` is symmetric positive definite (a damped BFGS matrix, or an exact Hessian
//! shifted by the caller until its Cholesky factorization succeeds).
//!
//! # Method
//!
//! The dual active-set method of Goldfarb and Idnani (1983), as derived in
//! `docs/20_SQP_MATHEMATICS.md` §2.3. Every general row side and every bound is a
//! constraint `n^T x >= b`; equality rows are added first and never dropped. The
//! method starts at the unconstrained minimizer `-H^{-1} g`, keeps a working set
//! whose constraints hold as equalities with non-negative multipliers (dual
//! feasibility), and repeatedly adds the most violated constraint: a *full step*
//! makes it active, a *partial step* first drops the working constraint whose
//! multiplier would turn negative. It needs no feasible starting point, certifies
//! infeasibility (a violated constraint in the span of the working set with no
//! multiplier able to decrease), and terminates finitely for strictly convex
//! `H`. The factorizations maintained are `J = L^{-T} Q` and `R` from
//! `L^{-1} N = Q [R; 0]`, updated by Givens rotations; the only inversion is the
//! Cholesky factor `L` of `H`, once.
//!
//! Multipliers are returned in mincon's convention: stationarity is
//! `H x + g + A^T lambda - z_l + z_u = 0` with `lambda_i >= 0` when the upper
//! side of row `i` is active, `lambda_i <= 0` when the lower side is active,
//! free for an equality row, and `z_l, z_u >= 0`.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::fmt;

/// Which constraint a working-set entry refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Constraint {
    /// Row `i` at its lower bound (`a_i^T x >= a_L_i`).
    RowLower(usize),
    /// Row `i` at its upper bound (`a_i^T x <= a_U_i`).
    RowUpper(usize),
    /// Equality row `i` (`a_L_i = a_U_i`).
    RowEquality(usize),
    /// Variable `j` at its lower bound.
    BoundLower(usize),
    /// Variable `j` at its upper bound.
    BoundUpper(usize),
}

/// Why a solve did not return an optimal point.
#[derive(Debug, Clone, PartialEq)]
pub enum QpError {
    /// The Hessian is not (numerically) positive definite; the message names
    /// the pivot. The caller should shift `H` and retry.
    NotPositiveDefinite(String),
    /// The constraints are inconsistent. Carries the certificate constraint.
    Infeasible(Constraint),
    /// The iteration cap was hit (should not happen for a positive definite `H`
    /// unless the problem is severely degenerate); the caller falls back.
    IterationLimit(usize),
    /// Inconsistent dimensions or non-finite data.
    InvalidData(String),
}

impl fmt::Display for QpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QpError::NotPositiveDefinite(m) => write!(f, "QP Hessian not positive definite: {m}"),
            QpError::Infeasible(c) => write!(f, "QP constraints inconsistent (certificate {c:?})"),
            QpError::IterationLimit(k) => write!(f, "QP active-set iteration limit {k} reached"),
            QpError::InvalidData(m) => write!(f, "invalid QP data: {m}"),
        }
    }
}

impl std::error::Error for QpError {}

/// A dense QP in the form documented at the crate root. All slices are borrowed;
/// `h` and `a` are row-major.
#[derive(Debug, Clone)]
pub struct DenseQp<'a> {
    /// Number of variables.
    pub n: usize,
    /// Number of general rows.
    pub m: usize,
    /// Symmetric positive definite Hessian, `n * n`, row-major.
    pub h: &'a [f64],
    /// Linear term, `n`.
    pub g: &'a [f64],
    /// Constraint matrix, `m * n`, row-major.
    pub a: &'a [f64],
    /// Row lower bounds, `m` (`-inf` allowed).
    pub a_l: &'a [f64],
    /// Row upper bounds, `m` (`+inf` allowed).
    pub a_u: &'a [f64],
    /// Variable lower bounds, `n` (`-inf` allowed).
    pub x_l: &'a [f64],
    /// Variable upper bounds, `n` (`+inf` allowed).
    pub x_u: &'a [f64],
}

/// The outcome of a QP solve.
#[derive(Debug, Clone)]
pub struct QpSolution {
    /// The minimizer.
    pub x: Vec<f64>,
    /// Row multipliers (`m`), signed as documented at the crate root.
    pub lambda: Vec<f64>,
    /// Lower-bound multipliers (`n`), non-negative.
    pub z_l: Vec<f64>,
    /// Upper-bound multipliers (`n`), non-negative.
    pub z_u: Vec<f64>,
    /// The active set at the solution, in working-set order.
    pub active: Vec<Constraint>,
    /// Constraint additions and deletions performed.
    pub iterations: usize,
    /// Objective value `1/2 x^T H x + g^T x`.
    pub objective: f64,
}

/// Solver options.
#[derive(Debug, Clone)]
pub struct QpOptions {
    /// A constraint is violated when `b - n^T x > tol * (1 + |b|) * ||n||`.
    pub feasibility_tol: f64,
    /// Relative threshold below which `z` (the step direction when a constraint
    /// is added) is treated as zero, i.e. the constraint is linearly dependent
    /// on the working set.
    pub dependence_tol: f64,
    /// Cap on additions plus deletions; `None` means `10 * (n + number of constraints) + 100`.
    pub max_iterations: Option<usize>,
}

impl Default for QpOptions {
    fn default() -> Self {
        Self {
            feasibility_tol: 1e-10,
            dependence_tol: 1e-12,
            max_iterations: None,
        }
    }
}

/// Solve a dense QP with the dual active-set method.
///
/// `hint` lists constraints expected to be active (from the previous SQP
/// iteration, say); violated constraints in the hint are added before other
/// violated constraints, which is a safe form of warm start for a dual method.
///
/// # Errors
/// See [`QpError`].
pub fn solve_dense(
    qp: &DenseQp<'_>,
    hint: &[Constraint],
    opts: &QpOptions,
) -> Result<QpSolution, QpError> {
    let (n, m) = (qp.n, qp.m);
    if qp.h.len() != n * n
        || qp.g.len() != n
        || qp.a.len() != m * n
        || qp.a_l.len() != m
        || qp.a_u.len() != m
        || qp.x_l.len() != n
        || qp.x_u.len() != n
    {
        return Err(QpError::InvalidData("dimension mismatch".into()));
    }
    if qp.h.iter().chain(qp.g).chain(qp.a).any(|v| !v.is_finite()) {
        return Err(QpError::InvalidData("non-finite H, g or A".into()));
    }
    if n == 0 {
        return Ok(QpSolution {
            x: vec![],
            lambda: vec![0.0; m],
            z_l: vec![],
            z_u: vec![],
            active: vec![],
            iterations: 0,
            objective: 0.0,
        });
    }

    // ---- constraints as n^T x >= b ------------------------------------------
    let mut cons: Vec<Constraint> = Vec::new();
    for i in 0..m {
        if qp.a_l[i] == qp.a_u[i] {
            cons.push(Constraint::RowEquality(i));
        }
    }
    let n_eq = cons.len();
    for i in 0..m {
        if qp.a_l[i] != qp.a_u[i] {
            if qp.a_l[i].is_finite() {
                cons.push(Constraint::RowLower(i));
            }
            if qp.a_u[i].is_finite() {
                cons.push(Constraint::RowUpper(i));
            }
        }
    }
    for j in 0..n {
        if qp.x_l[j].is_finite() {
            cons.push(Constraint::BoundLower(j));
        }
        if qp.x_u[j].is_finite() {
            cons.push(Constraint::BoundUpper(j));
        }
    }
    let p = cons.len();
    let max_iter = opts.max_iterations.unwrap_or(10 * (n + p) + 100);

    // normal vector (into `out`) and right-hand side of constraint k
    let normal = |k: usize, out: &mut [f64]| -> f64 {
        out.fill(0.0);
        match cons[k] {
            Constraint::RowEquality(i) | Constraint::RowLower(i) => {
                out.copy_from_slice(&qp.a[i * n..(i + 1) * n]);
                qp.a_l[i]
            }
            Constraint::RowUpper(i) => {
                for j in 0..n {
                    out[j] = -qp.a[i * n + j];
                }
                -qp.a_u[i]
            }
            Constraint::BoundLower(j) => {
                out[j] = 1.0;
                qp.x_l[j]
            }
            Constraint::BoundUpper(j) => {
                out[j] = -1.0;
                -qp.x_u[j]
            }
        }
    };
    let norms: Vec<f64> = {
        let mut buf = vec![0.0; n];
        (0..p)
            .map(|k| {
                normal(k, &mut buf);
                buf.iter()
                    .map(|v| v * v)
                    .sum::<f64>()
                    .sqrt()
                    .max(f64::MIN_POSITIVE)
            })
            .collect()
    };

    // ---- Cholesky H = L L^T, J = L^{-T} ---------------------------------------
    let l = cholesky(qp.h, n)?;
    // J starts as L^{-T}: solve L^T J = I column by column (J is n x n, row-major).
    let mut j_mat = vec![0.0; n * n];
    for col in 0..n {
        // solve L^T y = e_col by back substitution
        let mut y = vec![0.0; n];
        for i in (0..n).rev() {
            let mut s = if i == col { 1.0 } else { 0.0 };
            for k in i + 1..n {
                s -= l[k * n + i] * y[k];
            }
            y[i] = s / l[i * n + i];
        }
        for i in 0..n {
            j_mat[i * n + col] = y[i];
        }
    }
    // x = -H^{-1} g = -J J^T g
    let mut x = vec![0.0; n];
    {
        let mut jt_g = vec![0.0; n];
        for col in 0..n {
            jt_g[col] = (0..n).map(|i| j_mat[i * n + col] * qp.g[i]).sum();
        }
        for i in 0..n {
            x[i] = -(0..n)
                .map(|col| j_mat[i * n + col] * jt_g[col])
                .sum::<f64>();
        }
    }

    // working set
    let mut active: Vec<usize> = Vec::new(); // indices into cons
    let mut u: Vec<f64> = Vec::new(); // multipliers, same order
    let mut r_mat = vec![0.0; n * n]; // R (upper triangular, first q columns used), row-major
    let mut iterations = 0usize;

    let mut np = vec![0.0; n];
    let mut d = vec![0.0; n];
    let mut z = vec![0.0; n];
    let mut r = vec![0.0; n];

    // Order in which violated constraints are considered: hinted first.
    let hint_rank = |k: usize| -> usize {
        hint.iter()
            .position(|h| *h == cons[k])
            .map_or(usize::MAX, |pos| pos)
    };

    // ---- main loop ----------------------------------------------------------
    let mut forced_pending: Vec<usize> = (0..n_eq).collect(); // equalities to add first
    loop {
        // choose the constraint to add
        let chosen: Option<(usize, f64)> = if let Some(k) = forced_pending.first().copied() {
            let b = normal(k, &mut np);
            let s = b - dot(&np, &x);
            Some((k, s))
        } else {
            // (k, violation, hint rank, scaled violation): hinted constraints
            // first, then the most violated relative to its normal's length
            let mut best: Option<(usize, f64, usize, f64)> = None;
            for k in n_eq..p {
                if active.contains(&k) {
                    continue;
                }
                let b = normal(k, &mut np);
                let s = b - dot(&np, &x);
                let tol = opts.feasibility_tol * (1.0 + b.abs()) * norms[k];
                if s > tol {
                    let rank = hint_rank(k);
                    let scaled = s / norms[k];
                    let better = match best {
                        None => true,
                        Some((_, _, brank, bscaled)) => {
                            rank < brank || (rank == brank && scaled > bscaled)
                        }
                    };
                    if better {
                        best = Some((k, s, rank, scaled));
                    }
                }
            }
            best.map(|(k, s, _, _)| (k, s))
        };
        let Some((kp, mut s_p)) = chosen else {
            break; // no violated constraint: optimal
        };
        let forced = !forced_pending.is_empty();
        let b_p = normal(kp, &mut np);
        let mut u_p = 0.0;

        // inner loop: partial steps until a full step adds kp
        loop {
            if iterations >= max_iter {
                return Err(QpError::IterationLimit(iterations));
            }
            let q = active.len();
            // d = J^T n_p ; z = J_2 d_2 ; r = R^{-1} d_1
            for col in 0..n {
                d[col] = (0..n).map(|i| j_mat[i * n + col] * np[i]).sum();
            }
            for i in 0..n {
                z[i] = (q..n).map(|col| j_mat[i * n + col] * d[col]).sum();
            }
            for i in (0..q).rev() {
                let mut s = d[i];
                for k in i + 1..q {
                    s -= r_mat[i * n + k] * r[k];
                }
                r[i] = s / r_mat[i * n + i];
            }
            // Dependence: the part of d = J^T n_p outside the working set's span
            // (d_2, which generates z) is negligible relative to d itself.
            let d_norm = d.iter().map(|v| v * v).sum::<f64>().sqrt();
            let d2_norm = d[q..].iter().map(|v| v * v).sum::<f64>().sqrt();
            let np_norm = norms[kp];
            let z_zero = d2_norm <= opts.dependence_tol * d_norm.max(f64::MIN_POSITIVE);

            // t1: first working inequality multiplier to reach zero
            let mut t1 = f64::INFINITY;
            let mut l_drop: Option<usize> = None;
            for (idx, &k) in active.iter().enumerate() {
                if k < n_eq {
                    continue; // equality: multiplier free
                }
                if r[idx] > 0.0 {
                    let t = u[idx] / r[idx];
                    if t < t1 {
                        t1 = t;
                        l_drop = Some(idx);
                    }
                }
            }
            // t2: step to satisfy kp
            let ztnp = dot(&z, &np);
            let t2 = if z_zero || ztnp <= 0.0 {
                f64::INFINITY
            } else {
                s_p / ztnp
            };
            if forced {
                // equality row: take the step whatever its sign; dependence means
                // the equality is redundant (consistent) or inconsistent.
                if z_zero {
                    if s_p.abs() <= opts.feasibility_tol.sqrt() * (1.0 + b_p.abs()) * np_norm {
                        // redundant and consistent: skip it
                        forced_pending.remove(0);
                        iterations += 1;
                        break;
                    }
                    return Err(QpError::Infeasible(cons[kp]));
                }
                let t = s_p / ztnp;
                for i in 0..n {
                    x[i] += t * z[i];
                }
                for (idx, ui) in u.iter_mut().enumerate() {
                    *ui -= t * r[idx];
                }
                u_p += t;
                add_constraint(&mut j_mat, &mut r_mat, &mut d, n, q)?;
                active.push(kp);
                u.push(u_p);
                forced_pending.remove(0);
                iterations += 1;
                break;
            }
            if t1.is_infinite() && t2.is_infinite() {
                return Err(QpError::Infeasible(cons[kp]));
            }
            if t2.is_infinite() || t1 < t2 {
                // partial step: drop the blocking constraint, keep trying to add kp
                let t = t1;
                let idx = l_drop.expect("finite t1 has a blocking index");
                if !t2.is_infinite() {
                    for i in 0..n {
                        x[i] += t * z[i];
                    }
                    s_p -= t * ztnp;
                }
                for (i2, ui) in u.iter_mut().enumerate() {
                    *ui -= t * r[i2];
                }
                u_p += t;
                delete_constraint(&mut j_mat, &mut r_mat, n, q, idx);
                active.remove(idx);
                u.remove(idx);
                iterations += 1;
                continue;
            }
            // full step
            let t = t2;
            for i in 0..n {
                x[i] += t * z[i];
            }
            for (idx, ui) in u.iter_mut().enumerate() {
                *ui -= t * r[idx];
            }
            u_p += t;
            add_constraint(&mut j_mat, &mut r_mat, &mut d, n, q)?;
            active.push(kp);
            u.push(u_p);
            iterations += 1;
            break;
        }
    }

    // ---- assemble multipliers --------------------------------------------------
    let mut lambda = vec![0.0; m];
    let mut z_l = vec![0.0; n];
    let mut z_u = vec![0.0; n];
    for (idx, &k) in active.iter().enumerate() {
        let ui = u[idx];
        match cons[k] {
            Constraint::RowEquality(i) => lambda[i] = -ui,
            Constraint::RowLower(i) => lambda[i] = -ui.max(0.0),
            Constraint::RowUpper(i) => lambda[i] = ui.max(0.0),
            Constraint::BoundLower(j) => z_l[j] = ui.max(0.0),
            Constraint::BoundUpper(j) => z_u[j] = ui.max(0.0),
        }
    }
    let objective = {
        let mut hx = vec![0.0; n];
        for i in 0..n {
            hx[i] = (0..n).map(|j| qp.h[i * n + j] * x[j]).sum();
        }
        0.5 * dot(&x, &hx) + dot(qp.g, &x)
    };
    Ok(QpSolution {
        x,
        lambda,
        z_l,
        z_u,
        active: active.iter().map(|&k| cons[k]).collect(),
        iterations,
        objective,
    })
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Dense Cholesky of a row-major symmetric matrix; returns the lower factor.
fn cholesky(h: &[f64], n: usize) -> Result<Vec<f64>, QpError> {
    let mut l = vec![0.0; n * n];
    let scale = h
        .iter()
        .fold(0.0_f64, |a, v| a.max(v.abs()))
        .max(f64::MIN_POSITIVE);
    for j in 0..n {
        let mut s = h[j * n + j];
        for k in 0..j {
            s -= l[j * n + k] * l[j * n + k];
        }
        if s.is_nan() || s <= 1e-15 * scale {
            return Err(QpError::NotPositiveDefinite(format!(
                "pivot {j} is {s:.3e} (matrix scale {scale:.3e})"
            )));
        }
        let ljj = s.sqrt();
        l[j * n + j] = ljj;
        for i in j + 1..n {
            let mut s = h[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = s / ljj;
        }
    }
    Ok(l)
}

/// Add the constraint whose transformed normal is `d = J^T n_p` as working
/// constraint number `q`: rotate `d[q+1..n]` into `d[q]` with Givens rotations
/// applied to the columns of `J`, then store `d[..=q]` as column `q` of `R`.
fn add_constraint(
    j_mat: &mut [f64],
    r_mat: &mut [f64],
    d: &mut [f64],
    n: usize,
    q: usize,
) -> Result<(), QpError> {
    for i in (q + 1..n).rev() {
        let (a, b) = (d[i - 1], d[i]);
        if b == 0.0 {
            continue;
        }
        let h = a.hypot(b);
        let (c, s) = (a / h, b / h);
        d[i - 1] = h;
        d[i] = 0.0;
        // columns i-1 and i of J: [col_{i-1}, col_i] <- [c*col_{i-1} + s*col_i, -s*col_{i-1} + c*col_i]
        for row in 0..n {
            let x = j_mat[row * n + i - 1];
            let y = j_mat[row * n + i];
            j_mat[row * n + i - 1] = c * x + s * y;
            j_mat[row * n + i] = -s * x + c * y;
        }
    }
    if q >= n {
        return Err(QpError::IterationLimit(q)); // cannot add more than n constraints
    }
    for i in 0..=q {
        r_mat[i * n + q] = d[i];
    }
    Ok(())
}

/// Remove working constraint `idx` (of `q`): delete column `idx` of `R`, shift
/// the later columns left and re-triangularize with Givens rotations, applying
/// each rotation to the corresponding columns of `J`.
fn delete_constraint(j_mat: &mut [f64], r_mat: &mut [f64], n: usize, q: usize, idx: usize) {
    // shift columns idx+1..q of R to idx..q-1
    for col in idx + 1..q {
        for row in 0..n {
            r_mat[row * n + col - 1] = r_mat[row * n + col];
        }
    }
    for row in 0..n {
        r_mat[row * n + q - 1] = 0.0;
    }
    // R now has a subdiagonal in columns idx..q-1 (rows k+1 for column k); zero them.
    for k in idx..q - 1 {
        let (a, b) = (r_mat[k * n + k], r_mat[(k + 1) * n + k]);
        if b == 0.0 {
            continue;
        }
        let h = a.hypot(b);
        let (c, s) = (a / h, b / h);
        // rows k and k+1 of R
        for col in k..q - 1 {
            let x = r_mat[k * n + col];
            let y = r_mat[(k + 1) * n + col];
            r_mat[k * n + col] = c * x + s * y;
            r_mat[(k + 1) * n + col] = -s * x + c * y;
        }
        // columns k and k+1 of J (J <- J G^T)
        for row in 0..n {
            let x = j_mat[row * n + k];
            let y = j_mat[row * n + k + 1];
            j_mat[row * n + k] = c * x + s * y;
            j_mat[row * n + k + 1] = -s * x + c * y;
        }
    }
}

/// Whether this solver is available.
#[must_use]
pub fn is_available() -> bool {
    true
}

#[cfg(test)]
mod tests;
