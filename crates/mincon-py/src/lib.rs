//! Python bindings.
//!
//! # API shape: be a drop-in for `scipy.optimize.minimize`
//!
//! The point of this package is that someone with a working SciPy script can
//! change one import and get a better answer. So the signature, the
//! `constraints` dict format (`{'type': 'eq'|'ineq', 'fun': ...}`, with `ineq`
//! meaning `fun(x) >= 0`, which is SciPy's convention and the *opposite* of
//! `fmincon`'s), and the result object's field names all follow SciPy.
//!
//! Where SciPy and `fmincon` disagree we follow SciPy, because that is who the
//! caller is, and we document the `fmincon` difference in the docstring.
//!
//! # The GIL is the performance story
//!
//! Every objective evaluation crosses into Python and needs the GIL, so
//! finite differences cannot be parallelized for a Python callback and
//! `parallel_safe` is `false`. This is why a model expressed in NumPy still
//! spends most of its time in Python, and why the Rust core being fast is only
//! half the win. The other half — batched evaluation, so `n` finite-difference
//! probes cross the boundary once instead of `n` times — is specified in
//! `docs/07_API_DESIGN.md` and is the single biggest speedup available on the
//! Python side.

use std::sync::Mutex;

use mincon_core::{
    Algorithm, BarrierUpdate, Capabilities, EvalError, ExitFlag, FdType, Nlp, NlpDims, Options,
    ScalingMode, Sparsity, Tolerances, INF_BOUND,
};
use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

/// One constraint block: (callable, is_equality, block length, optional Jacobian callable).
type Block = (Py<PyAny>, bool, usize, Option<Py<PyAny>>);

/// A model whose pieces are Python callables.
struct PyNlp {
    n: usize,
    m: usize,
    fun: Py<PyAny>,
    jac: Option<Py<PyAny>>,
    /// One entry per constraint block: (callable, is_equality, block length, optional Jacobian callable).
    blocks: Vec<Block>,
    /// Dense `m x n` structure, present only when every block supplied a Jacobian.
    jac_structure: Option<Sparsity>,
    lb: Vec<f64>,
    ub: Vec<f64>,
    cl: Vec<f64>,
    cu: Vec<f64>,
    x0: Vec<f64>,
    /// Set when a callback raised, so the original exception can be re-raised
    /// rather than turned into a meaningless "solver failed".
    error: Mutex<Option<String>>,
}

impl PyNlp {
    fn record(&self, e: &PyErr) {
        if let Ok(mut slot) = self.error.lock() {
            if slot.is_none() {
                *slot = Some(e.to_string());
            }
        }
    }
}

impl Nlp for PyNlp {
    fn dims(&self) -> NlpDims {
        NlpDims {
            n: self.n,
            m: self.m,
        }
    }
    fn x_bounds(&self) -> (&[f64], &[f64]) {
        (&self.lb, &self.ub)
    }
    fn c_bounds(&self) -> (&[f64], &[f64]) {
        (&self.cl, &self.cu)
    }
    fn x0(&self) -> &[f64] {
        &self.x0
    }
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            gradient: self.jac.is_some(),
            jacobian: self.jac_structure.is_some(),
            // Every call needs the GIL, so concurrent evaluation buys nothing.
            parallel_safe: false,
            ..Capabilities::none()
        }
    }

    fn jacobian_structure(&self) -> Option<&Sparsity> {
        self.jac_structure.as_ref()
    }

    /// Dense Jacobian assembled block by block, written in the column-major
    /// order of the dense structure. Each block's callable may return a
    /// `(len, n)` array, an `(n,)` array for a single row, or nested lists.
    fn jacobian(&self, x: &[f64], out: &mut [f64]) -> Result<(), EvalError> {
        let (n, m) = (self.n, self.m);
        if m == 0 {
            return Ok(());
        }
        Python::attach(|py| {
            let arr = x.to_pyarray(py);
            let mut row0 = 0usize;
            for (_, _, len, jac) in &self.blocks {
                let Some(jac) = jac else {
                    return Err(EvalError::Failed("a constraint block has no jac".into()));
                };
                let v = jac.call1(py, (arr.clone(),)).map_err(|e| {
                    self.record(&e);
                    EvalError::Failed(e.to_string())
                })?;
                // Accept 2-D (len x n), or 1-D (n) when len == 1, via numpy's ravel of a float array.
                let flat: Vec<f64> = match v.extract::<Vec<Vec<f64>>>(py) {
                    Ok(rows) => rows.into_iter().flatten().collect(),
                    Err(_) => match v.extract::<Vec<f64>>(py) {
                        Ok(row) => row,
                        Err(_) => {
                            // numpy arrays: go through np.asarray(v, float).ravel()
                            let np = py
                                .import("numpy")
                                .map_err(|e| EvalError::Failed(e.to_string()))?;
                            let a = np
                                .call_method1("asarray", (v.bind(py), "float64"))
                                .and_then(|a| a.call_method0("ravel"))
                                .map_err(|e| EvalError::Failed(e.to_string()))?;
                            a.extract::<Vec<f64>>().map_err(|e| {
                                EvalError::Failed(format!(
                                    "constraint jac did not return numbers: {e}"
                                ))
                            })?
                        }
                    },
                };
                if flat.len() != len * n {
                    return Err(EvalError::Failed(format!(
                        "a constraint jac returned {} values; expected {len} x {n}",
                        flat.len()
                    )));
                }
                for i in 0..*len {
                    for j in 0..n {
                        out[j * m + row0 + i] = flat[i * n + j];
                    }
                }
                row0 += len;
            }
            if out.iter().all(|v| v.is_finite()) {
                Ok(())
            } else {
                Err(EvalError::NonFinite(None))
            }
        })
    }

    fn objective(&self, x: &[f64]) -> Result<f64, EvalError> {
        Python::attach(|py| {
            let arr = x.to_pyarray(py);
            match self.fun.call1(py, (arr,)) {
                Ok(v) => match v.extract::<f64>(py) {
                    Ok(f) if f.is_finite() => Ok(f),
                    Ok(_) => Err(EvalError::NonFinite(None)),
                    Err(e) => {
                        self.record(&e);
                        Err(EvalError::Failed(format!(
                            "objective did not return a float: {e}"
                        )))
                    }
                },
                Err(e) => {
                    self.record(&e);
                    Err(EvalError::Failed(e.to_string()))
                }
            }
        })
    }

    fn gradient(&self, x: &[f64], out: &mut [f64]) -> Result<(), EvalError> {
        let Some(jac) = &self.jac else {
            return Err(EvalError::Failed("no jac supplied".into()));
        };
        Python::attach(|py| {
            let arr = x.to_pyarray(py);
            match jac.call1(py, (arr,)) {
                Ok(v) => match v.extract::<Vec<f64>>(py) {
                    Ok(g) if g.len() == out.len() => {
                        out.copy_from_slice(&g);
                        if out.iter().all(|v| v.is_finite()) {
                            Ok(())
                        } else {
                            Err(EvalError::NonFinite(None))
                        }
                    }
                    Ok(g) => Err(EvalError::Failed(format!(
                        "jac returned {} values, expected {}",
                        g.len(),
                        out.len()
                    ))),
                    Err(e) => {
                        self.record(&e);
                        Err(EvalError::Failed(e.to_string()))
                    }
                },
                Err(e) => {
                    self.record(&e);
                    Err(EvalError::Failed(e.to_string()))
                }
            }
        })
    }

    fn constraints(&self, x: &[f64], out: &mut [f64]) -> Result<(), EvalError> {
        if self.m == 0 {
            return Ok(());
        }
        Python::attach(|py| {
            let arr = x.to_pyarray(py);
            let mut off = 0usize;
            for (f, _, len, _) in &self.blocks {
                let v = f.call1(py, (arr.clone(),)).map_err(|e| {
                    self.record(&e);
                    EvalError::Failed(e.to_string())
                })?;
                let vals: Vec<f64> = v
                    .extract::<Vec<f64>>(py)
                    .or_else(|_| v.extract::<f64>(py).map(|s| vec![s]))
                    .map_err(|e| {
                        self.record(&e);
                        EvalError::Failed(format!("constraint did not return numbers: {e}"))
                    })?;
                if vals.len() != *len {
                    return Err(EvalError::Failed(format!(
                        "a constraint returned {} values but returned {len} at the starting point; \
                         the number of constraints must not change between calls",
                        vals.len()
                    )));
                }
                out[off..off + len].copy_from_slice(&vals);
                off += len;
            }
            if out.iter().all(|v| v.is_finite()) {
                Ok(())
            } else {
                Err(EvalError::NonFinite(None))
            }
        })
    }
}

/// Probe a constraint callable once to learn how many values it returns.
fn block_length(py: Python<'_>, f: &Py<PyAny>, x0: &[f64]) -> PyResult<usize> {
    let arr = x0.to_pyarray(py);
    let v = f.call1(py, (arr,))?;
    if let Ok(vals) = v.extract::<Vec<f64>>(py) {
        Ok(vals.len())
    } else if v.extract::<f64>(py).is_ok() {
        Ok(1)
    } else {
        Err(PyValueError::new_err(
            "a constraint's 'fun' must return a float or a 1-D array of floats",
        ))
    }
}

fn parse_options(py: Python<'_>, options: Option<&Bound<'_, PyDict>>) -> PyResult<Options> {
    let mut o = Options::default();
    let Some(d) = options else { return Ok(o) };

    macro_rules! get {
        ($key:expr, $ty:ty) => {
            match d.get_item($key)? {
                Some(v) if !v.is_none() => Some(v.extract::<$ty>()?),
                _ => None,
            }
        };
    }

    if let Some(v) = get!("maxiter", usize) {
        o.max_iterations = v;
    }
    if let Some(v) = get!("maxfev", u64) {
        o.max_evaluations = Some(v);
    }
    if let Some(v) = get!("maxtime", f64) {
        o.max_seconds = Some(v);
    }
    if let Some(v) = get!("tol", f64) {
        o.tol = Tolerances {
            optimality: v,
            feasibility: v,
            complementarity: v,
            ..o.tol
        };
    }
    if let Some(v) = get!("ftol", f64) {
        o.tol.optimality = v;
    }
    if let Some(v) = get!("ctol", f64) {
        o.tol.feasibility = v;
    }
    if let Some(v) = get!("threads", usize) {
        o.threads = Some(v);
    }
    if let Some(v) = get!("seed", u64) {
        o.seed = v;
    }
    if let Some(v) = get!("check_derivatives", bool) {
        o.check_derivatives = v;
    }
    if let Some(v) = get!("scaling", String) {
        o.scaling = match v.as_str() {
            "none" => ScalingMode::None,
            "gradient" | "gradient-based" => ScalingMode::GradientBased,
            "equilibration" => ScalingMode::Equilibration,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown scaling '{other}'; use 'none', 'gradient' or 'equilibration'"
                )))
            }
        };
    }
    if let Some(v) = get!("bfgs_scaling", bool) {
        o.bfgs_guarded_scaling = v;
    }
    if let Some(v) = get!("bfgs_rescale", f64) {
        o.bfgs_curvature_rescale = if v > 1.0 { v } else { f64::INFINITY };
    }
    if let Some(v) = get!("fd_error_aware", bool) {
        o.fd_error_aware = v;
    }
    if let Some(v) = get!("barrier", String) {
        o.barrier_update = match v.as_str() {
            "monotone" => BarrierUpdate::Monotone,
            "adaptive" => BarrierUpdate::Adaptive,
            "adaptive-then-monotone" | "auto" => BarrierUpdate::AdaptiveThenMonotone,
            other => {
                return Err(PyValueError::new_err(format!(
                "unknown barrier '{other}'; use 'monotone', 'adaptive' or 'adaptive-then-monotone'"
            )))
            }
        };
    }
    if let Some(v) = get!("finite_diff", String) {
        o.fd_type = match v.as_str() {
            "forward" => FdType::Forward,
            "central" => FdType::Central,
            "adaptive" => FdType::Adaptive,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown finite_diff '{other}'; use 'forward', 'central' or 'adaptive'"
                )))
            }
        };
    }
    let _ = py;
    Ok(o)
}

fn parse_algorithm(method: Option<&str>) -> PyResult<Algorithm> {
    Ok(match method.map(str::to_ascii_lowercase).as_deref() {
        None | Some("auto") => Algorithm::Auto,
        Some("interior-point" | "ip") => Algorithm::InteriorPoint,
        Some("sqp") => Algorithm::Sqp,
        Some(other) => {
            return Err(PyValueError::new_err(format!(
                "unknown method '{other}'; use 'auto', 'interior-point' or 'sqp'"
            )))
        }
    })
}

/// Minimize a scalar function subject to bounds and constraints.
///
/// Deliberately shaped like `scipy.optimize.minimize`, so an existing script
/// needs only a changed import.
#[pyfunction]
#[pyo3(signature = (fun, x0, jac=None, bounds=None, constraints=None, method=None, options=None))]
#[allow(clippy::too_many_arguments, clippy::needless_pass_by_value)]
fn minimize(
    py: Python<'_>,
    fun: Py<PyAny>,
    x0: PyReadonlyArray1<'_, f64>,
    jac: Option<Py<PyAny>>,
    bounds: Option<Bound<'_, PyAny>>,
    constraints: Option<Bound<'_, PyAny>>,
    method: Option<&str>,
    options: Option<Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let x0v: Vec<f64> = x0.as_slice()?.to_vec();
    let n = x0v.len();
    if n == 0 {
        return Err(PyValueError::new_err("x0 must not be empty"));
    }

    // --- bounds ---
    let mut lb = vec![-INF_BOUND; n];
    let mut ub = vec![INF_BOUND; n];
    if let Some(b) = &bounds {
        if !b.is_none() {
            let seq = b.cast::<PyList>().map(Bound::to_owned).or_else(|_| {
                b.cast::<PyTuple>()
                    .map(|t| PyList::new(py, t.iter()).expect("list from tuple"))
            });
            let Ok(seq) = seq else {
                return Err(PyValueError::new_err(
                    "bounds must be a sequence of (low, high) pairs",
                ));
            };
            if seq.len() != n {
                return Err(PyValueError::new_err(format!(
                    "bounds has {} entries but x0 has {n}",
                    seq.len()
                )));
            }
            for (i, item) in seq.iter().enumerate() {
                let pair: Vec<Option<f64>> = item.extract()?;
                if pair.len() != 2 {
                    return Err(PyValueError::new_err(
                        "each bound must be a (low, high) pair",
                    ));
                }
                lb[i] = pair[0].unwrap_or(-INF_BOUND);
                ub[i] = pair[1].unwrap_or(INF_BOUND);
                if lb[i].is_nan() || ub[i].is_nan() || lb[i] > ub[i] {
                    return Err(PyValueError::new_err(format!(
                        "variable {i} has lower bound {} above upper bound {}",
                        lb[i], ub[i]
                    )));
                }
            }
        }
    }

    if x0v.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err("x0 must contain finite values"));
    }
    // Python needs a constraint-size probe before constructing Nlp. Honor the
    // same physical box as the Rust setup and all subsequent evaluations.
    let probe: Vec<f64> = (0..n).map(|i| x0v[i].clamp(lb[i], ub[i])).collect();
    // --- constraints ---
    let mut blocks: Vec<Block> = Vec::new();
    let mut cl: Vec<f64> = Vec::new();
    let mut cu: Vec<f64> = Vec::new();
    if let Some(c) = &constraints {
        if !c.is_none() {
            let items: Vec<Bound<'_, PyAny>> = if c.cast::<PyDict>().is_ok() {
                vec![c.clone()]
            } else {
                c.try_iter()?.collect::<PyResult<Vec<_>>>()?
            };
            for item in items {
                let d = item.cast::<PyDict>().map_err(|_| {
                    PyValueError::new_err(
                        "each constraint must be a dict with 'type' and 'fun' keys",
                    )
                })?;
                let ty: String = d
                    .get_item("type")?
                    .ok_or_else(|| PyValueError::new_err("constraint is missing 'type'"))?
                    .extract()?;
                let f: Py<PyAny> = d
                    .get_item("fun")?
                    .ok_or_else(|| PyValueError::new_err("constraint is missing 'fun'"))?
                    .unbind();
                let jac_fn: Option<Py<PyAny>> = match d.get_item("jac")? {
                    Some(j) if !j.is_none() => {
                        if !j.is_callable() {
                            return Err(PyValueError::new_err(
                                "a constraint's 'jac' must be callable",
                            ));
                        }
                        Some(j.unbind())
                    }
                    _ => None,
                };
                let len = block_length(py, &f, &probe)?;
                let is_eq = match ty.as_str() {
                    "eq" => true,
                    "ineq" => false,
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "constraint type must be 'eq' or 'ineq', got '{other}'"
                        )))
                    }
                };
                for _ in 0..len {
                    if is_eq {
                        cl.push(0.0);
                        cu.push(0.0);
                    } else {
                        // SciPy convention: 'ineq' means fun(x) >= 0.
                        cl.push(0.0);
                        cu.push(INF_BOUND);
                    }
                }
                blocks.push((f, is_eq, len, jac_fn));
            }
        }
    }

    let m = cl.len();
    let mut opts = parse_options(py, options.as_ref())?;
    opts.algorithm = parse_algorithm(method)?;
    // Analytic Jacobian only when every block supplies one: a partial Jacobian
    // would silently mix exact and approximate rows.
    let jac_structure = if m > 0 && blocks.iter().all(|b| b.3.is_some()) {
        Some(Sparsity::dense(m, n))
    } else {
        None
    };

    let nlp = PyNlp {
        n,
        m,
        fun,
        jac,
        blocks,
        jac_structure,
        lb,
        ub,
        cl,
        cu,
        x0: x0v,
        error: Mutex::new(None),
    };

    // Release the GIL for the solve; every callback re-acquires it. Without
    // this, a solve blocks every other Python thread for its whole duration.
    let outcome = py.detach(|| mincon::minimize(&nlp, &opts));

    let report = match outcome {
        Ok(r) => r,
        Err(e) => {
            if let Ok(slot) = nlp.error.lock() {
                if let Some(msg) = slot.as_ref() {
                    return Err(PyRuntimeError::new_err(format!(
                        "a callback raised: {msg} (solver reported: {e})"
                    )));
                }
            }
            return Err(PyRuntimeError::new_err(e.to_string()));
        }
    };

    let d = PyDict::new(py);
    d.set_item("x", report.solution.x.to_pyarray(py))?;
    d.set_item("fun", report.solution.f)?;
    d.set_item("success", report.exit_flag.is_success())?;
    d.set_item("status", report.exit_flag as i32)?;
    d.set_item("message", report.exit_flag.message())?;
    d.set_item("nit", report.iterations)?;
    d.set_item("nfev", report.f_evals)?;
    d.set_item("njev", report.g_evals)?;
    d.set_item("ncev", report.c_evals)?;
    d.set_item("ncjev", report.j_evals)?;
    d.set_item("maxcv", report.constraint_violation)?;
    d.set_item("optimality", report.optimality)?;
    d.set_item("usable", report.exit_flag.returned_usable_point())?;
    d.set_item("algorithm", format!("{:?}", report.algorithm))?;
    d.set_item("con", report.solution.c.to_pyarray(py))?;
    d.set_item("lambda", report.solution.lambda.to_pyarray(py))?;
    d.set_item("z_l", report.solution.z_l.to_pyarray(py))?;
    d.set_item("z_u", report.solution.z_u.to_pyarray(py))?;
    d.set_item("notes", PyList::new(py, &report.notes)?)?;
    let trace = PyList::empty(py);
    for t in &report.trace {
        let row = PyDict::new(py);
        row.set_item("iter", t.iter)?;
        row.set_item("nfev", t.f_count)?;
        row.set_item("f", t.f)?;
        row.set_item("maxcv", t.constraint_violation)?;
        row.set_item("optimality", t.optimality)?;
        row.set_item("step_norm", t.step_norm)?;
        row.set_item("alpha", t.alpha)?;
        row.set_item("mu", t.mu)?;
        row.set_item("in_restoration", t.in_restoration)?;
        row.set_item("delta_w", t.delta_w)?;
        row.set_item("delta_c", t.delta_c)?;
        row.set_item("soc", t.soc_count)?;
        trace.append(row)?;
    }
    d.set_item("trace", trace)?;
    d.set_item("time", report.timings.total.as_secs_f64())?;
    d.set_item("model_time", report.timings.model.as_secs_f64())?;
    Ok(d.into_any().unbind())
}

/// Check analytic derivatives against finite differences.
#[pyfunction]
#[pyo3(signature = (fun, x0, jac, num_points=3, tol=1e-5))]
fn check_gradients(
    py: Python<'_>,
    fun: Py<PyAny>,
    x0: PyReadonlyArray1<'_, f64>,
    jac: Py<PyAny>,
    num_points: usize,
    tol: f64,
) -> PyResult<Py<PyAny>> {
    let x0v: Vec<f64> = x0.as_slice()?.to_vec();
    let n = x0v.len();
    let nlp = PyNlp {
        n,
        m: 0,
        fun,
        jac: Some(jac),
        blocks: Vec::new(),
        jac_structure: None,
        lb: vec![-INF_BOUND; n],
        ub: vec![INF_BOUND; n],
        cl: Vec::new(),
        cu: Vec::new(),
        x0: x0v,
        error: Mutex::new(None),
    };
    let report = py
        .detach(|| mincon::check_derivatives(&nlp, tol, num_points))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let d = PyDict::new(py);
    d.set_item("passed", report.passed())?;
    d.set_item("conclusive", report.conclusive())?;
    d.set_item("comparisons", report.comparisons)?;
    d.set_item("nonfinite", report.nonfinite)?;
    d.set_item("max_relative_error", report.max_relative)?;
    d.set_item("message", report.message())?;
    Ok(d.into_any().unbind())
}

/// The exit-flag integers, exposed so callers can branch on them by name.
#[pyfunction]
fn exit_flags(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let d = PyDict::new(py);
    for (name, flag) in [
        ("OPTIMAL", ExitFlag::Optimal),
        ("STEP_TOLERANCE", ExitFlag::StepTolerance),
        ("FUNCTION_TOLERANCE", ExitFlag::FunctionTolerance),
        ("ACCEPTABLE", ExitFlag::Acceptable),
        ("MAX_REACHED", ExitFlag::MaxReached),
        ("STOPPED_BY_USER", ExitFlag::StoppedByUser),
        ("INFEASIBLE", ExitFlag::Infeasible),
        ("UNBOUNDED", ExitFlag::Unbounded),
        ("LOCALLY_INFEASIBLE", ExitFlag::LocallyInfeasible),
        ("NUMERICAL_FAILURE", ExitFlag::NumericalFailure),
    ] {
        d.set_item(name, flag as i32)?;
    }
    Ok(d.into_any().unbind())
}

/// Silence the unused-import warning for `PyArray1` while keeping the type in
/// scope for future signatures that return arrays directly.
#[allow(dead_code)]
fn _type_anchor(py: Python<'_>) -> Bound<'_, PyArray1<f64>> {
    [0.0f64].to_pyarray(py)
}

#[allow(dead_code)]
fn _sparsity_anchor() -> Option<Sparsity> {
    None
}

#[pymodule]
fn _mincon(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(minimize, m)?)?;
    m.add_function(wrap_pyfunction!(check_gradients, m)?)?;
    m.add_function(wrap_pyfunction!(exit_flags, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
