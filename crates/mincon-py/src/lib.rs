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
//! Every objective evaluation crosses into Python and needs the GIL, so the
//! engine's thread pool cannot run a Python callback concurrently and
//! `parallel_safe` is `false`. What it can do is hand Python a whole gradient's
//! finite-difference probes at once ([`Nlp::objective_batch`]): the Python
//! layer builds a batch callable when the caller asks for one, from a
//! vectorised model (`vectorized=True`: one call on a `(k, n)` array) or from a
//! pool of worker processes (`workers=k`), and the probes cross the boundary
//! once per gradient instead of `n` times. The points are those of the serial
//! path; without either option no batch callable exists and nothing changes.

use std::sync::{Arc, Mutex};

use mincon_core::{
    Algorithm, BarrierUpdate, Capabilities, DerivativeCheck, EvalError, ExitFlag, FdType,
    IterationCallback, IterationRecord, Nlp, NlpDims, Options, PivotSigns, QuadraticBands,
    QuadraticBuild, QuadraticRows, SaddleStep, ScalingMode, Sparsity, Tolerances, VariableScaling,
    ZeroStep, INF_BOUND,
};
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArrayDyn, ToPyArray};
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
    /// `hess(x, lam) -> (n, n)`: the Hessian of the Lagrangian `f + sum lam_i c_i`.
    hess: Option<Py<PyAny>>,
    /// Dense lower triangle, present when `hess` is.
    hess_structure: Option<Sparsity>,
    /// One entry per constraint block: (callable, is_equality, block length, optional Jacobian callable).
    blocks: Vec<Block>,
    /// Dense `m x n` structure, present only when every block supplied a Jacobian.
    jac_structure: Option<Sparsity>,
    /// `fun_batch(X) -> (values, errors)` for a `(k, n)` array of points: `k`
    /// floats, and `None` or a list of `k` entries, each `None` or the message
    /// of the exception the model raised at that point.
    fun_batch: Option<Py<PyAny>>,
    /// `cons_batch(X) -> (values, errors)` with a `(k, m)` array of values.
    cons_batch: Option<Py<PyAny>>,
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
        self.record_message(&e.to_string());
    }

    fn record_message(&self, message: &str) {
        if let Ok(mut slot) = self.error.lock() {
            if slot.is_none() {
                *slot = Some(message.to_string());
            }
        }
    }

    /// Call a batch callable on the `k` points of `xs` and hand `write` the
    /// flat values; the result is one status per point. A failure of the call
    /// as a whole fails every point, so the derivative layer retreats or gives
    /// up exactly as it does when a scalar callback raises.
    fn call_batch(
        &self,
        batch: &Py<PyAny>,
        xs: &[f64],
        width: usize,
        mut write: impl FnMut(&[f64]),
    ) -> Vec<Result<(), EvalError>> {
        let k = xs.len() / self.n;
        let all = |e: EvalError| -> Vec<Result<(), EvalError>> { vec![Err(e); k] };
        Python::attach(|py| {
            let points = match xs.to_pyarray(py).reshape([k, self.n]) {
                Ok(a) => a,
                Err(e) => return all(EvalError::Failed(e.to_string())),
            };
            let answer = match batch.call1(py, (points,)) {
                Ok(v) => v,
                Err(e) => {
                    self.record(&e);
                    return all(EvalError::Failed(e.to_string()));
                }
            };
            let parsed = answer
                .extract::<(Bound<'_, PyAny>, Option<Vec<Option<String>>>)>(py)
                .and_then(|(values, errors)| {
                    // a C-contiguous float64 array: (k,) for the objective, (k, m) for rows
                    let flat: Vec<f64> = values
                        .extract::<PyReadonlyArrayDyn<'_, f64>>()?
                        .as_slice()?
                        .to_vec();
                    Ok((flat, errors))
                });
            let (flat, errors) = match parsed {
                Ok(v) => v,
                Err(e) => {
                    self.record(&e);
                    return all(EvalError::Failed(format!(
                        "the batch evaluator returned something unexpected: {e}"
                    )));
                }
            };
            if flat.len() != k * width || errors.as_ref().is_some_and(|e| e.len() != k) {
                return all(EvalError::Failed(format!(
                    "the batch evaluator returned {} values for {k} points of {width}",
                    flat.len()
                )));
            }
            write(&flat);
            (0..k)
                .map(|i| {
                    if let Some(message) = errors.as_ref().and_then(|e| e[i].as_ref()) {
                        self.record_message(message);
                        return Err(EvalError::Failed(message.clone()));
                    }
                    if flat[i * width..(i + 1) * width]
                        .iter()
                        .all(|v| v.is_finite())
                    {
                        Ok(())
                    } else {
                        Err(EvalError::NonFinite(None))
                    }
                })
                .collect()
        })
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
            hessian: self.hess.is_some(),
            // Every call needs the GIL, so concurrent evaluation buys nothing.
            parallel_safe: false,
            // The Python layer built a batch callable (a vectorised model or a
            // pool of workers): a gradient's probes cross the boundary at once.
            batch: self.fun_batch.is_some() || self.cons_batch.is_some(),
            ..Capabilities::none()
        }
    }

    fn objective_batch(&self, xs: &[f64]) -> Vec<Result<f64, EvalError>> {
        let Some(batch) = &self.fun_batch else {
            return xs.chunks_exact(self.n).map(|x| self.objective(x)).collect();
        };
        let mut values = Vec::new();
        let status = self.call_batch(batch, xs, 1, |flat| values = flat.to_vec());
        status
            .into_iter()
            .enumerate()
            .map(|(i, s)| s.map(|()| values[i]))
            .collect()
    }

    fn constraints_batch(&self, xs: &[f64], out: &mut [f64]) -> Vec<Result<(), EvalError>> {
        let m = self.m;
        let Some(batch) = &self.cons_batch else {
            if m == 0 {
                return xs.chunks_exact(self.n).map(|_| Ok(())).collect();
            }
            return xs
                .chunks_exact(self.n)
                .zip(out.chunks_exact_mut(m))
                .map(|(x, o)| self.constraints(x, o))
                .collect();
        };
        self.call_batch(batch, xs, m, |flat| out.copy_from_slice(flat))
    }

    fn jacobian_structure(&self) -> Option<&Sparsity> {
        self.jac_structure.as_ref()
    }

    fn hessian_structure(&self) -> Option<&Sparsity> {
        self.hess_structure.as_ref()
    }

    /// `sigma * hess_f + sum lam_i hess_c_i` from the user's `hess(x, lam)`, which
    /// returns the Lagrangian Hessian at `sigma = 1`; other `sigma` values (the
    /// restoration phase asks for 0) cost a second call at `lam = 0`.
    fn hessian_lagrangian(
        &self,
        x: &[f64],
        sigma: f64,
        lambda: &[f64],
        out: &mut [f64],
    ) -> Result<(), EvalError> {
        let Some(hess) = &self.hess else {
            return Err(EvalError::Failed("no hess supplied".into()));
        };
        let n = self.n;
        Python::attach(|py| {
            let full = |lam: &[f64]| -> Result<Vec<f64>, EvalError> {
                let v = hess
                    .call1(py, (x.to_pyarray(py), lam.to_pyarray(py)))
                    .map_err(|e| {
                        self.record(&e);
                        EvalError::Failed(e.to_string())
                    })?;
                let np = py
                    .import("numpy")
                    .map_err(|e| EvalError::Failed(e.to_string()))?;
                let a = np
                    .call_method1("asarray", (v.bind(py), "float64"))
                    .and_then(|a| a.call_method0("ravel"))
                    .map_err(|e| EvalError::Failed(e.to_string()))?;
                let flat: Vec<f64> = a
                    .extract()
                    .map_err(|e| EvalError::Failed(format!("hess did not return numbers: {e}")))?;
                if flat.len() != n * n {
                    return Err(EvalError::Failed(format!(
                        "hess returned {} values; expected an ({n}, {n}) matrix",
                        flat.len()
                    )));
                }
                Ok(flat)
            };
            let h_lam = full(lambda)?;
            let vals = if (sigma - 1.0).abs() <= f64::EPSILON {
                h_lam
            } else {
                let h0 = full(&vec![0.0; lambda.len()])?;
                (0..n * n)
                    .map(|k| sigma * h0[k] + (h_lam[k] - h0[k]))
                    .collect()
            };
            let mut pos = 0usize;
            for j in 0..n {
                for i in j..n {
                    out[pos] = 0.5 * (vals[i * n + j] + vals[j * n + i]);
                    pos += 1;
                }
            }
            if out.iter().all(|v| v.is_finite()) {
                Ok(())
            } else {
                Err(EvalError::NonFinite(None))
            }
        })
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

/// Dense lower-triangular pattern in column-major order, matching the fill
/// order of `PyNlp::hessian_lagrangian`.
fn lower_triangle(n: usize) -> Sparsity {
    let mut triplets = Vec::with_capacity(n * (n + 1) / 2);
    for j in 0..n {
        for i in j..n {
            triplets.push((i, j));
        }
    }
    Sparsity::from_triplets(n, n, &triplets).expect("valid lower triangle")
}

/// The iteration record as the Python-side dict (`result.trace` rows and the
/// callback's argument).
fn record_to_dict<'py>(py: Python<'py>, t: &IterationRecord) -> PyResult<Bound<'py, PyDict>> {
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
    Ok(row)
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
        o.max_iterations = Some(v);
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
    if let Some(v) = get!("xtol", f64) {
        o.tol.step = v;
    }
    if let Some(v) = get!("threads", usize) {
        o.threads = Some(v);
    }
    if let Some(v) = get!("seed", u64) {
        o.seed = v;
    }
    if let Some(v) = d.get_item("check_derivatives")? {
        if !v.is_none() {
            o.check_derivatives = if let Ok(b) = v.extract::<bool>() {
                if b {
                    DerivativeCheck::Full
                } else {
                    DerivativeCheck::Off
                }
            } else {
                match v.extract::<String>()?.as_str() {
                    "off" | "none" => DerivativeCheck::Off,
                    "auto" | "directional" => DerivativeCheck::Directional,
                    "full" => DerivativeCheck::Full,
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "unknown check_derivatives '{other}'; use True (full), False (off), 'auto' (directional, the default) or 'full'"
                        )))
                    }
                }
            };
        }
    }
    if let Some(v) = get!("scaling", String) {
        o.scaling = match v.as_str() {
            "none" => ScalingMode::None,
            "gradient" | "gradient-based" => ScalingMode::GradientBased,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown scaling '{other}'; use 'none' or 'gradient'"
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
    if let Some(v) = get!("quadratic_probe", bool) {
        o.quadratic_probe = v;
    }
    if let Some(v) = d.get_item("quadratic_build")? {
        if !v.is_none() {
            o.quadratic_build = match v.extract::<String>()?.as_str() {
                "structured" => QuadraticBuild::Structured,
                "dense" => QuadraticBuild::Dense,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "unknown quadratic_build '{other}'; use 'structured' or 'dense'"
                    )))
                }
            };
        }
    }
    if let Some(v) = get!("quadratic_bands", String) {
        o.quadratic_bands = match v.as_str() {
            "fixed" => QuadraticBands::Fixed,
            "decaying" => QuadraticBands::Decaying,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown quadratic_bands '{other}'; use 'fixed' or 'decaying'"
                )))
            }
        };
    }
    if let Some(v) = d.get_item("quadratic_rows")? {
        if !v.is_none() {
            o.quadratic_rows = if let Ok(b) = v.extract::<bool>() {
                if b {
                    QuadraticRows::Values
                } else {
                    QuadraticRows::Off
                }
            } else {
                match v.extract::<String>()?.as_str() {
                    "off" => QuadraticRows::Off,
                    "jacobian" => QuadraticRows::Jacobian,
                    "values" | "all" => QuadraticRows::Values,
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "unknown quadratic_rows '{other}'; use False, 'jacobian' or 'values'"
                        )))
                    }
                }
            };
        }
    }
    if let Some(v) = get!("zero_step", String) {
        o.zero_step = match v.as_str() {
            "norm" => ZeroStep::Norm,
            "decrease" => ZeroStep::Decrease,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown zero_step '{other}'; use 'norm' or 'decrease'"
                )))
            }
        };
    }
    if let Some(v) = get!("saddle_step", String) {
        o.saddle_step = match v.as_str() {
            "scale" => SaddleStep::Scale,
            "linearized" => SaddleStep::Linearized,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown saddle_step '{other}'; use 'scale' or 'linearized'"
                )))
            }
        };
    }
    if let Some(v) = d.get_item("kkt_pivot_signs")? {
        if !v.is_none() {
            o.kkt_pivot_signs = match v.extract::<String>()?.as_str() {
                "auto" => PivotSigns::Auto,
                "expected" => PivotSigns::Expected,
                "free" | "counted" => PivotSigns::Free,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "unknown kkt_pivot_signs '{other}'; use 'auto', 'expected' or 'free'"
                    )))
                }
            };
        }
    }
    if let Some(v) = d.get_item("scale_variables")? {
        if !v.is_none() {
            o.scale_variables = if let Ok(b) = v.extract::<bool>() {
                if b {
                    VariableScaling::On
                } else {
                    VariableScaling::Off
                }
            } else {
                match v.extract::<String>()?.as_str() {
                    "off" | "none" => VariableScaling::Off,
                    "auto" => VariableScaling::Auto,
                    "on" | "always" => VariableScaling::On,
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "unknown scale_variables '{other}'; use True, False or 'auto'"
                        )))
                    }
                }
            };
        }
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
#[pyo3(signature = (fun, x0, jac=None, bounds=None, constraints=None, method=None, options=None, hess=None, callback=None, warm_start=None, fun_batch=None, cons_batch=None))]
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
    hess: Option<Py<PyAny>>,
    callback: Option<Py<PyAny>>,
    warm_start: Option<Bound<'_, PyDict>>,
    fun_batch: Option<Py<PyAny>>,
    cons_batch: Option<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    for (name, f) in [("fun_batch", &fun_batch), ("cons_batch", &cons_batch)] {
        if f.as_ref().is_some_and(|f| !f.bind(py).is_callable()) {
            return Err(PyValueError::new_err(format!("{name} must be callable")));
        }
    }
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
    if let Some(w) = &warm_start {
        let vec_of = |key: &str| -> PyResult<Vec<f64>> {
            match w.get_item(key)? {
                Some(v) if !v.is_none() => Ok(v
                    .extract::<PyReadonlyArray1<'_, f64>>()?
                    .as_slice()?
                    .to_vec()),
                _ => Ok(Vec::new()),
            }
        };
        let lambda = vec_of("lambda")?;
        let z_l = vec_of("z_l")?;
        let z_u = vec_of("z_u")?;
        if lambda.len() != m || z_l.len() != n || z_u.len() != n {
            return Err(PyValueError::new_err(format!(
                "warm_start does not match this problem: it carries {} constraint and {} / {} bound multipliers, the problem has {m} constraint rows and {n} variables",
                lambda.len(),
                z_l.len(),
                z_u.len()
            )));
        }
        let mu = match w.get_item("mu")? {
            Some(v) if !v.is_none() => Some(v.extract::<f64>()?),
            _ => None,
        };
        let quasi_newton = match w.get_item("hess_approx")? {
            Some(v) if !v.is_none() => {
                let np = py.import("numpy")?;
                let flat: Vec<f64> = np
                    .call_method1("asarray", (v, "float64"))
                    .and_then(|a| a.call_method0("ravel"))?
                    .extract()?;
                if flat.len() != n * n {
                    return Err(PyValueError::new_err(format!(
                        "warm_start['hess_approx'] has {} values; expected an ({n}, {n}) matrix",
                        flat.len()
                    )));
                }
                Some(flat)
            }
            _ => None,
        };
        opts.warm_start = Some(mincon_core::WarmStart {
            lambda,
            z_l,
            z_u,
            mu,
            quasi_newton,
        });
    }
    // The user's callback runs with the GIL re-acquired for the call only; an
    // exception inside it stops the solve and is re-raised afterwards.
    let callback_error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    if let Some(cb) = callback {
        if !cb.bind(py).is_callable() {
            return Err(PyValueError::new_err("callback must be callable"));
        }
        let slot = callback_error.clone();
        opts.callback = Some(IterationCallback::new(move |rec: &IterationRecord| {
            Python::attach(|py| {
                let stop = record_to_dict(py, rec)
                    .and_then(|row| cb.call1(py, (row,)))
                    .and_then(|v| v.bind(py).is_truthy());
                match stop {
                    Ok(s) => s,
                    Err(e) => {
                        if let Ok(mut g) = slot.lock() {
                            if g.is_none() {
                                *g = Some(e.to_string());
                            }
                        }
                        true
                    }
                }
            })
        }));
    }
    if let Some(h) = &hess {
        if !h.bind(py).is_callable() {
            return Err(PyValueError::new_err("hess must be callable"));
        }
    }
    let hess_structure = hess.as_ref().map(|_| lower_triangle(n));
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
        hess,
        hess_structure,
        blocks,
        jac_structure,
        fun_batch,
        // Without constraint rows there is nothing to batch.
        cons_batch: cons_batch.filter(|_| m > 0),
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

    if let Ok(slot) = callback_error.lock() {
        if let Some(msg) = slot.as_ref() {
            return Err(PyRuntimeError::new_err(format!(
                "the callback raised: {msg}"
            )));
        }
    }
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
    d.set_item("message", report.message())?;
    d.set_item("limit", report.limit.map(mincon_core::Limit::name))?;
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
    match &report.quasi_newton {
        Some(q) => {
            let n = report.solution.x.len();
            let arr = q.to_pyarray(py);
            d.set_item("hess_approx", arr.call_method1("reshape", ((n, n),))?)?;
        }
        None => d.set_item("hess_approx", py.None())?,
    }
    d.set_item("notes", PyList::new(py, &report.notes)?)?;
    let trace = PyList::empty(py);
    for t in &report.trace {
        trace.append(record_to_dict(py, t)?)?;
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
        hess: None,
        hess_structure: None,
        blocks: Vec::new(),
        jac_structure: None,
        fun_batch: None,
        cons_batch: None,
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
