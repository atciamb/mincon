# mincon

A nonlinear constrained optimizer in Rust, aimed at what MATLAB's `fmincon`
does well — solving the problem a working scientist actually has, without being
told how — with Rust computations and Python wheels built using maturin.
Experimental version 0.1.0 is [available on PyPI](https://pypi.org/project/mincon/0.1.0/).
Install with `pip install mincon`. Prebuilt wheels support Windows and Linux x86_64.

```
minimize    f(x)
subject to  c_L <= c(x) <= c_U
            x_L <=   x  <= x_U
```

```python
from mincon import fmincon

r = fmincon(lambda x: ((x - 1)**2).sum(), [0., 0.],
            nonlcon=lambda x: ([x.sum() - 1], []))
print(r.x, r.fun, r.success)  # approximately [0.5, 0.5], 0.5, True
```

```rust
use mincon::{minimize, Options, Problem};

let p = Problem::new(2, |x| 100.0*(x[1]-x[0]*x[0]).powi(2) + (1.0-x[0]).powi(2))
    .start_at(&[-1.2, 1.0])
    .inequality(1, |x, c| c[0] = x[0]*x[0] + x[1]*x[1] - 1.0);
let r = minimize(&p, &Options::default())?;
```

> **Status: early.** The development gate now passes all 54 expected outcomes
> using independent objective/feasibility checks. This includes usable points
> and expected failure diagnostics; it does not mean 54 certified optima.
> Currently 40/54 report strict Optimal; 151 Rust tests and 18 Python tests pass locally.
> Soft and reduced-elastic restoration are implemented. CUTEst qualification,
> SQP and large sparse work remain. **No superiority claim against fmincon.**
> [Measurements and limitations](bench/results/restoration-stopping/README.md).

`fmincon` accepts optional `A, b, Aeq, beq, lb, ub, nonlcon`; its nonlinear
callback returns `(c, ceq)` with `c <= 0`. The existing `minimize` interface
uses SciPy's `ineq >= 0` convention. Both have automatic defaults and return
an `OptimizeResult`. See the [Python guide](crates/mincon-py/README.md) and
[release procedure](docs/13_RELEASE.md). A fresh PyPI installation passes all 18 Python tests.

---

## Why

`fmincon` is the standard because it works untuned. Its advantage is not
algorithmic — its interior-point method is the published KNITRO design, its SQP
is textbook Han–Powell — it is two decades of defensive engineering and
defaults that work. That is reproducible. Meanwhile it has four fixable
deficits, and it costs money and cannot be embedded.

| | `fmincon` | `mincon` |
|---|---|---|
| Problem scaling | **off by default** | gradient-based, **on by default** |
| Sparsity of nonlinear Jacobians | must be declared | detected, and CPR-colored |
| Algorithms per call | one, user chooses | a portfolio, raced |
| Derivatives | finite differences for function handles | analytic, AD-bridged, or FD |
| Licence | proprietary | MIT / Apache-2.0 |
| Install | MATLAB + toolbox | `pip install mincon` |

Measured consequences of the first two: `TORTURE_SCALING` (twelve orders of
magnitude between objective and constraint gradients) solves in **one
iteration** with scaling on; a tridiagonal Jacobian costs **3 finite-difference
evaluations regardless of `n`**.

Full analysis: [`docs/01_FMINCON_ANATOMY.md`](docs/01_FMINCON_ANATOMY.md).

---

## What works today

* **Interior point** — primal-dual, filter line search, Wächter–Biegler
  Algorithm IC inertia correction, second-order corrections,
  fraction-to-boundary, bound-multiplier resets, scaled `E_mu` termination.
* **Feasibility restoration** — guarded soft steps, then analytically reduced
  elastic minimization with Gauss-Newton curvature and filter re-entry.
* **Sparse `LDL^T`** written from scratch: dynamic regularization, inertia
  certified by Sylvester's law, element-growth detection, iterative refinement.
  **No HSL, no MUMPS, no Fortran** — which is what makes the wheel possible.
* **Derivatives** — bounds-aware finite differences with realized retreat
  displacements and second-order inward boundary stencils, graph coloring,
  sparsity detection, a multi-point derivative checker.
* **Gradient-based scaling**, on by default.
* **Algorithm portfolio** with deterministic ranking.
* **Python bindings** — `abi3` wheel, SciPy-compatible `minimize`.
* **Test set** — 44 Hock–Schittkowski problems plus 10 torture problems with
  independent feasibility/reference-value checks and explicit failure fixtures.
* **Benchmark harness** — 1075 CUTEst problems via S2MPJ, Dolan–Moré
  performance profiles, Moré–Wild data profiles, and a MATLAB script to
  generate the `fmincon` baseline.

### Known gaps, in priority order

1. **Restoration qualification.** The local gates pass; the required CUTEst
   improvement remains unmeasured. Inertia-free acceptance is still pending.
2. **Automatic differentiation.** Finite differences cap achievable accuracy at
   `sqrt(eps)` and cost `n` evaluations per gradient.
3. **AMD ordering.** `Ordering::Amd` falls back to RCM.
4. **Limited-memory BFGS.** Dense BFGS caps usable `n` at a couple of thousand.
5. **SQP.** Specified, not built. Measured: we use ~5x more evaluations per
   portfolio member than SciPy's SLSQP on small dense problems — that regime
   belongs to SQP.

Each is a milestone with an objective gate in
[`docs/10_ROADMAP.md`](docs/10_ROADMAP.md).

---

## Layout

```
AGENT_PROMPT.md          the brief for continuing this work
docs/                    mission, competitive analysis, specs, resources, roadmap, pitfalls
crates/
  mincon-core/           problem model, options, results
  mincon-linalg/         sparse LDL^T with certified inertia, KKT assembly
  mincon-diff/           finite differences, coloring, sparsity detection, checker
  mincon-ip/             interior point
  mincon-sqp/            SQP  (specified, not implemented)
  mincon-qp/             QP subsolver  (specified, not implemented)
  mincon/                public API, algorithm portfolio
  mincon-py/             PyO3 bindings + the Python package
  mincon-testset/        Hock-Schittkowski and torture problems
bench/                   CUTEst harness, performance profiles, fmincon baseline
```

---

## Building

```bash
cargo test --workspace                                   # 122 tests
cargo run --release -p mincon-ip --example run_testset    # the regression table
cargo run --release -p mincon-ip --example diagnose HS71  # one problem, in detail

pip install maturin
cd crates/mincon-py && maturin develop --release
```

Benchmarks:

```bash
pip install -e 'crates/mincon-py[bench]'
python bench/runner.py --set smoke --solvers mincon scipy-slsqp
python bench/profiles.py results.jsonl -o profiles.png
```

---

## The honesty rule

This project measures itself against a product with a twenty-year head start,
and the temptation to flatter the numbers is constant. So:

* Benchmark success is decided by the harness, never by the solver's own
  report — a rule that cuts against us as readily as for us.
* The portfolio's cost is reported across **all** members, not the winner's.
* `Acceptable` is not counted as success.
* Known gaps are listed above, not buried.
* When the CUTEst run happens, the README will name the problem classes where
  `fmincon` still wins.

A benchmark with no losses in it is a benchmark nobody believes.

---

## Licence

MIT or Apache-2.0, at your option. Every dependency is MIT, Apache-2.0 or BSD;
nothing here requires a licence, a Fortran toolchain, or HSL.

## Acknowledgements

Standing on: Wächter & Biegler's IPOPT paper, Chiang & Zavala's inertia-free
regularization, Curtis–Powell–Reid coloring, Dolan & Moré's performance
profiles, Gratton & Toint's S2MPJ, and Hock & Schittkowski's test set. Full
annotated bibliography in [`docs/09_RESOURCES.md`](docs/09_RESOURCES.md).
