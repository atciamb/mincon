# mincon

A nonlinear constrained optimizer in Rust, aimed at what MATLAB's `fmincon`
does well — solving the problem a working scientist actually has, without being
told how — with Rust computations and Python wheels built using maturin.
Experimental version 0.1.0 is [available on PyPI](https://pypi.org/project/mincon/0.1.0/).
Install with `pip install mincon`. Prebuilt wheels support Windows and Linux x86_64.
Source and contributions: [atciamb/mincon on GitHub](https://github.com/atciamb/mincon).

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

> **Status: experimental, measured.** Matched against the installed MATLAB
> R2025b `fmincon` on a 149-problem single-source corpus with an independent
> KKT oracle (`docs/15_BENCHMARK_PROTOCOL_V2.md`), two rounds of held-out
> qualification. At defaults mincon attained at least as many targets as
> `fmincon-interior-point` on every split (held-out: 19 vs 18 of 22, then
> 14 vs 12 of 16), used about the same number of model evaluations (0.88×
> to 1.43× across four sets; 0.88× [0.68, 1.10] on the latest held-out set)
> and returned **10–100× faster** in wall-clock time on problems up to a few
> hundred variables (same laptop, single thread). `fmincon-sqp` and SciPy
> SLSQP need 0.6–0.8× the evaluations of any interior-point code on small
> dense problems; mincon has no SQP yet, and dense BFGS makes it slow past
> n ≈ 200. **The preregistered superiority contract is not met at these
> sample sizes**; see `bench/results/s6v2-final2/README.md`,
> `docs/17_CLAIM_AUDIT.md` and `docs/16_FAILURE_ATLAS.md` for exactly where
> fmincon wins. Development gate: 160 Rust tests, 55/55 fixtures with
> independent checks, 23 Python tests.

`fmincon` accepts optional `A, b, Aeq, beq, lb, ub, nonlcon`; its nonlinear
callback returns `(c, ceq)` with `c <= 0`. The existing `minimize` interface
uses SciPy's `ineq >= 0` convention. Both have automatic defaults and return
an `OptimizeResult`. See the [Python guide](crates/mincon-py/README.md) and
[release procedure](docs/13_RELEASE.md). A fresh PyPI installation passes all 18 Python tests.

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
  sparsity detection, a multi-point derivative checker that cannot pass
  without evidence; exact objective gradients and constraint Jacobians from
  Python (`jac` in SciPy-style constraint dicts, `nonlcon_jac` in `fmincon`).
* **Gradient-based scaling**, on by default.
* **Adaptive barrier update** (LOQO centrality rule) with a bounded fallback
  to the monotone schedule; **algorithm portfolio** with deterministic
  ranking, sequential early exit for models that cannot evaluate concurrently
  (every Python model) and shared member budgets.
* **Python bindings** — `abi3` wheel, SciPy-compatible `minimize`.
* **Test set** — 44 Hock–Schittkowski problems plus 10 torture problems with
  independent feasibility/reference-value checks and explicit failure fixtures.
* **Benchmark corpus and harness** — 133 problems defined once in SymPy
  (89 Hock–Schittkowski, adversarial closed-form cases, scalable structured
  families, engineering designs), generated NumPy and MATLAB models that agree
  at 2,660 probe quantities, an independent KKT oracle, supervised MATLAB and
  Python workers with hard timeouts, and paired family-bootstrap analysis
  (`bench/harness/README.md`).

### Known gaps, in priority order (measured, see `docs/16_FAILURE_ATLAS.md`)

1. **SQP.** Specified, not built. `fmincon-sqp`/SLSQP use 0.6–0.8× the
   evaluations of every interior-point code on small dense problems.
2. **Dense BFGS from the identity at n ≳ 100** (ELLIPSOID2_200: ~800
   iterations; QUADSPHERE2_300 not solved within budget): hundreds of
   iterations on long curved valleys for every BFGS-based solver, and
   ~25 ms/iteration at n = 200. Pass exact derivatives; a guarded initial
   scaling is the next candidate fix (an unguarded one was rejected).
3. **Adaptive barrier stalls** on a few problems (HS63: 39 vs 8 iterations
   monotone) before the monotone fallback fires; a primal-infeasibility floor
   was rejected as neutral overall.
4. **Finite-difference accuracy on sensitive models**: the solver now stops
   at the estimated derivative accuracy (HS62: 100 evaluations, was 504) but
   the certificate is only as good as the derivatives; pass `jac` and
   constraint `jac` when you can.
5. **Nonlinear infeasibility is not always diagnosed** (INFEASIBLE_NL runs to
   the iteration limit; so do fmincon and SLSQP).
6. **AMD ordering** falls back to RCM; inertia-free acceptance is not wired.

---

## Layout

```
docs/                    algorithm specifications, resources, implementation notes, release guide
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
