# Resources

An annotated library. Entries are ordered by how much you need them, not
alphabetically, and each says **what to take from it**. Anything marked
★ is required reading before touching the relevant subsystem.

---

## 1. The four papers this project is built on

### ★ Wächter & Biegler (2006) — the interior-point method
*On the implementation of an interior-point filter line-search algorithm for
large-scale nonlinear programming.* Math. Prog. 106(1):25–57.
[PDF](https://optimization-online.org/wp-content/uploads/2004/03/836.pdf) ·
[Springer](https://link.springer.com/article/10.1007/s10107-004-0559-y)

The IPOPT paper. Unusually implementation-focused for a Math Prog paper — it
gives the constants, the heuristics and the failure handling, not just the
theorem. `crates/mincon-ip` follows it closely enough to read side by side.

**Take:** the barrier update, Algorithm IC (inertia correction), the filter
acceptance rules and their constants, second-order corrections, the feasibility
restoration phase, the `E_mu` termination test with `s_d`/`s_c` scaling, and the
initialization. Sections 3.1–3.4 are the whole globalization story.

### ★ Chiang & Zavala (2016) — how to avoid needing HSL
*An inertia-free filter line-search algorithm for large-scale nonlinear
programming.* Comput. Optim. Appl. 64.
[PDF](https://www.mcs.anl.gov/papers/P5197-0914.pdf)

Replaces the inertia test with a curvature test on the computed direction, so
any linear solver works — including iterative ones, GPU ones, and ours.

**Take:** conditions (RG3a) and (3.11), the IFR regularization loop, and the
empirical result that it needs **56–69% fewer regularizations** than the
inertia-based rule. This paper is why `mincon-linalg` can exist without MA57
and therefore why `pip install mincon` can exist.

### Byrd, Hribar & Nocedal (1999); Waltz, Morales, Nocedal & Orban (2006)
*An interior point algorithm for large-scale nonlinear programming*, SIAM J.
Optim. 9(4); *An interior algorithm for nonlinear optimization that combines
line search and trust region steps*, Math. Prog. 107(3).

The KNITRO line — **and the algorithm `fmincon`'s `interior-point` actually
implements.** Read them to know what the competitor does: barrier subproblems,
a direct Newton step with a trust-region conjugate-gradient fallback, and an
`l2` merit function.

**Take:** the direct/CG step switch. It is a genuinely good idea we do not yet
have, and it is a plausible answer to problems where our regularization loop
grinds.

### Curtis, Powell & Reid (1974); Coleman & Moré (1983, 1984)
*On the estimation of sparse Jacobian matrices*, J. Inst. Math. Appl. 13;
*Estimation of sparse Jacobian matrices and graph coloring problems*, SIAM J.
Numer. Anal. 20(1); *Estimation of sparse Hessian matrices and graph coloring
problems*, Math. Prog. 28.

**Take:** distance-1 coloring for Jacobians, **star coloring for Hessians**.
The distinction is not academic: a distance-1 coloring of a symmetric matrix
silently produces wrong second derivatives.

---

## 2. Books

### ★ Nocedal & Wright, *Numerical Optimization*, 2nd ed., Springer 2006
The standard reference and the one `fmincon`'s documentation cites for its SQP.
Chapters 15–19 are the relevant ones: 16 (QP), 17 (penalty and augmented
Lagrangian), 18 (SQP), 19 (interior point). **Theorem 16.3** is the inertia
condition the KKT regularization exists to enforce — read it before touching
`factor_with_correction`.

### ★ Davis, *Direct Methods for Sparse Linear Systems*, SIAM 2006
Everything in `mincon-linalg`. The elimination tree, up-looking factorization,
symbolic analysis, AMD. Short, dense, and the code in it is CSparse, which is
the clearest sparse linear algebra ever written.

### Gill, Murray & Wright, *Practical Optimization*, Academic Press 1981
Reissued by SIAM in 2019. Where `fmincon`'s `active-set` comes from. The
chapters on numerical stability and on what goes wrong in practice have aged
extremely well.

### Conn, Gould & Toint, *Trust-Region Methods*, SIAM 2000
Definitive if the trust-region path is ever taken seriously. Also the origin of
CUTE/CUTEr/CUTEst.

### Fletcher, *Practical Methods of Optimization*, 2nd ed., Wiley 1987
The `l1` merit function, the Maratos effect, and SQP as its inventors saw it.

### Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM 2002
For the growth-factor argument in `mincon-linalg`, iterative refinement, and
why `LDL^T` without pivoting is unstable in general.

---

## 3. Benchmarking

### ★ Dolan & Moré (2002) — performance profiles
*Benchmarking optimization software with performance profiles.* Math. Prog.
91(2):201–213.

**Take:** the whole construction. `rho_s(1)` is how often a solver was fastest,
`rho_s(inf)` is how often it solved the problem at all. Implemented in
`bench/profiles.py`.

### ★ Moré & Wild (2009) — data profiles
*Benchmarking derivative-free optimization algorithms.* SIAM J. Optim.
20(1):172–191. [Data and code](https://www.mcs.anl.gov/~more/dfo/)

Performance profiles hide how much *budget* a solver needed. Data profiles plot
fraction-solved against budget in units of `n+1` evaluations — the right view
when the model is expensive, which is nearly every real user.

### Gould & Scott (2016) — the caveats
*A note on performance profiles for benchmarking software.* ACM TOMS 43(2).
[PDF](https://centaur.reading.ac.uk/74694/1/perform_toms.pdf)

Read before publishing any profile. Explains how the construction misleads, and
why "a solver cannot buy a good profile by failing fast" needs the `r = inf`
convention to actually hold.

### ★ Gratton & Toint (2024) — S2MPJ
*S2MPJ and CUTEst optimization problems for Matlab, Python and Julia.*
[arXiv:2407.07812](https://arxiv.org/abs/2407.07812) ·
[GitHub](https://github.com/GrattonToint/S2MPJ) · BSD-3-Clause

1075 CUTEst problems as pure Python/Matlab/Julia source. **This is what makes
the real benchmark reachable** — no Fortran, no SIF decoder, no `pycutest`
cache. `bench/s2mpj_bridge.py` is built on it and self-checks against HS71.

### Hock & Schittkowski (1981); Schittkowski (1987)
*Test Examples for Nonlinear Programming Codes*, LNEMS 187; *More Test Examples
for Nonlinear Programming Codes*, LNEMS 282.

The 119 + 188 problems every NLP paper since 1981 reports on. About 45 are
transcribed in `crates/mincon-testset`. Reference objective values have known
errata — see the `notes` field on each problem, and `HS16` in particular, where
the published global optimum is not what a local method finds from the
published starting point.

### Kronqvist, Bernal, Lundell & Grossmann (2022) — the number to beat
*Nonlinear Programming Solvers for Unconstrained and Constrained Optimization
Problems: a Benchmark Analysis.* [arXiv:2204.05297](https://arxiv.org/pdf/2204.05297)

23 algorithms, 60 problems, three settings (plug-and-play, high accuracy, quick).
**`fmincon`'s interior-point solves 75.9% of the constrained set under
plug-and-play settings**, ahead of KNITRO-IP (74.6%) and SNOPT (72.1%). Small
set, different machine — a target, not a scoreboard, but it is the most
directly comparable public figure.

### Bussieck, Drud & Meeraus — GAMS World / MINLPLib; Mittelmann's benchmarks
[plato.asu.edu/bench.html](https://plato.asu.edu/bench.html)

Hans Mittelmann's independent benchmarks are the closest thing the field has to
a referee. Read the methodology; it is the model for `bench/README.md`'s rule
that success is decided by the harness.

---

## 4. Codebases worth reading

| Project | Licence | Read it for |
|---|---|---|
| [**IPOPT**](https://github.com/coin-or/Ipopt) | EPL-2.0 | The reference implementation of §1's first paper. **Work from the paper and the options page, not the source: EPL is not compatible with MIT/Apache.** If a source file is ever consulted, the spec that uses the idea records which file and why (rule below). The September 2026 audit found no IPOPT code or identifiers in `mincon-ip` (`docs/IP_AUDIT.md`, F-11). |
| [**IPOPT options**](https://coin-or.github.io/Ipopt/OPTIONS.html) | docs | Every default in `docs/02` was cross-checked here. The single most useful page in the field. |
| [**QDLDL**](https://github.com/osqp/qdldl) | Apache-2.0 | The up-looking `LDL^T` kernel, ~300 lines, extremely clear. `mincon-linalg/src/ldlt.rs` adapts its elimination tree and the structure of its numeric kernel; the file header and `NOTICE` carry QDLDL's copyright notice, as Apache-2.0 requires. |
| [**Clarabel.rs**](https://github.com/oxfordcontrol/Clarabel.rs) | Apache-2.0 | A serious interior-point solver in idiomatic Rust. Read for dynamic regularization in practice, and for how to organize a Rust numerical crate. |
| [**OSQP**](https://github.com/osqp/osqp) | Apache-2.0 | ADMM QP. The right fallback QP solver, and the paper is a model of clear exposition. |
| [**CSparse**](https://github.com/DrTimothyAldenDavis/SuiteSparse) | LGPL / BSD by module | AMD is **BSD-3-Clause** and therefore usable. The `amd` Rust crate (v0.2.2) is a port of it. CSparse, LDL and CHOLMOD are **LGPL**: read Davis's book, not their source (rule below). |
| [**faer**](https://docs.rs/faer/latest/faer/sparse/linalg/cholesky/index.html) | MIT | Dense and sparse linear algebra, including sparse LDLT and intranodal Bunch–Kaufman. Evaluate its numerical contracts as an alternative backend. |
| [**NLopt**](https://github.com/stevengj/nlopt) | LGPL/MIT | Breadth of algorithms; a good source of ideas, weaker on constrained methods. |
| [**SciPy `_slsqp`**](https://github.com/scipy/scipy) | BSD-3 | Kraft's 1988 Fortran SLSQP. The thing to beat on small dense problems; currently ~5x cheaper than us in evaluations. |
| [**CasADi**](https://github.com/casadi/casadi) | LGPL | Best-in-class AD for optimization. The model for what an AD bridge should feel like. |
| [**Percival.jl / JSO**](https://jso.dev) | MPL | The Julia optimization ecosystem's benchmarking discipline is worth stealing wholesale. |

### A licence rule, stated once

Everything shipped must be **MIT / Apache-2.0 / BSD**. Copyleft and
proprietary-licensed code may be **read and cited, never vendored or
translated**: GPL (R's `quadprog`), LGPL (`eiquadprog`, CSparse, LDL, CHOLMOD
and the other LGPL SuiteSparse modules, NLopt's LGPL parts, CasADi), EPL
(IPOPT), and MUMPS and HSL under their own terms. If a design decision comes
from reading such code, record the file read in the spec that uses the idea,
describe the idea in prose there, and implement from the description and the
paper. Permissive code that is adapted (QDLDL, Apache-2.0) keeps its copyright
notice in the file and in `NOTICE`.

**MathWorks.** Only MathWorks' public documentation is read. No MathWorks
code, P-code or toolbox file is opened, printed, stepped through in a debugger
or copied, and nothing is reverse-engineered from `fmincon`'s behaviour beyond
what its public interface returns. Documentation is quoted briefly, in
quotation marks, with the page and the date read (`docs/20` §10). Benchmark
records keep `fmincon`'s exit flag and numbers, not its messages or console
text (`scripts/scrub_matlab_records.py`). Names (`fmincon`, option names, exit
flag integers) are used only to describe compatibility.

---

## 5. Rust ecosystem, with versions as of this writing

| Crate | Version | Licence | Use |
|---|---|---|---|
| `faer` | 0.24.4 | MIT | dense and sparse linear algebra, including LDLT |
| `amd` | 0.2.2 | BSD-3 | AMD ordering — **the pending `mincon-linalg` task** |
| `clarabel` | 0.11.1 | Apache-2.0 | reference reading; possible conic backend later |
| `nalgebra` | 0.35 | Apache-2.0 | general linear algebra; heavier than needed here |
| `ndarray` | 0.17 | MIT/Apache | pulled in transitively by `numpy` |
| `sprs` | 0.11.5 | MIT/Apache | sparse containers; our own CSC is lighter |
| `rayon` | 1.12 | MIT/Apache | parallel finite differences |
| `pyo3` | 0.29 | Apache-2.0 | Python bindings. **0.29 renamed `Python::with_gil` → `attach` and `downcast` → `cast`.** |
| `numpy` (rust-numpy) | 0.29 | BSD-2 | zero-copy NumPy interop |
| `maturin` | ≥1.7 | MIT/Apache | wheel building; `abi3-py39` for one wheel per platform |
| `osqp-rust` | 0.6.2 | Apache-2.0 | candidate fallback QP |
| `argmin` | 0.11 | MIT/Apache | mostly unconstrained; not a fit |

---

## 6. Specific techniques, with their source

| Technique | Source | Where it lands |
|---|---|---|
| Quasi-definite matrices factor under any permutation | Vanderbei, SIAM J. Optim. 5(1), 1995 | why regularizing makes `LDL^T` safe |
| Sylvester's law of inertia | any linear algebra text | inertia from the signs of `D`, free |
| Powell damping for BFGS on the Lagrangian | Powell, 1978 | `mincon_ip::bfgs`; **not optional** |
| Compact L-BFGS representation | Byrd, Nocedal & Schnabel, Math. Prog. 63, 1994 | the L-BFGS milestone; low-rank update into the KKT matrix |
| Filter methods | Fletcher & Leyffer, Math. Prog. 91, 2002 | `mincon_ip::filter` |
| Maratos effect and second-order corrections | Maratos 1978; Fletcher 1987 §14.4 | why SOC is mandatory |
| Watchdog technique | Chamberlain et al., Math. Prog. 16, 1982 | unimplemented; IPOPT §3.4 Case II |
| Ruiz equilibration | Ruiz, RAL-TR-2001-034 | not implemented (`ScalingMode::Equilibration` was removed in Phase B) |
| Elastic / always-feasible QP | Fletcher 1985 (Sℓ1QP); Gill, Murray & Saunders, SIAM Review 47, 2005 (SNOPT) | `mincon_sqp`; `docs/20` §3 |
| Anti-cycling in active-set QP | Bland 1977; lexicographic rules | the QP solver |
| Bound relaxation for `lb == ub` | IPOPT `bound_relax_factor` | why `TORTURE_PINNED` is solvable |

### Components with no single published source

Recorded after the September 2026 audit (`docs/IP_AUDIT.md` §3.2, F-14). Each
is textbook practice with the citation below, or the project's own measured
heuristic, marked *original*. None comes from MathWorks software.

| Component | Source | Where it lands |
|---|---|---|
| Algorithm portfolio (racing configurations) | Rice, "The algorithm selection problem", Adv. Comput. 15, 1976; Gomes & Selman, Artif. Intell. 126, 2001 | `mincon::portfolio` |
| Quadratic-program probe (third differences on random lines), structured band Hessian build, decaying bands, convex quadratic rows | *original*, on textbook finite differences (Nocedal & Wright §8.1; Boyd & Vandenberghe, *Convex Optimization*, §3.1) | `mincon::quadratic` |
| Variable scaling by the start's magnitudes | Gill, Murray & Wright 1981 §7.5; Dennis & Schnabel, *Numerical Methods for Unconstrained Optimization and Nonlinear Equations*, 1983, §7.1 | `mincon::scaled` |
| Finite-difference steps `sqrt(eps)` forward, `eps^(1/3)` central, scaled by `max(|x|, typical)` | Dennis & Schnabel 1983 §5.4; Nocedal & Wright §8.1 | `mincon_diff::fd` |
| Bounds-respecting probes, retreat on non-finite values, unequal-step and one-sided stencils | Fornberg, Math. Comp. 51, 1988 (unequal steps); derivation in `docs/05` §2 | `mincon_diff::fd` |
| Forward-to-central escalation; error-aware termination | *original* | `mincon_diff::fd`; `mincon_ip::solver` |
| Black-box sparsity detection by probing | *original*; related work Griewank & Mitev, Math. Prog. 93, 2002 | `mincon_diff::detect` |
| Derivative checks (component-wise, directional) | Nocedal & Wright §8.1 | `mincon_diff::check` |
| Reverse Cuthill–McKee ordering | Cuthill & McKee, ACM 1969; George & Liu, *Computer Solution of Large Sparse Positive Definite Systems*, 1981 | `mincon_linalg::ordering` |
| Exact-Hessian regularization along the constraint normals | Debreu, Econometrica 20, 1952; Nocedal & Wright §17.3 | `mincon_sqp` |
| Initial penalty `max(1, ‖g0‖∞ / ‖J0‖∞)` | *original* (a scale-aware start for the ℓ1 penalty of Han 1977) | `mincon_sqp` |
| Saddle probe and escape, zero-step re-test, degenerate-multiplier verdict, line-search failure cascade | Moré & Sorensen, Math. Prog. 16, 1979 (negative curvature); the rest *original* (`docs/20` §11, `docs/22` §7.18) | `mincon_sqp` |
| Dense kernels: Gaussian elimination, Gram–Schmidt, cyclic Jacobi | Golub & Van Loan, *Matrix Computations* | `mincon_sqp` |
| QP constraint selection by scaled violation; warm-start hint ordering | Goldfarb & Idnani 1983 (selection rule); hint ordering *original* | `mincon_qp` |
| Multistart | Rinnooy Kan & Timmer, Math. Prog. 39, 1987 | `mincon.multistart` |
| XorShift and LCG generators | Marsaglia, J. Stat. Softw. 8, 2003; Knuth, TAOCP vol. 2 | probes and tests |
| KKT fixture checker `relative_stationarity` | Nocedal & Wright ch. 12 | `mincon_testset` |
| Positive diagonal congruence retry, natural-order fallback, certificate guards (D9, D11), divergence and stall exits, adaptive-to-monotone barrier fallback, curvature-drift BFGS rebuild, torture problems | *original* (`docs/16`, `docs/21`, `docs/22`) | `mincon_ip`; `mincon_testset` |

---

## 7. Where to ask

* **optimization-online.org** — preprints, and where implementation papers land.
* **COIN-OR Discourse / IPOPT issue tracker** — Andreas Wächter answers
  questions there. The archives are a goldmine of "why does my problem do this".
* **NA Digest** — the numerical analysis mailing list.
* **`scipy.optimize` issue tracker** — a long record of exactly what users get
  wrong and what they need. Read it as requirements, not as bug reports.
