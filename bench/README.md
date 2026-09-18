# The stress test

Three tiers, each answering a different question.

| Tier | What it is | Runs | Answers |
|---|---|---|---|
| **1. Unit + testset** | `cargo test`, `cargo run -p mincon-ip --example run_testset`, `cargo run -p mincon --example run_testset_portfolio -- auto` | every commit, seconds | Is the algorithm still correct? Did it start lying? |
| **2. Benchmark** | `runner.py --set constrained-small` | every PR, minutes | Did robustness or cost regress? |
| **3. Full CUTEst** | `runner.py --set all --solvers all` | nightly, hours | Are we beating `fmincon` yet? |

Tier 1 is the inner loop and lives in Rust. Tiers 2 and 3 live here.

## Quick start

```bash
pip install -e 'crates/mincon-py[bench]'      # mincon + scipy + matplotlib
python bench/runner.py --set smoke --solvers mincon scipy-slsqp
python bench/profiles.py results.jsonl -o profiles.png
```

The `smoke` set downloads four problems on demand and takes seconds. Anything
larger needs the full S2MPJ checkout:

```bash
python -c "import sys; sys.path.insert(0,'bench'); import s2mpj_bridge as b; b.ensure_s2mpj()"
```

## Where the problems come from

[**S2MPJ**](https://github.com/GrattonToint/S2MPJ) (Gratton & Toint,
[arXiv:2407.07812](https://arxiv.org/abs/2407.07812), BSD-3-Clause) translates
1075 CUTEst problems into pure Python. No Fortran, no SIF decoder, no
`pycutest` cache — `git clone` and `import`.

This matters more than it sounds. The reason most open-source optimizers are
never measured against the real benchmark is that setting up CUTEst is a
day's work. S2MPJ makes the full 1075-problem run a `pip install` away, which
means it can actually happen on every nightly instead of once before a paper.

`bench/s2mpj_bridge.py` maps S2MPJ's interface onto ours; the mapping is
one-to-one because both use the canonical form `c_L <= c(x) <= c_U`,
`x_L <= x <= x_U`. Running the bridge as a script self-checks against HS71's
published values.

## The rule that makes this credible

**Success is decided by the harness, never by the solver.**

Every solver reports its own status, with its own tolerances and its own idea
of "converged". Scoring on `result.success` measures reporting culture, not
solution quality. So for every returned point the harness recomputes the
objective and the maximum constraint violation from the problem's own
callables, and then calls a problem *solved* when:

* the point is feasible to `--feas-tol` (default `1e-5`), **and**
* its objective is within `--obj-tol` relative (default `1e-4`) of the best
  objective any solver reached on that problem while feasible.

This is the Dolan–Moré / Mittelmann convention. It cuts against us as readily
as for us: if `mincon` reports success at a point another solver beats, the
harness scores it a failure.

The summary table has a **`lied`** column: runs where the solver reported
success and returned an infeasible point. Treat any non-zero entry there as a
release blocker, whoever it belongs to. A solver that fails honestly costs a
user an afternoon; one that lies costs them a paper.

## Reading the plots

`profiles.py` produces three panels:

* **Performance profile by evaluations** (Dolan & Moré 2002). `rho_s(1)` is how
  often solver `s` was cheapest; `rho_s(inf)` is how often it solved the problem
  at all. The right-hand asymptote is the **robustness** number and it is the
  one this project is trying to win.
* **Performance profile by wall clock.** Where being written in Rust shows up.
* **Data profile** (Moré & Wild 2009). Fraction solved against a budget in units
  of `n+1` evaluations — the right view when the model is expensive, which is
  the situation nearly every real user is in.

## Do not tune on the reported set

Split the problems and keep the split:

* **Development set** — `constrained-small`, plus `mincon-testset`. Tune here.
* **Held-out set** — everything else. Run it before a release and report it.
  Never adjust a constant because of what it says.

Anyone can win a performance profile by fitting to it. The whole value of this
directory is that its numbers can be trusted, and that is a discipline, not a
feature.

## Comparing against fmincon

`fmincon` needs a MATLAB licence, so it cannot run in CI. `bench/matlab/fmincon_baseline.m`
generates its results in the same JSONL schema; commit the output and
`profiles.py` will treat `fmincon` as one more solver.

The fairness rules are written at the top of that file and are worth repeating:
**no gradients supplied to anyone**, default options otherwise, and the success
criterion applied afterwards by the harness. The plug-and-play case is where
`fmincon` is strongest and it is the case we claim to beat; comparing against a
hand-tuned `fmincon` would be worthless.

### The number to beat

The most directly comparable published figure is **75.9%** — `fmincon`'s
interior-point convergence rate on the 30 constrained problems of Kronqvist et
al.'s "plug-and-play" comparison ([arXiv:2204.05297](https://arxiv.org/pdf/2204.05297)),
against KNITRO-IP at 74.6% and SNOPT at 72.1%. It is a small set on a
different machine, so it is a target, not a scoreboard. The scoreboard is the
CUTEst run produced here.

## Current restoration increment

The development gate passes **54/54 expected outcomes**, with returned
objective and feasibility independently evaluated. This is not a strict
convergence rate. See [raw baseline/current tables and limitations](results/restoration/README.md).
The CUTEst gate has not run; the figures below are scaffold history.

## Historical scaffold measurements

From `cargo run --release -p mincon-ip --example run_testset`:

* **52 of 54** problems in `mincon-testset` pass; **0 false reports of success**.
* Failures: `HS13` (MFCQ fails at the solution — needs feasibility
  restoration) and `TORTURE_INFEASIBLE` (correctly refuses to claim success,
  but reports `NumericalFailure` instead of `Infeasible` — also restoration).

From the four-problem smoke benchmark against SciPy:

* Same objective on all four, feasible on all four.
* **`mincon` uses roughly 5x more objective evaluations per portfolio member
  than SLSQP** on these small dense problems (851 median across three members
  versus 52).

That last line is the most useful number in this document. It is not a bug —
an interior-point method takes more iterations than SQP on small dense problems
with few active constraints, and each iteration pays for a finite-difference
gradient and Jacobian. It is a precise statement of where the work is:

1. **SQP** (milestone M5) is the algorithm that owns this regime.
2. **Automatic differentiation** removes the per-iteration finite-difference
   cost entirely, which is most of the gap.
3. The portfolio triples the count; on a multi-core machine it costs wall clock
   rather than time, but on one core it is a real 3x and the early-exit path
   matters.

These small historical samples do not establish a general robustness or
wall-clock advantage. Re-run matched benchmarks before making either claim.
