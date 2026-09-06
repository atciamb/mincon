# Benchmark protocol

Operational detail is in `bench/README.md`. This document is the *discipline*:
what makes a number publishable.

---

## 1. The three tiers

| Tier | Command | Frequency | Question |
|---|---|---|---|
| 1 | `cargo test --workspace` + `cargo run --release -p mincon-ip --example run_testset` | every commit | Is it still correct? Did it start lying? |
| 2 | `python bench/runner.py --set constrained-small --solvers mincon scipy-slsqp scipy-trust-constr` | every PR | Did robustness or cost regress? |
| 3 | `python bench/runner.py --set all --solvers all` | nightly / pre-release | Are we beating `fmincon` yet? |

Tier 1 must stay under a minute or it stops being run.

---

## 2. Success is decided by the harness

For every returned point, recompute from the problem's own callables:

* the objective,
* the maximum constraint violation, **including variable bounds**.

Then: *solved* means feasible to `1e-5` **and** objective within `1e-4`
relative of the best any solver achieved on that problem while feasible.

This is the Dolan–Moré / Mittelmann convention. It cuts against us as readily
as for us, which is the point. A solver's own `success` flag is never used to
score it — only to populate the `lied` column.

---

## 3. What must be reported alongside any number

A performance profile without these is not evidence:

1. The **problem set**, by name and count.
2. The **tolerances** (`--feas-tol`, `--obj-tol`).
3. The **budget** (`--maxfev`, `--maxtime`).
4. Whether **derivatives were supplied** — and to whom. This is the single
   biggest lever on the result and the most common way comparisons are rigged.
5. The **machine** and solver versions.
6. **Total** evaluations for portfolio runs, not the winner's.

`runner.py` writes a metadata record as the first line of every results file so
these travel with the data.

---

## 4. The development / held-out split

* **Development:** `mincon-testset` and `constrained-small`. Tune here freely.
* **Held-out:** everything else. Run before a release, report it, and **never
  adjust a constant because of what it says.**

Anyone can win a performance profile by fitting to it. This split is the only
thing standing between an honest benchmark and a flattering one, and it is a
discipline rather than a feature — nothing enforces it but the person doing the
work.

---

## 5. Statistical care

**Timings are noisy.** Report medians, not means, and never compare timings
across machines. Run tier 3 on a quiet machine with a fixed CPU governor.
Evaluation counts are deterministic and are the primary metric for that reason.

**Small sets say little.** A 30-problem comparison distinguishes 76% from 90%;
it does not distinguish 76% from 79%. Do not claim a difference the set cannot
support. The full 1075-problem run is what supports a headline number.

**Failures are data.** A solver that crashes on a problem gets a record with
`ok: false` and the exception text, not a silently missing row. Missing rows
inflate rates.

**Determinism.** `Options::seed` fixes every stochastic decision, and the
portfolio's ranking breaks ties by evaluation count and member order so the
winner does not depend on which thread finished first. A benchmark that is not
reproducible is an anecdote.

---

## 6. Comparing against `fmincon`

`fmincon` needs a MATLAB licence and cannot run in CI.
`bench/matlab/fmincon_baseline.m` emits the same JSONL schema; commit the
output and `profiles.py` treats `fmincon` as one more solver.

**Fairness rules, non-negotiable:**

1. **No gradients supplied to anyone.** Plug-and-play is where `fmincon` is
   strongest and the case we claim to beat.
2. Default options otherwise, except a common time limit.
3. Success judged afterwards, by the harness.
4. Record the MATLAB version — `fmincon` changes between releases.

Also run `Options::fmincon_compatible()` (scaling off, dense BFGS, forward
differences, monotone barrier, single-threaded) as its own column. It answers
the question a reviewer will certainly ask: **how much of the win comes from
better defaults, and how much from better algorithms?** Both are legitimate
wins, but they are different claims and should be reported separately.

---

## 7. The reference figure

`fmincon`'s interior-point solved **75.9%** of the 30 constrained problems in
Kronqvist et al. (arXiv:2204.05297) under plug-and-play settings; KNITRO-IP
74.6%, SNOPT 72.1%.

Small set, different machine, different problems. **A target, not a scoreboard.**
The scoreboard is the CUTEst run produced here, and until that run exists with
a `fmincon` column in it, no comparative claim belongs in the README.

---

## 8. Current standing

From `run_testset` (tier 1): **52/54, zero false successes.** Failures are
`HS13` (MFCQ fails at the solution) and `TORTURE_INFEASIBLE` (refuses to lie
but reports the wrong flag) — both waiting on feasibility restoration, M1.

From the smoke benchmark against SciPy: same objective and feasibility on all
four problems; **roughly 5x more objective evaluations per portfolio member
than SLSQP** on small dense problems.

That last number is the most useful one currently available. It is not a bug —
interior point takes more iterations than SQP in that regime and each iteration
pays for a finite-difference gradient and Jacobian — and it points at exactly
three pieces of work: SQP (M5), automatic differentiation (M2), and batched
evaluation (M7).

Publish it. A project that only reports the numbers that flatter it is not
measuring anything.
