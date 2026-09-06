# Specification: the linear algebra layer

Implemented in `crates/mincon-linalg`. This is the layer the whole project
rests on, and the one where the interesting decision was made.

---

## 1. The problem, and why it forces a decision

An interior-point NLP solver must factor a symmetric **indefinite** KKT matrix
and know its inertia. The established answer is HSL's `MA27`/`MA57`.

HSL is not redistributable under a permissive licence. That single fact is why
`pip install ipopt` does not exist, why `cyipopt` builds are fragile, and why a
large share of people who would benefit from IPOPT use SciPy instead. Since
`pip install mincon` is a load-bearing goal (`docs/00_MISSION.md`), HSL is out.

What is left:

| Option | Verdict |
|---|---|
| MUMPS | Fortran toolchain, awkward licence, heavy |
| PARDISO | proprietary |
| `faer` 0.24 (MIT) | sparse `LLT`, `LU`, `QR` — **no sparse `LDL^T` or Bunch–Kaufman**. Verified against the 0.24.4 docs. |
| `qdldl` via `clarabel` | Apache-2.0 and good, but it is a quasi-definite `LDL^T` with no inertia contract and no per-pivot sign expectations |
| **Write it** | ~600 lines, no dependency, full control |

So we write it. This is not the compromise it looks like: it is what makes the
distribution story possible at all.

---

## 2. The design

### 2.1 Sylvester's law does the work

`A = L D L^T` with `L` unit lower triangular is a **congruence transformation**,
so by Sylvester's law of inertia the signs of `D` *are* the inertia of `A`. No
2x2 pivots, no Bunch–Kaufman search, no eigenvalues — the Wächter–Biegler
inertia test comes out of the factorization for free.

The catch is that a symmetric indefinite matrix need not *have* an `LDL^T` with
1x1 pivots. The factorization can break down on a zero pivot, and
near-breakdown is numerically poisonous.

### 2.2 Quasi-definiteness closes the loop

A matrix

```
[  H   A  ]      with H positive definite and E positive definite
[ A^T -E  ]
```

is **quasi-definite**, and Vanderbei (1995) showed a quasi-definite matrix
admits an `LDL^T` factorization under *any* symmetric permutation — no pivoting
search needed, unconditionally.

The interior-point KKT matrix becomes quasi-definite exactly when `delta_w` is
large enough to make `W + Sigma + delta_w I` positive definite and `delta_c > 0`.

So the loop closes:

* an uncertified factorization is itself the signal to raise `delta_w`;
* raising `delta_w` is what makes the factorization certifiable.

### 2.3 Element growth must be detected, not assumed away

`LDL^T` without pivoting is unstable in general. The textbook counterexample:

```
[ 1e-14   1 ]
[   1     2 ]
```

`d_0 = 1e-14`, `L_10 = 1e14`, and the computed factorization has nothing to do
with the input.

The factorization therefore tracks `max |L_ij|` and
`Factorization::inertia_is_certified()` is false when it exceeds
`DEFAULT_GROWTH_LIMIT = 1e10`. The caller's response is to raise `delta_w` —
which pushes the matrix toward quasi-definiteness, where growth is bounded.

*This guard was added because a test failed.* The first version reported a
confident inertia on that 2x2 and produced a solution off by 30 orders of
magnitude. The test `catastrophic_element_growth_is_detected_not_hidden` exists
so it cannot come back.

### 2.4 Dynamic regularization and iterative refinement

Pivots below `eps = 1e-13` are perturbed to `±delta = 1e-7` with the sign the
caller expects (`+1` for primal rows, `-1` for dual rows). The values are in
the range OSQP and Clarabel use.

Because that silently changes the matrix, **every solve is refined against the
unperturbed one** (`solve_refined`, default one step). `regularized_pivots()`
reports whether it fired.

---

## 3. Implementation

**Symbolic** (`Symbolic::analyse`, once per solve):

1. Validate square upper-triangular input.
2. Compute the fill-reducing permutation.
3. Build the permuted upper-triangular pattern with the diagonal forced
   present, plus a `value_map` from original entries to permuted slots.
4. Elimination tree and column counts (the `QDLDL` formulation).

**Numeric** (`Factorization::factor`, allocation-free, once or more per
iteration): up-looking `LDL^T`. For each row `k`, gather the pattern as the
reach of column `k` of `A` through the elimination tree, **sorted ascending** —
which is a valid topological order because `etree[j] > j` always.

> `QDLDL` avoids the sort with a reverse-topological trick. Replacing the sort
> is a legitimate optimization; do it *after* the benchmark says the sort
> matters, and keep `reconstructs_the_matrix_exactly` passing.

---

## 4. Ordering — the known gap

`Ordering::Amd` currently falls back to `Rcm`.

AMD (Amestoy–Davis–Duff) is what MA57, CHOLMOD and QDLDL all use, and on large
KKT systems it is worth an order of magnitude in fill against anything else.
It is deliberately not hand-rolled here: a minimum-degree implementation is a
multi-week project whose failure mode is "quietly 10x slower".

**The task:** wire in the `amd` crate (v0.2.2, a port of SuiteSparse AMD,
BSD-3-Clause, licence-compatible) behind `Ordering::Amd`, and gate the switch
on measured fill (`Symbolic::fill_ratio`) and factorization time.

RCM is genuinely good on the banded systems that come out of discretized
optimal control, which is a large share of real NLPs, and it is sixty obviously
correct lines. Ordering affects speed only, never the answer, and every test in
the crate runs under all available orderings.

---

## 5. Dense path

`LinearSolverKind::DenseLblt` is not implemented; `Auto` always chooses sparse.
Below roughly `n + m < 200` a dense Bunch–Kaufman with 2x2 pivots would be
faster *and* more numerically robust, since it needs no growth guard. Worth
having, and the crossover must be measured rather than guessed.

---

## 6. Test contract

Nothing here may be changed without these still passing:

| Test | What it protects |
|---|---|
| `inertia_matches_an_independent_eigenvalue_oracle` | 25 random quasi-definite matrices, inertia checked against a **cyclic Jacobi eigensolver sharing no code with the factorization** |
| `reconstructs_the_matrix_exactly` | dense `L D L^T` reassembled and compared entrywise to `P A P^T` |
| `all_orderings_agree_on_the_solution` | ordering cannot change the answer |
| `catastrophic_element_growth_is_detected_not_hidden` | the growth guard |
| `raising_delta_w_restores_the_certificate` | the regularization loop actually terminates |
| `iterative_refinement_delivers_a_small_residual_on_an_ill_conditioned_system` | Hilbert matrix, `n = 10`, residual `< 1e-14` |

The independent oracle is the important one. A factorization tested only
against itself is tested against nothing.

## Sources

- Vanderbei, *Symmetric quasi-definite matrices*, SIAM J. Optim. 5(1), 1995.
- Amestoy, Davis & Duff, *An approximate minimum degree ordering algorithm*, SIAM J. Matrix Anal. Appl. 17(4), 1996.
- Davis, *Direct Methods for Sparse Linear Systems*, SIAM, 2006.
- [QDLDL / OSQP](https://github.com/osqp/qdldl) — Apache-2.0, the reference for the up-looking kernel.
- [faer](https://docs.rs/faer/latest/faer/) — MIT; confirmed to lack sparse `LDL^T` at 0.24.4.
- [Clarabel.rs](https://github.com/oxfordcontrol/Clarabel.rs) — Apache-2.0; a good reference for dynamic regularization in practice.
