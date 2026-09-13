# Round 5 held-out set (final5): seal record

Sealed September 13, 2026, in the same step that launched the qualification run; the corpus
files, the equivalence gate and targets v6 were finished in the minutes before, and no mincon
candidate had run on any problem below before this record. Written by the session that landed
the structured Hessian build (`abl-i5-build`); nothing in the solver was tuned on, or run
against, any problem below before this record. The only computations on these problems before
the seal were the reference derivations at generation (SciPy least squares, SLSQP, BVLS and
L-BFGS-B starts, all polished and verified by independent KKT arithmetic) and the oracle's
probe evaluations for the MATLAB equivalence gate.

## Contents

Thirteen problems in seven new families (`bench/corpus/heldout5.py`, split `final5` in
`bench/corpus/split.py`), from the round-5 plan (`docs/22` §5, gate G3) and from the two gaps the
quadratic-program probe opened (a QP with a quadratic constraint row, which the probe declines by
design, and a nonconvex QP with several local minima, which its convexity check declines):

| problem | n | m | reference | kind | how verified |
|---|---:|---:|---|---|---|
| PKFIT_CLEAN | 4 | 1 | 0 | closed form | two-compartment pharmacokinetics, k2 / k1 = 1000 (stiff), noiseless, truth (2, 0.05, 5, 50) feasible |
| PKFIT_NOISY2 | 4 | 1 | 0.0205819534 | best-known | 2 % multiplicative noise; Newton-polished least squares from the truth, stationarity 1e-14 |
| LOGISTIC_CLEAN | 3 | 0 | 0 | closed form | logistic growth, noiseless, truth (50, 0.8, 6) |
| LOGISTIC_NOISY2 | 3 | 0 | 12.4210808646 | best-known | 2 % noise; polished least squares, stationarity 4e-13 (tolerance 1e-10 relative to f*) |
| TCPORT_20 / _100 | 20 / 100 | 2 | −0.1258414254 / −0.1345121050 | exact | quadratic transaction costs, budget row, variance row x'Σx ≤ σ² active; active-set KKT Newton polish, stationarity < 1e-10 |
| DECONV_60 / _200 | 60 / 200 | 0 | 0.0090414265 / 0.0339821967 | exact | bounded deconvolution (box_lsq at n); BVLS polished on the free block, projected gradient < 1e-10 |
| NOISYQP_20 | 20 | 0 | −9.6575553647 | exact (noiseless optimum) | convex box QP + 1e-7 (sin(1e3 a'x) + sin(3e4 b'x)); KKT polish of the noiseless QP; the perturbation moves f by < 1e-5 relative |
| NCBOXQP_8 / _12 | 8 / 12 | 0 | −8.4808888430 / −12.5382796787 | exact (global) | indefinite QP on [−1, 1]^n; every local minimum enumerated over the 3^n active-set patterns; tagged multiple-local-minima |
| CORRUGATED_BULKHEAD | 4 | 6 | 6.8429 | best-known | Kim and Lee corrugated bulkhead; the published point checks feasible to 2e-3 with f within 5e-3 |
| I_BEAM | 4 | 2 | 0.0130741 | best-known | Gold and Krishnamurty I-beam deflection; the published point checks feasible with f within 1e-6 |

The ODE fits use the closed forms of their models (the corpus needs SymPy expressions), the
simulator noise is a deterministic high-frequency term for the same reason (noted in the
problem), and both design problems start infeasible, as a user's first guess would.

## Gates passed before sealing

* Equivalence NumPy vs MATLAB at 4 probe points per problem plus the start: **3700 quantities
  on 185 problems, 0 mismatches** (`bench/corpus/matlab_probes.json`).
* Every pre-existing problem's manifest and probe entries are byte-identical to the committed
  ones (the new entries were generated one per process and merged in; `git diff` of
  `manifest.json` is 251 added lines and no removed line).
* Targets v6 = v5 plus 13 entries (`targets.py extend v6 --base v5`): 9 `closed-form-or-published`,
  4 `best-known-published`.

## Hashes (SHA-256, first 16 hex digits)

| file | hash |
|---|---|
| `bench/corpus/targets_v6.json` | 17a469303e038cfc |
| `bench/corpus/manifest.json` | a9c9c15a15eefe38 |
| `bench/corpus/probes.json` | 66edf581f56a7d7d |
| `bench/corpus/heldout5.py` | 37d2e0a4607ae6df |
| candidate wheel `mincon-0.1.0-cp39-abi3-win_amd64.whl` (the tree of commit `f0d847d`: round-5 I1-I9, S-E, I5 with the structured build, the no-scale note, I6; installed in both venvs) | 0c47a714011fa423 |

## Protocol for the run

Seven solvers (`mincon`, `mincon-ip`, `mincon-sqp`, `scipy-slsqp`, `scipy-trust-constr`,
`fmincon-interior-point`, `fmincon-sqp`), tracks A (finite differences, the minimal-input
first-try metric that matters first) and C (exact derivatives), defaults, single thread,
60 s / 100 000 evaluations, targets v6; then timing with three repeats for `mincon`,
`fmincon-interior-point`, `fmincon-sqp` on track A, reported with the solver / model split.
Results are reported as measured; after this run final5 is development material like its
predecessors, and only this run may change `docs/17` or the README status block.
