# Round 4 held-out set (final4): seal record

Sealed September 12, 2026, 00:01 local, in the same step that launched the qualification run; the corpus files, the equivalence gate and targets v5 were finished in the minutes before, and no candidate had run on these problems before any run of any mincon candidate on
these problems. Written by the session that reviewed C7; nothing in the
solver was tuned on, or run against, any problem below before this record.

## Contents

Eleven problems in four new families (`bench/corpus/heldout4.py`, split
`final4` in `bench/corpus/split.py`), every reference exact and verified at
generation (targets v5, kind `closed-form-or-published`):

| problem | n | m | reference | how verified |
|---|---:|---:|---|---|
| COVQP_30 / _120 / _300 | 30 / 120 / 300 | 2 | −0.0509364053 / −0.0588015821 / −0.0623946612 | active-set KKT Newton polish, stationarity < 1e-10; risk row active; dense Σ |
| DENSELAP_100 / _250 | 100 / 250 | 1 | 57.1200683191 / 150.413453752 | equality-constrained Newton, stationarity < 1e-10; dense Laplacian + exp terms; box inactive |
| SNL_24 / _60 / _150 | 24 / 60 / 150 | 1 | 0 | closed form (noiseless distances, true configuration feasible); dense, nonconvex |
| OBSTACLE_50 / _200 / _500 | 50 / 200 / 500 | 0 | 0.594226684752 / 0.586644577906 / 0.590324134557 | active-set solve of the bound-constrained QP, stationarity < 1e-12 |

Three families (covqp, denselap, snl) have dense coupled Hessians at n ≥ 100,
which the development corpus lacked (`docs/21` §7, G3). An analytic-centre
family was designed first and dropped before any run because its generated
MATLAB gradient was 69 MB at n = 100 (noted in `heldout4.py`).

## Gates passed before sealing

* Equivalence NumPy vs MATLAB at 4 probe points per problem: 3440 quantities
  on 172 problems, 0 mismatches (`bench/corpus/matlab_probes.json`).
* Every pre-existing problem's files are byte-identical to the committed
  ones (the new entries were merged in; a whole-corpus regeneration was not
  used, see the private worklog).

## Hashes (SHA-256, first 16 hex digits)

| file | hash |
|---|---|
| `bench/corpus/targets_v5.json` | 7185a5b5a9ea4ee2 |
| `bench/corpus/manifest.json` | 828a48245fb6e179 |
| `bench/corpus/probes.json` | 0848d35f372c51b1 |
| `bench/corpus/heldout4.py` | 6ec253c572e2bf60 |
| candidate wheel `mincon-0.1.0-cp39-abi3-win_amd64.whl` (C8 tree: C7 rule + route counters and rebuild log in the notes, numbers identical to `abl-c7` on 36 spot runs) | e821c1017de84cff… |

## Protocol for the run

Same as round 3 (`bench/results/s6v3-final3`): six solvers (`mincon`,
`mincon-ip`, `mincon-sqp`, `fmincon-interior-point`, `fmincon-sqp`,
`scipy-slsqp`), tracks A and C, defaults, single thread, 60 s / 100 000
evaluations, targets v5; then timing with three repeats for `mincon`,
`fmincon-interior-point`, `fmincon-sqp` on track A. Results are reported as
measured. After this run final4 is development material like its
predecessors.
