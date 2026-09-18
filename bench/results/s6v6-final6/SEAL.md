# Round 6 held-out set (final6): seal record

Sealed September 18, 2026, before any run of any benchmark solver on any problem below. The
candidate is the tree frozen for round 6 (`docs/23`, Phase E): commit `7a4be10` for everything the
wheel contains (`718318d` and `0f80bb2` after it touch only tests), wheel `d57902fa79492531`,
installed in both virtual environments and checked byte for byte against the wheel on the day of
sealing (extension `009c4499a3cf8c39`). The solver was not changed, run or consulted while the set
was designed. In particular the numpy replica of the quadratic probe's band search
(`band_trajectory.py` of Phase D) was **not** run on the two families written to test the band
rule: that would have shown how the candidate behaves on them before the seal. Their sizes were
set from the decay rates and from the measured cost of one objective call, nothing else.

The only computations on these problems before the seal: the reference derivations inside
`bench/corpus/heldout6.py` (closed forms, Lawson-Hanson NNLS, a barrier path, projected Newton,
Newton on KKT systems, an enumeration of KKT points, a gradient flow, and one SciPy SLSQP start on
the noiseless twin of each `stepnoise` problem, which is a reference computation and not a
record), `verify_refs.py` (the harness oracle at the stored points), the probe evaluations of the
MATLAB equivalence gate, and timings of the objective alone.

## Contents

Twenty-three problems in eleven new families (`bench/corpus/heldout6.py`, split `final6` in
`bench/corpus/split.py`). The module's docstring names, for every family, the nearest relative in
the development corpus or the friction audit, what differs, and **why the solvers are expected to
differ, written before any run**. Round 5's set separated no solver on five of its seven families
and its one discriminating family was a friction problem with the seed changed
(`../s6v5-final5/README.md`, section 8); nothing here is a development or friction problem with
its size or seed changed.

| problem | n | m | target | how the reference is known |
|---|---:|---:|---|---|
| EXPSPEC_30 / _90 | 30 / 90 | 0 | 24.9182562512 / 76.7654669000 | non-negative exponential spectrum, chi-square form; Lawson-Hanson NNLS polished on the free set; 25 and 84 bounds active; convex, so f* is unique |
| LOGGAS_30 / _120 | 30 / 120 | 1 | 53.8423572180 / 1482.5519880000 | closed form (Stieltjes): the zeros of the Hermite polynomial scaled to the sphere, multiplier (n - 1) / 4; convex on the ordered chamber |
| PROCRUSTES_3 / _5 | 9 / 25 | 9 / 25 | 1.3019562999 / 4.3113516726 | closed form: X = U V' from the SVD of A'B; all k^2 orthogonality equalities, k (k - 1) / 2 duplicates |
| EIGSUB_6X2 | 12 | 4 | -8.5 | closed form: minus the two largest eigenvalues; one duplicate equality; the minimisers are an orbit |
| CSTR_3 / _8 | 9 / 24 | 3 / 8 | 21.2285049078 / 13.3019732643 | closed form: every reactor at the temperature cap, equal residence times (AM-GM); the only KKT point; CSTR_8 starts with balance residuals of 1.4e3 |
| PROPFAIR_12 / _60 | 12 / 60 | 6 / 25 | 14.9621105863 / 98.0622974281 | strictly convex; barrier path, then Newton on the KKT system of the 4 and 17 saturated links, multipliers above 1e-4 |
| TVDENOISE_500 / _1000 | 500 / 1000 | 0 | 3.0697293304 / 4.9301569865 | strictly convex; projected Newton, then Newton on the final active set; 39 and 32 bounds active |
| WING_SIMPLE | 9 | 7 | 303.0751594495 | geometric program (a KKT point is global); Newton on the KKT system of the seven active rows; agrees with the published 303.1 N |
| PIPE_SIZING | 5 | 4 | 74196.0006488676 | geometric program with a closed form, D* = (4.75 b / 1.5 a)^(1/6.25) |
| STEPNOISE_6 / _12 | 6 / 12 | 1 | 14.7507357269 / 29.2936278450 | the noiseless optimum (convex; Newton on its KKT system, ball row active); the noise, 1e-9 and 1e-7, moves f by at most its amplitude |
| PHASESPLIT_6 / _10 | 6 / 10 | 1 | -1.9591132307 / -2.1768920618 | every KKT point enumerated over the cubic's branch patterns, the global one polished by Newton; the start's own minimum (gradient flow) is a different one |
| WINKLER_STIFF_250 / _SOFT_250 | 250 | 0 | -159.4416881971 / -71.2460287571 | strictly convex box QP; projected Newton and the exact solve on the final active set; 182 and 185 bounds active |
| LONGMEM_S08_200 / _S20_200 | 200 | 199 | 13.5585555287 / 25.2238047593 | strictly convex QP with monotonicity rows; pooling and splitting on the block system, multipliers by cumulative sums; 152 and 154 rows active |

No target is a published rounding, none carries the `best-known` tag, and every |f*| is above 1,
so the 1e-4 bar is relative on every problem of this set (in round 5 eight of thirteen targets
were below 1 and the bar was absolute there). Three problems are expected to be missed by every
local solver at defaults (STEPNOISE_12 and the two PHASESPLIT problems) and are in the set because
`docs/23` asks what the solvers say there, not only whether they arrive.

Costs of one objective call, the model alone, the minimum over eight batches of twenty calls
through the harness's counting model: WINKLER_STIFF_250 1.82 ms, WINKLER_SOFT_250 3.89 ms,
LONGMEM_S20_200 4.01 ms, LOGGAS_120 1.80 ms, TVDENOISE_1000 0.61 ms, EXPSPEC_90 0.76 ms;
DECONV_200 measures 4.10 ms the same way (4.1 to 4.34 ms in `docs/22`).

## Gates passed before sealing

* `verify_refs.py`, one problem per process, under the corpus model the workers use (`refs.jsonl`
  in this directory): every stored point is feasible to 1e-8 (worst 1.9e-9, WING_SIMPLE, on a row
  of 4e6), reproduces its target to 1e-6 (worst 3.4e-9, STEPNOISE_12, which is its noise) and is
  first-order stationary to 1e-10 of the gradient's scale by the oracle's own multiplier recovery
  (worst 2.6e-11, EXPSPEC_90). Round 5's generation assertion allowed 5e-3.
* **CORRUGATED_BULKHEAD's reference point recomputed** (`bench/corpus/heldout5.py`), as round 5's
  report asked before `final5` is reused. The point printed with the published value,
  (57.692, 34.148, 57.555, 1.05), is not the optimum of the model: its third coordinate gives
  f = 6.846036. The optimum is the vertex (57.692308, 34.147620, 57.692308, 1.05): t at its
  bound, both thickness rows and the second section row active, positive multipliers,
  f = 6.842958010080774, of which the published 6.8429 is a rounding. The model's checksum is
  unchanged (`8b94a1cfd4529dae`) and so are its probes. Targets v7 carries the recomputed value
  with a logged revision, which moves every archived record's gap by 8.5e-6 and changes no
  attainment (none of the 39 archived records of this problem lies between 5e-5 and 2e-4).
* Equivalence NumPy against MATLAB at the start and four probe points per problem: **4160
  quantities on 208 problems, 0 mismatches** (`bench/corpus/matlab_probes.json`). Probe points of
  the two families whose domain is not a box (loggas, propfair) are pulled towards the start until
  the model is finite (`generate_one.py`), so MATLAB is never asked for a complex value.
* Every pre-existing problem's manifest and probe entries keep their bytes: the new entries were
  generated one per process and merged in (`generate_one.py`). `git diff` of `manifest.json` is
  466 added lines and 3 removed, the three being CORRUGATED_BULKHEAD's `ref_f`, `has_ref_x` and
  `source`; each of the 185 old probe entries is found verbatim in the new `probes.json`.
* Targets v7 = v6 plus 23 entries, all `closed-form-or-published`, and the one recomputed entry.

## Protocol for the run: v3 (`docs/15`, Revisions, decided before this seal)

Nine arms: `mincon`, `mincon-ip`, `mincon-sqp`, `scipy-slsqp`, `scipy-trust-constr`,
`fmincon-interior-point`, `fmincon-sqp` as in rounds 1 to 5, plus
`fmincon-interior-point@MaxFunctionEvaluations=100000;MaxIterations=100000` and the same for
`fmincon-sqp`. Tracks A (finite differences, the minimal-input first try, the metric that matters
first) and C (exact derivatives), defaults otherwise, single thread, 60 s and 100 000 objective
evaluations **for every solver** (the harness now stops SciPy at the shared budget and reports
its last iterate), targets v7; then timing with three repeats for `mincon`,
`fmincon-interior-point` and `fmincon-sqp` on track A. Gate G1 of `docs/23` is read against all
four fmincon arms.

What v3 changed in the harness was checked on development problems only (HS71, DECONV_60,
TCPORT_20, all nine arms, both tracks): every count that round 5 also recorded is reproduced
(fmincon-interior-point 3050 on DECONV_60 at its factory cap, fmincon-sqp 6039, mincon 1552,
SLSQP 2562, trust-constr 19 520), the lifted arms run past the cap (12 569 and 9 403, both
converged), a SciPy run given 0.05 s or 40 evaluations stops there with `native_status = -98`,
and on track C fmincon-sqp's gradient count is now its own (15 against 22 objective calls on
TCPORT_20; it was equal to the objective count by construction).

Results are reported as measured. After this run final6 is development material like its
predecessors, and only this run may change `docs/17` or the README status block.

## Hashes (SHA-256, first 16 hex digits)

| file | hash |
|---|---|
| `bench/corpus/targets_v7.json` | 93ac5099abaed89d |
| `bench/corpus/manifest.json` | 6c31497c282a92e5 |
| `bench/corpus/probes.json` | cd593b87b4ca4da2 |
| `bench/corpus/matlab_probes.json` | f8b325569c8f818d |
| `bench/corpus/heldout6.py` | 73a1afb7af369ce3 |
| `bench/corpus/heldout5.py` (CORRUGATED_BULKHEAD's reference only) | af139a1a7b44249b |
| `bench/corpus/split.py` | 568b31d07065e457 |
| `bench/corpus/families.py` | fe2ca738f21cb2fe |
| `bench/corpus/verify_refs.py` | f2de47640ef77dc6 |
| `bench/corpus/generate_one.py` | 845df5566a501e83 |
| `bench/harness/worker_python.py` | 30d8b9e0fa44cad4 |
| `bench/harness/worker_matlab.m` | b05f338740d28f54 |
| `refs.jsonl` (this directory) | ac0c8035c9654b75 |
| candidate wheel `mincon-0.1.0-cp39-abi3-win_amd64.whl` | d57902fa79492531 |
