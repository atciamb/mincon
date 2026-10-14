# Round 6 protocol amendment v3a: the run without MATLAB

Decided October 14, 2026, before any record of the five arms below exists on `final6`, and
committed before the run. `SEAL.md` is not changed: this file amends its protocol paragraph and
protocol v3 (`docs/15_BENCHMARK_PROTOCOL_V2.md`, Revisions, v3a). Reason: this project no longer
runs MATLAB.

## What changes

1. **Arms.** Five: `mincon`, `mincon-ip`, `mincon-sqp`, `scipy-slsqp`, `scipy-trust-constr`, on
   tracks A and C. The four `fmincon` arms of v3 are not run.
2. **Timing.** Three repeats on track A for `mincon` and `scipy-slsqp` (v3: `mincon` and the two
   `fmincon` algorithms), read with `timing.py --baseline scipy-slsqp --targets v7`.
3. **Gate G1 of `docs/23` is read as G1':** two consecutive fresh sealed rounds of at least 20
   problems in at least 10 families each; track-A attainment at least that of `scipy-slsqp` and of
   `scipy-trust-constr`, both held to the shared budget; zero false certificates; every miss
   carrying a diagnosis in `notes`. Round 6 can be at most the first of the two rounds.
4. **`docs/15` section 6.** Items 2 and 3 (the primary comparison against
   `fmincon-interior-point`) are not evaluated on `final6`. Item 4's secondary baselines become
   `scipy-slsqp` and `scipy-trust-constr`. Items 1 and 5 are unchanged.
5. **Analysis.** `score.py --targets v7`; `analyze.py --baseline scipy-slsqp`, and again with
   `--baseline scipy-trust-constr` (their default baseline is an `fmincon` arm, which this run
   does not have).
6. **Record.** The supervisor is given `--matlab none`, so `experiment.json` shows that no MATLAB
   installation was used; `supervise.py` starts MATLAB only for `fmincon` solver names.

## What does not change

The set (23 problems in 11 families, `SEAL.md`), targets v7, the oracle, the tolerances, the
budgets (60 s and 100 000 objective evaluations, shared, with v3's stop of a SciPy run at the
budget), the hard factor 3, the grace of 30 s, one thread, seed `s1`, the tracks, the cost
metric, the frozen candidate (wheel `d57902fa79492531`, extension `009c4499a3cf8c39`) and the 13
hashed files. Each arm's problem order comes from a generator seeded by the seed, the solver
name and the repeat (`bench/harness/supervise.py`), so dropping arms moves no other arm's order.

No arm is added. An interior-point reference other than MATLAB's (IPOPT through `cyipopt`) would
need adapter code and options written after the seal, so it is registered for round 7 instead.

## What this bends, disclosed

* `docs/15` allows no revision of the final protocol once any final record exists. `final6`
  has records: the 53 of attempt 1 (`mincon`, `mincon-ip` and `mincon-sqp` on track A,
  `../s6v6-final6-attempt1`), used once to screen for false certificates. v3a only removes
  comparator arms, for a reason outside the benchmark, and no record of any comparator exists
  on `final6`, so it changes nothing those 53 records could have informed.
* `SEAL.md` reserves `docs/17` and the README status block for this run. v3a makes one
  exception before the run: statements resting on `fmincon` results may be withdrawn from them,
  marked as withdrawn and never replaced by a new claim. Nothing may be added or strengthened
  before this run is scored.
* G1' is not G1. Against SLSQP it is the harder half: on the last two sealed rounds mincon's
  track-A attainment was below SLSQP's (round 4: 9/11 against 11/11; round 5: 12/13 against
  13/13), so G1' would have failed both. trust-constr ran in round 5 only (11/13, against
  mincon's 12/13). How G1' compares with v3's four `fmincon` arms is not known and is not
  claimed.

## The run

On the benchmark laptop, once, on mains power, unattended, with sleep, hibernation and lid-close
disabled; about 3 to 4 hours. If it is interrupted, its files move to `s6v6-final6-attempt2`
and the decision is written down before any rerun. Results are reported as measured; afterwards
`final6` is development material like its predecessors.
