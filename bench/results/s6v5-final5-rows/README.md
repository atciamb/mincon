# s6v5-final5-rows: a disclosed re-run of the spent round-5 sealed set, September 13, 2026

**This is not a qualification run and it makes no new claim.** The round-5 claim
(`../s6v5-final5/README.md`, `docs/17` items 19-21) stands exactly as first measured, on the tree
of commit `d9a8ab8`. This directory exists because one increment landed *after* that run —
`quadratic_rows='values'`, commit `3aac1e5` — and two of the sealed problems (TCPORT_20,
TCPORT_100) are in the family it touches. Leaving the README silent about a default that postdates
the claim would have been misleading, so the mincon arm was re-run at HEAD and the difference is
reported here. See the caveat at the end for what this costs the set.

The sealed inputs were re-verified unchanged before the run: `targets_v6.json 17a469303e038cfc`,
`manifest.json a9c9c15a15eefe38`, `probes.json 66edf581f56a7d7d`, `heldout5.py 37d2e0a4607ae6df`
(`../s6v5-final5/SEAL.md`). The sealed directory's records were not touched.

Run: the same 13 problems, split `final5`, targets v6, 60 s / 100 000 evaluations, single thread,
both tracks, two arms — HEAD (`quadratic_rows='values'`) and the same wheel with
`quadratic_rows=off`, which is the sealed tree's behaviour. **The `off` arm reproduces the sealed
records bit for bit** — identical returned points on 12 of 13 track-A problems and all 13 on
track C — which is the check that the only difference between HEAD and the sealed tree is this
option. The exception is DECONV_200 on track A, a clock-bound record (60 s, status 0) whose
evaluation count moves with machine noise: 16 885 sealed, 11 458 off, 14 071 at HEAD.

| track | attainment, HEAD vs `off` | difference | cost ratio [95 % family bootstrap] | records that differ |
|---|---|---|---|---|
| A | 12 / 12 of 13 | **+0.0 pp [+0.0, +0.0]** | 1.00 [1.00, 1.01] on 12 | TCPORT_20, TCPORT_100 (+ DECONV_200, the clock) |
| C | 13 / 13 | **+0.0 pp [+0.0, +0.0]** | 0.93 [0.79, 1.00] on 13 | TCPORT_20, TCPORT_100 |

**Nothing changes outcome.** The rows build fires on exactly the two tcport problems, both of
which were already attained, and both remain attained with the same exit status. Per problem, in
model evaluations:

| problem | track A: `off` -> HEAD | track C: `off` -> HEAD |
|---|---|---|
| TCPORT_20 | 702 -> 632 (0.90x) | 57 -> 45 (0.79x) |
| TCPORT_100 | 5882 -> 6778 (**1.15x**) | 67 -> 31 (0.46x) |

TCPORT_100's track-A cost is the effect `abl-i5-rows` already reported and did not tune away: its
*objective* Hessian is dense, so the `n (n + 3) / 2` row build is paid on top of a build that was
already expensive. It is cheaper on track C, where the supplied Jacobian makes the row build
`n + 1` calls. The returned point moves by 7.1e-4 in the infinity norm on TCPORT_100 and 3.7e-6 on
TCPORT_20; both stay inside the frozen tolerance.

One recovered-KKT certificate is gained on each track (track A 8 -> 9, track C 12 -> 13), which is
the Newton path ending at a point the oracle can certify where the quasi-Newton path did not.

**The honest caveat, stated precisely: this is the third time the sealed set's records have been
produced since it was sealed.** (1) The qualification run, which is the claim, and whose records
were then read problem by problem for `docs/17` items 20 and 21. (2) `abl-i5-rows`, because the 13
problems were folded into the 185-problem development corpus after that run, the way every earlier
held-out set was — that ablation's verdict does not depend on them (removing the 13 gives 160/159
at 0.988 on track A and 162/162 at 0.908 on track C), but it is not independent of them either.
(3) This re-run. That is why this directory is filed as a disclosure and not as a claim, why no
tuning follows it, and why `docs/17` item 19 keeps its original numbers. **A further claim needs a
new sealed round, and `final5` can no longer provide one.**

Files: `mincon.{A,C}.jsonl` (HEAD), `mincon_quadratic_rows-off.{A,C}.jsonl` (the sealed tree's
behaviour), `scored-{A,C}.jsonl` with their analyses, `experiment.json`.
