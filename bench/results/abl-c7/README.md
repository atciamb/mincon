# Ablation C7: curvature-tracking rebuild of the quasi-Newton model

September 11, 2026. The increment of `docs/21_LARGE_N_CURVATURE_PLAN.md`,
ablated on the whole corpus (158 scorable non-diagnostic problems, track A,
defaults, single thread, targets v4). Candidate and baseline are the **same
build** run with the rule on (`bfgs_rescale = 10`, the default) and off
(`bfgs_rescale = 0`); off reproduces candidate C6 exactly. Wheel SHA-256
`7b5b7ca7…4d8d` (`.local-research/wheels-c7-final`).

## The rule

Inside the shared dense-BFGS update (`crates/mincon-ip/src/bfgs.rs`, used by
both members), after Powell damping: if the curvature the model predicts along
the accepted step, `sᵀBs`, is off from the measured `sᵀy` by more than a
factor of ten in either direction, the matrix is rebuilt from the
per-coordinate curvature quotients `yᵢ/sᵢ` (the C5 clamped diagonal) before the
ordinary update — but only when one of two tests confirms the diagonal is the
right move:

* **scale** — the quotients agree the diagonal itself is wrong on ≥ 80 % of
  the pair's weight `|sᵢyᵢ|` (a unit matrix on a curvature-1000 problem;
  curvature that grew 100× along the path);
* **shape** — the model's own diagonal explains the pair at least twice as
  well (in log terms) as the full matrix does, i.e. the off-diagonal part
  accumulated from earlier inconsistent pairs is cancelling the curvature
  along the directions the step now takes.

Rebuilds are rate-limited (a cooldown of five accepted updates that doubles
each time) and **gated to n ≥ 10**: the repair pathology costs of order n
iterations, so below ten variables a dense BFGS reaches the curvature in a
handful of updates and a mid-course rebuild only perturbs it. The gate is why
this version is clean where the first, un-gated one was not (see "The gate"
below).

## Result — rule on vs off, whole corpus, track A

| configuration | attained on / off | only-on | only-off (lost) | geo-mean evaluations on/off [95 % family bootstrap] | n ≥ 100 subset |
|---|---|---:|---:|---|---|
| **portfolio (`mincon`)** | **150 / 149** | 1 (ELLIPSOID_500) | **0** | **0.94 [0.78, 0.97]** | 0.81 |
| IP member | 148 / 147 | 1 (ELLIPSOID_500) | 0 | 0.96 [0.82, 0.99] | 0.81 |
| SQP member | 148 / 145 | 3 (QUADSPHERE2_300, QUADSPHERE_1000, ELLIPSOID_500) | 0 | 0.88 [0.58, 0.96] | 0.32 |

All three intervals exclude 1.0; no configuration loses an attainment. Gate G1
of `docs/21` is met on attainment and on the whole-corpus ratio, and **not met
on its n ≥ 100 subset criterion** (≤ 0.5×) for the portfolio and the IP member:
over the ten n ≥ 100 problems both arms attain, the subset ratio is 0.81×
(the newly attained ELLIPSOID_500 is outside it); the SQP member's is 0.32×.
The geo-mean and the subset ratio are over problems attained by both arms, so
the two PORTFOLIO problems (no target yet) contribute to neither.

### Where the evaluations go (portfolio)

Wins, largest first: MAXENT_200 **0.157×** (25 166 vs 160 520 model
evaluations; 59 vs 385 iterations — this is the problem that drove round 3's
1.13× against `fmincon-sqp`), ELLIPSOID_500 **attained** where the rule-off run
stalled at the budget (f = 2034.13 = target vs 1995.49 with violation 0.85),
QUADSPHERE_10 0.179×, HS118 0.322×, DISPATCH_20 0.447×, ELLIPSOID_50 0.509×,
MAXENT_10 0.510×, NNLS_SIMPLEX_30 0.607×, MAXENT_50 0.610×, ELLIPSOID2_20
0.645×, CHAINROSEN_EQ_10 0.667×. The one cost regression is ELLIPSOID2_200 at
1.035× (412 → 424 iterations after one rebuild under the IP member; the
problem is non-separable, and the rebuilt diagonal neither helps nor hurts
much). Exit-status changes: HS112 (portfolio, SQP) and DISPATCH_20 (IP)
`Acceptable` → `Optimal`; HS114 under the IP member `Optimal` →
`StepTolerance` after one rebuild, still attained.

### The SQP member alone

Larger swings because it has no interior-point fallback: QUADSPHERE_100
**0.022×** (2 vs 178 iterations), CHAINROSEN_EQ_200 0.109×, MAXENT_200 0.126×,
QUADSPHERE2_300 0.158× (newly attained), QUADSPHERE_1000 0.399× (newly
attained), ELLIPSOID2_200 0.467× (139 → 64 iterations after two rebuilds — the
rule does reach this non-separable problem for the SQP member; it does not for
the IP member). CHAINROSEN_BOX_200 rebuilds once and is unchanged (496 vs 494
iterations, budget exit either way). One real blemish — **PORTFOLIO_100 at
7.0×** (175 vs 23 iterations, exit `Optimal` → `Acceptable`): a **single**
rebuild through the **scale route** (route counters added after this run,
`bench/results/abl-c8-rejected/`) on a dense covariance QP, where the
quotients pass the 80 % consistency test yet are not the diagonal of anything,
and the damped updates that follow take 150 iterations to undo it. The problem
has no target (best-known pending), so it is outside the attainment counts and
the cost ratio above. The portfolio's record on it is byte-identical rule on
and off, because n = 100 > 20 routes it to the interior-point member; that
member's objective is 0.09 % above the SQP member's rule-off value, and
whether that is within the frozen tolerance is unknown until a target exists.
Filed against the scale route on coupled problems (`docs/16` cluster 15,
`docs/21` §7).

## The gate (why n ≥ 10)

The first version of the rule had no size gate. On the whole corpus it reached
0.947× evaluations but **lost two attainments, HS97 and HS98** (both n = 6):
the scale route fired early, when the matrix was still near-unit and the
quotients all exceeded the factor, and rebuilt to a diagonal that ignores the
coupling a dense BFGS would have learned in six updates
(`bench/results/abl-c7` was regenerated with the gate; the un-gated records
were not kept, contrary to the plan's `abl-c7-rejected/` rule, and only the
summary survives in the private worklog). Gating the rule to n ≥ 10 makes all
118 problems with n < 10 byte-identical rule-on vs rule-off in every
configuration while keeping every n ≥ 10 win (QUADSPHERE_10 at n = 10 is the
smallest problem the rule helps). The threshold was chosen after seeing that
it separates the two losses from the smallest win, i.e. it is fitted to this
corpus; the round-4 set is where that is tested. This is the same "helps
large, perturbs small" boundary that governed the earlier BFGS scaling
studies (`bench/results/s4-bfgs-scaling-rejected`,
`s5-bfgs-guarded-diagonal`).

## Status

**Rejected on round 4 (September 12, 2026).** On the sealed final4 set the
rule cost 1.24× [0.99, 1.67] evaluations for the portfolio (track A) and
1.13× [0.78, 1.82] (track C) with no attainment change, COVQP_120 3.6× worse
by the same first-update rebuild that hurt PORTFOLIO_100 here
(`../s6v4-final4/README.md` §2). The default is off; the option remains for
separable problems, where this record stands.

Before that: development-validated. This is the whole corpus (dev plus the former held-out
rounds, all development material). A **round-4
held-out set** was required before any superiority claim could be updated
(`docs/17_CLAIM_AUDIT.md`). The mechanism most affected — MAXENT_200 at
0.157× — is exactly the one that kept round 3's evaluations against
`fmincon-sqp` above 1, so round 4 is where that is tested, not asserted.

Reviewed September 11 (evening): every record here was regenerated from a
wheel rebuilt from the committed source on 18 spot runs (identical), the
rule-off arm equals the round-3 C6 build on the 12 final3 problems, and the
numbers above were recomputed from the raw records. Corrections made in that
review: the n ≥ 100 subset criterion, the PORTFOLIO_100 mechanism and scoring
status, ELLIPSOID2_200 and CHAINROSEN_BOX_200 do rebuild (see above), and the
exit-status changes. Since the review the notes carry the rebuild route split
(`docs/21` §7.1), which changes no numbers.
