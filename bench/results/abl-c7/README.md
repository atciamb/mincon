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
of `docs/21` is met: the paired evaluation ratio is below 1 with the interval
excluding 1 on the portfolio and both members, attainment does not fall, and
the n ≥ 100 subset improves (0.81× portfolio, 0.32× SQP member).

### Where the evaluations go (portfolio)

Wins, largest first: MAXENT_200 **0.157×** (25 166 vs 160 520 model
evaluations; 59 vs 385 iterations — this is the problem that drove round 3's
1.13× against `fmincon-sqp`), ELLIPSOID_500 **attained** where the rule-off run
stalled at the budget (f = 2034.13 = target vs 1995.49 with violation 0.85),
QUADSPHERE_10 0.179×, HS118 0.322×, DISPATCH_20 0.447×, ELLIPSOID_50 0.509×,
MAXENT_10 0.510×, NNLS_SIMPLEX_30 0.607×, MAXENT_50 0.610×, ELLIPSOID2_20
0.645×, CHAINROSEN_EQ_10 0.667×. The one cost regression is ELLIPSOID2_200 at
1.035×, within noise.

### The SQP member alone

Larger swings because it has no interior-point fallback: QUADSPHERE_100
**0.022×** (2 vs 178 iterations), CHAINROSEN_EQ_200 0.109×, MAXENT_200 0.126×,
QUADSPHERE2_300 0.158× (newly attained), QUADSPHERE_1000 0.399× (newly
attained). One real blemish — **PORTFOLIO_100 at 7.0×** (175 vs 23 iterations):
the rule fires on a dense covariance QP where the diagonal is not the right
model, and the repeated rebuilds thrash. It does **not** reach the portfolio,
which routes PORTFOLIO_100 (n = 100 > 20) to the interior-point member first
and attains it at 1.0×. It is filed against the shape test's weight threshold
(`docs/16` cluster 15) and is a candidate for the round-4 diagnosis.

## The gate (why n ≥ 10)

The first version of the rule had no size gate. On the whole corpus it reached
0.947× evaluations but **lost two attainments, HS97 and HS98** (both n = 6):
the scale route fired early, when the matrix was still near-unit and the
quotients all exceeded the factor, and rebuilt to a diagonal that ignores the
coupling a dense BFGS would have learned in six updates
(`bench/results/abl-c7` was regenerated with the gate; the un-gated numbers are
in the private worklog). Gating the rule to n ≥ 10 makes HS97, HS98, HS44 and
HS86 byte-identical rule-on vs rule-off while keeping every n ≥ 10 win
(QUADSPHERE_10 at n = 10 is the smallest problem the rule helps). This is the
same "helps large, perturbs small" boundary that governed the earlier BFGS
scaling studies (`bench/results/s4-bfgs-scaling-rejected`,
`s5-bfgs-guarded-diagonal`).

## Status

Development-validated. This is the whole corpus (dev plus the former held-out
rounds, all development material). The increment is kept in `auto`; a **round-4
held-out set** is required before any superiority claim is updated
(`docs/17_CLAIM_AUDIT.md`). The mechanism most affected — MAXENT_200 at
0.157× — is exactly the one that kept round 3's evaluations against
`fmincon-sqp` above 1, so round 4 is where that is tested, not asserted.
