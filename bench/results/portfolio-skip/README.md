# portfolio-skip: what the portfolio's shared budget actually costs, September 13, 2026

**A measurement, not an increment. Nothing was changed as a result of it.** It exists to decide
whether `docs/17` item 20a — mincon's one round-5 track-A miss being its own portfolio losing to
its own member — is worth a rule, and the answer it gives is *narrower than the item implies*.
Read this before writing anything against that item.

## 1. How often the portfolio skips a member at all

Scanned from the whole-corpus run at current defaults
(`../abl-i5-rows/mincon_quadratic_rows-values.A.jsonl`, 185 problems, track A, 60 s / 100 000
evaluations): **exactly five records have a later member skipped for budget**, and every one is
n >= 200. Always the same three are skipped (`sqp`, `ip-cautious`, `ip-unscaled`), always because
the first member consumed the shared budget. So the defect is **rare and concentrated**, not
systemic.

There are two different exhaustion causes, and the distinction turns out to be the whole story:

| problem | n | objective evaluations | % of the 100 000 offered | wall | what bound |
|---|---:|---:|---:|---:|---|
| DECONV_200 | 200 | 13 870 | **13.9 %** | 60.5 s | **the clock** |
| DENSELAP_250 | 250 | 8 311 | **8.3 %** | 60.9 s | **the clock** |
| COVQP_300 | 300 | 5 719 | **5.7 %** | 61.7 s | **the clock** |
| CHAINROSEN_BOX_200 | 200 | 100 154 | 100.2 % | 12.9 s | evaluations |
| ELLIPSOID_500 | 500 | 100 091 | 100.1 % | 32.2 s | evaluations |

On the three clock-bound records the solve stops after spending **6 % to 14 %** of the evaluation
budget it was offered. That asymmetry — the wall clock binding long before the evaluation budget,
at n >= 200 under finite differences, and binding *inside the first member* — is the real
mechanism behind item 20a.

## 2. Would a skipped member have attained?

The decisive question, run at HEAD on exactly those five: the portfolio against `mincon-ip` and
`mincon-sqp` standalone, both tracks, the same 60 s / 100 000 budget.

| problem | track A: portfolio | mincon-ip | mincon-sqp | reading |
|---|---|---|---|---|
| DECONV_200 | miss | miss | **attained**, 13 668 ev | **the one real loss** |
| DENSELAP_250 | attained | attained | attained (acceptable) | the skip costs nothing |
| COVQP_300 | miss | miss | miss | nobody attains; not an ordering problem |
| CHAINROSEN_BOX_200 | miss | miss | miss | nobody attains |
| ELLIPSOID_500 | **attained** | miss | miss | **the portfolio beats both members** |

On track C the portfolio attains all five, and no member beats it anywhere.

**So across the entire 185-problem corpus, the portfolio's shared budget costs exactly one
attainment, on one problem, on one track.** That is DECONV_200, which is the round-5 sealed
problem item 20a was written from — it reproduces at HEAD, `ip-default` answering after 60.5 s
and 13.9 % of the evaluation budget while `sqp`, skipped, attains the problem standalone in
13 668 evaluations.

**And the counter-example matters as much as the loss.** On ELLIPSOID_500 the skip is *correct*:
`ip-default` runs out, `sqp-quadratic` (the quadratic probe's member) produces the answer, the
remaining three are skipped on the evaluation budget, and the portfolio attains where both members
standalone miss. Any rule that hands later members a guaranteed share must not take that away.

## 3. What this says about the candidates, before any of them is written

* A **per-member share of the time budget** is the candidate the numbers point at, and the two
  budgets being separate is what makes it plausible: it would touch only the three clock-bound
  records and leave ELLIPSOID_500 and CHAINROSEN_BOX_200, which are *evaluation*-bound, alone.
  On DENSELAP_250 and COVQP_300 it has nothing to gain and something to lose, so the whole case
  rests on one problem.
* **Ordering members by the probe's findings** is not supported by anything measured here: the
  probe already changes the order on ELLIPSOID_500 and that is the record the current behaviour
  gets right.
* **An early hand-over when a member stalls** needs the stall diagnosed first (`docs/17` item 20b:
  DECONV_200 returns recovered stationarity 1.514e-2 and `mincon-ip` reaches the same value on
  track C with exact derivatives, stopping voluntarily after 1.9 s of its 60 s). That is the more
  interesting thread, and it is a numerical question, not a scheduling one.

The honest summary for whoever picks this up: **one attainment on the whole corpus is a thin
mandate for a scheduling rule.** If a candidate is tried, the falsifier is any attainment loss
anywhere and any change to the three diagnostics, and ELLIPSOID_500 is the specific record to
watch. It is entirely defensible to conclude that the stall (item 20b) is the better target and
that the scheduler should be left alone — that conclusion is supported by this table and should be
reported, not treated as a failure to deliver.

Files: `mincon.{A,C}.jsonl`, `mincon-ip.{A,C}.jsonl`, `mincon-sqp.{A,C}.jsonl`,
`scored-{A,C}.jsonl`, `experiment.json`. Launcher
`.local-research/portfolio_skip_launcher.cmd`.
