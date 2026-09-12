# G2: track C (exact derivatives) on the n ≥ 50 subset, candidate C7 (September 11, 2026)

Hypothesis H3 of `docs/21`: the interior-point member's 385-vs-144 iteration gap
between tracks A and C on MAXENT_200 (candidate C6, `r5-large-n`) was a
consequence of the quasi-Newton model error, not of derivative noise in the
pairs; after C7 the two tracks' iteration counts should agree within 1.5×.

Run: the 18 non-diagnostic problems with n ≥ 50, `mincon`, `mincon-ip`,
`mincon-sqp`, track C, defaults, single thread, targets v4; the same wheel as
the C7 records with the rebuild-route counters in the notes (numbers
identical to `abl-c7`). Track A figures below are the `abl-c7` rule-on records.

## Result

| member | A/C iteration ratio, median | max | problems with A/C > 1.5 |
|---|---:|---:|---|
| portfolio and IP member | 1.00 | 1.54 | ELLIPSOID_500 (123 vs 80) |
| SQP member | 1.00 | 1.35 | none |

MAXENT_200: IP member 59 (A) vs 49 (C) iterations, SQP member 46 vs 39 — the
gap that was 385 vs 144 before C7 is 1.2× after it. **H3 supported**: the
finite-difference gap was the model error. The residual A/C differences are
of the size the pair noise explains (ELLIPSOID_500 1.5×, POLYQP_100 1.36×,
ELLIPSOID_50 0.52× the other way).

Two things the track-C records also show:

* CHAINROSEN_BOX_200 is attained on track C by every configuration (915
  iterations, ≈ 1 000 objective evaluations) where track A ends at the
  100 000-evaluation budget after 496 iterations: with exact derivatives the
  problem is only slow, not unsolved. The iteration count stays the
  corpus' largest — the H4 limit (structured or exact Hessian) stands.
* PORTFOLIO_100 under the SQP member takes 175 iterations on track C as on
  track A, ending `Acceptable` both times: its post-rebuild mechanism is not
  derivative noise either (`docs/21` §7.3, S-C).

Attainment on this subset: portfolio and IP member 16 of 17 scorable
(CHAINROSEN_BOX_200 now attained; PORTFOLIO_100 has no target), SQP member
16 of 17 (CATENARY_40 not attained on either track). Files:
`mincon*.C.jsonl` raw, `scored-C.jsonl` with the oracle verdicts.
