# Claim audit

Every public statement about mincon versus fmincon, mapped to the experiment
that supports it, its scope and its uncertainty. Statements not in this table
are not claimed.

| # | statement | experiment | scope | uncertainty / caveat |
|---|---|---|---|---|
| 1 | At defaults mincon is at least as robust as fmincon-interior-point on this corpus | `s4-c2-dev` (77 vs 73 / 81), `s4-c2-val` (23 vs 21 / 24), `s6-final` (19 vs 18 / 22) | 127 scorable problems, n ≤ 1000, mostly small dense; track A | differences +4.5 to +8.3 pp with family-bootstrap intervals touching zero; not a superiority claim |
| 2 | With exact derivatives the same holds | `s3-c1-dev` C (77 vs 76), `s4-c2-val` C (23 vs 23), `s6-final` C (20 vs 19) | track C | same |
| 3 | Model evaluations: about the same on development material, 1.4× on the held-out split | `s4-c2-dev` 1.06 [0.85, 1.09]; `s4-c2-val` 0.91 [0.75, 0.96]; `s6-final` 1.43 [1.04, 1.59] | commonly attained problems only; counts at the model boundary | the held-out figure is the honest one; it fails the preregistered ≤ 1.0 target |
| 4 | mincon returns 30–100× faster in wall-clock time on problems up to a few hundred variables | `s6-timing` (3 repeats: 0.033 [0.019, 0.038]); `s4-c2-dev` single runs (0.008) | same laptop, single thread, Python callbacks vs MATLAB callbacks, solve time only (no import/startup); n ≤ 600 | includes each language's callback cost (mincon's callbacks are 29% of its wall time); MATLAB per-solve overhead dominates at small n; at n = 200 with hundreds of BFGS iterations mincon can be slower (CHAINROSEN_200: 10.7 s vs 1.1 s, where fmincon failed) |
| 5 | fmincon-sqp and SciPy SLSQP need fewer evaluations than any interior-point code here | all splits, track A/C: 0.6–0.8× of fmincon-ip | small dense problems | mincon has no SQP; this is where fmincon wins |
| 6 | No false success on the scorable corpus; one on the diagnostic track, fixed | oracle columns of every scored file; `s6-diagnostic` | UNBOUNDED_PAR reported `Optimal` at ||x|| ≈ 1e19 by C2; the RC reports `Unbounded` | the fix is tested but the held-out run predates it |
| 7 | The 1e-8 → 1e-6 tolerance change explains most of the B0 evaluation gap | `s3-tolerance` ablation (0.62× at equal attainment) | development split | a default change, not an algorithmic gain |
| 8 | The adaptive barrier saves ~7% evaluations and one attainment | `s4-barrier` (0.93 [0.85, 0.98]) | development split | regressions on HS100/HS38/HS63 |
| 9 | mincon's Python API supports exact constraint Jacobians and checks them | Python tests; track C runs | — | dense Jacobian only; no `jac_sparsity` yet |
| 10 | Statements **not** made: universal dominance; large-scale (n > 1000) performance; multi-core speed-ups; any result for CUTEst/S2MPJ (never acquired) | — | — | — |
