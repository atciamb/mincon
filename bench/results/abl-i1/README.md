# abl-i1: the certificate guard (D11), whole-corpus ablation, September 12, 2026

Increment I1 of `docs/22`: the interior-point member's certificate now also requires
complementarity in the user's units (`compl / d_f <= tol_compl * max(1, |f|)`), so a scaled
complementarity product that is small only because the objective factor is tiny no longer
passes; when the guard blocks and the gradient norm has drifted by more than 10x from where
the factor was chosen, the D9 rule rescales the objective and continues. Mechanism and the
record that found it: `bench/results/s7-friction` (bad_scaling), atlas cluster 19.

Run: track A, all 169 non-diagnostic corpus problems (the round-4 set final4 included as
development material), defaults, single thread, 60 s / 100 000 evaluations, targets v5, wheel
`wheels-i1` of the I1 tree. Baseline: the current default's records, which are the rule-off arms
`mincon*_bfgs_rescale-0.A.jsonl` of `abl-c7` (158 problems) and `s6v4-final4` (11), copied into
`cmp-*.jsonl` here under their original solver names.

| arm | attained candidate / baseline (of 166 scorable) | cost ratio, geo-mean on common [95 % family bootstrap] | records that changed |
|---|---|---|---|
| portfolio (`mincon`) | 159 / 158 | 1.00 [1.00, 1.00] on 158 | COVQP_300 and DENSELAP_250 only: both 60 s budget exits either way (COVQP_300 landed inside the target this time, a machine-speed effect, not the guard) |
| IP member (`mincon-ip`) | 156 / 156 | 1.00 [1.00, 1.01] on 156 | the two 60 s exits; HS16 +16 evaluations (unattained either way, cluster 6); HS17 +18 (attained either way); **UNITS**: `Optimal` at f = 5.009 against the target 0.5 (oracle: not KKT, relative stationarity 1.9e-3) became `Acceptable` at f = 5.002 (264 evaluations) — a second false certificate of the D11 kind that was already in the corpus |
| SQP member (`mincon-sqp`) | 154 / 154 | 1.00 [1.00, 1.00] on 154 | the five 60 s budget exits only (COVQP_300, ELLIPSOID_500, OBSTACLE_500, QUADSPHERE2_300, QUADSPHERE_1000), a control: the guard is interior-point code |

The run was not interrupted.

Verdict: H1 of `docs/22` stands. No attained record was lost, the cost ratio is 1.00 with an
interval of width 0.01, one corpus false certificate was removed, and the friction problem
that motivated the guard exits `Acceptable` with `success = False` (`s7-friction/mincon-fmincon-i1.jsonl`).
The problem is still not solved by any solver in the audit; that is H8 (variable scaling).

Files: `mincon*.A.jsonl` candidate records, `cmp-*.jsonl` candidate + baseline,
`scored-*.jsonl` oracle verdicts, `*.analysis.*` paired analyses, `run.log`.
