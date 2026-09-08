# Literature and evidence matrix

Sources consulted for design decisions in this cycle, the claim each was used
for, and what the experiments here established. "Supported" means measured in
this repository; "assumed" means taken from the source and not re-measured.

| source | claim used | status here | implication |
|---|---|---|---|
| Wächter & Biegler (2006), IPOPT paper | filter line search, inertia correction, monotone mu schedule, bound push, E_mu scaling | implemented earlier; tolerance semantics re-measured (`s3-tolerance`) | IPOPT's 1e-8 default assumes exact derivatives; with forward FD the achievable stationarity is ~1e-7 |
| Nocedal, Wächter, Waltz (2009), adaptive barrier update strategies; Vanderbei & Shanno (LOQO) | LOQO centrality rule `sigma = 0.1 min(0.05(1-xi)/xi, 2)^3` | supported: 0.93× evaluations, +1 attainment on dev; needs a bounded fallback (25/82 runs) | keep as default; earlier fallback criterion is an open experiment |
| MathWorks fmincon documentation (R2025b) | defaults: OptimalityTolerance 1e-6, ConstraintTolerance 1e-6, MaxFunctionEvaluations 3000 (interior-point) / 100 n (sqp); `CheckGradients` stops the solve; SQP uses damped BFGS; interior-point default uses BFGS | verified by behaviour: the 3000 cap fails n ≥ 50 FD problems (QUADSPHERE, CHAINROSEN); lambda is empty when an OutputFcn stops the run | track A keeps these defaults; the harness records them |
| Nocedal & Wright (2006) §6.1 eq. 6.20 | scale `B0` from the first curvature pair | **rejected** by ablation on 127 problems (`s4-bfgs-scaling-rejected`) | not uniformly better; keep as a guarded option idea |
| Powell (1978) damped BFGS | positive-definite updates on constrained problems | implemented earlier; assumed | — |
| Dolan & Moré (2002); Moré & Wild (2009) | performance profiles with failures kept; budget-dependent data profiles | implemented in `analyze.py` (profile points), data profiles not yet | — |
| Beiranvand, Hare, Lucet (2017) benchmarking best practice | preregistration, independent scoring, family-level uncertainty, exclusion ledger | followed: `docs/15_BENCHMARK_PROTOCOL_V2.md` | the contract failed honestly on the held-out set |
| Hock & Schittkowski (1981) | problem definitions and published optima | 89 transcribed; 44 cross-checked against an independent transcription; HS20/HS44/HS16/HS55 have alternative KKT points reached by local methods | targets for these are "published", not "the basin you will reach" |
| Adaptive finite-difference interval estimation (Shi, Xie, Xuan, Nocedal 2021) | error-aware step selection | not implemented; HS62 shows the need (13 evaluations exact vs 250–500 FD) | next increment candidate |
| Chiang & Zavala inertia-free regularization | inertia-free acceptance | not implemented; option remains a stub | unchanged |
| S2MPJ / CUTEst | large cross-language corpus | not acquired (declined earlier); replaced by the single-source SymPy corpus | held-out set is small (24); needs more families |
