# Direct MATLAB / mincon parity on tiny analytic problems

Five problems with closed-form solutions exercise each constraint kind:
linear inequality (P1), nonlinear inequality (P2), nonlinear equality (P3),
bounds (P4) and a ranged linear row split into two one-sided rows (P5).
`parity_matlab.m` runs `fmincon` with `interior-point` and `sqp` at defaults
(`Display='off'` only); `parity_mincon.py` runs `mincon.fmincon` (default
`auto` portfolio, `threads=1`) and `mincon.minimize(method='interior-point')`.
`parity_compare.py` verifies every returned point against the closed-form
solution and an independently coded feasibility function; it never uses a
solver's own success flag.

Result, September 7, 2026 (MATLAB R2025b, Optimization Toolbox 25.2, mincon
0.1.0 PyPI wheel, same laptop): **20/20 runs reach the closed-form solution**
(|x-x*| < 1e-4, |f-f*| < 1e-5, independent violation <= 1e-6).

Two lessons that changed the verifier *after* the first run, recorded here
because the protocol forbids silent re-scoring:

1. P3 (`min x1^2+x2^2 s.t. x1*x2=1`) has two global minimizers, ±(1,1).
   fmincon-SQP returned (-1,-1) with exit flag 2; the first verifier version
   knew only (1,1) and marked it FAIL. Reference solutions must be sets.
2. fmincon-interior-point at its default `OptimalityTolerance=1e-6` returned
   |f-f*| = 2.0e-6 on P5. The first verifier demanded 1e-6 on f, tighter than
   the competitor's declared accuracy. Target tolerances must be declared
   relative to the accuracy tracks, not chosen after seeing one solver's
   output. The threshold was loosened to 1e-5 for this smoke suite only.

Wall times here include MATLAB's first-call warm-up (0.37 s on P1) and are
not a benchmark. Evaluation counts: mincon 24–42 vs fmincon interior-point
15–35 and sqp 9–18 on these 2-variable problems.

Reproduce: `matlab -batch "run('bench/parity/parity_matlab.m')"`, then
`python bench/parity/parity_mincon.py`, then `python bench/parity/parity_compare.py`.
