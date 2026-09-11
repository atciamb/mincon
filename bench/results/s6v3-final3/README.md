# S6 round 3: held-out qualification of candidate C6 on `final3`

Frozen before the run: candidate C6 (commit `8ad564e`; Windows wheel SHA-256
`c75910af…1cc90`, built from that commit on the benchmark host — `cargo test`
clean, fixture gate 55/55 for `auto` and for the interior-point member, 54/55
for the SQP member alone (HS13, see `abl-sqp1`), Clippy and fmt clean); the
`final3` set (12 problems generated after C6 was frozen and never run by any
mincon build before this: NNLS_SIMPLEX_6/30/120, MAXENT_10/50/200,
LOGSUMEXP_10/20, ROSEN_SPHERE_4/10, SINFIT_CLEAN, HS114); targets `v4`
(exact or published references, two best-known from a 40-start reference at
generation; MATLAB/NumPy equivalence 3220 quantities, 0 mismatches). Same
laptop as every earlier round, single thread, MATLAB R2025b, Optimization
Toolbox 25.2. Six solvers: the portfolio (`mincon`), its two members alone
(`mincon-ip`, `mincon-sqp`), `fmincon` interior-point and sqp at defaults,
SciPy SLSQP.

## Track A (defaults, finite differences) — `analysis-A.txt`

| solver | attained / 12 | vs fmincon-ip: Δ (pp) [95 % CI] | evaluations vs fmincon-ip, common [95 % CI] | evaluations vs fmincon-sqp, common [95 % CI] |
|---|---:|---|---|---|
| fmincon-interior-point | 10 | — | 1.00 | 1.31 [0.98, 2.12] |
| fmincon-sqp | 12 | +16.7 [0, +28.6] | 0.77 [0.47, 1.02] | 1.00 |
| scipy-slsqp | 12 | +16.7 [0, +28.6] | 0.59 [0.37, 0.91] | 0.84 [0.66, 1.07] |
| mincon-ip (member alone) | 12 | +16.7 [0, +28.6] | 1.08 [0.81, 1.49] | 1.43 [1.17, 1.77] |
| mincon-sqp (member alone) | 12 | +16.7 [0, +28.6] | 0.77 [0.50, 0.98] | 1.09 [0.95, 1.29] |
| **mincon (portfolio)** | **12** | **+16.7 [0, +28.6]** | **0.81 [0.53, 1.05]** | **1.13 [0.97, 1.38]** |

fmincon-ip's two misses are its 3000-evaluation cap on MAXENT_200 and
NNLS_SIMPLEX_120 (n = 200 and 120 with finite differences). fmincon-sqp
attains all twelve but two of them by stopping at its own `100·n` cap inside
the 1e-4 target tolerance (MAXENT_200 at 20 017 evaluations, NNLS_SIMPLEX_120
at 12 032, both exit flag 0) — the cap that costs it problems elsewhere saves
it evaluations here. mincon spends 80 059 evaluations on MAXENT_200 to reach
`Optimal` (a separable objective on 200 variables with a dense BFGS model,
≈ 400 iterations); that single problem is most of the 1.13 against
fmincon-sqp. On the ten problems with n ≤ 50 the portfolio's evaluations are
0.73× fmincon-ip's and 0.99× fmincon-sqp's (geo-means).

## Track C (exact derivatives) — `analysis-C.txt`

Every solver attains 12/12. Evaluations vs fmincon-ip: mincon 0.88
[0.48, 1.71], mincon-ip 0.81 [0.55, 1.22], mincon-sqp 0.82 [0.50, 1.43],
fmincon-sqp 1.22 [0.83, 1.86], SLSQP 0.58 [0.43, 0.79]. The portfolio's
wide interval is MAXENT_200 again: the interior-point member stopped at the
step tolerance (849 gradients, target attained), so the SQP member ran as well
under the new early-exit rule and the two together cost 2259 against
fmincon-ip's 352.

## Wall time (`../s6v3-timing`, 3 repeats, same host) — `timing-vs-ip.md`, `timing-vs-sqp.md`

| | vs fmincon-interior-point | vs fmincon-sqp |
|---|---|---|
| problems attained by both in every repeat | 10 (MAXENT_200, NNLS_SIMPLEX_120 excluded: fmincon-ip 0/3) | 12 |
| geo-mean wall ratio [95 % CI] | **0.077 [0.034, 0.153]** | **0.35 [0.14, 0.63]** |
| median wall | 5.6 ms vs 143 ms | 7.9 ms vs 53 ms |
| slower than the baseline on | none | MAXENT_200 (7.3×), NNLS_SIMPLEX_120 (3.8×), NNLS_SIMPLEX_30 (2.1×), MAXENT_50 (1.4×) |

mincon's wall time is 63–82 % model callbacks (the models are NumPy
functions compiled from SymPy; fmincon's are the same expressions in MATLAB),
so on the two large problems the evaluation count is the wall time.

## Reading

Against fmincon at defaults on twelve unseen problems: mincon is at least as
reliable as either fmincon algorithm (12/12 against 10/12 and 12/12), uses
0.81× the evaluations of fmincon's default algorithm and 1.13× those of
fmincon-sqp (the interval [0.97, 1.38] includes 1; driven by one n = 200
problem where fmincon-sqp's cap stopped it inside the tolerance), and is an
order of magnitude faster in wall time against the default and 3× faster
against sqp, except on n ≥ 120 where its dense BFGS iteration count makes it
slower. The contract's evaluation interval (`docs/15` §6: ratio < 1 with the
interval excluding 1 against both algorithms) is met against neither at
n = 12; the attainment interval touches 0 against fmincon-ip and is exactly 0
against fmincon-sqp.

Per protocol this set is now development material; a fourth held-out round is
required for any further claim.
