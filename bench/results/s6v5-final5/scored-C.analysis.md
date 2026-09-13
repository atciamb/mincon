# Analysis of bench/results/s6v5-final5/scored-C.jsonl

13 problems, solvers: fmincon-interior-point, fmincon-sqp, mincon, mincon-ip, mincon-sqp, scipy-slsqp, scipy-trust-constr; duplicates overwritten: 0

## Target attainment (feasible to 1e-6 and objective within the frozen relative tolerance)

| solver | attained | of | rate | KKT (supplied) | KKT (recovered) | timeouts/crashes/errors |
|---|---:|---:|---:|---:|---:|---:|
| fmincon-interior-point | 13 | 13 | 1.000 | 9 | 4 | 0 |
| fmincon-sqp | 13 | 13 | 1.000 | 9 | 10 | 0 |
| mincon | 13 | 13 | 1.000 | 6 | 12 | 0 |
| mincon-ip | 12 | 13 | 0.923 | 4 | 5 | 0 |
| mincon-sqp | 13 | 13 | 1.000 | 6 | 12 | 0 |
| scipy-slsqp | 13 | 13 | 1.000 | 0 | 3 | 0 |
| scipy-trust-constr | 12 | 13 | 0.923 | 0 | 2 | 0 |

## Attainment by family

| family | n | fmincon-interior-point | fmincon-sqp | mincon | mincon-ip | mincon-sqp | scipy-slsqp | scipy-trust-constr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| deconv | 2 | 2/2 | 2/2 | 2/2 | 1/2 | 2/2 | 2/2 | 2/2 |
| engineering3 | 2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| logistic | 2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| ncboxqp | 2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| noisyqp | 1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| pkfit | 2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| tcport | 2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 1/2 |

## Paired comparison against `mincon` (metric for cost: total_model)

| solver | both | only solver | only baseline | neither | diff (pp) | 95% CI (family bootstrap) | geo-mean cost ratio solver/baseline (common) | 95% CI | n common |
|---|---:|---:|---:|---:|---:|---|---:|---|---:|
| fmincon-interior-point | 13 | 0 | 0 | 0 | +0.0 | [+0.0, +0.0] | 1.77 | [0.85, 4.75] | 13 |
| fmincon-sqp | 13 | 0 | 0 | 0 | +0.0 | [+0.0, +0.0] | 1.28 | [0.67, 3.44] | 13 |
| mincon-ip | 12 | 0 | 1 | 0 | -7.7 | [-23.1, +0.0] | 1.11 | [0.75, 2.01] | 12 |
| mincon-sqp | 13 | 0 | 0 | 0 | +0.0 | [+0.0, +0.0] | 1.30 | [0.75, 3.04] | 13 |
| scipy-slsqp | 13 | 0 | 0 | 0 | +0.0 | [+0.0, +0.0] | 0.65 | [0.41, 1.28] | 13 |
| scipy-trust-constr | 12 | 0 | 1 | 0 | -7.7 | [-23.1, +0.0] | 5.18 | [1.39, 16.83] | 12 |

## Performance profile points (total_model; fraction of problems solved within tau x best)

| tau | fmincon-interior-point | fmincon-sqp | mincon | mincon-ip | mincon-sqp | scipy-slsqp | scipy-trust-constr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.00 | 0.08 | 0.15 | 0.00 | 0.00 | 0.85 | 0.00 |
| 2 | 0.38 | 0.54 | 0.38 | 0.46 | 0.38 | 0.85 | 0.15 |
| 4 | 0.85 | 0.85 | 1.00 | 0.85 | 0.85 | 0.92 | 0.31 |
| 8 | 0.85 | 0.85 | 1.00 | 0.85 | 0.85 | 1.00 | 0.54 |
| 16 | 0.85 | 0.92 | 1.00 | 0.92 | 1.00 | 1.00 | 0.62 |
| 64 | 1.00 | 1.00 | 1.00 | 0.92 | 1.00 | 1.00 | 0.77 |
| 256 | 1.00 | 1.00 | 1.00 | 0.92 | 1.00 | 1.00 | 0.92 |

## Non-ok records (timeouts, crashes, errors) — all kept as failures above

