# Analysis of ../results/s6v3-final3/scored-A.jsonl

12 problems, solvers: fmincon-interior-point, fmincon-sqp, mincon, mincon-ip, mincon-sqp, scipy-slsqp; duplicates overwritten: 0

## Target attainment (feasible to 1e-6 and objective within the frozen relative tolerance)

| solver | attained | of | rate | KKT (supplied) | KKT (recovered) | timeouts/crashes/errors |
|---|---:|---:|---:|---:|---:|---:|
| fmincon-interior-point | 10 | 12 | 0.833 | 4 | 6 | 0 |
| fmincon-sqp | 12 | 12 | 1.000 | 4 | 5 | 0 |
| mincon | 12 | 12 | 1.000 | 4 | 8 | 0 |
| mincon-ip | 12 | 12 | 1.000 | 4 | 7 | 0 |
| mincon-sqp | 12 | 12 | 1.000 | 3 | 10 | 0 |
| scipy-slsqp | 12 | 12 | 1.000 | 0 | 2 | 0 |

## Attainment by family

| family | n | fmincon-interior-point | fmincon-sqp | mincon | mincon-ip | mincon-sqp | scipy-slsqp |
|---|---:|---:|---:|---:|---:|---:|---:|
| hs | 1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| logsumexp | 2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| maxent | 3 | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| nnls_simplex | 3 | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| rosen_sphere | 2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| sinfit | 1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |

## Paired comparison against `fmincon-sqp` (metric for cost: total_model)

| solver | both | only solver | only baseline | neither | diff (pp) | 95% CI (family bootstrap) | geo-mean cost ratio solver/baseline (common) | 95% CI | n common |
|---|---:|---:|---:|---:|---:|---|---:|---|---:|
| fmincon-interior-point | 10 | 0 | 2 | 0 | -16.7 | [-28.6, +0.0] | 1.31 | [0.98, 2.12] | 10 |
| mincon | 12 | 0 | 0 | 0 | +0.0 | [+0.0, +0.0] | 1.13 | [0.97, 1.38] | 12 |
| mincon-ip | 12 | 0 | 0 | 0 | +0.0 | [+0.0, +0.0] | 1.43 | [1.17, 1.77] | 12 |
| mincon-sqp | 12 | 0 | 0 | 0 | +0.0 | [+0.0, +0.0] | 1.09 | [0.95, 1.29] | 12 |
| scipy-slsqp | 12 | 0 | 0 | 0 | +0.0 | [+0.0, +0.0] | 0.84 | [0.66, 1.07] | 12 |

## Performance profile points (total_model; fraction of problems solved within tau x best)

| tau | fmincon-interior-point | fmincon-sqp | mincon | mincon-ip | mincon-sqp | scipy-slsqp |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.08 | 0.17 | 0.17 | 0.08 | 0.08 | 0.58 |
| 2 | 0.67 | 1.00 | 0.92 | 0.75 | 0.92 | 1.00 |
| 4 | 0.75 | 1.00 | 0.92 | 0.92 | 1.00 | 1.00 |
| 8 | 0.83 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 16 | 0.83 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 64 | 0.83 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 256 | 0.83 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

## Non-ok records (timeouts, crashes, errors) — all kept as failures above

