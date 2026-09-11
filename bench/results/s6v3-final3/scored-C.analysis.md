# Analysis of ../results/s6v3-final3/scored-C.jsonl

12 problems, solvers: fmincon-interior-point, fmincon-sqp, mincon, mincon-ip, mincon-sqp, scipy-slsqp; duplicates overwritten: 0

## Target attainment (feasible to 1e-6 and objective within the frozen relative tolerance)

| solver | attained | of | rate | KKT (supplied) | KKT (recovered) | timeouts/crashes/errors |
|---|---:|---:|---:|---:|---:|---:|
| fmincon-interior-point | 12 | 12 | 1.000 | 7 | 8 | 0 |
| fmincon-sqp | 12 | 12 | 1.000 | 5 | 5 | 0 |
| mincon | 12 | 12 | 1.000 | 7 | 9 | 0 |
| mincon-ip | 12 | 12 | 1.000 | 8 | 7 | 0 |
| mincon-sqp | 12 | 12 | 1.000 | 5 | 12 | 0 |
| scipy-slsqp | 12 | 12 | 1.000 | 0 | 2 | 0 |

## Attainment by family

| family | n | fmincon-interior-point | fmincon-sqp | mincon | mincon-ip | mincon-sqp | scipy-slsqp |
|---|---:|---:|---:|---:|---:|---:|---:|
| hs | 1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| logsumexp | 2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| maxent | 3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| nnls_simplex | 3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| rosen_sphere | 2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| sinfit | 1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |

## Paired comparison against `fmincon-interior-point` (metric for cost: total_model)

| solver | both | only solver | only baseline | neither | diff (pp) | 95% CI (family bootstrap) | geo-mean cost ratio solver/baseline (common) | 95% CI | n common |
|---|---:|---:|---:|---:|---:|---|---:|---|---:|
| fmincon-sqp | 12 | 0 | 0 | 0 | +0.0 | [+0.0, +0.0] | 1.22 | [0.83, 1.86] | 12 |
| mincon | 12 | 0 | 0 | 0 | +0.0 | [+0.0, +0.0] | 0.88 | [0.48, 1.71] | 12 |
| mincon-ip | 12 | 0 | 0 | 0 | +0.0 | [+0.0, +0.0] | 0.81 | [0.55, 1.22] | 12 |
| mincon-sqp | 12 | 0 | 0 | 0 | +0.0 | [+0.0, +0.0] | 0.82 | [0.50, 1.43] | 12 |
| scipy-slsqp | 12 | 0 | 0 | 0 | +0.0 | [+0.0, +0.0] | 0.58 | [0.43, 0.79] | 12 |

## Performance profile points (total_model; fraction of problems solved within tau x best)

| tau | fmincon-interior-point | fmincon-sqp | mincon | mincon-ip | mincon-sqp | scipy-slsqp |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.08 | 0.08 | 0.08 | 0.00 | 0.17 | 0.75 |
| 2 | 0.75 | 0.42 | 0.75 | 0.75 | 0.75 | 0.92 |
| 4 | 0.83 | 0.83 | 0.92 | 1.00 | 0.92 | 1.00 |
| 8 | 1.00 | 0.92 | 1.00 | 1.00 | 1.00 | 1.00 |
| 16 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 64 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 256 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

## Non-ok records (timeouts, crashes, errors) — all kept as failures above

