# Analysis of bench/results/s6v4-final4/scored-C.jsonl

11 problems, solvers: fmincon-interior-point, fmincon-sqp, mincon, mincon-ip, mincon-sqp, scipy-slsqp; duplicates overwritten: 0

## Target attainment (feasible to 1e-6 and objective within the frozen relative tolerance)

| solver | attained | of | rate | KKT (supplied) | KKT (recovered) | timeouts/crashes/errors |
|---|---:|---:|---:|---:|---:|---:|
| fmincon-interior-point | 10 | 11 | 0.909 | 10 | 3 | 0 |
| fmincon-sqp | 10 | 11 | 0.909 | 3 | 4 | 0 |
| mincon | 11 | 11 | 1.000 | 5 | 4 | 0 |
| mincon-ip | 11 | 11 | 1.000 | 5 | 4 | 0 |
| mincon-sqp | 10 | 11 | 0.909 | 5 | 7 | 0 |
| scipy-slsqp | 11 | 11 | 1.000 | 0 | 0 | 0 |

## Attainment by family

| family | n | fmincon-interior-point | fmincon-sqp | mincon | mincon-ip | mincon-sqp | scipy-slsqp |
|---|---:|---:|---:|---:|---:|---:|---:|
| covqp | 3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| denselap | 2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| obstacle | 3 | 2/3 | 2/3 | 3/3 | 3/3 | 2/3 | 3/3 |
| snl | 3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |

## Paired comparison against `scipy-slsqp` (metric for cost: total_model)

| solver | both | only solver | only baseline | neither | diff (pp) | 95% CI (family bootstrap) | geo-mean cost ratio solver/baseline (common) | 95% CI | n common |
|---|---:|---:|---:|---:|---:|---|---:|---|---:|
| fmincon-interior-point | 10 | 0 | 1 | 0 | -9.1 | [-25.0, +0.0] | 2.24 | [1.86, 2.69] | 10 |
| fmincon-sqp | 10 | 0 | 1 | 0 | -9.1 | [-25.0, +0.0] | 2.70 | [1.71, 4.26] | 10 |
| mincon | 11 | 0 | 0 | 0 | +0.0 | [+0.0, +0.0] | 1.16 | [0.53, 2.30] | 11 |
| mincon-ip | 11 | 0 | 0 | 0 | +0.0 | [+0.0, +0.0] | 1.16 | [0.53, 2.30] | 11 |
| mincon-sqp | 10 | 0 | 1 | 0 | -9.1 | [-25.0, +0.0] | 1.18 | [0.60, 2.38] | 10 |

## Performance profile points (total_model; fraction of problems solved within tau x best)

| tau | fmincon-interior-point | fmincon-sqp | mincon | mincon-ip | mincon-sqp | scipy-slsqp |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.00 | 0.00 | 0.27 | 0.27 | 0.27 | 0.45 |
| 2 | 0.27 | 0.45 | 0.82 | 0.82 | 0.73 | 0.82 |
| 4 | 0.55 | 0.55 | 0.91 | 0.91 | 0.82 | 1.00 |
| 8 | 0.91 | 0.64 | 1.00 | 1.00 | 0.91 | 1.00 |
| 16 | 0.91 | 0.82 | 1.00 | 1.00 | 0.91 | 1.00 |
| 64 | 0.91 | 0.91 | 1.00 | 1.00 | 0.91 | 1.00 |
| 256 | 0.91 | 0.91 | 1.00 | 1.00 | 0.91 | 1.00 |

## Non-ok records (timeouts, crashes, errors) — all kept as failures above

