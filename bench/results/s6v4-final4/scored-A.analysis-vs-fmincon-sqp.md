# Analysis of bench/results/s6v4-final4/scored-A.jsonl

11 problems, solvers: fmincon-interior-point, fmincon-sqp, mincon, mincon-ip, mincon-sqp, scipy-slsqp; duplicates overwritten: 0

## Target attainment (feasible to 1e-6 and objective within the frozen relative tolerance)

| solver | attained | of | rate | KKT (supplied) | KKT (recovered) | timeouts/crashes/errors |
|---|---:|---:|---:|---:|---:|---:|
| fmincon-interior-point | 5 | 11 | 0.455 | 1 | 0 | 0 |
| fmincon-sqp | 8 | 11 | 0.727 | 2 | 2 | 0 |
| mincon | 9 | 11 | 0.818 | 0 | 0 | 0 |
| mincon-ip | 9 | 11 | 0.818 | 0 | 0 | 0 |
| mincon-sqp | 9 | 11 | 0.818 | 1 | 2 | 0 |
| scipy-slsqp | 11 | 11 | 1.000 | 0 | 0 | 0 |

## Attainment by family

| family | n | fmincon-interior-point | fmincon-sqp | mincon | mincon-ip | mincon-sqp | scipy-slsqp |
|---|---:|---:|---:|---:|---:|---:|---:|
| covqp | 3 | 1/3 | 3/3 | 2/3 | 2/3 | 2/3 | 3/3 |
| denselap | 2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| obstacle | 3 | 0/3 | 1/3 | 2/3 | 2/3 | 2/3 | 3/3 |
| snl | 3 | 2/3 | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 |

## Paired comparison against `fmincon-sqp` (metric for cost: total_model)

| solver | both | only solver | only baseline | neither | diff (pp) | 95% CI (family bootstrap) | geo-mean cost ratio solver/baseline (common) | 95% CI | n common |
|---|---:|---:|---:|---:|---:|---|---:|---|---:|
| fmincon-interior-point | 5 | 0 | 3 | 3 | -27.3 | [-54.5, +0.0] | 0.80 | [0.43, 2.25] | 5 |
| mincon | 7 | 2 | 1 | 1 | +9.1 | [-20.0, +33.3] | 1.02 | [0.52, 2.13] | 7 |
| mincon-ip | 7 | 2 | 1 | 1 | +9.1 | [-20.0, +33.3] | 0.99 | [0.52, 2.13] | 7 |
| mincon-sqp | 7 | 2 | 1 | 1 | +9.1 | [-20.0, +33.3] | 0.77 | [0.39, 1.45] | 7 |
| scipy-slsqp | 8 | 3 | 0 | 0 | +27.3 | [+0.0, +54.5] | 0.58 | [0.34, 0.77] | 8 |

## Performance profile points (total_model; fraction of problems solved within tau x best)

| tau | fmincon-interior-point | fmincon-sqp | mincon | mincon-ip | mincon-sqp | scipy-slsqp |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.00 | 0.00 | 0.09 | 0.09 | 0.18 | 0.73 |
| 2 | 0.18 | 0.45 | 0.45 | 0.45 | 0.55 | 0.73 |
| 4 | 0.45 | 0.64 | 0.64 | 0.73 | 0.82 | 1.00 |
| 8 | 0.45 | 0.73 | 0.82 | 0.82 | 0.82 | 1.00 |
| 16 | 0.45 | 0.73 | 0.82 | 0.82 | 0.82 | 1.00 |
| 64 | 0.45 | 0.73 | 0.82 | 0.82 | 0.82 | 1.00 |
| 256 | 0.45 | 0.73 | 0.82 | 0.82 | 0.82 | 1.00 |

## Non-ok records (timeouts, crashes, errors) — all kept as failures above

