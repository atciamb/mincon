# Analysis of bench/results/s6v4-final4/cmp-onoff-C.jsonl

11 problems, solvers: fmincon-interior-point, fmincon-sqp, mincon, mincon-ip, mincon-ip@bfgs_rescale=0, mincon-sqp, mincon-sqp@bfgs_rescale=0, mincon@bfgs_rescale=0, scipy-slsqp; duplicates overwritten: 0

## Target attainment (feasible to 1e-6 and objective within the frozen relative tolerance)

| solver | attained | of | rate | KKT (supplied) | KKT (recovered) | timeouts/crashes/errors |
|---|---:|---:|---:|---:|---:|---:|
| fmincon-interior-point | 10 | 11 | 0.909 | 10 | 3 | 0 |
| fmincon-sqp | 10 | 11 | 0.909 | 3 | 4 | 0 |
| mincon | 11 | 11 | 1.000 | 5 | 4 | 0 |
| mincon-ip | 11 | 11 | 1.000 | 5 | 4 | 0 |
| mincon-ip@bfgs_rescale=0 | 11 | 11 | 1.000 | 7 | 4 | 0 |
| mincon-sqp | 10 | 11 | 0.909 | 5 | 7 | 0 |
| mincon-sqp@bfgs_rescale=0 | 10 | 11 | 0.909 | 7 | 9 | 0 |
| mincon@bfgs_rescale=0 | 11 | 11 | 1.000 | 7 | 4 | 0 |
| scipy-slsqp | 11 | 11 | 1.000 | 0 | 0 | 0 |

## Attainment by family

| family | n | fmincon-interior-point | fmincon-sqp | mincon | mincon-ip | mincon-ip@bfgs_rescale=0 | mincon-sqp | mincon-sqp@bfgs_rescale=0 | mincon@bfgs_rescale=0 | scipy-slsqp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| covqp | 3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| denselap | 2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| obstacle | 3 | 2/3 | 2/3 | 3/3 | 3/3 | 3/3 | 2/3 | 2/3 | 3/3 | 3/3 |
| snl | 3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |

## Paired comparison against `mincon-sqp@bfgs_rescale=0` (metric for cost: total_model)

| solver | both | only solver | only baseline | neither | diff (pp) | 95% CI (family bootstrap) | geo-mean cost ratio solver/baseline (common) | 95% CI | n common |
|---|---:|---:|---:|---:|---:|---|---:|---|---:|
| fmincon-interior-point | 10 | 0 | 0 | 1 | +0.0 | [+0.0, +0.0] | 2.04 | [1.21, 3.43] | 10 |
| fmincon-sqp | 10 | 0 | 0 | 1 | +0.0 | [+0.0, +0.0] | 2.46 | [0.99, 6.10] | 10 |
| mincon | 10 | 1 | 0 | 0 | +9.1 | [+0.0, +25.0] | 1.07 | [0.86, 1.39] | 10 |
| mincon-ip | 10 | 1 | 0 | 0 | +9.1 | [+0.0, +25.0] | 1.07 | [0.86, 1.39] | 10 |
| mincon-ip@bfgs_rescale=0 | 10 | 1 | 0 | 0 | +9.1 | [+0.0, +25.0] | 0.95 | [0.73, 1.11] | 10 |
| mincon-sqp | 10 | 0 | 0 | 1 | +0.0 | [+0.0, +0.0] | 1.07 | [0.77, 1.46] | 10 |
| mincon@bfgs_rescale=0 | 10 | 1 | 0 | 0 | +9.1 | [+0.0, +25.0] | 0.95 | [0.73, 1.11] | 10 |
| scipy-slsqp | 10 | 1 | 0 | 0 | +9.1 | [+0.0, +25.0] | 0.91 | [0.58, 1.63] | 10 |

## Performance profile points (total_model; fraction of problems solved within tau x best)

| tau | fmincon-interior-point | fmincon-sqp | mincon | mincon-ip | mincon-ip@bfgs_rescale=0 | mincon-sqp | mincon-sqp@bfgs_rescale=0 | mincon@bfgs_rescale=0 | scipy-slsqp |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.00 | 0.00 | 0.09 | 0.09 | 0.36 | 0.27 | 0.09 | 0.36 | 0.36 |
| 2 | 0.18 | 0.45 | 0.82 | 0.82 | 0.91 | 0.73 | 0.82 | 0.91 | 0.82 |
| 4 | 0.55 | 0.55 | 0.91 | 0.91 | 1.00 | 0.82 | 0.91 | 1.00 | 1.00 |
| 8 | 0.91 | 0.64 | 1.00 | 1.00 | 1.00 | 0.91 | 0.91 | 1.00 | 1.00 |
| 16 | 0.91 | 0.82 | 1.00 | 1.00 | 1.00 | 0.91 | 0.91 | 1.00 | 1.00 |
| 64 | 0.91 | 0.91 | 1.00 | 1.00 | 1.00 | 0.91 | 0.91 | 1.00 | 1.00 |
| 256 | 0.91 | 0.91 | 1.00 | 1.00 | 1.00 | 0.91 | 0.91 | 1.00 | 1.00 |

## Non-ok records (timeouts, crashes, errors) — all kept as failures above

