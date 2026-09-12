# Analysis of bench/results/s6v4-final4/cmp-onoff-A.jsonl

11 problems, solvers: fmincon-interior-point, fmincon-sqp, mincon, mincon-ip, mincon-ip@bfgs_rescale=0, mincon-sqp, mincon-sqp@bfgs_rescale=0, mincon@bfgs_rescale=0, scipy-slsqp; duplicates overwritten: 0

## Target attainment (feasible to 1e-6 and objective within the frozen relative tolerance)

| solver | attained | of | rate | KKT (supplied) | KKT (recovered) | timeouts/crashes/errors |
|---|---:|---:|---:|---:|---:|---:|
| fmincon-interior-point | 5 | 11 | 0.455 | 1 | 0 | 0 |
| fmincon-sqp | 8 | 11 | 0.727 | 2 | 2 | 0 |
| mincon | 9 | 11 | 0.818 | 0 | 0 | 0 |
| mincon-ip | 9 | 11 | 0.818 | 0 | 0 | 0 |
| mincon-ip@bfgs_rescale=0 | 9 | 11 | 0.818 | 1 | 0 | 0 |
| mincon-sqp | 9 | 11 | 0.818 | 1 | 2 | 0 |
| mincon-sqp@bfgs_rescale=0 | 9 | 11 | 0.818 | 2 | 3 | 0 |
| mincon@bfgs_rescale=0 | 9 | 11 | 0.818 | 1 | 0 | 0 |
| scipy-slsqp | 11 | 11 | 1.000 | 0 | 0 | 0 |

## Attainment by family

| family | n | fmincon-interior-point | fmincon-sqp | mincon | mincon-ip | mincon-ip@bfgs_rescale=0 | mincon-sqp | mincon-sqp@bfgs_rescale=0 | mincon@bfgs_rescale=0 | scipy-slsqp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| covqp | 3 | 1/3 | 3/3 | 2/3 | 2/3 | 2/3 | 2/3 | 2/3 | 2/3 | 3/3 |
| denselap | 2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| obstacle | 3 | 0/3 | 1/3 | 2/3 | 2/3 | 2/3 | 2/3 | 2/3 | 2/3 | 3/3 |
| snl | 3 | 2/3 | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |

## Paired comparison against `mincon@bfgs_rescale=0` (metric for cost: total_model)

| solver | both | only solver | only baseline | neither | diff (pp) | 95% CI (family bootstrap) | geo-mean cost ratio solver/baseline (common) | 95% CI | n common |
|---|---:|---:|---:|---:|---:|---|---:|---|---:|
| fmincon-interior-point | 5 | 0 | 4 | 2 | -36.4 | [-58.3, -11.1] | 1.16 | [0.60, 1.94] | 5 |
| fmincon-sqp | 7 | 1 | 2 | 1 | -9.1 | [-33.3, +20.0] | 1.23 | [0.82, 1.71] | 7 |
| mincon | 9 | 0 | 0 | 2 | +0.0 | [+0.0, +0.0] | 1.24 | [0.99, 1.67] | 9 |
| mincon-ip | 9 | 0 | 0 | 2 | +0.0 | [+0.0, +0.0] | 1.21 | [0.96, 1.62] | 9 |
| mincon-ip@bfgs_rescale=0 | 9 | 0 | 0 | 2 | +0.0 | [+0.0, +0.0] | 1.00 | [1.00, 1.01] | 9 |
| mincon-sqp | 9 | 0 | 0 | 2 | +0.0 | [+0.0, +0.0] | 0.91 | [0.69, 1.30] | 9 |
| mincon-sqp@bfgs_rescale=0 | 9 | 0 | 0 | 2 | +0.0 | [+0.0, +0.0] | 0.79 | [0.66, 0.96] | 9 |
| scipy-slsqp | 9 | 2 | 0 | 0 | +18.2 | [+0.0, +33.3] | 0.80 | [0.43, 1.46] | 9 |

## Performance profile points (total_model; fraction of problems solved within tau x best)

| tau | fmincon-interior-point | fmincon-sqp | mincon | mincon-ip | mincon-ip@bfgs_rescale=0 | mincon-sqp | mincon-sqp@bfgs_rescale=0 | mincon@bfgs_rescale=0 | scipy-slsqp |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.00 | 0.00 | 0.09 | 0.09 | 0.09 | 0.18 | 0.18 | 0.09 | 0.73 |
| 2 | 0.18 | 0.45 | 0.45 | 0.45 | 0.55 | 0.55 | 0.64 | 0.55 | 0.73 |
| 4 | 0.45 | 0.64 | 0.64 | 0.73 | 0.82 | 0.82 | 0.82 | 0.82 | 1.00 |
| 8 | 0.45 | 0.73 | 0.82 | 0.82 | 0.82 | 0.82 | 0.82 | 0.82 | 1.00 |
| 16 | 0.45 | 0.73 | 0.82 | 0.82 | 0.82 | 0.82 | 0.82 | 0.82 | 1.00 |
| 64 | 0.45 | 0.73 | 0.82 | 0.82 | 0.82 | 0.82 | 0.82 | 0.82 | 1.00 |
| 256 | 0.45 | 0.73 | 0.82 | 0.82 | 0.82 | 0.82 | 0.82 | 0.82 | 1.00 |

## Non-ok records (timeouts, crashes, errors) — all kept as failures above

