# Analysis of bench\results\s6v2-final2\scored-C.jsonl

16 problems, solvers: fmincon-interior-point, fmincon-sqp, mincon, scipy-slsqp; duplicates overwritten: 0

## Target attainment (feasible to 1e-6 and objective within the frozen relative tolerance)

| solver | attained | of | rate | KKT (supplied) | KKT (recovered) | timeouts/crashes/errors |
|---|---:|---:|---:|---:|---:|---:|
| fmincon-interior-point | 15 | 16 | 0.938 | 9 | 13 | 0 |
| fmincon-sqp | 15 | 16 | 0.938 | 7 | 11 | 0 |
| mincon | 15 | 16 | 0.938 | 11 | 13 | 0 |
| scipy-slsqp | 13 | 16 | 0.812 | 0 | 5 | 0 |

## Attainment by family

| family | n | fmincon-interior-point | fmincon-sqp | mincon | scipy-slsqp |
|---|---:|---:|---:|---:|---:|
| catenary | 2 | 2/2 | 2/2 | 2/2 | 2/2 |
| dispatch | 2 | 2/2 | 2/2 | 2/2 | 2/2 |
| ellipsoid2 | 2 | 2/2 | 2/2 | 2/2 | 2/2 |
| engineering2 | 1 | 1/1 | 1/1 | 1/1 | 1/1 |
| expfit2 | 2 | 2/2 | 2/2 | 2/2 | 2/2 |
| hs | 3 | 2/3 | 2/3 | 2/3 | 1/3 |
| polyqp | 2 | 2/2 | 2/2 | 2/2 | 2/2 |
| quadsphere2 | 2 | 2/2 | 2/2 | 2/2 | 1/2 |

## Paired comparison against `fmincon-interior-point` (metric for cost: total_model)

| solver | both | only solver | only baseline | neither | diff (pp) | 95% CI (family bootstrap) | geo-mean cost ratio solver/baseline (common) | 95% CI | n common |
|---|---:|---:|---:|---:|---:|---|---:|---|---:|
| fmincon-sqp | 15 | 0 | 0 | 1 | +0.0 | [+0.0, +0.0] | 0.95 | [0.64, 1.49] | 15 |
| mincon | 15 | 0 | 0 | 1 | +0.0 | [+0.0, +0.0] | 0.82 | [0.62, 1.06] | 15 |
| scipy-slsqp | 13 | 0 | 2 | 1 | -12.5 | [-26.3, +0.0] | 0.56 | [0.49, 0.65] | 13 |

## Performance profile points (total_model; fraction of problems solved within tau x best)

| tau | fmincon-interior-point | fmincon-sqp | mincon | scipy-slsqp |
|---:|---:|---:|---:|---:|
| 1 | 0.00 | 0.07 | 0.27 | 0.67 |
| 2 | 0.60 | 0.73 | 0.73 | 0.87 |
| 4 | 0.93 | 0.93 | 0.93 | 0.87 |
| 8 | 1.00 | 0.93 | 1.00 | 0.87 |
| 16 | 1.00 | 0.93 | 1.00 | 0.87 |
| 64 | 1.00 | 1.00 | 1.00 | 0.87 |
| 256 | 1.00 | 1.00 | 1.00 | 0.87 |

## Non-ok records (timeouts, crashes, errors) — all kept as failures above

