# Analysis of bench\results\s6v2-final2\scored-A.jsonl

16 problems, solvers: fmincon-interior-point, fmincon-sqp, mincon, scipy-slsqp; duplicates overwritten: 0

## Target attainment (feasible to 1e-6 and objective within the frozen relative tolerance)

| solver | attained | of | rate | KKT (supplied) | KKT (recovered) | timeouts/crashes/errors |
|---|---:|---:|---:|---:|---:|---:|
| fmincon-interior-point | 12 | 16 | 0.750 | 3 | 9 | 0 |
| fmincon-sqp | 13 | 16 | 0.812 | 6 | 11 | 0 |
| mincon | 14 | 16 | 0.875 | 3 | 9 | 0 |
| scipy-slsqp | 14 | 16 | 0.875 | 0 | 4 | 0 |

## Attainment by family

| family | n | fmincon-interior-point | fmincon-sqp | mincon | scipy-slsqp |
|---|---:|---:|---:|---:|---:|
| catenary | 2 | 1/2 | 1/2 | 2/2 | 2/2 |
| dispatch | 2 | 2/2 | 2/2 | 2/2 | 2/2 |
| ellipsoid2 | 2 | 1/2 | 1/2 | 2/2 | 2/2 |
| engineering2 | 1 | 1/1 | 1/1 | 1/1 | 1/1 |
| expfit2 | 2 | 2/2 | 2/2 | 2/2 | 2/2 |
| hs | 3 | 2/3 | 2/3 | 2/3 | 1/3 |
| polyqp | 2 | 2/2 | 2/2 | 2/2 | 2/2 |
| quadsphere2 | 2 | 1/2 | 2/2 | 1/2 | 2/2 |

## Paired comparison against `fmincon-interior-point` (metric for cost: total_model)

| solver | both | only solver | only baseline | neither | diff (pp) | 95% CI (family bootstrap) | geo-mean cost ratio solver/baseline (common) | 95% CI | n common |
|---|---:|---:|---:|---:|---:|---|---:|---|---:|
| fmincon-sqp | 12 | 1 | 0 | 3 | +6.2 | [+0.0, +20.0] | 0.67 | [0.53, 0.83] | 12 |
| mincon | 12 | 2 | 0 | 2 | +12.5 | [+0.0, +28.6] | 0.88 | [0.68, 1.10] | 12 |
| scipy-slsqp | 11 | 3 | 1 | 1 | +12.5 | [-10.5, +37.5] | 0.58 | [0.51, 0.68] | 11 |

## Performance profile points (total_model; fraction of problems solved within tau x best)

| tau | fmincon-interior-point | fmincon-sqp | mincon | scipy-slsqp |
|---:|---:|---:|---:|---:|
| 1 | 0.00 | 0.07 | 0.13 | 0.80 |
| 2 | 0.40 | 0.87 | 0.73 | 0.93 |
| 4 | 0.73 | 0.87 | 0.87 | 0.93 |
| 8 | 0.80 | 0.87 | 0.93 | 0.93 |
| 16 | 0.80 | 0.87 | 0.93 | 0.93 |
| 64 | 0.80 | 0.87 | 0.93 | 0.93 |
| 256 | 0.80 | 0.87 | 0.93 | 0.93 |

## Non-ok records (timeouts, crashes, errors) — all kept as failures above

