# Analysis of bench\results\s6-final\scored-C.jsonl

24 problems, solvers: fmincon-interior-point, fmincon-sqp, mincon, scipy-slsqp; duplicates overwritten: 0

## Target attainment (feasible to 1e-6 and objective within the frozen relative tolerance)

| solver | attained | of | rate | KKT (supplied) | KKT (recovered) | timeouts/crashes/errors |
|---|---:|---:|---:|---:|---:|---:|
| fmincon-interior-point | 19 | 22 | 0.864 | 10 | 16 | 0 |
| fmincon-sqp | 20 | 22 | 0.909 | 14 | 22 | 0 |
| mincon | 20 | 22 | 0.909 | 18 | 18 | 0 |
| scipy-slsqp | 19 | 22 | 0.864 | 0 | 13 | 0 |

## Attainment by family

| family | n | fmincon-interior-point | fmincon-sqp | mincon | scipy-slsqp |
|---|---:|---:|---:|---:|---:|
| adversarial | 2 | 1/2 | 1/2 | 1/2 | 1/2 |
| ellipsoid | 3 | 2/3 | 2/3 | 3/3 | 2/3 |
| engineering | 3 | 3/3 | 3/3 | 3/3 | 2/3 |
| hs | 14 | 13/14 | 14/14 | 13/14 | 14/14 |
| portfolio | 2 | 0/0 | 0/0 | 0/0 | 0/0 |

## Paired comparison against `fmincon-interior-point` (metric for cost: total_model)

| solver | both | only solver | only baseline | neither | diff (pp) | 95% CI (family bootstrap) | geo-mean cost ratio solver/baseline (common) | 95% CI | n common |
|---|---:|---:|---:|---:|---:|---|---:|---|---:|
| fmincon-sqp | 19 | 1 | 0 | 2 | +4.5 | [+0.0, +6.7] | 0.80 | [0.59, 1.05] | 19 |
| mincon | 19 | 1 | 0 | 2 | +4.5 | [+0.0, +25.0] | 1.14 | [0.72, 1.28] | 19 |
| scipy-slsqp | 18 | 1 | 1 | 2 | +0.0 | [-25.0, +6.7] | 0.60 | [0.57, 0.75] | 18 |

## Performance profile points (total_model; fraction of problems solved within tau x best)

| tau | fmincon-interior-point | fmincon-sqp | mincon | scipy-slsqp |
|---:|---:|---:|---:|---:|
| 1 | 0.05 | 0.14 | 0.19 | 0.76 |
| 2 | 0.67 | 0.81 | 0.57 | 0.86 |
| 4 | 0.76 | 0.95 | 0.81 | 0.86 |
| 8 | 0.86 | 0.95 | 0.90 | 0.90 |
| 16 | 0.90 | 0.95 | 0.90 | 0.90 |
| 64 | 0.90 | 0.95 | 0.95 | 0.90 |
| 256 | 0.90 | 0.95 | 0.95 | 0.90 |

## Non-ok records (timeouts, crashes, errors) — all kept as failures above

