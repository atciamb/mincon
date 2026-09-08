# Analysis of bench\results\s6-final\scored-A.jsonl

24 problems, solvers: fmincon-interior-point, fmincon-sqp, mincon, scipy-slsqp; duplicates overwritten: 0

## Target attainment (feasible to 1e-6 and objective within the frozen relative tolerance)

| solver | attained | of | rate | KKT (supplied) | KKT (recovered) | timeouts/crashes/errors |
|---|---:|---:|---:|---:|---:|---:|
| fmincon-interior-point | 18 | 22 | 0.818 | 9 | 16 | 0 |
| fmincon-sqp | 20 | 22 | 0.909 | 13 | 22 | 0 |
| mincon | 19 | 22 | 0.864 | 14 | 17 | 0 |
| scipy-slsqp | 20 | 22 | 0.909 | 0 | 13 | 0 |

## Attainment by family

| family | n | fmincon-interior-point | fmincon-sqp | mincon | scipy-slsqp |
|---|---:|---:|---:|---:|---:|
| adversarial | 2 | 1/2 | 1/2 | 1/2 | 1/2 |
| ellipsoid | 3 | 1/3 | 2/3 | 2/3 | 3/3 |
| engineering | 3 | 3/3 | 3/3 | 3/3 | 2/3 |
| hs | 14 | 13/14 | 14/14 | 13/14 | 14/14 |
| portfolio | 2 | 0/0 | 0/0 | 0/0 | 0/0 |

## Paired comparison against `fmincon-interior-point` (metric for cost: total_model)

| solver | both | only solver | only baseline | neither | diff (pp) | 95% CI (family bootstrap) | geo-mean cost ratio solver/baseline (common) | 95% CI | n common |
|---|---:|---:|---:|---:|---:|---|---:|---|---:|
| fmincon-sqp | 18 | 2 | 0 | 2 | +9.1 | [+0.0, +25.0] | 0.77 | [0.59, 0.98] | 18 |
| mincon | 18 | 1 | 0 | 3 | +4.5 | [+0.0, +25.0] | 1.43 | [1.04, 1.59] | 18 |
| scipy-slsqp | 17 | 3 | 1 | 1 | +9.1 | [-20.0, +41.7] | 0.61 | [0.56, 0.93] | 17 |

## Performance profile points (total_model; fraction of problems solved within tau x best)

| tau | fmincon-interior-point | fmincon-sqp | mincon | scipy-slsqp |
|---:|---:|---:|---:|---:|
| 1 | 0.10 | 0.14 | 0.00 | 0.76 |
| 2 | 0.52 | 0.86 | 0.48 | 0.95 |
| 4 | 0.81 | 0.95 | 0.71 | 0.95 |
| 8 | 0.86 | 0.95 | 0.86 | 0.95 |
| 16 | 0.86 | 0.95 | 0.90 | 0.95 |
| 64 | 0.86 | 0.95 | 0.90 | 0.95 |
| 256 | 0.86 | 0.95 | 0.90 | 0.95 |

## Non-ok records (timeouts, crashes, errors) — all kept as failures above

