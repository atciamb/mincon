# Analysis of bench\results\s3-c1-dev\scored-A.jsonl

82 problems, solvers: fmincon-interior-point, fmincon-sqp, mincon, mincon-ip, scipy-slsqp; duplicates overwritten: 0

## Target attainment (feasible to 1e-6 and objective within the frozen relative tolerance)

| solver | attained | of | rate | KKT (supplied) | KKT (recovered) | timeouts/crashes/errors |
|---|---:|---:|---:|---:|---:|---:|
| fmincon-interior-point | 73 | 81 | 0.901 | 44 | 62 | 0 |
| fmincon-sqp | 70 | 81 | 0.864 | 53 | 70 | 0 |
| mincon | 76 | 81 | 0.938 | 53 | 63 | 0 |
| mincon-ip | 76 | 81 | 0.938 | 51 | 63 | 0 |
| scipy-slsqp | 72 | 81 | 0.889 | 0 | 37 | 0 |

## Attainment by family

| family | n | fmincon-interior-point | fmincon-sqp | mincon | mincon-ip | scipy-slsqp |
|---|---:|---:|---:|---:|---:|---:|
| adversarial | 7 | 7/7 | 7/7 | 7/7 | 7/7 | 6/7 |
| chainrosen | 6 | 2/6 | 0/6 | 5/6 | 5/6 | 6/6 |
| expfit | 3 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| hs | 63 | 59/63 | 58/63 | 59/63 | 59/63 | 55/63 |
| lqtraj | 3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |

## Paired comparison against `fmincon-interior-point` (metric for cost: total_model)

| solver | both | only solver | only baseline | neither | diff (pp) | 95% CI (family bootstrap) | geo-mean cost ratio solver/baseline (common) | 95% CI | n common |
|---|---:|---:|---:|---:|---:|---|---:|---|---:|
| fmincon-sqp | 68 | 2 | 5 | 6 | -3.7 | [-25.0, +0.0] | 0.73 | [0.69, 1.13] | 68 |
| mincon | 71 | 5 | 2 | 3 | +3.7 | [+0.0, +37.5] | 1.16 | [0.95, 1.35] | 71 |
| mincon-ip | 71 | 5 | 2 | 3 | +3.7 | [+0.0, +37.5] | 1.12 | [0.88, 1.26] | 71 |
| scipy-slsqp | 67 | 5 | 6 | 3 | -1.2 | [-9.1, +48.4] | 0.61 | [0.60, 0.73] | 67 |

## Performance profile points (total_model; fraction of problems solved within tau x best)

| tau | fmincon-interior-point | fmincon-sqp | mincon | mincon-ip | scipy-slsqp |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.10 | 0.18 | 0.04 | 0.04 | 0.70 |
| 2 | 0.59 | 0.81 | 0.61 | 0.63 | 0.89 |
| 4 | 0.87 | 0.89 | 0.81 | 0.82 | 0.91 |
| 8 | 0.91 | 0.89 | 0.91 | 0.94 | 0.91 |
| 16 | 0.92 | 0.89 | 0.95 | 0.95 | 0.91 |
| 64 | 0.92 | 0.89 | 0.96 | 0.96 | 0.91 |
| 256 | 0.92 | 0.89 | 0.96 | 0.96 | 0.91 |

## Non-ok records (timeouts, crashes, errors) — all kept as failures above

