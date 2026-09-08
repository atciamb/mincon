# Analysis of bench\results\s3-c1-dev\scored-C.jsonl

82 problems, solvers: fmincon-interior-point, fmincon-sqp, mincon, mincon-ip, scipy-slsqp; duplicates overwritten: 0

## Target attainment (feasible to 1e-6 and objective within the frozen relative tolerance)

| solver | attained | of | rate | KKT (supplied) | KKT (recovered) | timeouts/crashes/errors |
|---|---:|---:|---:|---:|---:|---:|
| fmincon-interior-point | 76 | 81 | 0.938 | 54 | 71 | 0 |
| fmincon-sqp | 74 | 81 | 0.914 | 60 | 73 | 0 |
| mincon | 77 | 81 | 0.951 | 57 | 66 | 0 |
| mincon-ip | 77 | 81 | 0.951 | 57 | 66 | 0 |
| scipy-slsqp | 71 | 81 | 0.877 | 0 | 37 | 0 |

## Attainment by family

| family | n | fmincon-interior-point | fmincon-sqp | mincon | mincon-ip | scipy-slsqp |
|---|---:|---:|---:|---:|---:|---:|
| adversarial | 7 | 7/7 | 7/7 | 7/7 | 7/7 | 6/7 |
| chainrosen | 6 | 3/6 | 4/6 | 6/6 | 6/6 | 6/6 |
| expfit | 3 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| hs | 63 | 61/63 | 58/63 | 59/63 | 59/63 | 54/63 |
| lqtraj | 3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |

## Paired comparison against `fmincon-interior-point` (metric for cost: total_model)

| solver | both | only solver | only baseline | neither | diff (pp) | 95% CI (family bootstrap) | geo-mean cost ratio solver/baseline (common) | 95% CI | n common |
|---|---:|---:|---:|---:|---:|---|---:|---|---:|
| fmincon-sqp | 71 | 3 | 5 | 2 | -2.5 | [-4.5, +12.5] | 0.73 | [0.69, 1.19] | 71 |
| mincon | 73 | 4 | 3 | 1 | +1.2 | [-3.0, +37.5] | 0.92 | [0.67, 1.09] | 73 |
| mincon-ip | 73 | 4 | 3 | 1 | +1.2 | [-3.0, +37.5] | 0.91 | [0.67, 1.09] | 73 |
| scipy-slsqp | 68 | 3 | 8 | 2 | -6.2 | [-11.5, +35.5] | 0.60 | [0.57, 0.67] | 68 |

## Performance profile points (total_model; fraction of problems solved within tau x best)

| tau | fmincon-interior-point | fmincon-sqp | mincon | mincon-ip | scipy-slsqp |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.15 | 0.23 | 0.20 | 0.20 | 0.69 |
| 2 | 0.61 | 0.79 | 0.64 | 0.64 | 0.84 |
| 4 | 0.89 | 0.89 | 0.90 | 0.90 | 0.88 |
| 8 | 0.94 | 0.91 | 0.95 | 0.95 | 0.88 |
| 16 | 0.94 | 0.93 | 0.95 | 0.96 | 0.89 |
| 64 | 0.95 | 0.93 | 0.96 | 0.96 | 0.89 |
| 256 | 0.95 | 0.93 | 0.96 | 0.96 | 0.89 |

## Non-ok records (timeouts, crashes, errors) — all kept as failures above

