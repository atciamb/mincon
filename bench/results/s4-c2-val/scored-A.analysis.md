# Analysis of bench\results\s4-c2-val\scored-A.jsonl

24 problems, solvers: fmincon-interior-point, fmincon-sqp, mincon, scipy-slsqp; duplicates overwritten: 0

## Target attainment (feasible to 1e-6 and objective within the frozen relative tolerance)

| solver | attained | of | rate | KKT (supplied) | KKT (recovered) | timeouts/crashes/errors |
|---|---:|---:|---:|---:|---:|---:|
| fmincon-interior-point | 21 | 24 | 0.875 | 8 | 18 | 0 |
| fmincon-sqp | 21 | 24 | 0.875 | 10 | 18 | 0 |
| mincon | 23 | 24 | 0.958 | 13 | 18 | 0 |
| scipy-slsqp | 22 | 24 | 0.917 | 0 | 12 | 0 |

## Attainment by family

| family | n | fmincon-interior-point | fmincon-sqp | mincon | scipy-slsqp |
|---|---:|---:|---:|---:|---:|
| adversarial | 6 | 5/6 | 4/6 | 5/6 | 5/6 |
| engineering | 3 | 3/3 | 3/3 | 3/3 | 2/3 |
| hs | 12 | 12/12 | 12/12 | 12/12 | 12/12 |
| quadsphere | 3 | 1/3 | 2/3 | 3/3 | 3/3 |

## Paired comparison against `fmincon-interior-point` (metric for cost: total_model)

| solver | both | only solver | only baseline | neither | diff (pp) | 95% CI (family bootstrap) | geo-mean cost ratio solver/baseline (common) | 95% CI | n common |
|---|---:|---:|---:|---:|---:|---|---:|---|---:|
| fmincon-sqp | 20 | 1 | 1 | 2 | +0.0 | [-11.1, +16.7] | 0.61 | [0.41, 0.73] | 20 |
| mincon | 20 | 3 | 1 | 0 | +8.3 | [+0.0, +40.0] | 0.91 | [0.75, 0.96] | 20 |
| scipy-slsqp | 20 | 2 | 1 | 1 | +4.2 | [-14.3, +40.0] | 0.54 | [0.36, 0.68] | 20 |

## Performance profile points (total_model; fraction of problems solved within tau x best)

| tau | fmincon-interior-point | fmincon-sqp | mincon | scipy-slsqp |
|---:|---:|---:|---:|---:|
| 1 | 0.04 | 0.25 | 0.12 | 0.62 |
| 2 | 0.50 | 0.83 | 0.62 | 0.92 |
| 4 | 0.71 | 0.88 | 0.79 | 0.92 |
| 8 | 0.88 | 0.88 | 0.96 | 0.92 |
| 16 | 0.88 | 0.88 | 0.96 | 0.92 |
| 64 | 0.88 | 0.88 | 0.96 | 0.92 |
| 256 | 0.88 | 0.88 | 0.96 | 0.92 |

## Non-ok records (timeouts, crashes, errors) — all kept as failures above

