# Analysis of bench\results\s4-c2-val\scored-C.jsonl

24 problems, solvers: fmincon-interior-point, fmincon-sqp, mincon, scipy-slsqp; duplicates overwritten: 0

## Target attainment (feasible to 1e-6 and objective within the frozen relative tolerance)

| solver | attained | of | rate | KKT (supplied) | KKT (recovered) | timeouts/crashes/errors |
|---|---:|---:|---:|---:|---:|---:|
| fmincon-interior-point | 23 | 24 | 0.958 | 15 | 18 | 0 |
| fmincon-sqp | 21 | 24 | 0.875 | 14 | 19 | 0 |
| mincon | 23 | 24 | 0.958 | 18 | 18 | 0 |
| scipy-slsqp | 22 | 24 | 0.917 | 0 | 11 | 0 |

## Attainment by family

| family | n | fmincon-interior-point | fmincon-sqp | mincon | scipy-slsqp |
|---|---:|---:|---:|---:|---:|
| adversarial | 6 | 5/6 | 4/6 | 5/6 | 5/6 |
| engineering | 3 | 3/3 | 3/3 | 3/3 | 2/3 |
| hs | 12 | 12/12 | 12/12 | 12/12 | 12/12 |
| quadsphere | 3 | 3/3 | 2/3 | 3/3 | 3/3 |

## Paired comparison against `fmincon-interior-point` (metric for cost: total_model)

| solver | both | only solver | only baseline | neither | diff (pp) | 95% CI (family bootstrap) | geo-mean cost ratio solver/baseline (common) | 95% CI | n common |
|---|---:|---:|---:|---:|---:|---|---:|---|---:|
| fmincon-sqp | 21 | 0 | 2 | 1 | -8.3 | [-25.0, +0.0] | 0.67 | [0.39, 1.03] | 21 |
| mincon | 22 | 1 | 1 | 0 | +0.0 | [+0.0, +0.0] | 0.85 | [0.74, 1.16] | 22 |
| scipy-slsqp | 22 | 0 | 1 | 1 | -4.2 | [-20.0, +0.0] | 0.52 | [0.37, 0.69] | 22 |

## Performance profile points (total_model; fraction of problems solved within tau x best)

| tau | fmincon-interior-point | fmincon-sqp | mincon | scipy-slsqp |
|---:|---:|---:|---:|---:|
| 1 | 0.04 | 0.33 | 0.17 | 0.75 |
| 2 | 0.54 | 0.67 | 0.71 | 0.92 |
| 4 | 0.79 | 0.83 | 0.75 | 0.92 |
| 8 | 0.96 | 0.83 | 0.96 | 0.92 |
| 16 | 0.96 | 0.88 | 0.96 | 0.92 |
| 64 | 0.96 | 0.88 | 0.96 | 0.92 |
| 256 | 0.96 | 0.88 | 0.96 | 0.92 |

## Non-ok records (timeouts, crashes, errors) — all kept as failures above

