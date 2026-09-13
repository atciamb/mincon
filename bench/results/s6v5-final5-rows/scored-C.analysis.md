# Analysis of ../results/s6v5-final5-rows/scored-C.jsonl

13 problems, solvers: mincon, mincon@quadratic_rows=off; duplicates overwritten: 0

## Target attainment (feasible to 1e-6 and objective within the frozen relative tolerance)

| solver | attained | of | rate | KKT (supplied) | KKT (recovered) | timeouts/crashes/errors |
|---|---:|---:|---:|---:|---:|---:|
| mincon | 13 | 13 | 1.000 | 6 | 13 | 0 |
| mincon@quadratic_rows=off | 13 | 13 | 1.000 | 6 | 12 | 0 |

## Attainment by family

| family | n | mincon | mincon@quadratic_rows=off |
|---|---:|---:|---:|
| deconv | 2 | 2/2 | 2/2 |
| engineering3 | 2 | 2/2 | 2/2 |
| logistic | 2 | 2/2 | 2/2 |
| ncboxqp | 2 | 2/2 | 2/2 |
| noisyqp | 1 | 1/1 | 1/1 |
| pkfit | 2 | 2/2 | 2/2 |
| tcport | 2 | 2/2 | 2/2 |

## Paired comparison against `mincon@quadratic_rows=off` (metric for cost: total_model)

| solver | both | only solver | only baseline | neither | diff (pp) | 95% CI (family bootstrap) | geo-mean cost ratio solver/baseline (common) | 95% CI | n common |
|---|---:|---:|---:|---:|---:|---|---:|---|---:|
| mincon | 13 | 0 | 0 | 0 | +0.0 | [+0.0, +0.0] | 0.93 | [0.79, 1.00] | 13 |

## Performance profile points (total_model; fraction of problems solved within tau x best)

| tau | mincon | mincon@quadratic_rows=off |
|---:|---:|---:|
| 1 | 1.00 | 0.85 |
| 2 | 1.00 | 0.92 |
| 4 | 1.00 | 1.00 |
| 8 | 1.00 | 1.00 |
| 16 | 1.00 | 1.00 |
| 64 | 1.00 | 1.00 |
| 256 | 1.00 | 1.00 |

## Non-ok records (timeouts, crashes, errors) — all kept as failures above

