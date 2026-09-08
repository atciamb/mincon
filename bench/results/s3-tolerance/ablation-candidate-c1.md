# Analysis of results/abl-c1/scored.jsonl

82 problems, solvers: b0-mincon-ip, mincon, mincon-ip; duplicates overwritten: 0

## Target attainment (feasible to 1e-6 and objective within the frozen relative tolerance)

| solver | attained | of | rate | KKT (supplied) | KKT (recovered) | timeouts/crashes/errors |
|---|---:|---:|---:|---:|---:|---:|
| b0-mincon-ip | 76 | 81 | 0.938 | 51 | 69 | 0 |
| mincon | 76 | 81 | 0.938 | 53 | 63 | 0 |
| mincon-ip | 76 | 81 | 0.938 | 51 | 63 | 0 |

## Attainment by family

| family | n | b0-mincon-ip | mincon | mincon-ip |
|---|---:|---:|---:|---:|
| adversarial | 7 | 7/7 | 7/7 | 7/7 |
| chainrosen | 6 | 5/6 | 5/6 | 5/6 |
| expfit | 3 | 2/2 | 2/2 | 2/2 |
| hs | 63 | 59/63 | 59/63 | 59/63 |
| lqtraj | 3 | 3/3 | 3/3 | 3/3 |

## Paired comparison against `b0-mincon-ip` (metric for cost: total_model)

| solver | both | only solver | only baseline | neither | diff (pp) | 95% CI (family bootstrap) | geo-mean cost ratio solver/baseline (common) | 95% CI | n common |
|---|---:|---:|---:|---:|---:|---|---:|---|---:|
| mincon | 76 | 0 | 0 | 5 | +0.0 | [+0.0, +0.0] | 0.65 | [0.37, 0.96] | 76 |
| mincon-ip | 76 | 0 | 0 | 5 | +0.0 | [+0.0, +0.0] | 0.61 | [0.37, 0.76] | 76 |

## Performance profile points (total_model; fraction of problems solved within tau x best)

| tau | b0-mincon-ip | mincon | mincon-ip |
|---:|---:|---:|---:|
| 1 | 0.14 | 0.89 | 0.96 |
| 2 | 0.68 | 0.93 | 0.99 |
| 4 | 0.89 | 1.00 | 1.00 |
| 8 | 0.99 | 1.00 | 1.00 |
| 16 | 1.00 | 1.00 | 1.00 |
| 64 | 1.00 | 1.00 | 1.00 |
| 256 | 1.00 | 1.00 | 1.00 |

## Non-ok records (timeouts, crashes, errors) — all kept as failures above

