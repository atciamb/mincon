# Analysis of results/abl-tol/scored.jsonl

82 problems, solvers: mincon-ip, mincon-ip@ftol=1e-6; duplicates overwritten: 0

## Target attainment (feasible to 1e-6 and objective within the frozen relative tolerance)

| solver | attained | of | rate | KKT (supplied) | KKT (recovered) | timeouts/crashes/errors |
|---|---:|---:|---:|---:|---:|---:|
| mincon-ip | 76 | 81 | 0.938 | 51 | 69 | 0 |
| mincon-ip@ftol=1e-6 | 76 | 81 | 0.938 | 47 | 63 | 0 |

## Attainment by family

| family | n | mincon-ip | mincon-ip@ftol=1e-6 |
|---|---:|---:|---:|
| adversarial | 7 | 7/7 | 7/7 |
| chainrosen | 6 | 5/6 | 5/6 |
| expfit | 3 | 2/2 | 2/2 |
| hs | 63 | 59/63 | 59/63 |
| lqtraj | 3 | 3/3 | 3/3 |

## Paired comparison against `mincon-ip` (metric for cost: total_model)

| solver | both | only solver | only baseline | neither | diff (pp) | 95% CI (family bootstrap) | geo-mean cost ratio solver/baseline (common) | 95% CI | n common |
|---|---:|---:|---:|---:|---:|---|---:|---|---:|
| mincon-ip@ftol=1e-6 | 76 | 0 | 0 | 5 | +0.0 | [+0.0, +0.0] | 0.62 | [0.36, 0.76] | 76 |

## Performance profile points (total_model; fraction of problems solved within tau x best)

| tau | mincon-ip | mincon-ip@ftol=1e-6 |
|---:|---:|---:|
| 1 | 0.16 | 0.96 |
| 2 | 0.70 | 0.99 |
| 4 | 0.88 | 1.00 |
| 8 | 0.99 | 1.00 |
| 16 | 1.00 | 1.00 |
| 64 | 1.00 | 1.00 |
| 256 | 1.00 | 1.00 |

## Non-ok records (timeouts, crashes, errors) — all kept as failures above

