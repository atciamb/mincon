# Analysis of results/abl-c2/scored.jsonl

82 problems, solvers: mincon-ip, mincon-ip@barrier=monotone; duplicates overwritten: 0

## Target attainment (feasible to 1e-6 and objective within the frozen relative tolerance)

| solver | attained | of | rate | KKT (supplied) | KKT (recovered) | timeouts/crashes/errors |
|---|---:|---:|---:|---:|---:|---:|
| mincon-ip | 77 | 81 | 0.951 | 50 | 65 | 0 |
| mincon-ip@barrier=monotone | 76 | 81 | 0.938 | 51 | 63 | 0 |

## Attainment by family

| family | n | mincon-ip | mincon-ip@barrier=monotone |
|---|---:|---:|---:|
| adversarial | 7 | 7/7 | 7/7 |
| chainrosen | 6 | 5/6 | 5/6 |
| expfit | 3 | 2/2 | 2/2 |
| hs | 63 | 60/63 | 59/63 |
| lqtraj | 3 | 3/3 | 3/3 |

## Paired comparison against `mincon-ip@barrier=monotone` (metric for cost: total_model)

| solver | both | only solver | only baseline | neither | diff (pp) | 95% CI (family bootstrap) | geo-mean cost ratio solver/baseline (common) | 95% CI | n common |
|---|---:|---:|---:|---:|---:|---|---:|---|---:|
| mincon-ip | 76 | 1 | 0 | 4 | +1.2 | [+0.0, +1.5] | 0.93 | [0.85, 0.98] | 76 |

## Performance profile points (total_model; fraction of problems solved within tau x best)

| tau | mincon-ip | mincon-ip@barrier=monotone |
|---:|---:|---:|
| 1 | 0.82 | 0.49 |
| 2 | 0.99 | 0.95 |
| 4 | 1.00 | 0.97 |
| 8 | 1.00 | 0.99 |
| 16 | 1.00 | 0.99 |
| 64 | 1.00 | 0.99 |
| 256 | 1.00 | 0.99 |

## Non-ok records (timeouts, crashes, errors) — all kept as failures above

