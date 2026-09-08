# Analysis of results/abl-c3/scored.jsonl

130 problems, solvers: c2-windows, fmincon-interior-point, mincon-ip; duplicates overwritten: 0

## Target attainment (feasible to 1e-6 and objective within the frozen relative tolerance)

| solver | attained | of | rate | KKT (supplied) | KKT (recovered) | timeouts/crashes/errors |
|---|---:|---:|---:|---:|---:|---:|
| c2-windows | 119 | 127 | 0.937 | 77 | 100 | 0 |
| fmincon-interior-point | 112 | 127 | 0.882 | 61 | 96 | 0 |
| mincon-ip | 116 | 127 | 0.913 | 77 | 102 | 0 |

## Attainment by family

| family | n | c2-windows | fmincon-interior-point | mincon-ip |
|---|---:|---:|---:|---:|
| adversarial | 15 | 13/15 | 13/15 | 13/15 |
| chainrosen | 6 | 5/6 | 2/6 | 4/6 |
| ellipsoid | 3 | 2/3 | 1/3 | 2/3 |
| engineering | 6 | 6/6 | 6/6 | 6/6 |
| expfit | 3 | 2/2 | 2/2 | 2/2 |
| hs | 89 | 85/89 | 84/89 | 83/89 |
| lqtraj | 3 | 3/3 | 3/3 | 3/3 |
| portfolio | 2 | 0/0 | 0/0 | 0/0 |
| quadsphere | 3 | 3/3 | 1/3 | 3/3 |

## Paired comparison against `c2-windows` (metric for cost: total_model)

| solver | both | only solver | only baseline | neither | diff (pp) | 95% CI (family bootstrap) | geo-mean cost ratio solver/baseline (common) | 95% CI | n common |
|---|---:|---:|---:|---:|---:|---|---:|---|---:|
| fmincon-interior-point | 110 | 2 | 9 | 6 | -5.5 | [-32.3, -1.3] | 0.93 | [0.90, 1.12] | 110 |
| mincon-ip | 116 | 0 | 3 | 8 | -2.4 | [-6.5, +0.0] | 1.03 | [0.94, 1.13] | 116 |

## Performance profile points (total_model; fraction of problems solved within tau x best)

| tau | c2-windows | fmincon-interior-point | mincon-ip |
|---:|---:|---:|---:|
| 1 | 0.31 | 0.55 | 0.27 |
| 2 | 0.93 | 0.90 | 0.89 |
| 4 | 0.97 | 0.92 | 0.94 |
| 8 | 0.98 | 0.92 | 0.95 |
| 16 | 0.98 | 0.93 | 0.96 |
| 64 | 0.98 | 0.93 | 0.96 |
| 256 | 0.98 | 0.93 | 0.96 |

## Non-ok records (timeouts, crashes, errors) — all kept as failures above

