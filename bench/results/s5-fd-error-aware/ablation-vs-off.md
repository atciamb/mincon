# Analysis of results/abl-c4/scored.jsonl

130 problems, solvers: c2-windows, fmincon-interior-point, mincon-ip, mincon-ip@fd_error_aware=false; duplicates overwritten: 0

## Target attainment (feasible to 1e-6 and objective within the frozen relative tolerance)

| solver | attained | of | rate | KKT (supplied) | KKT (recovered) | timeouts/crashes/errors |
|---|---:|---:|---:|---:|---:|---:|
| c2-windows | 119 | 127 | 0.937 | 77 | 100 | 0 |
| fmincon-interior-point | 112 | 127 | 0.882 | 61 | 96 | 0 |
| mincon-ip | 119 | 127 | 0.937 | 76 | 100 | 0 |
| mincon-ip@fd_error_aware=false | 119 | 127 | 0.937 | 78 | 100 | 0 |

## Attainment by family

| family | n | c2-windows | fmincon-interior-point | mincon-ip | mincon-ip@fd_error_aware=false |
|---|---:|---:|---:|---:|---:|
| adversarial | 15 | 13/15 | 13/15 | 13/15 | 13/15 |
| chainrosen | 6 | 5/6 | 2/6 | 5/6 | 5/6 |
| ellipsoid | 3 | 2/3 | 1/3 | 2/3 | 2/3 |
| engineering | 6 | 6/6 | 6/6 | 6/6 | 6/6 |
| expfit | 3 | 2/2 | 2/2 | 2/2 | 2/2 |
| hs | 89 | 85/89 | 84/89 | 85/89 | 85/89 |
| lqtraj | 3 | 3/3 | 3/3 | 3/3 | 3/3 |
| portfolio | 2 | 0/0 | 0/0 | 0/0 | 0/0 |
| quadsphere | 3 | 3/3 | 1/3 | 3/3 | 3/3 |

## Paired comparison against `mincon-ip@fd_error_aware=false` (metric for cost: total_model)

| solver | both | only solver | only baseline | neither | diff (pp) | 95% CI (family bootstrap) | geo-mean cost ratio solver/baseline (common) | 95% CI | n common |
|---|---:|---:|---:|---:|---:|---|---:|---|---:|
| c2-windows | 119 | 0 | 0 | 8 | +0.0 | [+0.0, +0.0] | 0.99 | [0.99, 1.00] | 119 |
| fmincon-interior-point | 110 | 2 | 9 | 6 | -5.5 | [-32.3, -1.3] | 0.92 | [0.89, 1.12] | 110 |
| mincon-ip | 119 | 0 | 0 | 8 | +0.0 | [+0.0, +0.0] | 0.97 | [0.96, 1.01] | 119 |

## Performance profile points (total_model; fraction of problems solved within tau x best)

| tau | c2-windows | fmincon-interior-point | mincon-ip | mincon-ip@fd_error_aware=false |
|---:|---:|---:|---:|---:|
| 1 | 0.37 | 0.60 | 0.30 | 0.37 |
| 2 | 0.93 | 0.90 | 0.95 | 0.93 |
| 4 | 0.96 | 0.92 | 0.98 | 0.96 |
| 8 | 0.98 | 0.92 | 0.98 | 0.98 |
| 16 | 0.98 | 0.93 | 0.98 | 0.98 |
| 64 | 0.98 | 0.93 | 0.98 | 0.98 |
| 256 | 0.98 | 0.93 | 0.98 | 0.98 |

## Non-ok records (timeouts, crashes, errors) — all kept as failures above

