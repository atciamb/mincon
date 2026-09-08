# Analysis of results/abl-c5/scored.jsonl

130 problems, solvers: c4, fmincon-interior-point, mincon-ip@_floor=0.01, mincon-ip@_floor=0.1; duplicates overwritten: 0

## Target attainment (feasible to 1e-6 and objective within the frozen relative tolerance)

| solver | attained | of | rate | KKT (supplied) | KKT (recovered) | timeouts/crashes/errors |
|---|---:|---:|---:|---:|---:|---:|
| c4 | 119 | 127 | 0.937 | 76 | 100 | 0 |
| fmincon-interior-point | 112 | 127 | 0.882 | 61 | 96 | 0 |
| mincon-ip@_floor=0.01 | 119 | 127 | 0.937 | 78 | 100 | 0 |
| mincon-ip@_floor=0.1 | 119 | 127 | 0.937 | 75 | 100 | 0 |

## Attainment by family

| family | n | c4 | fmincon-interior-point | mincon-ip@_floor=0.01 | mincon-ip@_floor=0.1 |
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

## Paired comparison against `c4` (metric for cost: total_model)

| solver | both | only solver | only baseline | neither | diff (pp) | 95% CI (family bootstrap) | geo-mean cost ratio solver/baseline (common) | 95% CI | n common |
|---|---:|---:|---:|---:|---:|---|---:|---|---:|
| fmincon-interior-point | 110 | 2 | 9 | 6 | -5.5 | [-32.3, -1.3] | 0.94 | [0.92, 1.11] | 110 |
| mincon-ip@_floor=0.01 | 119 | 0 | 0 | 8 | +0.0 | [+0.0, +0.0] | 1.00 | [0.98, 1.10] | 119 |
| mincon-ip@_floor=0.1 | 119 | 0 | 0 | 8 | +0.0 | [+0.0, +0.0] | 1.00 | [0.98, 1.19] | 119 |

## Performance profile points (total_model; fraction of problems solved within tau x best)

| tau | c4 | fmincon-interior-point | mincon-ip@_floor=0.01 | mincon-ip@_floor=0.1 |
|---:|---:|---:|---:|---:|
| 1 | 0.37 | 0.57 | 0.35 | 0.35 |
| 2 | 0.95 | 0.90 | 0.94 | 0.94 |
| 4 | 0.98 | 0.92 | 0.98 | 0.98 |
| 8 | 0.98 | 0.92 | 0.98 | 0.98 |
| 16 | 0.98 | 0.93 | 0.98 | 0.98 |
| 64 | 0.98 | 0.93 | 0.98 | 0.98 |
| 256 | 0.98 | 0.93 | 0.98 | 0.98 |

## Non-ok records (timeouts, crashes, errors) — all kept as failures above

