# Analysis of bench/results/abl-c6/scored.jsonl

146 problems, solvers: fmincon-interior-point, mincon-ip, mincon-ip@bfgs_scaling=false, mincon-ip@bfgs_scaling=true; duplicates overwritten: 0

## Target attainment (feasible to 1e-6 and objective within the frozen relative tolerance)

| solver | attained | of | rate | KKT (supplied) | KKT (recovered) | timeouts/crashes/errors |
|---|---:|---:|---:|---:|---:|---:|
| fmincon-interior-point | 112 | 127 | 0.882 | 61 | 96 | 0 |
| mincon-ip | 134 | 143 | 0.937 | 80 | 110 | 0 |
| mincon-ip@bfgs_scaling=false | 133 | 143 | 0.930 | 79 | 109 | 0 |
| mincon-ip@bfgs_scaling=true | 134 | 143 | 0.937 | 80 | 110 | 0 |

## Attainment by family

| family | n | fmincon-interior-point | mincon-ip | mincon-ip@bfgs_scaling=false | mincon-ip@bfgs_scaling=true |
|---|---:|---:|---:|---:|---:|
| adversarial | 15 | 13/15 | 13/15 | 13/15 | 13/15 |
| catenary | 2 | 0/0 | 2/2 | 2/2 | 2/2 |
| chainrosen | 6 | 2/6 | 5/6 | 5/6 | 5/6 |
| dispatch | 2 | 0/0 | 2/2 | 2/2 | 2/2 |
| ellipsoid | 3 | 1/3 | 2/3 | 2/3 | 2/3 |
| ellipsoid2 | 2 | 0/0 | 2/2 | 2/2 | 2/2 |
| engineering | 6 | 6/6 | 6/6 | 6/6 | 6/6 |
| engineering2 | 1 | 0/0 | 1/1 | 1/1 | 1/1 |
| expfit | 3 | 2/2 | 2/2 | 2/2 | 2/2 |
| expfit2 | 2 | 0/0 | 2/2 | 2/2 | 2/2 |
| hs | 92 | 84/89 | 87/92 | 87/92 | 87/92 |
| lqtraj | 3 | 3/3 | 3/3 | 3/3 | 3/3 |
| polyqp | 2 | 0/0 | 2/2 | 2/2 | 2/2 |
| portfolio | 2 | 0/0 | 0/0 | 0/0 | 0/0 |
| quadsphere | 3 | 1/3 | 3/3 | 3/3 | 3/3 |
| quadsphere2 | 2 | 0/0 | 2/2 | 1/2 | 2/2 |

## Paired comparison against `mincon-ip@bfgs_scaling=false` (metric for cost: total_model)

| solver | both | only solver | only baseline | neither | diff (pp) | 95% CI (family bootstrap) | geo-mean cost ratio solver/baseline (common) | 95% CI | n common |
|---|---:|---:|---:|---:|---:|---|---:|---|---:|
| fmincon-interior-point | 110 | 2 | 9 | 6 | -5.5 | [-32.3, -1.3] | 0.94 | [0.92, 1.11] | 110 |
| mincon-ip | 133 | 1 | 0 | 9 | +0.7 | [+0.0, +5.9] | 0.94 | [0.69, 1.01] | 133 |
| mincon-ip@bfgs_scaling=true | 133 | 1 | 0 | 9 | +0.7 | [+0.0, +5.9] | 0.94 | [0.69, 1.01] | 133 |

## Performance profile points (total_model; fraction of problems solved within tau x best)

| tau | fmincon-interior-point | mincon-ip | mincon-ip@bfgs_scaling=false | mincon-ip@bfgs_scaling=true |
|---:|---:|---:|---:|---:|
| 1 | 0.51 | 0.44 | 0.40 | 0.44 |
| 2 | 0.79 | 0.96 | 0.92 | 0.96 |
| 4 | 0.82 | 0.98 | 0.95 | 0.98 |
| 8 | 0.82 | 0.98 | 0.96 | 0.98 |
| 16 | 0.82 | 0.99 | 0.98 | 0.99 |
| 64 | 0.82 | 0.99 | 0.98 | 0.99 |
| 256 | 0.82 | 0.99 | 0.98 | 0.99 |

## Non-ok records (timeouts, crashes, errors) — all kept as failures above

