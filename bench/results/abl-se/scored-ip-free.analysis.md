# Analysis of C:\Users\andyc\OneDrive\Desktop\python_fmincon\mincon-research\bench\results\abl-se\scored-ip-free.jsonl

169 problems, solvers: mincon-ip@bfgs_rescale=0, mincon-ip@kkt_pivot_signs=free; duplicates overwritten: 0

## Target attainment (feasible to 1e-6 and objective within the frozen relative tolerance)

| solver | attained | of | rate | KKT (supplied) | KKT (recovered) | timeouts/crashes/errors |
|---|---:|---:|---:|---:|---:|---:|
| mincon-ip@bfgs_rescale=0 | 156 | 166 | 0.940 | 87 | 119 | 0 |
| mincon-ip@kkt_pivot_signs=free | 157 | 166 | 0.946 | 86 | 118 | 0 |

## Attainment by family

| family | n | mincon-ip@bfgs_rescale=0 | mincon-ip@kkt_pivot_signs=free |
|---|---:|---:|---:|
| adversarial | 15 | 14/15 | 14/15 |
| catenary | 2 | 2/2 | 2/2 |
| chainrosen | 6 | 5/6 | 5/6 |
| covqp | 3 | 2/3 | 3/3 |
| denselap | 2 | 2/2 | 2/2 |
| dispatch | 2 | 2/2 | 2/2 |
| ellipsoid | 3 | 2/3 | 2/3 |
| ellipsoid2 | 2 | 2/2 | 2/2 |
| engineering | 6 | 6/6 | 6/6 |
| engineering2 | 1 | 1/1 | 1/1 |
| expfit | 3 | 2/2 | 2/2 |
| expfit2 | 2 | 2/2 | 2/2 |
| hs | 93 | 88/93 | 88/93 |
| logsumexp | 2 | 2/2 | 2/2 |
| lqtraj | 3 | 3/3 | 3/3 |
| maxent | 3 | 3/3 | 3/3 |
| nnls_simplex | 3 | 3/3 | 3/3 |
| obstacle | 3 | 2/3 | 2/3 |
| polyqp | 2 | 2/2 | 2/2 |
| portfolio | 2 | 0/0 | 0/0 |
| quadsphere | 3 | 3/3 | 3/3 |
| quadsphere2 | 2 | 2/2 | 2/2 |
| rosen_sphere | 2 | 2/2 | 2/2 |
| sinfit | 1 | 1/1 | 1/1 |
| snl | 3 | 3/3 | 3/3 |

## Paired comparison against `mincon-ip@bfgs_rescale=0` (metric for cost: total_model)

| solver | both | only solver | only baseline | neither | diff (pp) | 95% CI (family bootstrap) | geo-mean cost ratio solver/baseline (common) | 95% CI | n common |
|---|---:|---:|---:|---:|---:|---|---:|---|---:|
| mincon-ip@kkt_pivot_signs=free | 156 | 1 | 0 | 9 | +0.6 | [+0.0, +3.6] | 1.02 | [1.00, 1.02] | 156 |

## Performance profile points (total_model; fraction of problems solved within tau x best)

| tau | mincon-ip@bfgs_rescale=0 | mincon-ip@kkt_pivot_signs=free |
|---:|---:|---:|
| 1 | 0.99 | 0.97 |
| 2 | 0.99 | 0.99 |
| 4 | 0.99 | 0.99 |
| 8 | 0.99 | 0.99 |
| 16 | 0.99 | 1.00 |
| 64 | 0.99 | 1.00 |
| 256 | 0.99 | 1.00 |

## Non-ok records (timeouts, crashes, errors) — all kept as failures above

