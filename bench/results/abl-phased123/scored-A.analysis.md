# Analysis of C:\Users\andyc\OneDrive\Desktop\python_fmincon\mincon-research\bench\results\abl-phased123\scored-A.jsonl

185 problems, solvers: mincon, mincon@barrier_safeguard=0.1, mincon@barrier_safeguard=1, mincon@quadratic_bands=decaying; duplicates overwritten: 0

## Target attainment (feasible to 1e-6 and objective within the frozen relative tolerance)

| solver | attained | of | rate | KKT (supplied) | KKT (recovered) | timeouts/crashes/errors |
|---|---:|---:|---:|---:|---:|---:|
| mincon | 172 | 179 | 0.961 | 111 | 148 | 0 |
| mincon@barrier_safeguard=0.1 | 172 | 179 | 0.961 | 111 | 148 | 0 |
| mincon@barrier_safeguard=1 | 172 | 179 | 0.961 | 111 | 148 | 0 |
| mincon@quadratic_bands=decaying | 173 | 179 | 0.966 | 112 | 149 | 0 |

## Attainment by family

| family | n | mincon | mincon@barrier_safeguard=0.1 | mincon@barrier_safeguard=1 | mincon@quadratic_bands=decaying |
|---|---:|---:|---:|---:|---:|
| adversarial | 18 | 14/15 | 14/15 | 14/15 | 14/15 |
| catenary | 2 | 2/2 | 2/2 | 2/2 | 2/2 |
| chainrosen | 6 | 5/6 | 5/6 | 5/6 | 5/6 |
| covqp | 3 | 2/3 | 2/3 | 2/3 | 2/3 |
| deconv | 2 | 1/2 | 1/2 | 1/2 | 2/2 |
| denselap | 2 | 2/2 | 2/2 | 2/2 | 2/2 |
| dispatch | 2 | 2/2 | 2/2 | 2/2 | 2/2 |
| ellipsoid | 3 | 3/3 | 3/3 | 3/3 | 3/3 |
| ellipsoid2 | 2 | 2/2 | 2/2 | 2/2 | 2/2 |
| engineering | 6 | 6/6 | 6/6 | 6/6 | 6/6 |
| engineering2 | 1 | 1/1 | 1/1 | 1/1 | 1/1 |
| engineering3 | 2 | 2/2 | 2/2 | 2/2 | 2/2 |
| expfit | 3 | 2/2 | 2/2 | 2/2 | 2/2 |
| expfit2 | 2 | 2/2 | 2/2 | 2/2 | 2/2 |
| hs | 93 | 90/93 | 90/93 | 90/93 | 90/93 |
| logistic | 2 | 2/2 | 2/2 | 2/2 | 2/2 |
| logsumexp | 2 | 2/2 | 2/2 | 2/2 | 2/2 |
| lqtraj | 3 | 3/3 | 3/3 | 3/3 | 3/3 |
| maxent | 3 | 3/3 | 3/3 | 3/3 | 3/3 |
| ncboxqp | 2 | 2/2 | 2/2 | 2/2 | 2/2 |
| nnls_simplex | 3 | 3/3 | 3/3 | 3/3 | 3/3 |
| noisyqp | 1 | 1/1 | 1/1 | 1/1 | 1/1 |
| obstacle | 3 | 3/3 | 3/3 | 3/3 | 3/3 |
| pkfit | 2 | 2/2 | 2/2 | 2/2 | 2/2 |
| polyqp | 2 | 2/2 | 2/2 | 2/2 | 2/2 |
| portfolio | 2 | 0/0 | 0/0 | 0/0 | 0/0 |
| quadsphere | 3 | 3/3 | 3/3 | 3/3 | 3/3 |
| quadsphere2 | 2 | 2/2 | 2/2 | 2/2 | 2/2 |
| rosen_sphere | 2 | 2/2 | 2/2 | 2/2 | 2/2 |
| sinfit | 1 | 1/1 | 1/1 | 1/1 | 1/1 |
| snl | 3 | 3/3 | 3/3 | 3/3 | 3/3 |
| tcport | 2 | 2/2 | 2/2 | 2/2 | 2/2 |

## Paired comparison against `mincon` (metric for cost: total_model)

| solver | both | only solver | only baseline | neither | diff (pp) | 95% CI (family bootstrap) | geo-mean cost ratio solver/baseline (common) | 95% CI | n common |
|---|---:|---:|---:|---:|---:|---|---:|---|---:|
| mincon@barrier_safeguard=0.1 | 172 | 0 | 0 | 7 | +0.0 | [+0.0, +0.0] | 1.00 | [1.00, 1.00] | 172 |
| mincon@barrier_safeguard=1 | 172 | 0 | 0 | 7 | +0.0 | [+0.0, +0.0] | 1.00 | [1.00, 1.01] | 172 |
| mincon@quadratic_bands=decaying | 172 | 1 | 0 | 6 | +0.6 | [+0.0, +2.7] | 1.00 | [1.00, 1.00] | 172 |

## Performance profile points (total_model; fraction of problems solved within tau x best)

| tau | mincon | mincon@barrier_safeguard=0.1 | mincon@barrier_safeguard=1 | mincon@quadratic_bands=decaying |
|---:|---:|---:|---:|---:|
| 1 | 0.95 | 0.95 | 0.95 | 0.96 |
| 2 | 0.99 | 0.99 | 0.99 | 1.00 |
| 4 | 0.99 | 0.99 | 0.99 | 1.00 |
| 8 | 0.99 | 0.99 | 0.99 | 1.00 |
| 16 | 0.99 | 0.99 | 0.99 | 1.00 |
| 64 | 0.99 | 0.99 | 0.99 | 1.00 |
| 256 | 0.99 | 0.99 | 0.99 | 1.00 |

## Non-ok records (timeouts, crashes, errors) — all kept as failures above

