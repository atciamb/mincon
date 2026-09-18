# Analysis of C:\Users\andyc\OneDrive\Desktop\python_fmincon\mincon-research\bench\results\abl-phased123\scored-ip-A.jsonl

185 problems, solvers: mincon-ip, mincon-ip@barrier_safeguard=1; duplicates overwritten: 0

## Target attainment (feasible to 1e-6 and objective within the frozen relative tolerance)

| solver | attained | of | rate | KKT (supplied) | KKT (recovered) | timeouts/crashes/errors |
|---|---:|---:|---:|---:|---:|---:|
| mincon-ip | 168 | 179 | 0.939 | 105 | 122 | 0 |
| mincon-ip@barrier_safeguard=1 | 167 | 179 | 0.933 | 104 | 115 | 0 |

## Attainment by family

| family | n | mincon-ip | mincon-ip@barrier_safeguard=1 |
|---|---:|---:|---:|
| adversarial | 18 | 14/15 | 14/15 |
| catenary | 2 | 2/2 | 2/2 |
| chainrosen | 6 | 5/6 | 5/6 |
| covqp | 3 | 2/3 | 2/3 |
| deconv | 2 | 1/2 | 1/2 |
| denselap | 2 | 2/2 | 2/2 |
| dispatch | 2 | 2/2 | 2/2 |
| ellipsoid | 3 | 2/3 | 2/3 |
| ellipsoid2 | 2 | 2/2 | 2/2 |
| engineering | 6 | 6/6 | 6/6 |
| engineering2 | 1 | 1/1 | 1/1 |
| engineering3 | 2 | 2/2 | 2/2 |
| expfit | 3 | 2/2 | 2/2 |
| expfit2 | 2 | 2/2 | 2/2 |
| hs | 93 | 88/93 | 87/93 |
| logistic | 2 | 2/2 | 2/2 |
| logsumexp | 2 | 2/2 | 2/2 |
| lqtraj | 3 | 3/3 | 3/3 |
| maxent | 3 | 3/3 | 3/3 |
| ncboxqp | 2 | 2/2 | 2/2 |
| nnls_simplex | 3 | 3/3 | 3/3 |
| noisyqp | 1 | 1/1 | 1/1 |
| obstacle | 3 | 2/3 | 2/3 |
| pkfit | 2 | 2/2 | 2/2 |
| polyqp | 2 | 2/2 | 2/2 |
| portfolio | 2 | 0/0 | 0/0 |
| quadsphere | 3 | 3/3 | 3/3 |
| quadsphere2 | 2 | 2/2 | 2/2 |
| rosen_sphere | 2 | 2/2 | 2/2 |
| sinfit | 1 | 1/1 | 1/1 |
| snl | 3 | 3/3 | 3/3 |
| tcport | 2 | 2/2 | 2/2 |

## Paired comparison against `mincon-ip` (metric for cost: total_model)

| solver | both | only solver | only baseline | neither | diff (pp) | 95% CI (family bootstrap) | geo-mean cost ratio solver/baseline (common) | 95% CI | n common |
|---|---:|---:|---:|---:|---:|---|---:|---|---:|
| mincon-ip@barrier_safeguard=1 | 166 | 1 | 2 | 10 | -0.6 | [-0.9, +0.0] | 1.18 | [1.14, 1.29] | 166 |

## Performance profile points (total_model; fraction of problems solved within tau x best)

| tau | mincon-ip | mincon-ip@barrier_safeguard=1 |
|---:|---:|---:|
| 1 | 0.88 | 0.38 |
| 2 | 0.99 | 0.92 |
| 4 | 0.99 | 0.99 |
| 8 | 0.99 | 0.99 |
| 16 | 0.99 | 0.99 |
| 64 | 0.99 | 0.99 |
| 256 | 0.99 | 0.99 |

## Non-ok records (timeouts, crashes, errors) — all kept as failures above

