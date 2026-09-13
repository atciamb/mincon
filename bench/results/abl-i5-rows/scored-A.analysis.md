# Analysis of C:\Users\andyc\OneDrive\Desktop\python_fmincon\mincon-research\bench\results\abl-i5-rows\scored-A.jsonl

185 problems, solvers: mincon, mincon@quadratic_rows=values; duplicates overwritten: 0

## Target attainment (feasible to 1e-6 and objective within the frozen relative tolerance)

| solver | attained | of | rate | KKT (supplied) | KKT (recovered) | timeouts/crashes/errors |
|---|---:|---:|---:|---:|---:|---:|
| mincon | 171 | 179 | 0.955 | 92 | 144 | 0 |
| mincon@quadratic_rows=values | 172 | 179 | 0.961 | 94 | 148 | 0 |

## Attainment by family

| family | n | mincon | mincon@quadratic_rows=values |
|---|---:|---:|---:|
| adversarial | 18 | 14/15 | 14/15 |
| catenary | 2 | 2/2 | 2/2 |
| chainrosen | 6 | 5/6 | 5/6 |
| covqp | 3 | 2/3 | 2/3 |
| deconv | 2 | 1/2 | 1/2 |
| denselap | 2 | 2/2 | 2/2 |
| dispatch | 2 | 2/2 | 2/2 |
| ellipsoid | 3 | 2/3 | 3/3 |
| ellipsoid2 | 2 | 2/2 | 2/2 |
| engineering | 6 | 6/6 | 6/6 |
| engineering2 | 1 | 1/1 | 1/1 |
| engineering3 | 2 | 2/2 | 2/2 |
| expfit | 3 | 2/2 | 2/2 |
| expfit2 | 2 | 2/2 | 2/2 |
| hs | 93 | 90/93 | 90/93 |
| logistic | 2 | 2/2 | 2/2 |
| logsumexp | 2 | 2/2 | 2/2 |
| lqtraj | 3 | 3/3 | 3/3 |
| maxent | 3 | 3/3 | 3/3 |
| ncboxqp | 2 | 2/2 | 2/2 |
| nnls_simplex | 3 | 3/3 | 3/3 |
| noisyqp | 1 | 1/1 | 1/1 |
| obstacle | 3 | 3/3 | 3/3 |
| pkfit | 2 | 2/2 | 2/2 |
| polyqp | 2 | 2/2 | 2/2 |
| portfolio | 2 | 0/0 | 0/0 |
| quadsphere | 3 | 3/3 | 3/3 |
| quadsphere2 | 2 | 2/2 | 2/2 |
| rosen_sphere | 2 | 2/2 | 2/2 |
| sinfit | 1 | 1/1 | 1/1 |
| snl | 3 | 3/3 | 3/3 |
| tcport | 2 | 2/2 | 2/2 |

## Paired comparison against `mincon` (metric for cost: total_model)

| solver | both | only solver | only baseline | neither | diff (pp) | 95% CI (family bootstrap) | geo-mean cost ratio solver/baseline (common) | 95% CI | n common |
|---|---:|---:|---:|---:|---:|---|---:|---|---:|
| mincon@quadratic_rows=values | 171 | 1 | 0 | 7 | +0.6 | [+0.0, +3.2] | 0.99 | [0.93, 1.01] | 171 |

## Performance profile points (total_model; fraction of problems solved within tau x best)

| tau | mincon | mincon@quadratic_rows=values |
|---:|---:|---:|
| 1 | 0.95 | 0.94 |
| 2 | 0.99 | 1.00 |
| 4 | 0.99 | 1.00 |
| 8 | 0.99 | 1.00 |
| 16 | 0.99 | 1.00 |
| 64 | 0.99 | 1.00 |
| 256 | 0.99 | 1.00 |

## Non-ok records (timeouts, crashes, errors) — all kept as failures above

