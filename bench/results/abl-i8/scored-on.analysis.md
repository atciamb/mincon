# Analysis of ../results/abl-i8/scored-on.jsonl

169 problems, solvers: mincon@bfgs_rescale=0, mincon@scale_variables=true; duplicates overwritten: 0

## Target attainment (feasible to 1e-6 and objective within the frozen relative tolerance)

| solver | attained | of | rate | KKT (supplied) | KKT (recovered) | timeouts/crashes/errors |
|---|---:|---:|---:|---:|---:|---:|
| mincon@bfgs_rescale=0 | 158 | 166 | 0.952 | 88 | 130 | 0 |
| mincon@scale_variables=true | 151 | 166 | 0.910 | 81 | 130 | 0 |

## Attainment by family

| family | n | mincon@bfgs_rescale=0 | mincon@scale_variables=true |
|---|---:|---:|---:|
| adversarial | 15 | 14/15 | 14/15 |
| catenary | 2 | 2/2 | 2/2 |
| chainrosen | 6 | 5/6 | 5/6 |
| covqp | 3 | 2/3 | 1/3 |
| denselap | 2 | 2/2 | 2/2 |
| dispatch | 2 | 2/2 | 2/2 |
| ellipsoid | 3 | 2/3 | 2/3 |
| ellipsoid2 | 2 | 2/2 | 2/2 |
| engineering | 6 | 6/6 | 6/6 |
| engineering2 | 1 | 1/1 | 1/1 |
| expfit | 3 | 2/2 | 2/2 |
| expfit2 | 2 | 2/2 | 2/2 |
| hs | 93 | 90/93 | 88/93 |
| logsumexp | 2 | 2/2 | 2/2 |
| lqtraj | 3 | 3/3 | 3/3 |
| maxent | 3 | 3/3 | 3/3 |
| nnls_simplex | 3 | 3/3 | 3/3 |
| obstacle | 3 | 2/3 | 2/3 |
| polyqp | 2 | 2/2 | 1/2 |
| portfolio | 2 | 0/0 | 0/0 |
| quadsphere | 3 | 3/3 | 2/3 |
| quadsphere2 | 2 | 2/2 | 1/2 |
| rosen_sphere | 2 | 2/2 | 2/2 |
| sinfit | 1 | 1/1 | 1/1 |
| snl | 3 | 3/3 | 2/3 |

## Paired comparison against `mincon@bfgs_rescale=0` (metric for cost: total_model)

| solver | both | only solver | only baseline | neither | diff (pp) | 95% CI (family bootstrap) | geo-mean cost ratio solver/baseline (common) | 95% CI | n common |
|---|---:|---:|---:|---:|---:|---|---:|---|---:|
| mincon@scale_variables=true | 151 | 0 | 7 | 8 | -4.2 | [-11.3, -2.1] | 1.09 | [1.00, 1.25] | 151 |

## Performance profile points (total_model; fraction of problems solved within tau x best)

| tau | mincon@bfgs_rescale=0 | mincon@scale_variables=true |
|---:|---:|---:|
| 1 | 0.73 | 0.61 |
| 2 | 0.98 | 0.89 |
| 4 | 0.99 | 0.92 |
| 8 | 1.00 | 0.94 |
| 16 | 1.00 | 0.95 |
| 64 | 1.00 | 0.96 |
| 256 | 1.00 | 0.96 |

## Non-ok records (timeouts, crashes, errors) — all kept as failures above

