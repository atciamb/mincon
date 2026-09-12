# Analysis of ../results/abl-i2/scored-sqp.jsonl

172 problems, solvers: mincon-sqp, mincon-sqp@bfgs_rescale=0; duplicates overwritten: 0

## Target attainment (feasible to 1e-6 and objective within the frozen relative tolerance)

| solver | attained | of | rate | KKT (supplied) | KKT (recovered) | timeouts/crashes/errors |
|---|---:|---:|---:|---:|---:|---:|
| mincon-sqp | 154 | 166 | 0.928 | 88 | 138 | 0 |
| mincon-sqp@bfgs_rescale=0 | 154 | 166 | 0.928 | 88 | 138 | 0 |

## Attainment by family

| family | n | mincon-sqp | mincon-sqp@bfgs_rescale=0 |
|---|---:|---:|---:|
| adversarial | 18 | 14/15 | 14/15 |
| catenary | 2 | 1/2 | 1/2 |
| chainrosen | 6 | 5/6 | 5/6 |
| covqp | 3 | 2/3 | 2/3 |
| denselap | 2 | 2/2 | 2/2 |
| dispatch | 2 | 2/2 | 2/2 |
| ellipsoid | 3 | 2/3 | 2/3 |
| ellipsoid2 | 2 | 2/2 | 2/2 |
| engineering | 6 | 6/6 | 6/6 |
| engineering2 | 1 | 1/1 | 1/1 |
| expfit | 3 | 2/2 | 2/2 |
| expfit2 | 2 | 2/2 | 2/2 |
| hs | 93 | 89/93 | 89/93 |
| logsumexp | 2 | 2/2 | 2/2 |
| lqtraj | 3 | 3/3 | 3/3 |
| maxent | 3 | 3/3 | 3/3 |
| nnls_simplex | 3 | 3/3 | 3/3 |
| obstacle | 3 | 2/3 | 2/3 |
| polyqp | 2 | 2/2 | 2/2 |
| portfolio | 2 | 0/0 | 0/0 |
| quadsphere | 3 | 2/3 | 2/3 |
| quadsphere2 | 2 | 1/2 | 1/2 |
| rosen_sphere | 2 | 2/2 | 2/2 |
| sinfit | 1 | 1/1 | 1/1 |
| snl | 3 | 3/3 | 3/3 |

## Paired comparison against `mincon-sqp@bfgs_rescale=0` (metric for cost: total_model)

| solver | both | only solver | only baseline | neither | diff (pp) | 95% CI (family bootstrap) | geo-mean cost ratio solver/baseline (common) | 95% CI | n common |
|---|---:|---:|---:|---:|---:|---|---:|---|---:|
| mincon-sqp | 154 | 0 | 0 | 12 | +0.0 | [+0.0, +0.0] | 1.00 | [1.00, 1.00] | 154 |

## Performance profile points (total_model; fraction of problems solved within tau x best)

| tau | mincon-sqp | mincon-sqp@bfgs_rescale=0 |
|---:|---:|---:|
| 1 | 1.00 | 0.99 |
| 2 | 1.00 | 1.00 |
| 4 | 1.00 | 1.00 |
| 8 | 1.00 | 1.00 |
| 16 | 1.00 | 1.00 |
| 64 | 1.00 | 1.00 |
| 256 | 1.00 | 1.00 |

## Non-ok records (timeouts, crashes, errors) — all kept as failures above

