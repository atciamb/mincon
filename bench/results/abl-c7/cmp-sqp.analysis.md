# Analysis of ../results/abl-c7/cmp-sqp.jsonl

158 problems, solvers: mincon-sqp, mincon-sqp@bfgs_rescale=0; duplicates overwritten: 0

## Target attainment (feasible to 1e-6 and objective within the frozen relative tolerance)

| solver | attained | of | rate | KKT (supplied) | KKT (recovered) | timeouts/crashes/errors |
|---|---:|---:|---:|---:|---:|---:|
| mincon-sqp | 148 | 155 | 0.955 | 85 | 135 | 0 |
| mincon-sqp@bfgs_rescale=0 | 145 | 155 | 0.935 | 86 | 135 | 0 |

## Attainment by family

| family | n | mincon-sqp | mincon-sqp@bfgs_rescale=0 |
|---|---:|---:|---:|
| adversarial | 15 | 14/15 | 14/15 |
| catenary | 2 | 1/2 | 1/2 |
| chainrosen | 6 | 5/6 | 5/6 |
| dispatch | 2 | 2/2 | 2/2 |
| ellipsoid | 3 | 3/3 | 2/3 |
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
| polyqp | 2 | 2/2 | 2/2 |
| portfolio | 2 | 0/0 | 0/0 |
| quadsphere | 3 | 3/3 | 2/3 |
| quadsphere2 | 2 | 2/2 | 1/2 |
| rosen_sphere | 2 | 2/2 | 2/2 |
| sinfit | 1 | 1/1 | 1/1 |

## Paired comparison against `mincon-sqp@bfgs_rescale=0` (metric for cost: total_model)

| solver | both | only solver | only baseline | neither | diff (pp) | 95% CI (family bootstrap) | geo-mean cost ratio solver/baseline (common) | 95% CI | n common |
|---|---:|---:|---:|---:|---:|---|---:|---|---:|
| mincon-sqp | 145 | 3 | 0 | 7 | +1.9 | [+0.0, +9.6] | 0.88 | [0.58, 0.96] | 145 |

## Performance profile points (total_model; fraction of problems solved within tau x best)

| tau | mincon-sqp | mincon-sqp@bfgs_rescale=0 |
|---:|---:|---:|
| 1 | 0.99 | 0.84 |
| 2 | 1.00 | 0.91 |
| 4 | 1.00 | 0.95 |
| 8 | 1.00 | 0.97 |
| 16 | 1.00 | 0.97 |
| 64 | 1.00 | 0.98 |
| 256 | 1.00 | 0.98 |

## Non-ok records (timeouts, crashes, errors) — all kept as failures above

