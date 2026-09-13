# Analysis of C:\Users\andyc\OneDrive\Desktop\python_fmincon\mincon-research\bench\results\abl-i5-build\scored-structured.jsonl

172 problems, solvers: mincon@bfgs_rescale=0, mincon@quadratic_build=structured; duplicates overwritten: 0

## Target attainment (feasible to 1e-6 and objective within the frozen relative tolerance)

| solver | attained | of | rate | KKT (supplied) | KKT (recovered) | timeouts/crashes/errors |
|---|---:|---:|---:|---:|---:|---:|
| mincon@bfgs_rescale=0 | 158 | 166 | 0.952 | 88 | 130 | 0 |
| mincon@quadratic_build=structured | 159 | 166 | 0.958 | 88 | 136 | 0 |

## Attainment by family

| family | n | mincon@bfgs_rescale=0 | mincon@quadratic_build=structured |
|---|---:|---:|---:|
| adversarial | 18 | 14/15 | 14/15 |
| catenary | 2 | 2/2 | 2/2 |
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
| hs | 93 | 90/93 | 90/93 |
| logsumexp | 2 | 2/2 | 2/2 |
| lqtraj | 3 | 3/3 | 3/3 |
| maxent | 3 | 3/3 | 3/3 |
| nnls_simplex | 3 | 3/3 | 3/3 |
| obstacle | 3 | 2/3 | 3/3 |
| polyqp | 2 | 2/2 | 2/2 |
| portfolio | 2 | 0/0 | 0/0 |
| quadsphere | 3 | 3/3 | 3/3 |
| quadsphere2 | 2 | 2/2 | 2/2 |
| rosen_sphere | 2 | 2/2 | 2/2 |
| sinfit | 1 | 1/1 | 1/1 |
| snl | 3 | 3/3 | 3/3 |

## Paired comparison against `mincon@bfgs_rescale=0` (metric for cost: total_model)

| solver | both | only solver | only baseline | neither | diff (pp) | 95% CI (family bootstrap) | geo-mean cost ratio solver/baseline (common) | 95% CI | n common |
|---|---:|---:|---:|---:|---:|---|---:|---|---:|
| mincon@quadratic_build=structured | 158 | 1 | 0 | 7 | +0.6 | [+0.0, +3.7] | 0.93 | [0.62, 1.04] | 158 |

## Performance profile points (total_model; fraction of problems solved within tau x best)

| tau | mincon@bfgs_rescale=0 | mincon@quadratic_build=structured |
|---:|---:|---:|
| 1 | 0.84 | 0.16 |
| 2 | 0.94 | 0.98 |
| 4 | 0.94 | 0.99 |
| 8 | 0.97 | 0.99 |
| 16 | 0.99 | 1.00 |
| 64 | 0.99 | 1.00 |
| 256 | 0.99 | 1.00 |

## Non-ok records (timeouts, crashes, errors) — all kept as failures above

