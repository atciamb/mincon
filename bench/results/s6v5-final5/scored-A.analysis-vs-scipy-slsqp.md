# Analysis of C:\Users\andyc\OneDrive\Desktop\python_fmincon\mincon-research\bench\results\s6v5-final5\scored-A.jsonl

13 problems, solvers: fmincon-interior-point, fmincon-sqp, mincon, mincon-ip, mincon-sqp, scipy-slsqp, scipy-trust-constr; duplicates overwritten: 0

## Target attainment (feasible to 1e-6 and objective within the frozen relative tolerance)

| solver | attained | of | rate | KKT (supplied) | KKT (recovered) | timeouts/crashes/errors |
|---|---:|---:|---:|---:|---:|---:|
| fmincon-interior-point | 10 | 13 | 0.769 | 3 | 2 | 0 |
| fmincon-sqp | 13 | 13 | 1.000 | 5 | 7 | 0 |
| mincon | 12 | 13 | 0.923 | 4 | 8 | 0 |
| mincon-ip | 12 | 13 | 0.923 | 2 | 3 | 0 |
| mincon-sqp | 13 | 13 | 1.000 | 4 | 8 | 0 |
| scipy-slsqp | 13 | 13 | 1.000 | 0 | 3 | 0 |
| scipy-trust-constr | 11 | 13 | 0.846 | 0 | 1 | 1 |

## Attainment by family

| family | n | fmincon-interior-point | fmincon-sqp | mincon | mincon-ip | mincon-sqp | scipy-slsqp | scipy-trust-constr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| deconv | 2 | 0/2 | 2/2 | 1/2 | 1/2 | 2/2 | 2/2 | 1/2 |
| engineering3 | 2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| logistic | 2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| ncboxqp | 2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| noisyqp | 1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| pkfit | 2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| tcport | 2 | 1/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 1/2 |

## Paired comparison against `scipy-slsqp` (metric for cost: total_model)

| solver | both | only solver | only baseline | neither | diff (pp) | 95% CI (family bootstrap) | geo-mean cost ratio solver/baseline (common) | 95% CI | n common |
|---|---:|---:|---:|---:|---:|---|---:|---|---:|
| fmincon-interior-point | 10 | 0 | 3 | 0 | -23.1 | [-53.8, +0.0] | 2.20 | [1.72, 2.91] | 10 |
| fmincon-sqp | 13 | 0 | 0 | 0 | +0.0 | [+0.0, +0.0] | 1.42 | [1.21, 1.73] | 13 |
| mincon | 12 | 0 | 1 | 0 | -7.7 | [-23.1, +0.0] | 1.52 | [1.13, 2.04] | 12 |
| mincon-ip | 12 | 0 | 1 | 0 | -7.7 | [-23.1, +0.0] | 1.88 | [1.44, 2.40] | 12 |
| mincon-sqp | 13 | 0 | 0 | 0 | +0.0 | [+0.0, +0.0] | 1.47 | [1.17, 1.95] | 13 |
| scipy-trust-constr | 11 | 0 | 2 | 0 | -15.4 | [-33.4, +0.0] | 7.33 | [4.18, 18.81] | 11 |

## Performance profile points (total_model; fraction of problems solved within tau x best)

| tau | fmincon-interior-point | fmincon-sqp | mincon | mincon-ip | mincon-sqp | scipy-slsqp | scipy-trust-constr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.00 | 0.00 | 0.08 | 0.00 | 0.15 | 0.92 | 0.00 |
| 2 | 0.38 | 0.92 | 0.77 | 0.54 | 0.77 | 1.00 | 0.00 |
| 4 | 0.77 | 1.00 | 0.92 | 0.85 | 0.92 | 1.00 | 0.23 |
| 8 | 0.77 | 1.00 | 0.92 | 0.92 | 1.00 | 1.00 | 0.62 |
| 16 | 0.77 | 1.00 | 0.92 | 0.92 | 1.00 | 1.00 | 0.69 |
| 64 | 0.77 | 1.00 | 0.92 | 0.92 | 1.00 | 1.00 | 0.77 |
| 256 | 0.77 | 1.00 | 0.92 | 0.92 | 1.00 | 1.00 | 0.85 |

## Non-ok records (timeouts, crashes, errors) — all kept as failures above

- DECONV_200 / scipy-trust-constr: timeout: killed by supervisor after 211s (hard limit 210s)
