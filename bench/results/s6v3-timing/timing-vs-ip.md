# Wall-time comparison mincon / fmincon-interior-point (10 problems attained by both in every repeat; 2 excluded)

geo-mean ratio 0.0766  95% family bootstrap [0.0335, 0.1534]; median 0.0631; 90th pct 0.6782; max 0.7218
median wall: mincon 5.63 ms, fmincon-interior-point 142.74 ms
callback share of wall (median over runs): mincon 0.63, fmincon-interior-point 0.09

| problem | family | n | mincon median s | fmincon-interior-point median s | ratio | repeats |
|---|---|---:|---:|---:|---:|---:|
| MAXENT_50 | maxent | 50 | 0.2012 | 0.2787 | 0.7218 | 3/3 |
| NNLS_SIMPLEX_30 | nnls_simplex | 30 | 0.1380 | 0.2050 | 0.6733 | 3/3 |
| LOGSUMEXP_20 | logsumexp | 20 | 0.0860 | 0.2156 | 0.3987 | 3/3 |
| LOGSUMEXP_10 | logsumexp | 10 | 0.0092 | 0.0385 | 0.2393 | 3/3 |
| ROSEN_SPHERE_10 | rosen_sphere | 10 | 0.0066 | 0.0771 | 0.0853 | 3/3 |
| HS114 | hs | 10 | 0.0047 | 0.1144 | 0.0409 | 3/3 |
| NNLS_SIMPLEX_6 | nnls_simplex | 6 | 0.0006 | 0.0301 | 0.0200 | 3/3 |
| MAXENT_10 | maxent | 10 | 0.0029 | 0.1592 | 0.0183 | 3/3 |
| SINFIT_CLEAN | sinfit | 4 | 0.0019 | 0.1263 | 0.0151 | 3/3 |
| ROSEN_SPHERE_4 | rosen_sphere | 4 | 0.0018 | 0.2341 | 0.0077 | 3/3 |

Excluded from the ratio (listed, never dropped silently):
- MAXENT_200: not attained in every repeat (mincon: 3/3, fmincon-interior-point: 0/3)
- NNLS_SIMPLEX_120: not attained in every repeat (mincon: 3/3, fmincon-interior-point: 0/3)
