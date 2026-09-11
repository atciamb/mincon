# Wall-time comparison mincon / fmincon-sqp (12 problems attained by both in every repeat; 0 excluded)

geo-mean ratio 0.3518  95% family bootstrap [0.1360, 0.6311]; median 0.2567; 90th pct 3.6451; max 7.2662
median wall: mincon 7.89 ms, fmincon-sqp 53.08 ms
callback share of wall (median over runs): mincon 0.82, fmincon-sqp 0.19

| problem | family | n | mincon median s | fmincon-sqp median s | ratio | repeats |
|---|---|---:|---:|---:|---:|---:|
| MAXENT_200 | maxent | 200 | 19.3843 | 2.6677 | 7.2662 | 3/3 |
| NNLS_SIMPLEX_120 | nnls_simplex | 120 | 8.7018 | 2.2773 | 3.8210 | 3/3 |
| NNLS_SIMPLEX_30 | nnls_simplex | 30 | 0.1380 | 0.0669 | 2.0613 | 3/3 |
| MAXENT_50 | maxent | 50 | 0.2012 | 0.1456 | 1.3817 | 3/3 |
| LOGSUMEXP_20 | logsumexp | 20 | 0.0860 | 0.1252 | 0.6868 | 3/3 |
| LOGSUMEXP_10 | logsumexp | 10 | 0.0092 | 0.0266 | 0.3458 | 3/3 |
| ROSEN_SPHERE_10 | rosen_sphere | 10 | 0.0066 | 0.0392 | 0.1677 | 3/3 |
| HS114 | hs | 10 | 0.0047 | 0.0336 | 0.1393 | 3/3 |
| SINFIT_CLEAN | sinfit | 4 | 0.0019 | 0.0237 | 0.0808 | 3/3 |
| NNLS_SIMPLEX_6 | nnls_simplex | 6 | 0.0006 | 0.0099 | 0.0608 | 3/3 |
| ROSEN_SPHERE_4 | rosen_sphere | 4 | 0.0018 | 0.0352 | 0.0512 | 3/3 |
| MAXENT_10 | maxent | 10 | 0.0029 | 0.0893 | 0.0326 | 3/3 |
