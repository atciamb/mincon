# Wall-time comparison mincon / fmincon-sqp (12 problems attained by both in every repeat; 1 excluded)

geo-mean ratio 0.2832  95% family bootstrap [0.1112, 0.7819]; median 0.2348; 90th pct 1.2334; max 4.6710
median wall: mincon 4.14 ms, fmincon-sqp 31.55 ms
callback share of wall (median over runs): mincon 0.80, fmincon-sqp 0.43

| problem | family | n | mincon median s | fmincon-sqp median s | ratio | repeats |
|---|---|---:|---:|---:|---:|---:|
| TCPORT_100 | tcport | 100 | 2.8857 | 0.6178 | 4.6710 | 3/3 |
| DECONV_60 | deconv | 60 | 0.5149 | 0.4000 | 1.2872 | 3/3 |
| NOISYQP_20 | noisyqp | 20 | 0.0275 | 0.0367 | 0.7497 | 3/3 |
| TCPORT_20 | tcport | 20 | 0.0220 | 0.0324 | 0.6781 | 3/3 |
| NCBOXQP_12 | ncboxqp | 12 | 0.0039 | 0.0134 | 0.2880 | 3/3 |
| PKFIT_CLEAN | pkfit | 4 | 0.0044 | 0.0178 | 0.2485 | 3/3 |
| LOGISTIC_CLEAN | logistic | 3 | 0.0024 | 0.0109 | 0.2210 | 3/3 |
| LOGISTIC_NOISY2 | logistic | 3 | 0.0031 | 0.0142 | 0.2204 | 3/3 |
| NCBOXQP_8 | ncboxqp | 8 | 0.0014 | 0.0070 | 0.2083 | 3/3 |
| PKFIT_NOISY2 | pkfit | 4 | 0.0048 | 0.0569 | 0.0840 | 3/3 |
| CORRUGATED_BULKHEAD | engineering3 | 4 | 0.0013 | 0.0307 | 0.0418 | 3/3 |
| I_BEAM | engineering3 | 4 | 0.0017 | 0.0498 | 0.0342 | 3/3 |

Excluded from the ratio (listed, never dropped silently):
- DECONV_200: not attained in every repeat (mincon: 0/3, fmincon-sqp: 3/3)
