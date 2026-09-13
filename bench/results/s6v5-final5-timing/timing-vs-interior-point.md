# Wall-time comparison mincon / fmincon-interior-point (10 problems attained by both in every repeat; 3 excluded)

geo-mean ratio 0.0499  95% family bootstrap [0.0269, 0.1066]; median 0.0512; 90th pct 0.1769; max 0.3451
median wall: mincon 3.50 ms, fmincon-interior-point 80.03 ms
callback share of wall (median over runs): mincon 0.76, fmincon-interior-point 0.07

| problem | family | n | mincon median s | fmincon-interior-point median s | ratio | repeats |
|---|---|---:|---:|---:|---:|---:|
| NOISYQP_20 | noisyqp | 20 | 0.0275 | 0.0797 | 0.3451 | 3/3 |
| PKFIT_CLEAN | pkfit | 4 | 0.0044 | 0.0279 | 0.1582 | 3/3 |
| NCBOXQP_12 | ncboxqp | 12 | 0.0039 | 0.0685 | 0.0564 | 3/3 |
| NCBOXQP_8 | ncboxqp | 8 | 0.0014 | 0.0279 | 0.0519 | 3/3 |
| PKFIT_NOISY2 | pkfit | 4 | 0.0048 | 0.0927 | 0.0515 | 3/3 |
| TCPORT_20 | tcport | 20 | 0.0220 | 0.4323 | 0.0508 | 3/3 |
| LOGISTIC_NOISY2 | logistic | 3 | 0.0031 | 0.0754 | 0.0416 | 3/3 |
| LOGISTIC_CLEAN | logistic | 3 | 0.0024 | 0.1042 | 0.0232 | 3/3 |
| I_BEAM | engineering3 | 4 | 0.0017 | 0.0804 | 0.0212 | 3/3 |
| CORRUGATED_BULKHEAD | engineering3 | 4 | 0.0013 | 0.1144 | 0.0112 | 3/3 |

Excluded from the ratio (listed, never dropped silently):
- DECONV_200: not attained in every repeat (mincon: 0/3, fmincon-interior-point: 0/3)
- DECONV_60: not attained in every repeat (mincon: 3/3, fmincon-interior-point: 0/3)
- TCPORT_100: not attained in every repeat (mincon: 3/3, fmincon-interior-point: 0/3)
