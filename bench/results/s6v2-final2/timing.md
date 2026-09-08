# Wall-time comparison mincon / fmincon-interior-point (12 problems attained by both in every repeat; 4 excluded)

geo-mean ratio 0.0773  95% family bootstrap [0.0316, 0.2337]; median 0.0889; 90th pct 0.2348; max 5.4547
median wall: mincon 5.61 ms, fmincon-interior-point 109.05 ms
callback share of wall (median over runs): mincon 0.53, fmincon-interior-point 0.06

| problem | family | n | mincon median s | fmincon-interior-point median s | ratio | repeats |
|---|---|---:|---:|---:|---:|---:|
| POLYQP_100 | polyqp | 100 | 13.9601 | 2.5593 | 5.4547 | 3/3 |
| POLYQP_10 | polyqp | 10 | 0.0097 | 0.0399 | 0.2437 | 3/3 |
| QUADSPHERE2_30 | quadsphere2 | 30 | 0.0350 | 0.2257 | 0.1550 | 3/3 |
| CATENARY_10 | catenary | 18 | 0.0085 | 0.0658 | 0.1291 | 3/3 |
| DISPATCH_20 | dispatch | 20 | 0.0223 | 0.1808 | 0.1234 | 3/3 |
| ELLIPSOID2_20 | ellipsoid2 | 20 | 0.0309 | 0.2657 | 0.1164 | 3/3 |
| EXPFIT2_CLEAN | expfit2 | 4 | 0.0026 | 0.0417 | 0.0615 | 3/3 |
| HS56 | hs | 7 | 0.0021 | 0.0550 | 0.0382 | 3/3 |
| EXPFIT2_NOISY | expfit2 | 4 | 0.0027 | 0.1312 | 0.0208 | 3/3 |
| TUBULAR_COLUMN | engineering2 | 2 | 0.0008 | 0.0391 | 0.0207 | 3/3 |
| DISPATCH_5 | dispatch | 5 | 0.0014 | 0.0869 | 0.0165 | 3/3 |
| HS84 | hs | 5 | 0.0022 | 0.3032 | 0.0071 | 3/3 |

Excluded from the ratio (listed, never dropped silently):
- CATENARY_40: not attained in every repeat (mincon: 3/3, fmincon-interior-point: 0/3)
- ELLIPSOID2_200: not attained in every repeat (mincon: 3/3, fmincon-interior-point: 0/3)
- HS25: not attained in every repeat (mincon: 0/3, fmincon-interior-point: 0/3)
- QUADSPHERE2_300: not attained in every repeat (mincon: 0/3, fmincon-interior-point: 0/3)
