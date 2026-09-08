# Wall-time comparison mincon / fmincon-interior-point (18 problems attained by both in every repeat; 6 excluded)

geo-mean ratio 0.0329  95% family bootstrap [0.0193, 0.0384]; median 0.0275; 90th pct 0.0960; max 0.2707
median wall: mincon 1.73 ms, fmincon-interior-point 60.22 ms
callback share of wall (median over runs): mincon 0.29, fmincon-interior-point 0.05

| problem | family | n | mincon median s | fmincon-interior-point median s | ratio | repeats |
|---|---|---:|---:|---:|---:|---:|
| HS117 | hs | 15 | 0.0094 | 0.0346 | 0.2707 | 3/3 |
| HS96 | hs | 6 | 0.0020 | 0.0196 | 0.1033 | 3/3 |
| HS98 | hs | 6 | 0.0035 | 0.0381 | 0.0928 | 3/3 |
| HS76 | hs | 4 | 0.0009 | 0.0134 | 0.0653 | 3/3 |
| HS79 | hs | 5 | 0.0017 | 0.0298 | 0.0577 | 3/3 |
| HS63 | hs | 3 | 0.0032 | 0.0718 | 0.0442 | 3/3 |
| HS51 | hs | 5 | 0.0008 | 0.0177 | 0.0430 | 3/3 |
| SPEED_REDUCER | engineering | 7 | 0.0023 | 0.0647 | 0.0357 | 3/3 |
| ELLIPSOID_5 | ellipsoid | 5 | 0.0017 | 0.0593 | 0.0292 | 3/3 |
| HS48 | hs | 5 | 0.0013 | 0.0488 | 0.0258 | 3/3 |
| BIGMULT | adversarial | 2 | 0.0004 | 0.0144 | 0.0250 | 3/3 |
| HS93 | hs | 6 | 0.0014 | 0.0684 | 0.0201 | 3/3 |
| HS62 | hs | 3 | 0.0030 | 0.1930 | 0.0154 | 3/3 |
| HS77 | hs | 5 | 0.0021 | 0.1474 | 0.0146 | 3/3 |
| HS75 | hs | 4 | 0.0014 | 0.0980 | 0.0145 | 3/3 |
| HS80 | hs | 5 | 0.0009 | 0.0612 | 0.0143 | 3/3 |
| PRESSURE_VESSEL | engineering | 4 | 0.0009 | 0.0643 | 0.0140 | 3/3 |
| WELDED_BEAM | engineering | 4 | 0.0019 | 0.1490 | 0.0125 | 3/3 |

Excluded from the ratio (listed, never dropped silently):
- ELLIPSOID_50: not attained in every repeat (mincon: 3/3, fmincon-interior-point: 0/3)
- ELLIPSOID_500: not attained in every repeat (mincon: 0/3, fmincon-interior-point: 0/3)
- HS108: not attained in every repeat (mincon: 0/3, fmincon-interior-point: 0/3)
- PORTFOLIO_100: not attained in every repeat (mincon: 0/3, fmincon-interior-point: 0/3)
- PORTFOLIO_20: not attained in every repeat (mincon: 0/3, fmincon-interior-point: 0/3)
- UNITS: not attained in every repeat (mincon: 0/3, fmincon-interior-point: 0/3)
