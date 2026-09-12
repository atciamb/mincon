# Wall-time comparison mincon / fmincon-interior-point (5 problems attained by both in every repeat; 6 excluded)

geo-mean ratio 1.3920  95% family bootstrap [0.5001, 4.7256]; median 1.8661; 90th pct 5.6363; max 7.3753
median wall: mincon 890.19 ms, fmincon-interior-point 477.03 ms
callback share of wall (median over runs): mincon 0.97, fmincon-interior-point 0.36

| problem | family | n | mincon median s | fmincon-interior-point median s | ratio | repeats |
|---|---|---:|---:|---:|---:|---:|
| DENSELAP_250 | denselap | 250 | 61.3661 | 8.3205 | 7.3753 | 3/3 |
| DENSELAP_100 | denselap | 100 | 3.7897 | 1.2516 | 3.0279 | 3/3 |
| SNL_60 | snl | 60 | 0.8902 | 0.4770 | 1.8661 | 3/3 |
| COVQP_30 | covqp | 30 | 0.0754 | 0.1508 | 0.5001 | 3/3 |
| SNL_24 | snl | 24 | 0.1136 | 0.4530 | 0.2508 | 3/3 |

Excluded from the ratio (listed, never dropped silently):
- COVQP_120: not attained in every repeat (mincon: 3/3, fmincon-interior-point: 0/3)
- COVQP_300: not attained in every repeat (mincon: 0/3, fmincon-interior-point: 0/3)
- OBSTACLE_200: not attained in every repeat (mincon: 3/3, fmincon-interior-point: 0/3)
- OBSTACLE_50: not attained in every repeat (mincon: 3/3, fmincon-interior-point: 0/3)
- OBSTACLE_500: not attained in every repeat (mincon: 0/3, fmincon-interior-point: 0/3)
- SNL_150: not attained in every repeat (mincon: 3/3, fmincon-interior-point: 0/3)
