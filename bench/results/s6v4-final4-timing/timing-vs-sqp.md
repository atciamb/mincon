# Wall-time comparison mincon / fmincon-sqp (7 problems attained by both in every repeat; 4 excluded)

geo-mean ratio 1.8716  95% family bootstrap [1.3929, 2.3357]; median 1.7302; 90th pct 7.5203; max 13.8929
median wall: mincon 890.19 ms, fmincon-sqp 475.65 ms
callback share of wall (median over runs): mincon 0.97, fmincon-sqp 0.44

| problem | family | n | mincon median s | fmincon-sqp median s | ratio | repeats |
|---|---|---:|---:|---:|---:|---:|
| COVQP_120 | covqp | 120 | 18.1094 | 1.3035 | 13.8929 | 3/3 |
| DENSELAP_100 | denselap | 100 | 3.7897 | 1.1582 | 3.2720 | 3/3 |
| SNL_60 | snl | 60 | 0.8902 | 0.4757 | 1.8715 | 3/3 |
| DENSELAP_250 | denselap | 250 | 61.3661 | 35.4668 | 1.7302 | 3/3 |
| OBSTACLE_50 | obstacle | 50 | 0.1938 | 0.1499 | 1.2927 | 3/3 |
| SNL_24 | snl | 24 | 0.1136 | 0.1017 | 1.1171 | 3/3 |
| COVQP_30 | covqp | 30 | 0.0754 | 0.1993 | 0.3784 | 3/3 |

Excluded from the ratio (listed, never dropped silently):
- COVQP_300: not attained in every repeat (mincon: 0/3, fmincon-sqp: 3/3)
- OBSTACLE_200: not attained in every repeat (mincon: 3/3, fmincon-sqp: 0/3)
- OBSTACLE_500: not attained in every repeat (mincon: 0/3, fmincon-sqp: 0/3)
- SNL_150: not attained in every repeat (mincon: 3/3, fmincon-sqp: 0/3)
