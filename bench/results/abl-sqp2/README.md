# Ablation SQP-2: the portfolio with the SQP member (candidate C6)

Build: as SQP-1 plus the portfolio routing: SQP first when n ≤ 20
(`SQP_FIRST_MAX_N`), interior point first otherwise, the other as the next
member; a member's answer stops the sequence only when it is `Optimal` or
`Acceptable` and feasible (a step-tolerance point is kept as a candidate but
the next member still runs); `Acceptable` ranks above unverified usable
points when the final answer is chosen. Whole corpus, track A, defaults,
same host and budget as SQP-1. Compared with the I0 build (`abl-i0`, whose
`auto` was the interior-point portfolio).

| | I0 (`auto`, IP portfolio) | C6 (`auto`, SQP + IP portfolio) |
|---|---|---|
| attained (of 143) | 135 | **137** (+HS25, +HS108; no loss) |
| paired evaluation ratio C6 / I0, 135 jointly attained | — | geo-mean **0.775**, median 0.90, bootstrap 95 % [0.711, 0.843] |
| n ≤ 20 (127 problems) | 121 attained | 123 attained; 0.75× evaluations |
| n > 20 (16 problems) | 14 attained | 14 attained; 1.05× evaluations |
| member that produced the answer | — | sqp 128, ip-default 19, ip-cautious 1, ip-unscaled 1 |
| fixture gate (`run_testset auto`) | 55/55 | 55/55 |

Costlier cases: CHAINROSEN_BOX_50 (2.0×: the interior-point member stopped at
the step tolerance with the right answer, so SQP ran as well and certified
it `Optimal`), HS106 (3.5×: SQP first, 1445 evaluations against 414),
QUADSPHERE_10 (2.2×). Cheaper by 3–7×: the vertex and many-constraint
problems (HS32, HS36, HS44, HS83, HS95–98, MANY_INEQ, DEGEN_LICQ, NEARDEP).

Decision (gate G2 of the plan): met on the development material — the
portfolio attains strictly more than either member alone (137 vs 135 and
134) and costs 0.78× the previous candidate. The validation of record is the
round-3 held-out set.
