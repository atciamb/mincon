# S5 increment C5: guarded diagonal initial scaling of the BFGS matrix

Development material: all 146 problems (dev, validation, former final-1 and
former final-2; 143 scorable), targets v3, Linux evaluation counts,
Linux-vs-Linux ablation with the same build (`options={'bfgs_scaling': ...}`).

Mechanism: the unit initial BFGS matrix misjudges the scale of badly scaled
models; the first line search cuts the step hard and every later update
repairs one direction at a time (QUADSPHERE_100: 28k evaluations,
QUADSPHERE_1000: 200k, CHAINROSEN_EQ_200: 129k).

Three variants were measured on the same 146 problems:

| variant | attained / 143 | evaluations vs off [95% CI] | worst losses |
|---|---:|---|---|
| unconditional scalar `B0 = (y'y / s'y) I` (`../s4-bfgs-scaling-rejected`) | 116 (−3) | 1.02 | PORTFOLIO_100 4.4×, HS112 3.3× — rejected |
| guarded scalar (only when the first step was cut below 1/8) | 132 (−1) | 1.04 [0.99, 1.20] | QUADSPHERE2_300 4.6×, HS93 2.6× — rejected |
| **guarded diagonal** `B0 = diag(clamp(y_i/s_i, γ/1e4, 1e4 γ))` | **134 (+1)** | **0.94 [0.69, 1.01]** | HS93 2.6×, HS100 1.5×, PRESSURE_VESSEL 1.4× |

Wins of the kept variant: QUADSPHERE_100 0.18×, QUADSPHERE_1000 0.25×,
CHAINROSEN_EQ_200 0.24×, QUADSPHERE2_300 0.32×, QUADSPHERE2_30 0.45×,
CHAINROSEN_EQ_50 0.52×, HS19/HS37/HS83 0.5–0.6×. A clamp of two orders of
magnitude (first try) lost QUADSPHERE2_300 (cond 1e5) at 1.9×; four orders
keep it. ELLIPSOID2_200 is unchanged (its first step is not cut; the
pathology there is the Lagrangian curvature, not the objective scale — atlas
#15 stays open).

Against fmincon-interior-point on the 130 problems that have Windows records:
1.05× [0.82, 1.07] (C4: 1.06 [0.90, 1.08]); the large-n wins are on problems
fmincon does not attain, so they do not enter the common-problem ratio.
Fixture gate: 55/55, 51 strict Optimal; the 55 tiny fixtures cost 4% more
evaluations in total (4028 vs 3876), the corpus 6% less.
