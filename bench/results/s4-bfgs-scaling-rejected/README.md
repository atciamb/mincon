# Rejected increment: BFGS initial-matrix scaling

Hypothesis (from ELLIPSOID_50, 125 iterations with exact derivatives on a
diagonal-Hessian convex problem): the unit initial BFGS matrix forces the
line search to shrink early steps, so rescaling `B0 = (y^T y / s^T y) I`
from the first curvature pair (Nocedal–Wright eq. 6.20) should cut
iterations.

Measured on all 127 scorable problems (dev + validation + former final, now
all development material; Linux evaluation counts against the Windows C2
records): attainment 116 vs 119 (lost CHAINROSEN_EQ_200, HS97 and HS44 — the
last to its documented local minimum), geometric-mean evaluations 1.02× C2
overall with large swings both ways: CHAINROSEN_EQ_200 0.17×, HS100 0.42×,
HS62 0.50× but PORTFOLIO_100 4.4×, HS112 3.3×, HS93 2.6×, HS37 2.3×, HS13
2.1×. The fixture gate dropped to 53/54 (HS44 basin change). Not a net
improvement; reverted. The single-instance diagnosis was right about the
mechanism (ELLIPSOID_50 fell from 128 to about 40 iterations) but wrong about
its generality: a scaled `B0` is a different starting model, not a uniformly
better one. Kept as evidence against re-trying it without a safeguard (e.g.
scaling only when the first step is cut back by the line search).
