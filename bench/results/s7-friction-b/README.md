# Friction audit at the Phase B wheel, with a fifteenth problem (September 18, 2026)

The same audit as `../s7-friction` (fourteen realistic problems, a user's minimal inputs, five
solvers, every returned point judged by the independent oracle), re-run at the wheel of the Phase
B tree (`docs/22_ROUND5_ROBUSTNESS_PLAN.md` section 7.16) with one problem added:
`heatflux_design`, the owner's heat-flux surface design in surrogate form -- 19 Fourier
coefficients, 1384 linear rows keeping the surface in a band, a smooth stand-in for a PDE
objective. Its reference is trust-constr with exact derivatives from the same start, corrected
onto the 20 rows active at the optimum by a minimum-norm step and KKT-verified (stationarity
7e-15): the optimum is a degenerate vertex, 20 active rows for 19 variables, consistent by the
problem's symmetry. The problem is not convex, so the reference is the local optimum from this
start. `summary.md` and `scored.jsonl` are the scored output, `*.jsonl` the raw records,
`matlab_run.log` the MATLAB run. MATLAB and Python definitions agree at x0 to 3.6e-11 relative over
all 30 MATLAB records.

Solvers: mincon at the Phase B tree (`mincon.fmincon`, default everything; wheel sha256
`29ff4239ff5bc8a7`), SciPy SLSQP and trust-constr (defaults), MATLAB R2025b `fmincon`
interior-point and sqp (`Display='off'` only). One run each, single machine.

## Attainment

| solver | attained | change from `s7-friction` |
|---|---|---|
| mincon | **14/15** | 13/14 before; the new problem attained; `wrong_gradient` is still the refusal with the component named |
| fmincon-interior-point | 12/15 | 11/14 before |
| fmincon-sqp | 12/15 | 11/14 before |
| SciPy SLSQP | 11/15 | 10/14 before |
| SciPy trust-constr | 12/15 | 11/14 before |

Every solver attains the new problem, so it separates nobody on attainment. It separates them on
cost and on accuracy:

| solver | evaluations | iterations | gap to the reference | distance to the reference point |
|---|---:|---:|---:|---:|
| mincon | 477 | 16 | 4.2e-8 | 5.2e-8 |
| fmincon-sqp | 120 | -- | 0.0 | 1.8e-2 |
| fmincon-interior-point | 382 | -- | 1.7e-7 | 1.8e-2 |
| SciPy SLSQP | 40 | 2 | 2.1e-5 | 9.1e-3 |
| SciPy trust-constr | 620 | 37 | 2.3e-6 | 1.8e-2 |

Both `fmincon` points and trust-constr's sit at a different vertex with the same objective (the
problem's near-symmetry); SLSQP stops 2.1e-5 above it, inside the audit's 1e-4 bar.

**Why mincon paid 477.** The portfolio's SQP member (first at n = 19) stopped at `StepTolerance`
with f = -47.45625, a gap of 2.1e-5 -- inside the audit's bar, but a `StepTolerance` exit is not
"usable" to the portfolio, so `ip-default` ran next and finished the vertex in 16 iterations and
5.9 s (0.37 s an iteration on 1384 dense rows in the KKT system). Robust, and the most accurate
answer in the table, at 4x fmincon-sqp's evaluations and 12x SLSQP's. This is the cost finding
that goes to Phase D (`docs/23`): the SQP member's active-set QP on a degenerate vertex with many
rows, and whether a step-tolerance point this close should count as usable.

## The fourteen earlier records

Thirteen are identical to the evaluation (`f_model`, iterations, status, objective).
`noisy_simulator` went from 200 to 202 evaluations and 12 to 13 iterations, still `Optimal` and
attained: the facade now passes the exact Jacobian of its linear row, which changes the last
digits of the path. No other record was touched by Phase B's changes (the exact linear Jacobian,
the shared iteration cap when the caller sets one, the counting wrapper, the probe's decline note,
the `limit` field, the option removals).
