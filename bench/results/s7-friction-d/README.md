# Friction audit at the Phase D defaults (September 18, 2026)

The audit of `../s7-friction-c` (fifteen realistic problems, a user's minimal inputs, five solvers,
every returned point judged by the independent oracle) re-run after two SQP options became defaults
(`docs/22_ROUND5_ROBUSTNESS_PLAN.md` section 7.18: `zero_step='decrease'`, `saddle_step='linearized'`;
extension module of wheel `c49470ea73886c0a`). MATLAB and Python definitions agree at x0 to 3.6e-11.

mincon attains **14/15** as before (`wrong_gradient` is the refusal with the component named). The
other four solvers' sixty records are identical to `s7-friction-c` in every non-clock field.
Thirteen of mincon's records are identical too. Two changed, both for the better:

| problem | before | now |
|---|---|---|
| `heatflux_design` | 477 evaluations, 16 iterations: the SQP member stopped at `StepTolerance` on a degenerate vertex and `ip-default` finished; 5.9 s; 5e-8 from the reference point | **87 evaluations, 5 iterations**, the SQP member alone, 0.05 s, 2e-13 from the reference point |
| `noisy_simulator` | 202 evaluations, 13 iterations: the quadratic member ended `Acceptable` and a second member ran | **89 evaluations, 7 iterations**, the quadratic member alone, certified at the accuracy of its finite differences; 2.7e-6 from the reference point instead of 2.2e-5 |

On `heatflux_design` the other solvers need 120 (`fmincon-sqp`), 382 (`fmincon-interior-point`), 40
(SLSQP, to a gap of 2.1e-5) and 620 (trust-constr) evaluations.
