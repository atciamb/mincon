| problem | mincon-fmincon |
|---|---|
| chainrosen20 | Y 2483 |
| odefit | Y 96 |
| portfolio_risk | Y 111 |
| pressure_vessel | Y 57 |
| nan_region | Y 22 |
| hs71 | Y 32 |
| linear_only | Y 121 |
| with_args | Y 99 |
| wrong_gradient | err 11 |
| infeasible_start_far | Y 99 |
| bad_scaling | nk 15 |
| noisy_simulator | Y 86 |
| box_lsq | Y 8595 |
| equality_circle | Y 26 |
| **attained** | **12/14** |

Cell: Y attained (feasible to 1e-6, objective within 1e-4 relative of the reference) or n / err, then model-boundary objective evaluations. `!` marks a false certificate (success reported at a point the oracle does not certify); `k` marks success reported at a point the oracle certifies as first-order stationary but which is not the reference optimum; `?` marks attainment the solver did not report as success.

### mincon-fmincon

| problem | attained | reported | exit | KKT | f_model | c_model | nit | wall s | message |
|---|---|---|---|---|---:|---:|---:|---:|---|
| chainrosen20 | Y gap 6.2e-11 | True | 1 | n | 2483 | 0 | 113 | 0.02 | First-order optimality and constraint tolerances satisfied. |
| odefit | Y gap 2.3e-13 (param err 2.3e-07, ok) | True | 1 | n | 96 | 0 | 16 | 0.55 | First-order optimality and constraint tolerances satisfied. |
| portfolio_risk | Y gap 0.0e+00 | True | 1 | Y | 111 | 129 | 11 | 0.01 | First-order optimality and constraint tolerances satisfied. |
| pressure_vessel | Y gap 5.2e-13 | True | 1 | Y | 57 | 47 | 9 | 0.00 | First-order optimality and constraint tolerances satisfied. |
| nan_region | Y gap -4.9e-11 | True | 1 | Y | 22 | 26 | 5 | 0.00 | First-order optimality and constraint tolerances satisfied. |
| hs71 | Y gap -1.3e-12 | True | 1 | Y | 32 | 42 | 5 | 0.00 | First-order optimality and constraint tolerances satisfied. |
| linear_only | Y gap -1.9e-16 | True | 1 | Y | 121 | 0 | 11 | 0.00 | First-order optimality and constraint tolerances satisfied. |
| with_args | Y gap 8.5e-14 (param err 2.6e-07, ok) | True | 1 | Y | 99 | 0 | 19 | 0.00 | First-order optimality and constraint tolerances satisfied. |
| wrong_gradient | n | None |  | n | 11 | 0 |  | 0.00 | RuntimeError: invalid problem: The supplied derivatives disagree with finite differences a |
| infeasible_start_far | Y gap -4.3e-12 | True | 1 | Y | 99 | 0 | 7 | 0.00 | First-order optimality and constraint tolerances satisfied. |
| bad_scaling | n gap 1.6e+01 | True | 1 | Y | 15 | 21 | 9 | 0.00 | First-order optimality and constraint tolerances satisfied. |
| noisy_simulator | Y gap -9.8e-10 | True | 1 | n | 86 | 0 | 12 | 0.00 | First-order optimality and constraint tolerances satisfied. |
| box_lsq | Y gap 4.2e-06 | True | 1 | n | 8595 | 0 | 166 | 0.08 | First-order optimality and constraint tolerances satisfied. |
| equality_circle | Y gap -6.5e-14 | True | 1 | Y | 26 | 34 | 5 | 0.00 | First-order optimality and constraint tolerances satisfied. |

