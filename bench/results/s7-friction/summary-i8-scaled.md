| problem | mincon-fmincon |
|---|---|
| chainrosen20 | Y 2411 |
| odefit | Y 108 |
| portfolio_risk | Y 247 |
| pressure_vessel | Y 128 |
| nan_region | Y 22 |
| hs71 | Y 42 |
| linear_only | Y 124 |
| with_args | Y 99 |
| wrong_gradient | err 11 |
| infeasible_start_far | Y 62 |
| bad_scaling | Y 27 |
| noisy_simulator | Y 112 |
| box_lsq | Y 12677 |
| equality_circle | Y 22 |
| **attained** | **13/14** |

Cell: Y attained (feasible to 1e-6, objective within 1e-4 relative of the reference) or n / err, then model-boundary objective evaluations. `!` marks a false certificate (success reported at a point the oracle does not certify); `k` marks success reported at a point the oracle certifies as first-order stationary but which is not the reference optimum; `?` marks attainment the solver did not report as success.

### mincon-fmincon

| problem | attained | reported | exit | KKT | f_model | c_model | nit | wall s | message |
|---|---|---|---|---|---:|---:|---:|---:|---|
| chainrosen20 | Y gap 7.6e-11 | True | 1 | n | 2411 | 0 | 110 | 0.02 | First-order optimality and constraint tolerances satisfied. |
| odefit | Y gap 2.1e-14 (param err 7.4e-08, ok) | True | 1 | n | 108 | 0 | 20 | 0.57 | First-order optimality and constraint tolerances satisfied. |
| portfolio_risk | Y gap 1.4e-17 | True | 1 | Y | 247 | 265 | 26 | 0.01 | First-order optimality and constraint tolerances satisfied. |
| pressure_vessel | Y gap 4.5e-08 | True | 1 | n | 128 | 98 | 14 | 0.00 | First-order optimality and constraint tolerances satisfied. |
| nan_region | Y gap -4.9e-11 | True | 1 | Y | 22 | 26 | 5 | 0.00 | First-order optimality and constraint tolerances satisfied. |
| hs71 | Y gap -1.3e-13 | True | 1 | Y | 42 | 52 | 7 | 0.00 | First-order optimality and constraint tolerances satisfied. |
| linear_only | Y gap 2.1e-14 | True | 1 | Y | 124 | 0 | 12 | 0.00 | First-order optimality and constraint tolerances satisfied. |
| with_args | Y gap 8.5e-14 (param err 2.6e-07, ok) | True | 1 | Y | 99 | 0 | 19 | 0.00 | First-order optimality and constraint tolerances satisfied. |
| wrong_gradient | n | None |  | n | 11 | 0 |  | 0.00 | RuntimeError: invalid problem: The supplied derivatives disagree with finite differences a |
| infeasible_start_far | Y gap -2.1e-10 | True | 1 | Y | 62 | 0 | 3 | 0.00 | First-order optimality and constraint tolerances satisfied. |
| bad_scaling | Y gap -2.7e-13 | True | 1 | Y | 27 | 33 | 7 | 0.00 | First-order optimality and constraint tolerances satisfied. |
| noisy_simulator | Y gap 2.9e-09 | True | 1 | n | 112 | 0 | 17 | 0.00 | First-order optimality and constraint tolerances satisfied. |
| box_lsq | Y gap 4.2e-06 | True | 1 | n | 12677 | 0 | 246 | 0.10 | First-order optimality and constraint tolerances satisfied. |
| equality_circle | Y gap -3.4e-07 | True | 1 | Y | 22 | 30 | 4 | 0.00 | First-order optimality and constraint tolerances satisfied. |

