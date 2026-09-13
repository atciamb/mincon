| problem | mincon-fmincon |
|---|---|
| chainrosen20 | Y 2487 |
| odefit | Y 100 |
| portfolio_risk | Y 115 |
| pressure_vessel | Y 61 |
| nan_region | Y 26 |
| hs71 | Y 36 |
| linear_only | Y 78 |
| with_args | Y 103 |
| wrong_gradient | err 11 |
| infeasible_start_far | Y 71 |
| bad_scaling | Y 31 |
| noisy_simulator | Y 200 |
| box_lsq | Y 1232 |
| equality_circle | Y 30 |
| **attained** | **13/14** |

Cell: Y attained (feasible to 1e-6, objective within 1e-4 relative of the reference) or n / err, then model-boundary objective evaluations. `!` marks a false certificate (success reported at a point the oracle does not certify); `k` marks success reported at a point the oracle certifies as first-order stationary but which is not the reference optimum; `?` marks attainment the solver did not report as success.

### mincon-fmincon

| problem | attained | reported | exit | KKT | f_model | c_model | nit | wall s | message |
|---|---|---|---|---|---:|---:|---:|---:|---|
| chainrosen20 | Y gap 6.2e-11 | True | 1 | n | 2487 | 0 | 113 | 0.02 | First-order optimality and constraint tolerances satisfied. |
| odefit | Y gap 2.3e-13 (param err 2.3e-07, ok) | True | 1 | n | 100 | 0 | 16 | 0.47 | First-order optimality and constraint tolerances satisfied. |
| portfolio_risk | Y gap 0.0e+00 | True | 1 | Y | 115 | 133 | 11 | 0.00 | First-order optimality and constraint tolerances satisfied. |
| pressure_vessel | Y gap 5.2e-13 | True | 1 | Y | 61 | 47 | 9 | 0.00 | First-order optimality and constraint tolerances satisfied. |
| nan_region | Y gap -4.9e-11 | True | 1 | Y | 26 | 26 | 5 | 0.00 | First-order optimality and constraint tolerances satisfied. |
| hs71 | Y gap -1.3e-12 | True | 1 | Y | 36 | 42 | 5 | 0.00 | First-order optimality and constraint tolerances satisfied. |
| linear_only | Y gap 1.9e-16 | True | 1 | Y | 78 | 0 | 1 | 0.00 | First-order optimality and constraint tolerances satisfied. |
| with_args | Y gap 8.5e-14 (param err 2.6e-07, ok) | True | 1 | Y | 103 | 0 | 19 | 0.00 | First-order optimality and constraint tolerances satisfied. |
| wrong_gradient | n | None |  | n | 11 | 0 |  | 0.00 | RuntimeError: invalid problem: The supplied derivatives disagree with finite differences a |
| infeasible_start_far | Y gap -2.1e-16 | True | 1 | Y | 71 | 0 | 2 | 0.00 | First-order optimality and constraint tolerances satisfied. |
| bad_scaling | Y gap -2.7e-13 | True | 1 | Y | 31 | 37 | 7 | 0.00 | First-order optimality and constraint tolerances satisfied. |
| noisy_simulator | Y gap -9.8e-10 | True | 1 | n | 200 | 0 | 12 | 0.00 | First-order optimality and constraint tolerances satisfied. |
| box_lsq | Y gap 2.1e-17 | True | 1 | Y | 1232 | 0 | 2 | 0.01 | First-order optimality and constraint tolerances satisfied. |
| equality_circle | Y gap -6.5e-14 | True | 1 | Y | 30 | 38 | 5 | 0.00 | First-order optimality and constraint tolerances satisfied. |

