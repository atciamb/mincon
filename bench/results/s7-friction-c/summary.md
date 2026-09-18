| problem | fmincon-interior-point | fmincon-sqp | mincon-fmincon | scipy-slsqp | scipy-trust-constr |
|---|---|---|---|---|---|
| chainrosen20 | Y? 3014 | n 2014 | Y 2487 | n 2188 | Y 8001 |
| odefit | Y 160 | Y 135 | Y 100 | Y 104 | Y 376 |
| portfolio_risk | Y 250 | Y 111 | Y 60 | Y 99 | Y 153 |
| pressure_vessel | Y 63 | Y 32 | Y 61 | n 105 | Y 655 |
| nan_region | Y 32 | Y 18 | Y 26 | Y 31 | n! 21 |
| hs71 | Y 46 | Y 30 | Y 36 | Y 25 | Y 650 |
| linear_only | Y 142 | Y 92 | Y 78 | Y 61 | Y 175 |
| with_args | Y 86 | Y 77 | Y 103 | Y 61 | Y 100 |
| wrong_gradient | n! 45 | n! 49 | err 11 | n! 416 | n! 407 |
| infeasible_start_far | Y 199 | Y 33 | Y 71 | Y 33 | Y 462 |
| bad_scaling | nk 20 | n 11 | Y 31 | n 3 | nk 84 |
| noisy_simulator | Y 183 | Y 68 | Y 202 | Y 28 | Y 255 |
| box_lsq | n 3009 | Y? 5049 | Y 1232 | Y 3213 | Y 7803 |
| equality_circle | Y 44 | Y 24 | Y 30 | Y 20 | Y 44 |
| heatflux_design | Y 382 | Y 120 | Y 477 | Y 40 | Y 620 |
| **attained** | **12/15** | **12/15** | **14/15** | **11/15** | **12/15** |

Cell: Y attained (feasible to 1e-6, objective within 1e-4 relative of the reference) or n / err, then model-boundary objective evaluations. `!` marks a false certificate (success reported at a point the oracle does not certify); `k` marks success reported at a point the oracle certifies as first-order stationary but which is not the reference optimum; `?` marks attainment the solver did not report as success.

### fmincon-interior-point

| problem | attained | reported | exit | KKT | f_model | c_model | nit | wall s | message |
|---|---|---|---|---|---:|---:|---:|---:|---|
| chainrosen20 | Y gap 1.1e-06 | False | 0 | n | 3014 | 0 | 133 | 1.23 | Solver stopped prematurely. |
| odefit | Y gap 1.4e-13 (param err 1.7e-07, ok) | True | 1 | n | 160 | 0 | 35 | 0.19 | Local minimum found that satisfies the constraints. |
| portfolio_risk | Y gap 2.2e-06 | True | 1 | n | 250 | 250 | 26 | 0.15 | Local minimum found that satisfies the constraints. |
| pressure_vessel | Y gap 1.2e-08 | True | 1 | n | 63 | 63 | 7 | 0.16 | Local minimum found that satisfies the constraints. |
| nan_region | Y gap 2.0e-08 | True | 1 | Y | 32 | 32 | 9 | 0.18 | Local minimum found that satisfies the constraints. |
| hs71 | Y gap 2.4e-07 | True | 1 | Y | 46 | 46 | 8 | 0.13 | Local minimum found that satisfies the constraints. |
| linear_only | Y gap 1.2e-12 | True | 1 | n | 142 | 0 | 19 | 0.07 | Local minimum found that satisfies the constraints. |
| with_args | Y gap 5.1e-14 (param err 1.9e-07, ok) | True | 1 | Y | 86 | 0 | 18 | 0.10 | Local minimum found that satisfies the constraints. |
| wrong_gradient | n gap 1.0e+00 | True | 2 | n | 45 | 0 | 2 | 0.03 | Local minimum possible. Constraints satisfied. |
| infeasible_start_far | Y gap 7.8e-07 | True | 1 | n | 199 | 0 | 17 | 0.06 | Local minimum found that satisfies the constraints. |
| bad_scaling | n gap 1.6e+01 | True | 1 | Y | 20 | 20 | 5 | 0.05 | Local minimum found that satisfies the constraints. |
| noisy_simulator | Y gap 3.2e-07 | True | 1 | n | 183 | 0 | 33 | 0.09 | Local minimum found that satisfies the constraints. |
| box_lsq | n gap 1.3e-03 | False | 0 | n | 3009 | 0 | 58 | 0.11 | Solver stopped prematurely. |
| equality_circle | Y gap -1.4e-07 | True | 1 | n | 44 | 44 | 10 | 0.10 | Local minimum found that satisfies the constraints. |
| heatflux_design | Y gap 1.7e-07 | True | 1 | Y | 382 | 0 | 18 | 0.26 | Local minimum found that satisfies the constraints. |

### fmincon-sqp

| problem | attained | reported | exit | KKT | f_model | c_model | nit | wall s | message |
|---|---|---|---|---|---:|---:|---:|---:|---|
| chainrosen20 | n gap 4.5e+00 | False | 0 | n | 2014 | 0 | 86 | 0.15 | Solver stopped prematurely. |
| odefit | Y gap 2.0e-13 (param err 2.9e-07, ok) | True | 2 | n | 135 | 0 | 28 | 0.06 | Local minimum possible. Constraints satisfied. |
| portfolio_risk | Y gap -1.4e-17 | True | 1 | Y | 111 | 111 | 11 | 0.02 | Local minimum found that satisfies the constraints. |
| pressure_vessel | Y gap 1.5e-16 | True | 1 | Y | 32 | 32 | 5 | 0.02 | Local minimum found that satisfies the constraints. |
| nan_region | Y gap -6.1e-10 | True | 1 | Y | 18 | 18 | 4 | 0.02 | Local minimum found that satisfies the constraints. |
| hs71 | Y gap -1.3e-12 | True | 1 | Y | 30 | 30 | 5 | 0.01 | Local minimum found that satisfies the constraints. |
| linear_only | Y gap 4.5e-14 | True | 1 | Y | 92 | 0 | 11 | 0.01 | Local minimum found that satisfies the constraints. |
| with_args | Y gap 8.7e-14 (param err 2.8e-07, ok) | True | 1 | Y | 77 | 0 | 14 | 0.01 | Local minimum found that satisfies the constraints. |
| wrong_gradient | n gap 1.5e+00 | True | 2 | n | 49 | 0 | 2 | 0.01 | Local minimum possible. Constraints satisfied. |
| infeasible_start_far | Y gap 0.0e+00 | True | 1 | Y | 33 | 0 | 2 | 0.01 | Local minimum found that satisfies the constraints. |
| bad_scaling | n gap 2.8e-01 | False | -2 | n | 11 | 11 | 1 | 0.00 | Converged to an infeasible point. |
| noisy_simulator | Y gap 4.2e-09 | True | 2 | n | 68 | 0 | 11 | 0.01 | Local minimum possible. Constraints satisfied. |
| box_lsq | Y gap 2.5e-05 | False | 0 | n | 5049 | 0 | 98 | 0.13 | Solver stopped prematurely. |
| equality_circle | Y gap -6.5e-14 | True | 1 | Y | 24 | 24 | 5 | 0.01 | Local minimum found that satisfies the constraints. |
| heatflux_design | Y gap 0.0e+00 | True | 1 | Y | 120 | 0 | 5 | 0.05 | Local minimum found that satisfies the constraints. |

### mincon-fmincon

| problem | attained | reported | exit | KKT | f_model | c_model | nit | wall s | message |
|---|---|---|---|---|---:|---:|---:|---:|---|
| chainrosen20 | Y gap 6.2e-11 | True | 1 | n | 2487 | 0 | 113 | 0.02 | First-order optimality and constraint tolerances satisfied. |
| odefit | Y gap 2.3e-13 (param err 2.3e-07, ok) | True | 1 | n | 100 | 0 | 16 | 0.46 | First-order optimality and constraint tolerances satisfied. |
| portfolio_risk | Y gap -3.5e-10 | True | 1 | Y | 60 | 106 | 3 | 0.00 | First-order optimality and constraint tolerances satisfied. |
| pressure_vessel | Y gap 5.2e-13 | True | 1 | Y | 61 | 47 | 9 | 0.00 | First-order optimality and constraint tolerances satisfied. |
| nan_region | Y gap -4.9e-11 | True | 1 | Y | 26 | 26 | 5 | 0.00 | First-order optimality and constraint tolerances satisfied. |
| hs71 | Y gap -1.3e-12 | True | 1 | Y | 36 | 42 | 5 | 0.00 | First-order optimality and constraint tolerances satisfied. |
| linear_only | Y gap 1.9e-16 | True | 1 | Y | 78 | 0 | 1 | 0.00 | First-order optimality and constraint tolerances satisfied. |
| with_args | Y gap 8.5e-14 (param err 2.6e-07, ok) | True | 1 | Y | 103 | 0 | 19 | 0.00 | First-order optimality and constraint tolerances satisfied. |
| wrong_gradient | n | None |  | n | 11 | 0 |  | 0.00 | RuntimeError: invalid problem: The supplied derivatives disagree with finite differences a |
| infeasible_start_far | Y gap -2.1e-16 | True | 1 | Y | 71 | 0 | 2 | 0.00 | First-order optimality and constraint tolerances satisfied. |
| bad_scaling | Y gap -2.7e-13 | True | 1 | Y | 31 | 37 | 7 | 0.00 | First-order optimality and constraint tolerances satisfied. |
| noisy_simulator | Y gap 4.8e-10 | True | 1 | n | 202 | 0 | 13 | 0.00 | First-order optimality and constraint tolerances satisfied. |
| box_lsq | Y gap 2.1e-17 | True | 1 | Y | 1232 | 0 | 2 | 0.01 | First-order optimality and constraint tolerances satisfied. |
| equality_circle | Y gap -6.5e-14 | True | 1 | Y | 30 | 38 | 5 | 0.00 | First-order optimality and constraint tolerances satisfied. |
| heatflux_design | Y gap 4.2e-08 | True | 1 | Y | 477 | 0 | 16 | 6.21 | First-order optimality and constraint tolerances satisfied. |

### scipy-slsqp

| problem | attained | reported | exit | KKT | f_model | c_model | nit | wall s | message |
|---|---|---|---|---|---:|---:|---:|---:|---|
| chainrosen20 | n gap 4.0e+00 | False | 9 | n | 2188 | 0 | 100 | 0.04 | Iteration limit reached |
| odefit | Y gap 3.2e-08 (param err 8.9e-05, ok) | True | 0 | n | 104 | 0 | 23 | 0.44 | Optimization terminated successfully |
| portfolio_risk | Y gap -4.4e-08 | True | 0 | Y | 99 | 100 | 11 | 0.01 | Optimization terminated successfully |
| pressure_vessel | n gap -1.3e-02 | False | 8 | n | 105 | 106 | 16 | 0.00 | Positive directional derivative for linesearch |
| nan_region | Y gap -4.4e-09 | True | 0 | Y | 31 | 32 | 9 | 0.00 | Optimization terminated successfully |
| hs71 | Y gap -2.6e-09 | True | 0 | Y | 25 | 51 | 5 | 0.00 | Optimization terminated successfully |
| linear_only | Y gap 9.0e-10 | True | 0 | n | 61 | 0 | 8 | 0.00 | Optimization terminated successfully |
| with_args | Y gap 1.9e-08 (param err 5.0e-05, ok) | True | 0 | n | 61 | 0 | 14 | 0.00 | Optimization terminated successfully |
| wrong_gradient | n gap 1.3e+00 | True | 0 | n | 416 | 0 | 45 | 0.01 | Optimization terminated successfully |
| infeasible_start_far | Y gap -1.9e-15 | True | 0 | Y | 33 | 0 | 3 | 0.00 | Optimization terminated successfully |
| bad_scaling | n gap 2.8e-01 | False | 8 | n | 3 | 4 | 5 | 0.00 | Positive directional derivative for linesearch |
| noisy_simulator | Y gap 6.4e-09 | True | 0 | n | 28 | 0 | 5 | 0.00 | Optimization terminated successfully |
| box_lsq | Y gap 9.8e-05 | True | 0 | n | 3213 | 0 | 63 | 0.07 | Optimization terminated successfully |
| equality_circle | Y gap -3.4e-07 | True | 0 | Y | 20 | 41 | 5 | 0.00 | Optimization terminated successfully |
| heatflux_design | Y gap 2.1e-05 | True | 0 | Y | 40 | 0 | 2 | 0.01 | Optimization terminated successfully |

### scipy-trust-constr

| problem | attained | reported | exit | KKT | f_model | c_model | nit | wall s | message |
|---|---|---|---|---|---:|---:|---:|---:|---|
| chainrosen20 | Y gap 6.2e-11 | True | 1 | n | 8001 | 0 | 326 | 0.72 | `gtol` termination condition is satisfied. |
| odefit | Y gap 6.5e-15 (param err 1.8e-08, ok) | True | 2 | n | 376 | 0 | 98 | 2.32 | `xtol` termination condition is satisfied. |
| portfolio_risk | Y gap 4.3e-05 | True | 1 | n | 153 | 153 | 22 | 0.03 | `gtol` termination condition is satisfied. |
| pressure_vessel | Y gap 1.9e-07 | True | 1 | n | 655 | 655 | 74 | 0.10 | `gtol` termination condition is satisfied. |
| nan_region | n gap 1.6e-04 | True | 1 | n | 21 | 21 | 11 | 0.01 | `gtol` termination condition is satisfied. |
| hs71 | Y gap 1.5e-07 | True | 1 | Y | 650 | 650 | 136 | 0.14 | `gtol` termination condition is satisfied. |
| linear_only | Y gap 1.8e-14 | True | 1 | Y | 175 | 0 | 34 | 0.03 | `gtol` termination condition is satisfied. |
| with_args | Y gap 1.8e-14 (param err 1.2e-07, ok) | True | 1 | Y | 100 | 0 | 33 | 0.04 | `gtol` termination condition is satisfied. |
| wrong_gradient | n gap 1.1e+00 | True | 2 | n | 407 | 0 | 339 | 0.14 | `xtol` termination condition is satisfied. |
| infeasible_start_far | Y gap 2.7e-06 | True | 1 | n | 462 | 0 | 49 | 0.05 | `gtol` termination condition is satisfied. |
| bad_scaling | n gap 1.6e+01 | True | 2 | Y | 84 | 84 | 106 | 0.06 | `xtol` termination condition is satisfied. |
| noisy_simulator | Y gap 5.2e-08 | True | 1 | n | 255 | 0 | 47 | 0.04 | `gtol` termination condition is satisfied. |
| box_lsq | Y gap 2.2e-06 | True | 1 | n | 7803 | 0 | 162 | 0.39 | `gtol` termination condition is satisfied. |
| equality_circle | Y gap -1.7e-09 | True | 1 | n | 44 | 44 | 18 | 0.01 | `gtol` termination condition is satisfied. |
| heatflux_design | Y gap 2.3e-06 | True | 1 | Y | 620 | 0 | 37 | 24.61 | `gtol` termination condition is satisfied. |

MATLAB/Python definition agreement at x0 over 30 records: worst relative difference 3.6e-11; records above 1e-8: none.

