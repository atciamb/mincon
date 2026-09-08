"""Public convenience interface, checked against analytical solutions."""
import numpy as np
import pytest
import mincon


def test_just_function_start_and_nonlinear_constraints():
    r = mincon.fmincon(lambda x: np.sum((x-1.)**2), [0., 0.],
                      nonlcon=lambda x: ([x.sum()-1.], []))
    assert r.success, r
    np.testing.assert_allclose(r.x, [0.5, 0.5], atol=1e-6)
    assert abs(r.fun-0.5) < 1e-6
    np.testing.assert_allclose(r.multipliers["ineqnonlin"], [1.], atol=1e-5)


def test_linear_constraints_scalar_bounds_and_args():
    r = mincon.fmincon(lambda x, target: np.sum((x-target)**2), [0.5, 0.5],
                      A=[[1., 0.]], b=[1.], Aeq=[[0., 1.]], beq=[1.],
                      lb=0., ub=4., args=(np.array([2., 3.]),))
    assert r.success, r
    np.testing.assert_allclose(r.x, [1., 1.], atol=1e-6)
    np.testing.assert_allclose(r.multipliers["ineqlin"], [2.], atol=1e-5)
    np.testing.assert_allclose(r.multipliers["eqlin"], [4.], atol=1e-5)


def test_nonlinear_equality_and_empty_linear_pairs():
    r = mincon.fmincon(lambda x: (x[0]-2.)**2, [0.], A=[], b=[], Aeq=[], beq=[], lb=[], ub=[],
                      nonlcon=lambda x: ([], x[0]-0.5))
    assert r.success, r
    assert abs(r.x[0]-0.5) < 1e-6
    np.testing.assert_allclose(r.multipliers["eqnonlin"], [3.], atol=1e-5)


def test_bound_multiplier_and_no_outside_probes():
    def f(x):
        assert 0. <= x[0] <= 1.
        return (x[0]-2.)**2
    r = mincon.fmincon(f, [-3.], lb=0., ub=1.)
    assert r.success, r
    assert abs(r.x[0]-1.) < 1e-6
    np.testing.assert_allclose(r.multipliers["upper"], [2.], atol=1e-5)


@pytest.mark.parametrize("kwargs", [
    {"A": [[1., 2.]], "b": [1.]}, {"A": [[1.]]},
    {"Aeq": [[1.]], "beq": [1., 2.]}, {"lb": [0., 0.]},
    {"options": {"MaxIterations": 10}},
    {"lb": float("nan")}, {"lb": 2., "ub": 1.},
])
def test_invalid_input_is_rejected_before_callbacks(kwargs):
    def never(x):
        pytest.fail("invalid setup evaluated the model")
    with pytest.raises(ValueError):
        mincon.fmincon(never, [0.], **kwargs)


def test_nonlcon_is_called_once_per_point_and_jacobians_cut_evaluations():
    calls = {"n": 0}

    def nonlcon(x):
        calls["n"] += 1
        return ([x[0] * x[1] * x[2] * x[3] * -1 + 25.], [x @ x - 40.])

    def nonlcon_jac(x):
        return ([[-x[1]*x[2]*x[3], -x[0]*x[2]*x[3], -x[0]*x[1]*x[3], -x[0]*x[1]*x[2]]], [2*x])

    f = lambda x: x[0]*x[3]*(x[0]+x[1]+x[2]) + x[2]
    g = lambda x: np.array([x[3]*(2*x[0]+x[1]+x[2]), x[0]*x[3], x[0]*x[3]+1., x[0]*(x[0]+x[1]+x[2])])
    r_fd = mincon.fmincon(f, [1., 5., 5., 1.], lb=1., ub=5., nonlcon=nonlcon, options={"threads": 1})
    assert r_fd.success, r_fd
    # one model call per distinct point: the ineq and eq components share it
    assert calls["n"] <= r_fd.ncev + 2, (calls["n"], r_fd.ncev)
    calls["n"] = 0
    r = mincon.fmincon(f, [1., 5., 5., 1.], lb=1., ub=5., nonlcon=nonlcon, jac=g, nonlcon_jac=nonlcon_jac,
                       options={"threads": 1, "check_derivatives": True})
    assert r.success, r
    np.testing.assert_allclose(r.x, [1., 4.7429996, 3.8211500, 1.3794083], atol=1e-4)
    assert r.ncjev > 0 and r.ncev < r_fd.ncev / 2, (r.ncev, r_fd.ncev)
    assert r.nfev < r_fd.nfev / 2


def test_wrong_constraint_jacobian_is_rejected_when_checked():
    with pytest.raises(RuntimeError, match="FAILED"):
        mincon.fmincon(lambda x: (x[0]-1.)**2 + (x[1]-1.)**2, [0., 0.],
                       nonlcon=lambda x: ([x[0] + x[1] - 1.], []),
                       nonlcon_jac=lambda x: ([[3., 3.]], []),
                       options={"check_derivatives": True})


def test_check_derivatives_without_analytic_derivatives_is_inconclusive_not_silent():
    with pytest.raises(RuntimeError, match="INCONCLUSIVE"):
        mincon.fmincon(lambda x: (x[0]-1.)**2, [0.], options={"check_derivatives": True})
