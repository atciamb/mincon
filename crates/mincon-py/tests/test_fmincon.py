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
