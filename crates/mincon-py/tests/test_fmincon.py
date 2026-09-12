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


def _double_well(x):
    # Two basins along x[0]: minima near -1 (f = 0.5) and +1 (f = 0), plus a bowl in x[1].
    return float((x[0]**2 - 1.0)**2 + 0.25 * (x[0] - 1.0)**2 + x[1]**2)


def _double_well_grad(x):
    return [4.0 * x[0] * (x[0]**2 - 1.0) + 0.5 * (x[0] - 1.0), 2.0 * x[1]]


def test_scaling_accepts_booleans():
    r_true = mincon.minimize(_double_well, [0.5, 0.5], jac=_double_well_grad, options={"scaling": True})
    r_default = mincon.minimize(_double_well, [0.5, 0.5], jac=_double_well_grad)
    r_false = mincon.minimize(_double_well, [0.5, 0.5], jac=_double_well_grad, options={"scaling": False})
    assert r_true.success and r_false.success
    assert r_true.nit == r_default.nit and r_true.nfev == r_default.nfev  # True keeps the default
    np.testing.assert_allclose(r_false.x, [1.0, 0.0], atol=1e-5)


def test_disp_prints_iteration_table_and_final_line(capsys):
    r = mincon.minimize(_double_well, [0.5, 0.5], jac=_double_well_grad, options={"disp": True})
    out = capsys.readouterr().out
    assert "Iter" in out and "F-count" in out and "mincon (converged" in out
    assert out.count("\n") >= len(r.trace) + 2
    mincon.minimize(_double_well, [0.5, 0.5], jac=_double_well_grad, options={"display": "final"})
    out = capsys.readouterr().out
    assert "Iter" not in out and "mincon (converged" in out
    with pytest.raises(ValueError, match="display"):
        mincon.minimize(_double_well, [0.5, 0.5], options={"display": "loud"})


def test_multipliers_allow_attribute_access():
    r = mincon.fmincon(lambda x: np.sum((x - 1.0)**2), [0.0, 0.0], A=[[1.0, 1.0]], b=[1.0],
                       nonlcon=lambda x: ([], [x[0] - x[1]]))
    assert r.success, r
    assert isinstance(r.multipliers, mincon.Multipliers)
    np.testing.assert_allclose(r.multipliers.ineqlin, r.multipliers["ineqlin"])
    assert r.multipliers.ineqlin[0] > 0.9  # the linear constraint is active with multiplier 1
    with pytest.raises(AttributeError, match="no multiplier group"):
        r.multipliers.nonsense


def test_nonlcon_may_return_its_jacobians():
    calls = {"n": 0}

    def nonlcon4(x):
        calls["n"] += 1
        c = [x[0]**2 + x[1]**2 - 4.0]           # inside the circle of radius 2
        ceq = [x[0] - 2.0 * x[1]]
        return c, ceq, [[2.0 * x[0], 2.0 * x[1]]], [[1.0, -2.0]]

    f = lambda x: -(x[0] + x[1])  # noqa: E731
    jac = lambda x: [-1.0, -1.0]  # noqa: E731
    r4 = mincon.fmincon(f, [0.1, 0.1], nonlcon=nonlcon4, jac=jac, options={"check_derivatives": True})
    r2 = mincon.fmincon(f, [0.1, 0.1], nonlcon=lambda x: nonlcon4(x)[:2], jac=jac)
    assert r4.success and r2.success, (r4, r2)
    expect = [4.0 / np.sqrt(5.0), 2.0 / np.sqrt(5.0)]
    np.testing.assert_allclose(r4.x, expect, atol=1e-5)
    np.testing.assert_allclose(r2.x, expect, atol=1e-5)
    assert r4.ncjev > 0 and r2.ncjev == 0 and r4.ncev < r2.ncev  # the supplied Jacobians replace finite differences
    with pytest.raises(ValueError, match="do not also pass nonlcon_jac"):
        mincon.fmincon(f, [0.1, 0.1], nonlcon=nonlcon4, nonlcon_jac=lambda x: nonlcon4(x)[2:])


def test_multistart_finds_the_lower_basin_and_reports_the_others():
    single = mincon.minimize(_double_well, [-0.8, 0.3], jac=_double_well_grad)
    assert single.success and single.x[0] < 0  # the local method stays in the left basin
    r = mincon.multistart(_double_well, [(-2.0, 2.0), (-1.0, 1.0)], n_starts=8, x0=[-0.8, 0.3],
                          jac=_double_well_grad, seed=1)
    assert r.success and r.x[0] > 0 and abs(r.fun) < 1e-8, r
    assert len(r.starts) == 8 and r.distinct == 2 and r.nfev_total == sum(s.nfev for s in r.starts)
    np.testing.assert_allclose(r.starts[0].x, single.x, atol=1e-6)  # x0 is the first start
    threaded = mincon.multistart(_double_well, [(-2.0, 2.0), (None, None)], n_starts=4, x0=[0.0, 0.0],
                                 jac=_double_well_grad, seed=1, workers=2)
    assert threaded.success and threaded.x[0] > 0
    with pytest.raises(ValueError, match="finite"):
        mincon.multistart(_double_well, [(-2.0, 2.0), (None, None)], n_starts=2)
