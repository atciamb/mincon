"""Wheel-level regressions with objectives and constraints checked independently."""
import numpy as np

from mincon import ExitFlag, check_gradients, minimize


def test_hs71_without_derivatives():
    f = lambda x: x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]
    r = minimize(f, [1., 5., 5., 1.], bounds=[(1, 5)] * 4, constraints=[
        {"type": "ineq", "fun": lambda x: np.prod(x) - 25.},
        {"type": "eq", "fun": lambda x: 40. - x @ x},
    ])
    assert r.success, r
    assert abs(f(r.x) - 17.0140173) < 1e-5
    assert abs(r.x @ r.x - 40.) < 1e-6
    assert np.prod(r.x) >= 25. - 1e-6
    assert np.all((r.x >= 1.) & (r.x <= 5.))


def test_inconsistent_equalities_return_restoration_diagnostic():
    r = minimize(lambda x: x @ x, [0., 0.], method="interior-point", constraints=[
        {"type": "eq", "fun": lambda x: x[0] + x[1] - 1.},
        {"type": "eq", "fun": lambda x: x[0] + x[1] - 3.},
    ])
    assert not r.success
    assert r.status == ExitFlag.LOCALLY_INFEASIBLE, r
    assert max(abs(r.x.sum() - 1.), abs(r.x.sum() - 3.)) >= 1.


def test_strict_box_includes_python_constraint_size_probe():
    seen = []

    def domain(x):
        assert 0. <= x[0] <= 1.
        seen.append(x[0])
        return x[0]

    r = minimize(domain, [-2.], bounds=[(0., 1.)], method="interior-point",
                 constraints={"type": "ineq", "fun": domain})
    assert seen
    assert 0. <= r.x[0] < 1e-6


def test_degenerate_hs13_does_not_claim_kkt_success():
    f = lambda x: (x[0] - 2.)**2 + x[1]**2
    r = minimize(f, [-2., -2.], bounds=[(0., None), (0., None)],
                 method="interior-point", constraints={
                     "type": "ineq", "fun": lambda x: (1. - x[0])**3 - x[1]})
    assert abs(f(r.x) - 1.) < 1e-4, r
    assert (1. - r.x[0])**3 - r.x[1] >= -1e-6
    assert not r.success
    assert r.status in (ExitFlag.ACCEPTABLE, ExitFlag.STEP_TOLERANCE)


def test_scipy_inequality_multiplier_sign_and_objective_count():
    calls = []

    def f(x):
        calls.append(x[0])
        return (x[0] - 2.)**2

    r = minimize(f, [0.], method="interior-point",
                 constraints={"type": "ineq", "fun": lambda x: 1. - x[0]})
    assert abs(r.x[0] - 1.) < 1e-6
    # grad f + J^T lambda = 0, J=-1, grad f=-2.
    assert abs(r["lambda"][0] + 2.) < 1e-4
    assert r.nfev == len(calls)


def test_wrong_gradient_is_detected():
    result = check_gradients(lambda x: x[0]**3, [2.], lambda x: np.array([3*x[0]]))
    assert not result.passed


def test_correct_gradient_survives_asymmetric_domain_retreat():
    rejected = []

    def f(x):
        if not -2e-6 <= x[0] <= 1e-6:
            rejected.append(x[0])
            return np.nan
        return 3. * x[0] + 5. * x[0]**2

    r = check_gradients(f, [0.], lambda x: np.array([3. + 10. * x[0]]),
                        num_points=1, tol=1e-10)
    assert rejected  # Both nominal central probes must retreat.
    assert r.passed, r
