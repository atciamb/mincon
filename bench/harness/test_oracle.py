"""Adversarial tests: the oracle must reject what a flattering scorer would accept."""
import math
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "corpus"))
import numpy as np
import spec, families  # noqa: F401
from oracle import assess

S2 = math.sqrt(2)


def prob(name):
    return spec.get(name).numpy()


def test_hs71_reference_point_is_kkt_with_supplied_multipliers():
    p = prob("HS71")
    x = np.array([1.0, 4.7429996357, 3.8211499974, 1.3794082939])
    # canonical multipliers: row0 (>= 25 lower side active) lam <= 0; row1 (eq); bound x1 = 1 active: zl
    lam = np.array([-0.55229366, 0.16146857]); zl = np.array([1.08787115, 0, 0, 0])
    v = assess(p, x, lam, zl, np.zeros(4), target=17.0140172935, stat_tol=1e-5, feas_tol=1e-6)
    assert v.feasible and v.stationarity < 1e-5, v
    assert v.kkt_first_order and v.kkt_first_order_recovered and v.target_attained, v


def test_nan_point_is_invalid_not_success():
    p = prob("HS71")
    v = assess(p, [float("nan")] * 4, target=17.0)
    assert not v.valid and not v.feasible and v.target_attained is False and v.violation == float("inf")


def test_wrong_size_point_is_invalid():
    v = assess(prob("HS71"), [1.0, 2.0], target=17.0)
    assert not v.valid


def test_infeasible_point_with_better_objective_is_not_target_attained():
    p = prob("HS35")
    x = np.array([2.0, 2.0, 2.0])  # violates 3 - x1 - x2 - 2x3 >= 0
    v = assess(p, x, target=1 / 9)
    assert not v.feasible and v.target_attained is False


def test_feasible_non_stationary_point_fails_kkt_but_may_hit_target_only_if_close():
    p = prob("HS35")
    v = assess(p, p.x0, target=1 / 9)
    assert v.feasible and not v.kkt_first_order_recovered and v.target_attained is False


def test_wrong_sign_multiplier_is_flagged():
    p = prob("HS35")  # x* = (4/3, 7/9, 4/9), row active at lower side (>= 0) so lam <= 0 canonical
    x = np.array([4 / 3, 7 / 9, 4 / 9])
    g = p.grad(x); J = p.jac(x)
    lam_true = -(g @ J[0]) / (J[0] @ J[0])  # least squares on the single active row
    good = assess(p, x, [lam_true], stat_tol=1e-6)
    bad = assess(p, x, [-lam_true], stat_tol=1e-6)
    assert good.dual_sign_violation == 0.0 and good.kkt_first_order
    assert bad.dual_sign_violation > 0 and not bad.kkt_first_order


def test_missing_multipliers_use_recovery():
    p = prob("HS35")
    x = np.array([4 / 3, 7 / 9, 4 / 9])
    v = assess(p, x, stat_tol=1e-6)
    assert not v.multipliers_supplied and v.kkt_first_order_recovered and not v.kkt_first_order


def test_bound_active_solution_recovers_bound_multipliers():
    p = prob("HS45")  # every upper bound active at (1,2,3,4,5)
    v = assess(p, [1, 2, 3, 4, 5], target=1.0)
    assert v.kkt_first_order_recovered and v.target_attained and v.active_bounds == 5


def test_better_than_target_is_noted_for_target_revision():
    p = prob("HS35")
    v = assess(p, [4 / 3, 7 / 9, 4 / 9], target=0.5)
    assert any("beats the frozen target" in n for n in v.notes)


def test_json_safe_dict():
    import json
    v = assess(prob("HS35"), [float("nan")] * 3, target=1.0)
    d = v.as_dict()
    assert d["valid"] is False and math.isinf(d["violation"])
    json.dumps({k: (None if isinstance(x, float) and not math.isfinite(x) else x) for k, x in d.items()})
