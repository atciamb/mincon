"""Two defaults of the SQP member since Phase D, and the options that restore the behaviour
before it, measured on one problem (docs/22 sections 7.16 to 7.18): a
surface design with 19 Fourier coefficients and 1384 linear rows whose first SQP step lands on a
degenerate vertex (20 active rows for 19 variables) that is a saddle point of the Lagrangian.

``zero_step``: at that vertex the QP returns a step whose predicted decrease is 1e-13 on a merit of
47. No line search can verify it, the multipliers are still those of the previous step, and the
member reports a KKT error of 13 and stops at the step tolerance (``'norm'``, the old behaviour).
``'decrease'``, the default, adopts the QP's multipliers there and re-runs the test, which passes,
so the second-order probe runs and the member leaves the saddle.

``saddle_step``: leaving the saddle starts from a trial step of ``max |x|`` and halves it, two
evaluations a trial, nineteen evaluations here (``'scale'``, the old behaviour). ``'linearized'``,
the default, caps the first trial by the distance to the inactive rows along the direction, and
the first trial is accepted.
"""
import numpy as np
import pytest

import mincon

OPTIMUM = -47.45725946
VERTEX = -47.45625263


def design(operator):
    """The objective with its derivative operator as a matrix (``operator=True``) or as
    ``np.gradient``: the same function to rounding, and rounding decides which of the two
    behaviours the default shows."""
    modes, nx, L, hmin, hmax = 9, 345, 0.05, 0.01, 0.05

    def fourier(x):
        cols = [np.ones(x.size)]
        for i in range(1, modes + 1):
            cols += [np.cos(2 * np.pi * i * x / L), np.sin(2 * np.pi * i * x / L)]
        return np.column_stack(cols)

    xs = np.linspace(0., L, nx + 1)
    mesh, dx = fourier(xs), xs[1] - xs[0]
    D = np.gradient(np.eye(nx + 1), dx, axis=0)

    def f(c):
        h = mesh @ (0.01 * c)
        if np.any(h <= 0):
            return float("nan")
        hp = D @ h if operator else np.gradient(h, dx)
        hpp = D @ hp if operator else np.gradient(hp, dx)
        return float(-1400. * (np.mean((1. + 0.02 * hp ** 2) / h) - 2e-9 * np.mean(hpp ** 2)) / 7000.)

    C = fourier(np.linspace(0., L, 2 * (nx + 1), endpoint=False))
    x0 = np.zeros(2 * modes + 1)
    x0[0], x0[-2] = 3., (hmin + (hmax - hmin) / 6) / 0.01
    lb, ub = np.full(x0.size, -4.), np.full(x0.size, 4.)
    lb[0], ub[0] = 1., 5.
    return f, dict(x0=x0, A=np.vstack([C, -C]), lb=lb, ub=ub,
                   b=np.concatenate([np.full(C.shape[0], hmax), np.full(C.shape[0], -hmin)]) / 0.01)


def solve(operator, **options):
    f, kw = design(operator)
    return mincon.fmincon(f, method="sqp", options=options, **kw)


def test_zero_step_decrease_retests_with_the_qp_multipliers_at_a_degenerate_vertex():
    stalled = solve(True, zero_step="norm")
    assert stalled.status == mincon.ExitFlag.STEP_TOLERANCE and not stalled.success
    assert abs(stalled.fun - VERTEX) < 1e-7 and stalled.optimality > 1.0   # the stale multipliers
    fixed = solve(True)
    assert fixed.success and abs(fixed.fun - OPTIMUM) < 1e-7
    assert fixed.nfev <= stalled.nfev + 5
    assert any("left the saddle point" in note for note in fixed.notes)
    same = solve(True, zero_step="decrease")
    assert np.array_equal(fixed.x, same.x) and fixed.nfev == same.nfev
    # where the old rule already passed the test there, the two agree to the last bit
    a, b = solve(False, zero_step="norm"), solve(False)
    assert a.success and np.array_equal(a.x, b.x) and a.nfev == b.nfev
    # and the whole portfolio no longer needs its second member on this problem
    f, kw = design(True)
    auto = mincon.fmincon(f, **kw)
    assert auto.success and any("1 member(s) run" in note for note in auto.notes), auto.notes


def test_saddle_step_linearized_starts_inside_the_inactive_rows():
    full = solve(False, saddle_step="scale")
    capped = solve(False)
    assert full.success and capped.success
    assert abs(full.fun - OPTIMUM) < 1e-7 and abs(capped.fun - OPTIMUM) < 1e-7
    assert capped.nfev <= full.nfev - 10, (capped.nfev, full.nfev)

    def step(result):
        note = [n for n in result.notes if "left the saddle point" in n][0]
        return float(note.split("(step ")[1].split(")")[0])
    assert step(full) < 1e-2 < 3.0          # nine halvings from max |x| = 3
    assert 5e-3 < step(capped) < 2e-2       # the ratio test's first trial


def test_the_two_options_are_checked():
    f, kw = design(False)
    for name in ("zero_step", "saddle_step"):
        with pytest.raises(ValueError, match=name):
            mincon.fmincon(f, options={name: "sometimes"}, **kw)
