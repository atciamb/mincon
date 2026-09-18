"""``quadratic_bands``: how far the quadratic probe's band search goes when the dense Hessian build
does not fit the budget (docs/22 section 7.19).

The case is a bounded deconvolution, ``0.5 |K x - y|^2`` in a box with a Gaussian kernel: its
Hessian ``K'K`` decays away from the diagonal and is reproduced to the probe's tolerance at
half-bandwidth 27 of 59. With an evaluation budget that the dense build's 1770 remaining pairs do
not fit, ``'fixed'`` (the behaviour before Phase D) stops after two bands and the quasi-Newton path
runs; ``'decaying'``, the default, keeps going while the fit error at the probe's line points keeps
falling, builds the band the kernel needs and hands the SQP member an exact Hessian. A Hessian
without decay gets nothing from the option.
"""
import numpy as np
import pytest

import mincon

N = 60


def deconvolution(width=3.0):
    i = np.arange(N)
    K = np.exp(-0.5 * ((i[:, None] - i[None, :]) / width) ** 2)
    K /= K.sum(axis=1, keepdims=True)
    y = np.where((i >= N // 3) & (i < N // 3 + N // 4), 0.8, 0.1)
    return (lambda x: 0.5 * float(np.sum((K @ x - y) ** 2))), np.full(N, 0.5), [(0.0, 1.0)] * N


def probe_note(res):
    notes = [s for s in res.notes if "uadratic" in s]
    assert notes, res.notes
    return notes[0]


def test_fixed_declines_after_two_bands_when_the_dense_build_does_not_fit():
    f, x0, bounds = deconvolution()
    res = mincon.minimize(f, x0, bounds=bounds, options={"maxfev": 1700, "quadratic_bands": "fixed"})
    assert "within 2 off-diagonal bands" in probe_note(res)


def test_decaying_is_the_default_and_builds_the_band_the_kernel_needs():
    f, x0, bounds = deconvolution()
    res = mincon.minimize(f, x0, bounds=bounds, options={"maxfev": 1700})
    note = probe_note(res)
    assert "half-bandwidth 27" in note and "1362 objective evaluations" in note, note
    assert res.success, res.message
    assert res.nit <= 5
    # the same answer as the unbudgeted solve, which builds the same band on the way to the dense build
    ref = mincon.minimize(f, x0, bounds=bounds)
    assert "half-bandwidth 27" in probe_note(ref)
    assert abs(res.fun - ref.fun) <= 1e-9 * max(1.0, abs(ref.fun))


def test_a_hessian_without_decay_gets_nothing_from_the_option():
    # every off-diagonal entry equal: two bands remove 4 / N of the misfit, far from a quarter
    f = lambda x: float(np.sum(x * x) + 0.001 * np.sum(x) ** 2 - np.sum(x))  # noqa: E731
    kw = dict(bounds=[(0.0, 1.0)] * N)
    a = mincon.minimize(f, np.full(N, 0.5), options={"maxfev": 1700, "quadratic_bands": "fixed"}, **kw)
    b = mincon.minimize(f, np.full(N, 0.5), options={"maxfev": 1700, "quadratic_bands": "decaying"}, **kw)
    assert probe_note(a) == probe_note(b)
    assert "within 2 off-diagonal bands" in probe_note(a)
    assert a.nfev == b.nfev and np.array_equal(a.x, b.x)


def test_the_option_is_checked():
    f, x0, bounds = deconvolution()
    with pytest.raises(ValueError, match="quadratic_bands"):
        mincon.minimize(f, x0, bounds=bounds, options={"quadratic_bands": "wide"})
