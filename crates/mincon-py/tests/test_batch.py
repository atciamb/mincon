"""Batched finite-difference probes: ``workers=`` (the probes of a gradient on a pool) and
``vectorized=`` (the probes of a gradient in one call). The contract of both: the probes are the
points of the serial solve, so ``workers`` changes nothing but the wall time, and a model that
asks for neither is solved exactly as before."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import mincon

N = 12
X0 = np.array([-1.2 if i % 2 == 0 else 1. for i in range(N)])


def chain(x):
    """Chained Rosenbrock; module level, so a worker process can import it."""
    return float(np.sum(100. * (x[1:] - x[:-1] ** 2) ** 2 + (1. - x[:-1]) ** 2))


def chain_rows(X):
    """The same objective for a (k, n) array of points."""
    return np.sum(100. * (X[:, 1:] - X[:, :-1] ** 2) ** 2 + (1. - X[:, :-1]) ** 2, axis=1)


def ball(x):
    return [x @ x - 4.], []


def ball_rows(X):
    return (np.sum(X * X, axis=1) - 4.)[:, None], None


def walled(x, wall):
    """Undefined past ``wall`` in the first coordinate: probes there must retreat."""
    if x[0] > wall:
        raise ValueError("outside the model's domain")
    return float((x[0] - 2.) ** 2 + (x[1] + 1.) ** 2)


def same_solve(a, b):
    assert np.array_equal(a.x, b.x), (a.x, b.x)
    assert (a.fun, a.nit, a.nfev, a.status) == (b.fun, b.nit, b.nfev, b.status)
    assert [t["f"] for t in a.trace] == [t["f"] for t in b.trace]
    assert np.array_equal(a["lambda"], b["lambda"])


def test_worker_processes_take_the_iterates_of_the_serial_solve():
    kw = dict(lb=-2., ub=2., nonlcon=ball)
    serial = mincon.fmincon(chain, X0, **kw)
    assert serial.success
    pooled = mincon.fmincon(chain, X0, workers=2, **kw)
    same_solve(serial, pooled)
    assert any("2 worker processes" in note for note in pooled.notes), pooled.notes
    # the option dict is the other door, as in SciPy
    same_solve(serial, mincon.fmincon(chain, X0, options={"workers": 2}, **kw))
    with pytest.raises(ValueError, match="not both"):
        mincon.fmincon(chain, X0, workers=2, options={"workers": 2}, **kw)


def test_a_model_that_cannot_be_sent_to_a_worker_says_so():
    with pytest.raises(ValueError, match="picklable.*lambda"):
        mincon.fmincon(lambda x: chain(x), X0, workers=2)
    # MATLAB's switch is the same thing
    with pytest.raises(ValueError, match="picklable"):
        mincon.fmincon(lambda x: chain(x), X0, options={"UseParallel": True})
    assert mincon.fmincon(lambda x: chain(x), X0, options={"UseParallel": False}).success
    for bad in (0, -3, 2.5, True, "many"):
        with pytest.raises((ValueError, TypeError), match="workers"):
            mincon.fmincon(chain, X0, workers=bad)


def test_constraints_that_cannot_be_sent_stay_in_this_process_with_a_note():
    kw = dict(lb=-2., ub=2.)
    serial = mincon.fmincon(chain, X0, nonlcon=lambda x: ([x @ x - 4.], []), **kw)
    pooled = mincon.fmincon(chain, X0, nonlcon=lambda x: ([x @ x - 4.], []), workers=2, **kw)
    same_solve(serial, pooled)
    assert any("cannot be sent to a worker" in note for note in pooled.notes), pooled.notes


def test_a_map_like_callable_is_used_as_given_and_takes_a_lambda():
    kw = dict(lb=-2., ub=2., nonlcon=lambda x: ([x @ x - 4.], []))
    serial = mincon.fmincon(lambda x: chain(x), X0, **kw)
    seen = []

    with ThreadPoolExecutor(4) as pool:
        def mapper(f, points):
            points = list(points)
            seen.append(len(points))
            return pool.map(f, points)
        threaded = mincon.fmincon(lambda x: chain(x), X0, workers=mapper, **kw)
    same_solve(serial, threaded)
    assert max(seen) == N and sum(seen) > serial.nfev // 2


def test_a_map_that_fails_stops_the_solve_with_its_reason():
    calls = {"n": 0}

    def counted(x):
        calls["n"] += 1
        return chain(x)

    def broken(f, points):
        raise OSError("the cluster is down")

    with pytest.raises(RuntimeError, match="evaluating a batch of points failed.*cluster is down"):
        mincon.fmincon(counted, X0, workers=broken)
    assert calls["n"] < 10, "the model kept being called after the workers were lost"


def test_an_exception_on_a_worker_is_one_failed_probe():
    # The start sits just inside the wall, so the forward probe in x[0] lands outside it and
    # has to retreat; the serial solve and the pooled one must retreat alike.
    x0 = np.array([1.0 - 1e-9, 0.])
    serial = mincon.minimize(walled, x0, args=(1.0,), bounds=[(None, 1.0), (None, None)])
    with ThreadPoolExecutor(2) as pool:
        threaded = mincon.minimize(walled, x0, args=(1.0,), bounds=[(None, 1.0), (None, None)], workers=pool.map)
    pooled = mincon.minimize(walled, x0, args=(1.0,), bounds=[(None, 1.0), (None, None)], workers=2)
    same_solve(serial, threaded)
    same_solve(serial, pooled)
    assert serial.usable and abs(serial.x[0] - 1.) < 1e-6


def test_a_vectorised_model_crosses_into_python_once_per_gradient():
    # Fifty variables, not the module's twelve: a gradient is one crossing and a line-search point
    # is one too, so at n = 12 the ratio is between 4 and 6 and which side of the gate's 5 it falls
    # on depends on the path (4.6 on CI's Linux runner); at n = 50 it is about 18.
    n = 50
    x0 = np.array([-1.2 if i % 2 == 0 else 1. for i in range(n)])
    calls = {"scalar": 0, "rows": 0, "points": 0}

    def counted(x):
        calls["scalar"] += 1
        return chain(x)

    def counted_rows(X):
        assert X.ndim == 2 and X.shape[1] == n
        calls["rows"] += 1
        calls["points"] += X.shape[0]
        return chain_rows(X)

    serial = mincon.fmincon(counted, x0, lb=-2., ub=2.)
    fast = mincon.fmincon(counted_rows, x0, lb=-2., ub=2., vectorized=True)
    assert serial.success and fast.success
    assert abs(fast.fun - serial.fun) < 1e-8 and np.allclose(fast.x, serial.x, atol=1e-5)
    # the model's own counters (at this size two portfolio members run, and `nfev` is the winner's):
    # the same evaluations either way, in far fewer calls
    assert calls["points"] == calls["scalar"], calls
    assert 5 * calls["rows"] <= calls["scalar"], calls
    assert any("batches" in note for note in fast.notes)


def test_a_vectorised_nonlcon_and_a_vectorised_constraint_block():
    serial = mincon.fmincon(chain, X0, lb=-2., ub=2., nonlcon=ball)
    both = mincon.fmincon(chain_rows, X0, lb=-2., ub=2., nonlcon=ball_rows, vectorized=True)
    assert both.success and abs(both.fun - serial.fun) < 1e-7
    assert both.multipliers.ineqnonlin.shape == (1,)
    only_nonlcon = mincon.fmincon(chain, X0, lb=-2., ub=2., nonlcon=ball_rows, vectorized="nonlcon")
    assert only_nonlcon.success and abs(only_nonlcon.fun - serial.fun) < 1e-7
    block = mincon.minimize(chain, X0, bounds=[(-2., 2.)] * N,
                            constraints=[{"type": "ineq", "fun": lambda X: 4. - np.sum(X * X, axis=1),
                                          "vectorized": True}])
    assert block.success and abs(block.fun - serial.fun) < 1e-7


def test_a_model_that_is_not_vectorised_is_refused_not_misread():
    # np.sum over everything returns one number for k points
    with pytest.raises(ValueError, match="vectorized.*shape"):
        mincon.fmincon(lambda X: np.sum((X - 1.) ** 2), X0, vectorized=True)
    # a scalar model indexes coordinates, which are rows of a (1, n) array
    with pytest.raises((ValueError, RuntimeError, IndexError)):
        mincon.fmincon(lambda x: (x[0] - 1.) ** 2 + (x[1] - 2.) ** 2, [0., 0.], vectorized=True)
    with pytest.raises(ValueError, match="vectorized"):
        mincon.fmincon(chain_rows, X0, nonlcon=lambda X: (np.sum(X * X) - 4., None), vectorized=True)
    with pytest.raises(ValueError, match="not both"):
        mincon.fmincon(chain_rows, X0, vectorized=True, workers=2)
    with pytest.raises(ValueError, match="vectorized must be"):
        mincon.fmincon(chain_rows, X0, vectorized="yes")
