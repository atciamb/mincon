"""I6 warm-start study: interrupt a solve at a fraction of its evaluations, then resume warm
(x0 = res.x, warm_start = res) and cold (x0 = res.x only). Friction problems, defaults."""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import problems  # noqa: E402
import mincon  # noqa: E402

names = sys.argv[1:] or ["chainrosen20", "box_lsq", "portfolio_risk", "pressure_vessel", "hs71", "odefit",
                          "infeasible_start_far", "linear_only", "equality_circle", "with_args"]
fractions = (0.33, 0.66)
rows = []
for name in names:
    p = problems.REGISTRY[name]()
    kw = {}
    for key in ("A", "b", "Aeq", "beq", "lb", "ub", "nonlcon"):
        v = getattr(p, key, None)
        if v is not None:
            kw[key] = v
    if getattr(p, "args", None):
        kw["args"] = p.args
    opts = {"quadratic_probe": False}
    full = mincon.fmincon(p.fun, p.x0, options=dict(opts), **kw)
    tgt = p.target
    def att(r):
        return abs(r.fun - tgt) <= 1e-4 * max(1.0, abs(tgt)) and r.maxcv <= 1e-6

    for frac in fractions:
        budget = max(5, int(full.nfev * frac))
        cut = mincon.fmincon(p.fun, p.x0, options=dict(opts, maxfev=budget), **kw)
        warm = mincon.fmincon(p.fun, cut.x, options=dict(opts), warm_start=cut, **kw)
        cold = mincon.fmincon(p.fun, cut.x, options=dict(opts), **kw)
        rows.append(dict(problem=name, frac=frac, full=full.nfev, full_att=att(full), cut=cut.nfev, cut_success=cut.success,
                         warm_total=cut.nfev + warm.nfev, warm_att=att(warm), warm_success=warm.success,
                         cold_total=cut.nfev + cold.nfev, cold_att=att(cold), cold_success=cold.success))
        print(f"{name:22s} cut@{frac:.2f}: full {full.nfev:6d} ({'Y' if att(full) else 'n'}) | "
              f"warm {cut.nfev:5d}+{warm.nfev:5d}={cut.nfev + warm.nfev:6d} ({'Y' if att(warm) else 'n'}{'' if warm.success else '?'}) | "
              f"cold {cut.nfev:5d}+{cold.nfev:5d}={cut.nfev + cold.nfev:6d} ({'Y' if att(cold) else 'n'}{'' if cold.success else '?'})")
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "i6-warmstart", "measure.json")
json.dump(rows, open(out, "w"), indent=1)
