"""I6 follow-up: should an early resume blend the handed-over quasi-Newton model toward an
identity? Measured, not assumed, and measured with no code change: `warm_start` accepts a mapping,
so a modified `hess_approx` is passed straight through.

Each friction problem is solved at defaults (`quadratic_probe` off, so the quasi-Newton path is
what is measured), interrupted by an evaluation budget at six fractions of that cost, and resumed
from the interrupted point under five arms:

    warm     the model as handed over (the shipped behaviour)
    blend_i  0.5 H + 0.5 I
    blend_s  0.5 H + 0.5 (tr H / n) I      (shape halved, scale kept)
    scale    (tr H / n) I                  (shape dropped, scale kept)
    cold     no model at all

Each record also carries the interrupted run's iteration count and statistics of the model it
handed over -- eigenvalue range, condition number, and how far it is from the scaled identity --
so that a rule keyed on any of them can be looked for. Totals are the interrupted run's model
evaluations plus the resumed run's; every run is judged against the frozen reference.

    python warm_start_blend.py [problem ...]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import problems  # noqa: E402
import mincon  # noqa: E402

NAMES = ["chainrosen20", "box_lsq", "portfolio_risk", "pressure_vessel", "hs71", "odefit",
         "infeasible_start_far", "linear_only", "equality_circle", "with_args"]
FRACTIONS = (0.15, 0.25, 0.33, 0.5, 0.66, 0.8)
ARMS = ("warm", "blend_i", "blend_s", "scale", "cold")


def model_stats(H):
    """Eigenvalue range, condition number and distance from the scaled identity."""
    n = H.shape[0]
    S = 0.5 * (H + H.T)
    w = np.linalg.eigvalsh(S)
    s = float(np.trace(S) / n)
    return dict(n=n, tr_over_n=s, lmin=float(w.min()), lmax=float(w.max()),
                cond=float(w.max() / w.min()) if w.min() > 0 else float("inf"),
                aniso=float(np.linalg.norm(S - s * np.eye(n)) / max(np.linalg.norm(S), 1e-300)))


def main(names):
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

        def attained(r):
            return abs(r.fun - tgt) <= 1e-4 * max(1.0, abs(tgt)) and r.maxcv <= 1e-6

        for frac in FRACTIONS:
            budget = max(5, int(full.nfev * frac))
            cut = mincon.fmincon(p.fun, p.x0, options=dict(opts, maxfev=budget), **kw)
            rec = dict(problem=name, frac=frac, full=int(full.nfev), full_att=bool(attained(full)),
                       cut_nfev=int(cut.nfev), cut_nit=int(cut.nit or 0),
                       have_H=cut.hess_approx is not None)
            if not rec["have_H"]:
                rows.append(rec)
                continue
            H = np.asarray(cut.hess_approx, dtype=float)
            rec.update(model_stats(H))
            n, s = H.shape[0], rec["tr_over_n"]
            for arm, M in (("warm", H),
                           ("blend_i", 0.5 * H + 0.5 * np.eye(n)),
                           ("blend_s", 0.5 * H + 0.5 * s * np.eye(n)),
                           ("scale", s * np.eye(n))):
                ws = dict(cut)
                ws["hess_approx"] = M
                r = mincon.fmincon(p.fun, cut.x, options=dict(opts), warm_start=ws, **kw)
                rec[arm] = int(cut.nfev + r.nfev)
                rec[arm + "_att"] = bool(attained(r))
            cold = mincon.fmincon(p.fun, cut.x, options=dict(opts), **kw)
            rec["cold"] = int(cut.nfev + cold.nfev)
            rec["cold_att"] = bool(attained(cold))
            rows.append(rec)
            print(f'{name:22s} cut@{frac:4.2f} nit={rec["cut_nit"]:4d}  '
                  + "  ".join(f'{a} {rec[a]:6d}{"" if rec[a + "_att"] else "!"}' for a in ARMS))

    out = os.path.join(HERE, "..", "results", "i6-warmstart", "blend.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)
    print("wrote", os.path.normpath(out))


if __name__ == "__main__":
    main(sys.argv[1:] or NAMES)
